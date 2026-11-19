# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.15.0", "numpy==2.4.6"]
# ///
"""Generate PyTorch reference fixtures (CONTRACTS.md section 11).

Run with `pnpm torch-ref` (= `uv run tools/torch-ref/gen.py`). Output goes to
packages/engine/test/fixtures/<name>.json. Everything is float64 and seeded, so
a second run is byte-identical. Never edit the JSON by hand; change this file.

Conventions (fixed by CONTRACTS.md):
- params are written in HostParams layout: W row-major [in, out] (torch's
  nn.Linear.weight is [out, in] and is transposed on the way in and out).
- init is nn.Linear style: W and b uniform on +-1/sqrt(in).
- the objective is mean loss over the batch + lambda * R(W), with R summed over
  Linear weights only (explicit term, biases excluded).
- pred is post-link, [n, width] row-major.
- dropout: expect.dropoutMask is a 0/1 keep mask [n, width]; the output of the
  dropout node is x * mask / (1 - p).
- feature node output layout (see CONTRACTS.md section 3): the concatenation of
  x^p for p = 1..degree (power-major, then column), then sin(x) if sin, then
  cos(x) if cos, then x_i * x_j for i < j in row-major pair order if products.
- classification targets are class indices stored as floats (0 or 1).
- cases named prim_* are graph version 2 (CONTRACTS.md section 2): `predictions` and `loss` are pins,
  losses are built from primitives (a log loss takes probabilities, so a sigmoid link in front is
  evaluated literally here and fused by the engine). `expect.pred` is the first prediction pin
  (empty with none) and `expect.predictions` has every pin in order.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent.parent.parent / "packages/engine/test/fixtures"


# ---------- graph building ----------

def inp(d):
    return {"id": "x", "type": "input", "columns": [f"x{i + 1}" for i in range(d)]}


def lin(id, out, bias=True):
    return {"id": id, "type": "linear", "out": out, "bias": bias}


def act(id, fn):
    return {"id": id, "type": "activation", "fn": fn}


def outp(link):
    return {"id": "yhat", "type": "output", "link": link}


def loss(fn, delta=None, penalty=None):
    n = {"id": "loss", "type": "loss", "fn": fn}
    if fn == "huber":
        n["delta"] = delta if delta is not None else 1.0
    if penalty:
        n["penalty"] = penalty
    return n


TARGET = {"id": "y", "type": "target"}


def chain_edges(ids):
    return [[a, b] for a, b in zip(ids, ids[1:])]


def mlp(d, hidden, fn, link="identity", loss_node=None, bias=True, out=1):
    nodes = [inp(d)]
    ids = ["x"]
    for i, h in enumerate(hidden, 1):
        nodes += [lin(f"lin{i}", h, bias), act(f"act{i}", fn)]
        ids += [f"lin{i}", f"act{i}"]
    k = len(hidden) + 1
    nodes += [lin(f"lin{k}", out, bias), outp(link), TARGET, loss_node or loss("squared_error")]
    ids += [f"lin{k}", "yhat"]
    edges = chain_edges(ids) + [["yhat", "loss.pred"], ["y", "loss.target"]]
    return nodes, edges


def widths(nodes, edges):
    """Output width per node, by CONTRACTS section 3 rules."""
    src = {}
    for a, b in edges:
        a = a.split(".")[0]
        dst, _, port = b.partition(".")
        src[(dst, port or "in")] = a
    w = {}
    for n in nodes:
        t, i = n["type"], n["id"]
        if t == "input":
            w[i] = len(n["columns"])
        elif t == "target":
            w[i] = 1
        elif t == "loss":
            pass
        elif t == "add":
            w[i] = w[src[(i, "a")]]
        else:
            wi = w[src[(i, "in")]]
            if t == "feature":
                w[i] = (
                    wi * n["degree"]
                    + (wi if n["sin"] else 0)
                    + (wi if n["cos"] else 0)
                    + (wi * (wi - 1) // 2 if n["products"] else 0)
                )
            elif t == "linear":
                w[i] = n["out"]
            else:
                w[i] = wi
    return w, src


# ---------- torch evaluator ----------

class Model(nn.Module):
    def __init__(self, nodes, edges, params, dropout_mask=None):
        super().__init__()
        self.nodes, self.edges = nodes, edges
        self.w, self.src = widths(nodes, edges)
        self.mask = dropout_mask
        self.lins = nn.ModuleDict()
        for n in nodes:
            if n["type"] == "linear":
                i = n["id"]
                inw = self.w[self.src[(i, "in")]]
                m = nn.Linear(inw, n["out"], bias=n["bias"])
                # HostParams [in, out] -> torch weight [out, in]
                W = torch.tensor(np.array(params[i]["W"]).reshape(inw, n["out"]).T.copy())
                m.weight.data = W.contiguous()
                if n["bias"]:
                    m.bias.data = torch.tensor(params[i]["b"])
                self.lins[i] = m

    def forward(self, x, y):
        """Returns (pred post-link, objective)."""
        v, z, link, obj = {}, None, None, None
        for n in self.nodes:
            t, i = n["type"], n["id"]

            def get(port="in"):
                return v[self.src[(i, port)]]

            if t == "input":
                v[i] = x
            elif t == "target":
                v[i] = y
            elif t == "linear":
                v[i] = self.lins[i](get())
            elif t == "feature":
                a = get()
                parts = [a**p for p in range(1, n["degree"] + 1)]
                if n["sin"]:
                    parts.append(torch.sin(a))
                if n["cos"]:
                    parts.append(torch.cos(a))
                if n["products"]:
                    d = a.shape[1]
                    parts += [(a[:, p] * a[:, q]).unsqueeze(1) for p in range(d) for q in range(p + 1, d)]
                v[i] = torch.cat(parts, dim=1)
            elif t == "activation":
                f = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu,
                     "gelu": nn.GELU()}[n["fn"]]  # nn.GELU() default = exact erf form
                v[i] = f(get())
            elif t == "dropout":
                # mask None = eval mode (dropout off)
                v[i] = get() if self.mask is None else get() * self.mask / (1.0 - n["p"])
            elif t == "add":
                v[i] = get("a") + get("b")
            elif t == "output":
                z = get()
                link = n["link"]
                v[i] = {"identity": z, "sigmoid": torch.sigmoid(z),
                        "softmax": torch.softmax(z, dim=1)}[link]
            elif t == "loss":
                fn = n["fn"]
                if fn == "log_loss" and link == "sigmoid":
                    L = nn.BCEWithLogitsLoss()(z, y)
                elif fn == "cross_entropy" and link == "softmax":
                    L = nn.CrossEntropyLoss()(z, y.squeeze(1).long())
                else:
                    p = get("pred")
                    if fn == "squared_error":
                        L = nn.MSELoss()(p, y)
                    elif fn == "absolute_error":
                        L = nn.L1Loss()(p, y)
                    elif fn == "huber":
                        L = nn.HuberLoss(delta=n["delta"])(p, y)
                    elif fn == "log_loss":
                        L = F.binary_cross_entropy(p, y)
                    else:
                        raise ValueError(fn)
                pen = n.get("penalty")
                if pen and pen["lambda"] > 0:
                    ws = [m.weight for m in self.lins.values()]
                    R = sum((w**2).sum() if pen["kind"] == "l2" else w.abs().sum() for w in ws)
                    L = L + pen["lambda"] * R
                obj = L
        return v["yhat"], obj

    def host(self, what="params"):
        out = {}
        for i, m in self.lins.items():
            t = m.weight if what == "params" else m.weight.grad
            e = {"W": t.detach().T.contiguous().flatten().tolist()}
            if m.bias is not None:
                tb = m.bias if what == "params" else m.bias.grad
                e["b"] = tb.detach().flatten().tolist()
            out[i] = e
        return out


# ---------- case assembly ----------

def init_params(rng, nodes, edges):
    w, src = widths(nodes, edges)
    p = {}
    for n in nodes:
        if n["type"] == "linear":
            inw, o = w[src[(n["id"], "in")]], n["out"]
            s = 1.0 / math.sqrt(inw)
            e = {"W": rng.uniform(-s, s, size=inw * o).tolist()}
            if n["bias"]:
                e["b"] = rng.uniform(-s, s, size=o).tolist()
            p[n["id"]] = e
    return p


def make_case(name, seed, nodes, edges, *, n=6, d=1, ytype="reg",
              optimizer=None, steps=5, lr=0.1, mask_for=None, xscale=1.5,
              zero_weights=(), zero_x_rows=()):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-xscale, xscale, size=(n, d))
    if ytype == "reg":
        y = 0.7 * x[:, :1] + 0.3 * np.sin(2 * x[:, :1]) + 0.4 * rng.standard_normal((n, 1))
        y = y * 2.0  # spread so huber sees both regimes
    else:  # class index as float, 0/1; guarantee both classes
        y = (rng.uniform(size=(n, 1)) > 0.5).astype(float)
        y[0, 0], y[1, 0] = 0.0, 1.0
    for r in zero_x_rows:  # exact zeros in the input row -> exact zero pre-activations (bias-free)
        x[r, :] = 0.0
    params = init_params(rng, nodes, edges)
    for node, idx in zero_weights:  # exact zero weights (for sign(0) = 0 in the L1 penalty)
        params[node]["W"][idx] = 0.0
    mask = None
    if mask_for:
        dn = next(k for k in nodes if k["type"] == "dropout")
        mw = widths(nodes, edges)[0][dn["id"]]
        mask = (rng.uniform(size=(n, mw)) > dn["p"]).astype(float)
        assert 0 < mask.sum() < mask.size
    xt, yt = torch.tensor(x), torch.tensor(y)
    maskt = torch.tensor(mask) if mask is not None else None

    m = Model(nodes, edges, params, maskt)
    pred, obj = m(xt, yt)
    obj.backward()
    expect = {"pred": pred.detach().flatten().tolist(), "loss": obj.item(), "grads": m.host("grads")}
    if mask is not None:
        expect["dropoutMask"] = mask.tolist()
        # dropout-off (eval) forward: what predict()/evaluate() must return
        pe, oe = Model(nodes, edges, params, None)(xt, yt)
        expect["predEval"] = pe.detach().flatten().tolist()
        expect["lossEval"] = oe.item()

    if optimizer:
        m2 = Model(nodes, edges, params, maskt)
        opt = {"sgd": lambda ps: torch.optim.SGD(ps, lr=lr),
               "momentum": lambda ps: torch.optim.SGD(ps, lr=lr, momentum=0.9),
               "adam": lambda ps: torch.optim.Adam(ps, lr=lr)}[optimizer](m2.parameters())
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            _, o = m2(xt, yt)
            losses.append(o.item())  # loss before the update
            o.backward()
            opt.step()
        expect["after"] = {"steps": steps, "params": m2.host("params"), "losses": losses}

    # data/train blocks are placeholders (n=40 satisfies parseGraph, n >= 10); the batch above is explicit.
    graph = {
        "version": 1, "nodes": nodes, "edges": edges,
        "data": {"set": "blobs" if ytype == "cls" else {1: "sine", 2: "plane"}[d],
                 "n": 40, "noise": 0.1, "val": 0.25, "seed": 1, "standardize": False},
        "train": {"optimizer": optimizer or "sgd", "lr": lr, "batch": "full", "seed": 1},
    }
    fx = {"name": name, "graph": graph, "params": params,
          "batch": {"x": x.flatten().tolist(), "y": y.flatten().tolist(), "n": n}, "expect": expect}
    return fx, y


def write(fx):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{fx['name']}.json").write_text(json.dumps(fx, indent=1) + "\n")


# ---------- graph v2 cases (arbitrary DAG with pins; CONTRACTS.md section 2) ----------
#
# The v2 evaluator walks the graph literally, primitive by primitive, in float64. Losses are built
# from primitives (subtract, square, ..., log_loss on probabilities, mean), so the semantic forms
# here are compared with the engine's fused, numerically stable forms.

def n_(id, type, **kw):
    return {"id": id, "type": type, **kw}


def src_of(edges):
    """(node, input port) -> source node id. A bare destination is the port `in`."""
    s = {}
    for a, b in edges:
        dst, _, port = b.partition(".")
        s[(dst, port or "in")] = a.split(".")[0]
    return s


def widths_v2(nodes, edges, d):
    """Example width per node (None for scalars and weight matrices)."""
    src = src_of(edges)
    w = {}
    for n in nodes:
        t, i = n["type"], n["id"]
        if t == "input":
            w[i] = len(n["columns"])
        elif t == "target":
            w[i] = 1
        elif t == "column":
            w[i] = 1
        elif t in ("weights", "mean", "norm"):
            w[i] = None
        elif t in ("add", "subtract"):
            w[i] = w[src[(i, "a")]]
        elif t in ("log_loss", "cross_entropy"):
            w[i] = 1
        else:
            wi = w[src[(i, "in")]]
            if t == "feature":
                w[i] = (wi * n["degree"] + (wi if n["sin"] else 0) + (wi if n["cos"] else 0)
                        + (wi * (wi - 1) // 2 if n["products"] else 0))
            elif t == "linear":
                w[i] = n["out"]
            else:
                w[i] = wi
    return w, src


class ModelV2(nn.Module):
    def __init__(self, nodes, edges, params, w, src, dropout_mask=None):
        super().__init__()
        self.nodes, self.src, self.mask = nodes, src, dropout_mask
        self.nodes_by_id = {n["id"]: n for n in nodes}
        # every wire into a node's input port, in edge order (a Sum takes several into `in`)
        self.wires = {}
        for a, b in edges:
            dst, _, port = b.partition(".")
            self.wires.setdefault((dst, port or "in"), []).append(a.split(".")[0])
        # scalar parameters (scalew W [1, 1], bias b [1]) are plain tensors, in HostParams layout
        self.scal = nn.ParameterDict()
        for n in nodes:
            if n["type"] == "scalew":
                self.scal[n["id"]] = nn.Parameter(torch.tensor(params[n["id"]]["W"]).reshape(1, 1))
            elif n["type"] == "bias":
                self.scal[n["id"]] = nn.Parameter(torch.tensor(params[n["id"]]["b"]))
        self.lins = nn.ModuleDict()
        for n in nodes:
            if n["type"] == "linear":
                i = n["id"]
                inw = w[src[(i, "in")]]
                m = nn.Linear(inw, n["out"], bias=n["bias"])
                m.weight.data = torch.tensor(
                    np.array(params[i]["W"]).reshape(inw, n["out"]).T.copy()).contiguous()
                if n["bias"]:
                    m.bias.data = torch.tensor(params[i]["b"])
                self.lins[i] = m

    def forward(self, x, y, pins, loss_id):
        v = {}
        for n in self.nodes:
            t, i = n["type"], n["id"]

            def get(port="in"):
                return v[self.src[(i, port)]]

            if t == "input":
                v[i] = x
            elif t == "target":
                v[i] = y
            elif t == "linear":
                v[i] = self.lins[i](get())
            elif t == "column":
                v[i] = x[:, [int(n["name"][1:]) - 1]]
            elif t == "scalew":
                v[i] = get() * self.scal[i]
            elif t == "bias":
                v[i] = get() + self.scal[i]
            elif t == "sum":
                v[i] = torch.stack([v[a] for a in self.wires[(i, "in")]]).sum(0)
            elif t == "weights":
                v[i] = self.lins[n["of"]].weight.T  # W in HostParams layout [in, out]
            elif t == "norm":
                a = get()
                v[i] = (a ** 2).sum() if n["kind"] == "l2" else a.abs().sum()
            elif t == "feature":
                a = get()
                parts = [a ** p for p in range(1, n["degree"] + 1)]
                if n["sin"]:
                    parts.append(torch.sin(a))
                if n["cos"]:
                    parts.append(torch.cos(a))
                if n["products"]:
                    dd = a.shape[1]
                    parts += [(a[:, p] * a[:, q]).unsqueeze(1) for p in range(dd) for q in range(p + 1, dd)]
                v[i] = torch.cat(parts, dim=1)
            elif t == "activation":
                f = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu,
                     "gelu": nn.GELU()}[n["fn"]]
                v[i] = f(get())
            elif t == "dropout":
                v[i] = get() if self.mask is None else get() * self.mask / (1.0 - n["p"])
            elif t == "add":
                v[i] = get("a") + get("b")
            elif t == "subtract":
                v[i] = get("a") - get("b")
            elif t == "link":
                z = get()
                v[i] = torch.sigmoid(z) if n["fn"] == "sigmoid" else torch.softmax(z, dim=1)
            elif t == "square":
                v[i] = get() ** 2
            elif t == "abs":
                v[i] = get().abs()
            elif t == "huber":
                d_ = get()
                v[i] = F.huber_loss(d_, torch.zeros_like(d_), reduction="none", delta=n["delta"])
            elif t == "log_loss":
                v[i] = F.binary_cross_entropy(get("pred"), get("target"), reduction="none")
            elif t == "cross_entropy":
                v[i] = -torch.log(get("pred").gather(1, get("target").long()))
            elif t == "mean":
                v[i] = get().mean()
            elif t == "scale":
                v[i] = get() * n["c"]
            else:
                raise ValueError(t)
        return [v[p] for p in pins], v[loss_id]

    def host(self, what="params"):
        out = {}
        for i, m in self.lins.items():
            t = m.weight if what == "params" else m.weight.grad
            if t is None:  # only possible for grads of a never-used tensor: zero
                t = torch.zeros_like(m.weight)
            e = {"W": t.detach().T.contiguous().flatten().tolist()}
            if m.bias is not None:
                tb = m.bias if what == "params" else m.bias.grad
                if tb is None:
                    tb = torch.zeros_like(m.bias)
                e["b"] = tb.detach().flatten().tolist()
            out[i] = e
        for i, p in self.scal.items():
            t = p if what == "params" else p.grad
            if t is None:
                t = torch.zeros_like(p)
            key = "W" if self.nodes_by_id[i]["type"] == "scalew" else "b"
            out[i] = {key: t.detach().flatten().tolist()}
        return out


def make_case_v2(name, seed, nodes, edges, predictions, loss_id, *, n=6, d=1, ytype="reg",
                 optimizer=None, steps=5, lr=0.1, mask_for=None, xscale=1.5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-xscale, xscale, size=(n, d))
    if ytype == "reg":
        y = 0.7 * x[:, :1] + 0.3 * np.sin(2 * x[:, :1]) + 0.4 * rng.standard_normal((n, 1))
        y = y * 2.0
    else:
        y = (rng.uniform(size=(n, 1)) > 0.5).astype(float)
        y[0, 0], y[1, 0] = 0.0, 1.0
    w, src = widths_v2(nodes, edges, d)
    params = {}
    for nd in nodes:
        if nd["type"] == "linear":
            inw, o = w[src[(nd["id"], "in")]], nd["out"]
            s = 1.0 / math.sqrt(inw)
            e = {"W": rng.uniform(-s, s, size=inw * o).tolist()}
            if nd["bias"]:
                e["b"] = rng.uniform(-s, s, size=o).tolist()
            params[nd["id"]] = e
        elif nd["type"] in ("scalew", "bias"):
            # scalar models start at the scale of nn.Linear(d, 1): uniform on +-1/sqrt(d)
            sc = 1.0 / math.sqrt(max(1, d))
            key = "W" if nd["type"] == "scalew" else "b"
            params[nd["id"]] = {key: rng.uniform(-sc, sc, size=1).tolist()}
    mask = None
    if mask_for:
        dn = next(k for k in nodes if k["type"] == "dropout")
        mask = (rng.uniform(size=(n, w[dn["id"]])) > dn["p"]).astype(float)
        assert 0 < mask.sum() < mask.size
    xt, yt = torch.tensor(x), torch.tensor(y)
    maskt = torch.tensor(mask) if mask is not None else None
    pin_nodes = [p.split(".")[0] for p in predictions]
    loss_node = loss_id.split(".")[0]

    m = ModelV2(nodes, edges, params, w, src, maskt)
    preds, obj = m(xt, yt, pin_nodes, loss_node)
    assert obj.dim() == 0
    obj.backward()
    flat = [p.detach().flatten().tolist() for p in preds]
    expect = {"pred": flat[0] if flat else [], "predictions": flat, "loss": obj.item(),
              "grads": m.host("grads")}
    if mask is not None:
        expect["dropoutMask"] = mask.tolist()
        pe, oe = ModelV2(nodes, edges, params, w, src, None)(xt, yt, pin_nodes, loss_node)
        expect["predEval"] = pe[0].detach().flatten().tolist() if pe else []
        expect["lossEval"] = oe.item()
    if optimizer:
        m2 = ModelV2(nodes, edges, params, w, src, maskt)
        opt = {"sgd": lambda ps: torch.optim.SGD(ps, lr=lr),
               "momentum": lambda ps: torch.optim.SGD(ps, lr=lr, momentum=0.9),
               "adam": lambda ps: torch.optim.Adam(ps, lr=lr)}[optimizer](m2.parameters())
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            _, o = m2(xt, yt, pin_nodes, loss_node)
            losses.append(o.item())
            o.backward()
            opt.step()
        expect["after"] = {"steps": steps, "params": m2.host("params"), "losses": losses}

    graph = {
        "version": 2, "nodes": nodes, "edges": edges,
        "predictions": predictions, "loss": loss_id,
        "data": {"set": "blobs" if ytype == "cls" else {1: "sine", 2: "plane", 3: "plane"}[d],
                 "n": 40, "noise": 0.1, "val": 0.25, "seed": 1, "standardize": False},
        "train": {"optimizer": optimizer or "sgd", "lr": lr, "batch": "full", "seed": 1},
    }
    return {"name": name, "graph": graph, "params": params,
            "batch": {"x": x.flatten().tolist(), "y": y.flatten().tolist(), "n": n}, "expect": expect}


def v2_cases(add):
    """add(name, nodes, edges, predictions, loss, **kw)"""
    # regression stem: x -> lin1 -> act1 -> lin2 (the prediction), target y
    stem = [inp(2), lin("lin1", 3), act("act1", "tanh"), lin("lin2", 1), TARGET]
    stem_e = [["x", "lin1"], ["lin1", "act1"], ["act1", "lin2"]]

    def with_loss(extra_nodes, extra_edges, **kw):
        return stem + extra_nodes, stem_e + extra_edges

    # subtract -> square -> mean (the squared-error loss from primitives)
    nodes, edges = with_loss(
        [n_("diff", "subtract"), n_("err", "square"), n_("loss", "mean")],
        [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "loss"]])
    add("prim_square_mean", nodes, edges, ["lin2"], "loss", d=2)

    # abs
    nodes, edges = with_loss(
        [n_("diff", "subtract"), n_("err", "abs"), n_("loss", "mean")],
        [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "loss"]])
    add("prim_abs_mean", nodes, edges, ["lin2"], "loss", d=2)

    # huber with a small delta (n=8 so both regimes occur), checked inline below
    nodes, edges = with_loss(
        [n_("diff", "subtract"), n_("err", "huber", delta=0.5), n_("loss", "mean")],
        [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "loss"]])
    add("prim_huber_mean", nodes, edges, ["lin2"], "loss", d=2, n=8)

    # scale (negative c) and add of two scalars: L = mean(sq) + (-0.25) * mean(abs)
    nodes, edges = with_loss(
        [n_("diff", "subtract"), n_("sq", "square"), n_("ab", "abs"), n_("m1", "mean"),
         n_("m2", "mean"), n_("sc", "scale", c=-0.25), n_("loss", "add")],
        [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "sq"], ["diff", "ab"], ["sq", "m1"],
         ["ab", "m2"], ["m2", "sc"], ["m1", "loss.a"], ["sc", "loss.b"]])
    add("prim_scale_add", nodes, edges, ["lin2"], "loss", d=2)

    # Weights tap + norm: L2 and L1 penalties on two Linears, summed, scaled, added to the loss
    for kind in ("l2", "l1"):
        nodes, edges = with_loss(
            [n_("diff", "subtract"), n_("err", "square"), n_("mean1", "mean"),
             n_("w1", "weights", of="lin1"), n_("n1", "norm", kind=kind),
             n_("w2", "weights", of="lin2"), n_("n2", "norm", kind=kind),
             n_("both", "add"), n_("pen", "scale", c=0.1), n_("loss", "add")],
            [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "mean1"],
             ["w1", "n1"], ["w2", "n2"], ["n1", "both.a"], ["n2", "both.b"], ["both", "pen"],
             ["mean1", "loss.a"], ["pen", "loss.b"]])
        add(f"prim_weights_norm_{kind}", nodes, edges, ["lin2"], "loss", d=2)

    # a Linear read only through a Weights tap: no forward pass, but its W (and zero-grad bias) is live
    nodes, edges = with_loss(
        [lin("tap", 2), n_("diff", "subtract"), n_("err", "square"), n_("mean1", "mean"),
         n_("wt", "weights", of="tap"), n_("nt", "norm", kind="l2"), n_("pen", "scale", c=0.3),
         n_("loss", "add")],
        [["x", "tap"], ["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "mean1"],
         ["wt", "nt"], ["nt", "pen"], ["mean1", "loss.a"], ["pen", "loss.b"]])
    add("prim_weights_tap_only", nodes, edges, ["lin2"], "loss", d=2, optimizer="sgd", lr=0.1)

    # no prediction pin: the graph still trains; predict() returns nothing
    nodes, edges = with_loss(
        [n_("diff", "subtract"), n_("err", "square"), n_("loss", "mean")],
        [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "loss"]])
    add("prim_no_prediction", nodes, edges, [], "loss", d=2, optimizer="adam", lr=0.05)

    # two prediction pins: the output and a second head (a Linear the loss does not use); L on the output
    nodes, edges = with_loss(
        [lin("head2", 1), n_("diff", "subtract"), n_("err", "square"), n_("loss", "mean")],
        [["act1", "head2"], ["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "loss"]])
    add("prim_two_predictions", nodes, edges, ["lin2", "head2"], "loss", d=2, optimizer="sgd")

    # an auxiliary objective that is not part of the model: L = mean(sq(out - y)) + 0.3 mean(abs(hidden))
    nodes, edges = with_loss(
        [n_("diff", "subtract"), n_("err", "square"), n_("m1", "mean"), n_("hab", "abs"),
         n_("m2", "mean"), n_("sc", "scale", c=0.3), n_("loss", "add")],
        [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "m1"], ["act1", "hab"],
         ["hab", "m2"], ["m2", "sc"], ["m1", "loss.a"], ["sc", "loss.b"]])
    add("prim_aux_objective", nodes, edges, ["lin2"], "loss", d=2, optimizer="momentum", lr=0.05)

    # log loss on probabilities (sigmoid link in front): fused by the engine, semantic here
    nodes = [inp(2), lin("lin1", 3), act("act1", "tanh"), lin("lin2", 1),
             n_("p", "link", fn="sigmoid"), TARGET, n_("err", "log_loss"), n_("loss", "mean")]
    edges = [["x", "lin1"], ["lin1", "act1"], ["act1", "lin2"], ["lin2", "p"], ["p", "err.pred"],
             ["y", "err.target"], ["err", "loss"]]
    add("prim_log_loss_sigmoid", nodes, edges, ["p"], "loss", d=2, ytype="cls", optimizer="sgd")

    # cross-entropy on softmax probabilities
    nodes = [inp(2), lin("lin1", 3), act("act1", "tanh"), lin("lin2", 2),
             n_("p", "link", fn="softmax"), TARGET, n_("err", "cross_entropy"), n_("loss", "mean")]
    edges = [["x", "lin1"], ["lin1", "act1"], ["act1", "lin2"], ["lin2", "p"], ["p", "err.pred"],
             ["y", "err.target"], ["err", "loss"]]
    add("prim_cross_entropy_softmax", nodes, edges, ["p"], "loss", d=2, ytype="cls", optimizer="sgd")

    # the link also feeds something else (a second consumer), so it must still be computed in the loss pass
    nodes = [inp(2), lin("lin1", 3), act("act1", "tanh"), lin("lin2", 1),
             n_("p", "link", fn="sigmoid"), TARGET, n_("err", "log_loss"), n_("m1", "mean"),
             n_("sq", "square"), n_("m2", "mean"), n_("sc", "scale", c=0.5), n_("loss", "add")]
    edges = [["x", "lin1"], ["lin1", "act1"], ["act1", "lin2"], ["lin2", "p"], ["p", "err.pred"],
             ["y", "err.target"], ["err", "m1"], ["p", "sq"], ["sq", "m2"], ["m2", "sc"],
             ["m1", "loss.a"], ["sc", "loss.b"]]
    add("prim_link_shared", nodes, edges, ["p"], "loss", d=2, ytype="cls", optimizer="adam", lr=0.05)

    # dropout, a skip connection (add) and feature expansion together in one DAG, with a dropout mask
    nodes = [inp(2), n_("feat", "feature", degree=2, sin=True, cos=False, products=True),
             lin("lin1", 4), act("act1", "relu"), n_("drop1", "dropout", p=0.5),
             lin("lin2", 4), n_("skip", "add"), lin("lin3", 1), TARGET,
             n_("diff", "subtract"), n_("err", "huber", delta=1.0), n_("loss", "mean")]
    edges = [["x", "feat"], ["feat", "lin1"], ["lin1", "act1"], ["act1", "drop1"], ["drop1", "lin2"],
             ["lin2", "skip.a"], ["lin1", "skip.b"], ["skip", "lin3"], ["lin3", "diff.a"],
             ["y", "diff.b"], ["diff", "err"], ["err", "loss"]]
    add("prim_dag_dropout_skip", nodes, edges, ["lin3"], "loss", d=2, mask_for=True)

    # 300-step plain Adam on the primitive form with a penalty: long-run agreement
    nodes, edges = with_loss(
        [n_("diff", "subtract"), n_("err", "square"), n_("mean1", "mean"),
         n_("w1", "weights", of="lin1"), n_("n1", "norm", kind="l2"), n_("pen", "scale", c=0.01),
         n_("loss", "add")],
        [["lin2", "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "mean1"], ["w1", "n1"],
         ["n1", "pen"], ["mean1", "loss.a"], ["pen", "loss.b"]])
    add("prim_penalty_adam_300steps", nodes, edges, ["lin2"], "loss", d=2, optimizer="adam", lr=0.05,
        n=8, steps=300)


def scalar_cases(add, check_linear):
    """Scalar-piece graphs (Column, Weight, Bias, Sum). Each column x_i is scaled by its own Weight,
    the weighted columns are added by one Sum and an optional Bias shifts the result; the squared
    error is built from primitives. d = 3 has no dataset of its own: the graph names "plane" and the
    engine tests take the columns x1..x3 from the batch."""

    def regression(d, bias=True, fn=None, raw=()):
        nodes = [TARGET]
        edges = []
        for i in range(1, d + 1):
            nodes.append(n_(f"c{i}", "column", name=f"x{i}"))
            if i in raw:  # a column wired straight into the Sum, no Weight
                edges.append([f"c{i}", "s"])
            else:
                nodes.append(n_(f"w{i}", "scalew"))
                edges += [[f"c{i}", f"w{i}"], [f"w{i}", "s"]]
        nodes.append(n_("s", "sum"))
        last = "s"
        if bias:
            nodes.append(n_("b", "bias"))
            edges.append([last, "b"])
            last = "b"
        if fn:
            nodes.append(act("a", fn))
            edges.append([last, "a"])
            last = "a"
        nodes += [n_("diff", "subtract"), n_("err", "square"), n_("loss", "mean")]
        edges += [[last, "diff.a"], ["y", "diff.b"], ["diff", "err"], ["err", "loss"]]
        return nodes, edges, last

    for name, kw, d, extra in (
        ("scalar_regression_d1", {}, 1, {}),
        ("scalar_regression_d3", {}, 3, {}),
        ("scalar_regression_d3_nobias", {"bias": False}, 3, {}),
        ("scalar_sum_two_cols", {"bias": False, "raw": (2,)}, 2, {}),
        ("scalar_activation_after_sum", {"fn": "tanh"}, 2, {}),
        ("scalar_adam_5steps", {}, 3, {"optimizer": "adam", "lr": 0.05, "n": 8}),
    ):
        nodes, edges, last = regression(d, **kw)
        add(name, nodes, edges, [last], "loss", d=d, **extra)

    # the same numbers as a plain Linear: x @ W + b with W = [w1..wd] as a column (torch weight.T)
    check_linear(3)


def main():
    cases = []
    seed = [100]

    def add(name, nodes_edges, **kw):
        seed[0] += 1
        fx, y = make_case(name, seed[0], *nodes_edges, **kw)
        cases.append(fx)
        return fx, y

    # linear, with and without bias, d = 1 and 2
    for d in (1, 2):
        for bias in (False, True):
            add(f"linear_{'bias' if bias else 'nobias'}_d{d}", mlp(d, [], "tanh", bias=bias), n=5, d=d)

    # activations between two Linears
    for fn in ("sigmoid", "tanh", "relu", "gelu"):
        add(f"act_{fn}", mlp(2, [3], fn), n=6, d=2)

    # output links
    add("link_identity", mlp(1, [3], "tanh", "identity"))
    add("link_sigmoid", mlp(2, [3], "tanh", "sigmoid", loss("squared_error")), ytype="cls", d=2)
    add("link_softmax", mlp(2, [3], "tanh", "softmax", loss("cross_entropy"), out=2), ytype="cls", d=2)

    # losses
    add("loss_squared_error", mlp(2, [3], "tanh", loss_node=loss("squared_error")), d=2)
    add("loss_absolute_error", mlp(2, [3], "tanh", loss_node=loss("absolute_error")), d=2)
    for dl in (1.0, 0.5):
        fx, y = add(f"loss_huber_delta{dl:g}", mlp(2, [3], "tanh", loss_node=loss("huber", dl)), d=2, n=8)
        r = np.abs(np.array(fx["expect"]["pred"]) - y.flatten())
        assert (r < dl).any() and (r > dl).any(), f"huber delta {dl} must see both regimes: {r}"
    add("loss_log_loss", mlp(2, [3], "tanh", "sigmoid", loss("log_loss")), ytype="cls", d=2)
    add("loss_cross_entropy", mlp(2, [3], "tanh", "softmax", loss("cross_entropy"), out=2), ytype="cls", d=2)
    add("loss_squared_error_sigmoid", mlp(2, [3], "tanh", "sigmoid", loss("squared_error")), ytype="cls", d=2)

    # weight penalties (Linear weights only, lambda 0.1)
    for kind in ("l2", "l1"):
        add(f"penalty_{kind}",
            mlp(2, [3], "tanh", loss_node=loss("squared_error", penalty={"kind": kind, "lambda": 0.1})), d=2)

    # at-zero conventions: sign(0) = 0 for L1, relu'(0) = 0
    add("penalty_l1_zero_weight",
        mlp(2, [3], "tanh", loss_node=loss("squared_error", penalty={"kind": "l1", "lambda": 0.1})),
        d=2, zero_weights=[("lin1", 0), ("lin2", 1)])
    add("act_relu_zero_preact", mlp(2, [3], "relu", bias=False), d=2, zero_x_rows=[0, 3])

    # feature: degree 3, sin, cos, products, d = 2 -> width 11
    nodes = [inp(2),
             {"id": "feat", "type": "feature", "degree": 3, "sin": True, "cos": True, "products": True},
             lin("lin1", 1), outp("identity"), TARGET, loss("squared_error")]
    edges = chain_edges(["x", "feat", "lin1", "yhat"]) + [["yhat", "loss.pred"], ["y", "loss.target"]]
    add("feature_deg3_sin_cos_products", (nodes, edges), d=2)

    # add: skip connection lin1 -> act1 -> lin2, plus lin1 -> add.b
    nodes = [inp(2), lin("lin1", 3), act("act1", "tanh"), lin("lin2", 3), {"id": "add1", "type": "add"},
             lin("lin3", 1), outp("identity"), TARGET, loss("squared_error")]
    edges = [["x", "lin1"], ["lin1", "act1"], ["act1", "lin2"], ["lin2", "add1.a"], ["lin1", "add1.b"],
             ["add1", "lin3"], ["lin3", "yhat"], ["yhat", "loss.pred"], ["y", "loss.target"]]
    add("add_skip", (nodes, edges), d=2)

    # dropout p = 0.5 with a fixed keep mask
    nodes = [inp(2), lin("lin1", 4), act("act1", "relu"), {"id": "drop1", "type": "dropout", "p": 0.5},
             lin("lin2", 1), outp("identity"), TARGET, loss("squared_error")]
    edges = chain_edges(["x", "lin1", "act1", "drop1", "lin2", "yhat"]) + [
        ["yhat", "loss.pred"], ["y", "loss.target"]]
    add("dropout_p0.5", (nodes, edges), d=2, mask_for=True)

    # optimizers: 5 full-batch steps on a 2-layer tanh MLP
    for opt, lr in (("sgd", 0.1), ("momentum", 0.05), ("adam", 0.05)):
        add(f"mlp_tanh_{opt}_5steps", mlp(2, [3], "tanh"), d=2, optimizer=opt, lr=lr, n=8)

    # two-parameter model (1 weight, 1 bias) for the loss-surface view
    add("two_param", mlp(1, [], "tanh"), n=6, d=1)

    # long plain-Adam run (300 steps): the gradients shrink, so the second moment decays (tinker-df1)
    add("mlp_tanh_adam_300steps", mlp(2, [3], "tanh"), d=2, optimizer="adam", lr=0.1, n=8, steps=300)

    # graph v2 cases (primitives, Weights taps, pins); the cases above are version 1 graphs that the
    # engine tests migrate with parseGraph
    def add2(name, nodes, edges, predictions, loss_id, **kw):
        seed[0] += 1
        cases.append(make_case_v2(name, seed[0], nodes, edges, predictions, loss_id, **kw))

    v2_cases(add2)

    def check_linear(d):
        """scalar_regression_d{d} must equal nn.Linear(d, 1) with weight = [w_i] and bias = b."""
        fx = next(c for c in cases if c["name"] == f"scalar_regression_d{d}")
        x = torch.tensor(fx["batch"]["x"]).reshape(fx["batch"]["n"], d)
        y = torch.tensor(fx["batch"]["y"]).reshape(-1, 1)
        lin_ = nn.Linear(d, 1)
        lin_.weight.data = torch.tensor([[fx["params"][f"w{i}"]["W"][0] for i in range(1, d + 1)]])
        lin_.bias.data = torch.tensor(fx["params"]["b"]["b"])
        pred = lin_(x)
        loss_ = ((pred - y) ** 2).mean()
        loss_.backward()
        assert np.allclose(pred.detach().flatten().tolist(), fx["expect"]["pred"], atol=1e-12)
        assert abs(loss_.item() - fx["expect"]["loss"]) < 1e-12
        gw = lin_.weight.grad.flatten().tolist()
        want = [fx["expect"]["grads"][f"w{i}"]["W"][0] for i in range(1, d + 1)]
        assert np.allclose(gw, want, atol=1e-12)
        assert abs(lin_.bias.grad.item() - fx["expect"]["grads"]["b"]["b"][0]) < 1e-12

    scalar_cases(add2, check_linear)

    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names)
    for c in cases:
        write(c)
    print(f"wrote {len(cases)} fixtures to {OUT} (torch {torch.__version__})")


if __name__ == "__main__":
    main()

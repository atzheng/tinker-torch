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
                v[i] = get() * self.mask / (1.0 - n["p"])
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
              optimizer=None, steps=5, lr=0.1, mask_for=None, xscale=1.5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-xscale, xscale, size=(n, d))
    if ytype == "reg":
        y = 0.7 * x[:, :1] + 0.3 * np.sin(2 * x[:, :1]) + 0.4 * rng.standard_normal((n, 1))
        y = y * 2.0  # spread so huber sees both regimes
    else:  # class index as float, 0/1; guarantee both classes
        y = (rng.uniform(size=(n, 1)) > 0.5).astype(float)
        y[0, 0], y[1, 0] = 0.0, 1.0
    params = init_params(rng, nodes, edges)
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

    # data/train blocks are placeholders: the batch above is explicit.
    graph = {
        "version": 1, "nodes": nodes, "edges": edges,
        "data": {"set": "blobs" if ytype == "cls" else {1: "sine", 2: "plane"}[d],
                 "n": n, "noise": 0.1, "val": 0.25, "seed": 1, "standardize": False},
        "train": {"optimizer": optimizer or "sgd", "lr": lr, "batch": "full", "seed": 1},
    }
    fx = {"name": name, "graph": graph, "params": params,
          "batch": {"x": x.flatten().tolist(), "y": y.flatten().tolist(), "n": n}, "expect": expect}
    return fx, y


def write(fx):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{fx['name']}.json").write_text(json.dumps(fx, indent=1) + "\n")


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

    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names)
    for c in cases:
        write(c)
    print(f"wrote {len(cases)} fixtures to {OUT} (torch {torch.__version__})")


if __name__ == "__main__":
    main()

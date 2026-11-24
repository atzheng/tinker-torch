/** Torch code generation (CONTRACTS.md section 14, tinker-img.1): golden text for the model, objective and training parts. */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  fitTrees,
  sample,
  torchScript,
  treeLeafCounts,
  type DataConfig,
  type Edge,
  type Graph,
  type Node,
  type TorchCode,
  type TrainConfig,
} from '../src/index.js';

type Opts = {
  set?: DataConfig['set'];
  train?: Partial<TrainConfig>;
  data?: Partial<DataConfig>;
  predictions?: string[];
  loss?: string | null;
};

function graphOf(nodes: Node[], edges: Edge[], o: Opts = {}): Graph {
  return {
    version: 2,
    nodes,
    edges,
    predictions: o.predictions ?? [],
    loss: o.loss === undefined ? null : o.loss,
    data: {
      set: o.set ?? 'line',
      n: 10,
      noise: 0.1,
      val: 0.2,
      seed: 3,
      standardize: false,
      ...o.data,
    },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 5, ...o.train },
  };
}

const seen = new Set<Graph>();

function gen(g: Graph, steps?: number): TorchCode | null {
  seen.add(g);
  const ds = sample(g.data);
  const a = analyse(g, ds.info, treeLeafCounts(fitTrees(g, ds)));
  return torchScript(g, a, ds, steps === undefined ? undefined : { steps });
}

/** Everything from the Model class on: the part that depends on the graph (the data rows are separate). */
function script(g: Graph, steps?: number): string {
  const code = gen(g, steps);
  if (code === null) throw new Error('expected a script');
  return code.text.slice(code.text.indexOf('class Model'));
}

const sq = (): [Node[], Edge[]] => [
  [
    { id: 'diff', type: 'subtract' },
    { id: 'err', type: 'square' },
    { id: 'loss', type: 'mean' },
  ],
  [
    ['diff', 'err'],
    ['err', 'loss'],
  ],
];

/** in -> l1 (4) -> a1 -> l2 (1) as yhat; squared error against tgt. */
function mlp(o: Opts = {}): Graph {
  const [n, e] = sq();
  return graphOf(
    [
      { id: 'in', type: 'input', columns: ['x1'] },
      { id: 'l1', type: 'linear', out: 4, bias: true },
      { id: 'a1', type: 'activation', fn: 'tanh' },
      { id: 'l2', type: 'linear', out: 1, bias: true },
      { id: 'tgt', type: 'target' },
      ...n,
    ],
    [['in', 'l1'], ['l1', 'a1'], ['a1', 'l2'], ['l2', 'diff.a'], ['tgt', 'diff.b'], ...e],
    { predictions: ['l2'], loss: 'loss', ...o },
  );
}

describe('torchScript', () => {
  test('(a) 1-D MLP with squared error: whole script', () => {
    const code = gen(mlp());
    expect(code?.text).toMatchSnapshot();
  });

  test('(a) model and objective', () => {
    expect(script(mlp())).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.l1 = nn.Linear(1, 4, bias=True)
              self.l2 = nn.Linear(4, 1, bias=True)

          def forward(self, x):
              in_ = x[:, [0]]
              l1 = self.l1(in_)
              a1 = torch.tanh(l1)
              l2 = self.l2(a1)
              return l2


      def objective(model, x, y):
          l2 = model(x)
          tgt = y
          loss = F.mse_loss(l2, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(b) logistic: sigmoid link, log loss, fused into with-logits', () => {
    const g = graphOf(
      [
        { id: 'in', type: 'input', columns: ['x1', 'x2'] },
        { id: 'lin', type: 'linear', out: 1, bias: true },
        { id: 'p', type: 'link', fn: 'sigmoid' },
        { id: 'tgt', type: 'target' },
        { id: 'err', type: 'log_loss' },
        { id: 'loss', type: 'mean' },
      ],
      [
        ['in', 'lin'],
        ['lin', 'p'],
        ['p', 'err.pred'],
        ['tgt', 'err.target'],
        ['err', 'loss'],
      ],
      { set: 'blobs', predictions: ['p'], loss: 'loss' },
    );
    expect(script(g)).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.lin = nn.Linear(2, 1, bias=True)

          def forward(self, x):
              in_ = x[:, [0, 1]]
              lin = self.lin(in_)
              p = torch.sigmoid(lin)
              return p, lin


      def objective(model, x, y):
          p, lin = model(x)
          tgt = y
          loss = F.binary_cross_entropy_with_logits(lin, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(c) softmax and cross-entropy on three classes', () => {
    const g = graphOf(
      [
        { id: 'in', type: 'input', columns: ['x1', 'x2'] },
        { id: 'lin', type: 'linear', out: 3, bias: true },
        { id: 'p', type: 'link', fn: 'softmax' },
        { id: 'tgt', type: 'target' },
        { id: 'err', type: 'cross_entropy' },
        { id: 'loss', type: 'mean' },
      ],
      [
        ['in', 'lin'],
        ['lin', 'p'],
        ['p', 'err.pred'],
        ['tgt', 'err.target'],
        ['err', 'loss'],
      ],
      { set: 'blobs3', predictions: ['p'], loss: 'loss' },
    );
    expect(script(g)).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.lin = nn.Linear(2, 3, bias=True)

          def forward(self, x):
              in_ = x[:, [0, 1]]
              lin = self.lin(in_)
              p = torch.softmax(lin, dim=-1)
              return p, lin


      def objective(model, x, y):
          p, lin = model(x)
          tgt = y
          loss = F.cross_entropy(lin, tgt.long().squeeze(1))
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(d) skip connection with Add, dropout and a feature expansion', () => {
    const [n, e] = sq();
    const g = graphOf(
      [
        { id: 'in', type: 'input', columns: ['x1', 'x2'] },
        { id: 'feat', type: 'feature', degree: 2, sin: true, cos: false, products: true },
        { id: 'l1', type: 'linear', out: 4, bias: true },
        { id: 'drop', type: 'dropout', p: 0.25 },
        { id: 'l2', type: 'linear', out: 4, bias: false },
        { id: 'add1', type: 'add' },
        { id: 'act', type: 'activation', fn: 'gelu' },
        { id: 'l3', type: 'linear', out: 1, bias: true },
        { id: 'tgt', type: 'target' },
        ...n,
      ],
      [
        ['in', 'feat'],
        ['feat', 'l1'],
        ['l1', 'drop'],
        ['drop', 'l2'],
        ['l1', 'add1.a'],
        ['l2', 'add1.b'],
        ['add1', 'act'],
        ['act', 'l3'],
        ['l3', 'diff.a'],
        ['tgt', 'diff.b'],
        ...e,
      ],
      { set: 'plane', predictions: ['l3'], loss: 'loss' },
    );
    expect(script(g)).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.l1 = nn.Linear(7, 4, bias=True)
              self.l2 = nn.Linear(4, 4, bias=False)
              self.l3 = nn.Linear(4, 1, bias=True)

          def forward(self, x):
              in_ = x[:, [0, 1]]
              feat = torch.cat([in_, in_ ** 2, torch.sin(in_), in_[:, [0]] * in_[:, [1]]], dim=-1)
              l1 = self.l1(feat)
              drop = F.dropout(l1, p=0.25, training=self.training)
              l2 = self.l2(drop)
              add1 = l1 + l2
              act = F.gelu(add1)
              l3 = self.l3(act)
              return l3


      def objective(model, x, y):
          l3 = model(x)
          tgt = y
          loss = F.mse_loss(l3, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(e) weight penalty is an explicit added term', () => {
    const g = mlp();
    g.nodes.push(
      { id: 'w1', type: 'weights', of: 'l1' },
      { id: 'n1', type: 'norm', kind: 'l2' },
      { id: 'w2', type: 'weights', of: 'l2' },
      { id: 'n2', type: 'norm', kind: 'l1' },
      { id: 'pen', type: 'scale', c: 0.01 },
      { id: 'sum', type: 'add' },
      { id: 'total', type: 'add' },
    );
    g.edges.push(
      ['w1', 'n1'],
      ['w2', 'n2'],
      ['n1', 'sum.a'],
      ['n2', 'sum.b'],
      ['sum', 'pen'],
      ['loss', 'total.a'],
      ['pen', 'total.b'],
    );
    g.loss = 'total';
    expect(script(g)).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.l1 = nn.Linear(1, 4, bias=True)
              self.l2 = nn.Linear(4, 1, bias=True)

          def forward(self, x):
              in_ = x[:, [0]]
              l1 = self.l1(in_)
              a1 = torch.tanh(l1)
              l2 = self.l2(a1)
              return l2


      def objective(model, x, y):
          l2 = model(x)
          tgt = y
          w1 = model.l1.weight.T
          w2 = model.l2.weight.T
          n1 = w1.pow(2).sum()
          n2 = w2.abs().sum()
          sum = n1 + n2
          pen = 0.01 * sum
          loss = F.mse_loss(l2, tgt)
          total = loss + pen
          return total


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(f) scalar pieces: columns, weights, sum, bias', () => {
    const [n, e] = sq();
    const g = graphOf(
      [
        { id: 'c1', type: 'column', name: 'x1' },
        { id: 'c2', type: 'column', name: 'x2' },
        { id: 'w1', type: 'scalew' },
        { id: 'w2', type: 'scalew' },
        { id: 'tot', type: 'sum' },
        { id: 'b', type: 'bias' },
        { id: 'tgt', type: 'target' },
        ...n,
      ],
      [
        ['c1', 'w1'],
        ['c2', 'w2'],
        ['w1', 'tot'],
        ['w2', 'tot'],
        ['tot', 'b'],
        ['b', 'diff.a'],
        ['tgt', 'diff.b'],
        ...e,
      ],
      { set: 'plane', predictions: ['b'], loss: 'loss' },
    );
    expect(script(g)).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.w1 = nn.Parameter(torch.empty(1, 1).uniform_(-0.7071067811865475, 0.7071067811865475))
              self.w2 = nn.Parameter(torch.empty(1, 1).uniform_(-0.7071067811865475, 0.7071067811865475))
              self.b = nn.Parameter(torch.empty(1, 1).uniform_(-0.7071067811865475, 0.7071067811865475))

          def forward(self, x):
              c1 = x[:, [0]]
              c2 = x[:, [1]]
              w1 = self.w1 * c1
              w2 = self.w2 * c2
              tot = torch.stack([w1, w2]).sum(0)
              b = tot + self.b
              return b


      def objective(model, x, y):
          b = model(x)
          tgt = y
          loss = F.mse_loss(b, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(f) an unwired Bias is one number for every example', () => {
    const [n, e] = sq();
    const g = graphOf(
      [{ id: 'b', type: 'bias' }, { id: 'tgt', type: 'target' }, ...n],
      [['b', 'diff.a'], ['tgt', 'diff.b'], ...e],
      { predictions: ['b'], loss: 'loss' },
    );
    expect(script(g)).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.b = nn.Parameter(torch.empty(1, 1).uniform_(-1.0, 1.0))

          def forward(self, x):
              b = self.b.expand(x.shape[0], 1)
              return b


      def objective(model, x, y):
          b = model(x)
          tgt = y
          loss = F.mse_loss(b, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(g) category column: one-hot and lookup', () => {
    const [n, e] = sq();
    const g = graphOf(
      [
        { id: 'zone', type: 'input', columns: ['x4'] },
        { id: 'oh', type: 'onehot' },
        { id: 'lin', type: 'linear', out: 1, bias: false },
        { id: 'lk', type: 'lookup', width: 1 },
        { id: 'add1', type: 'add' },
        { id: 'tgt', type: 'target' },
        ...n,
      ],
      [
        ['zone', 'oh'],
        ['oh', 'lin'],
        ['zone', 'lk'],
        ['lin', 'add1.a'],
        ['lk', 'add1.b'],
        ['add1', 'diff.a'],
        ['tgt', 'diff.b'],
        ...e,
      ],
      { set: 'delivery_time', predictions: ['add1'], loss: 'loss' },
    );
    expect(script(g)).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.lin = nn.Linear(4, 1, bias=False)
              self.lk = nn.Embedding(4, 1)

          def forward(self, x):
              zone = x[:, [3]]
              oh = F.one_hot(zone[:, 0].long(), 4).float()
              lk = self.lk(zone[:, 0].long())
              lin = self.lin(oh)
              add1 = lin + lk
              return add1


      def objective(model, x, y):
          add1 = model(x)
          tgt = y
          loss = F.mse_loss(add1, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(h) tree: fitted before the model on the training rows', () => {
    const [n, e] = sq();
    const g = graphOf(
      [
        { id: 'in', type: 'input', columns: ['x1'] },
        {
          id: 'tr',
          type: 'tree',
          criterion: 'squared_error',
          depth: 2,
          minSplit: 2,
          minLeaf: 1,
          minDecrease: 0,
        },
        { id: 'lin', type: 'linear', out: 1, bias: true },
        { id: 'tgt', type: 'target' },
        ...n,
      ],
      [['in', 'tr'], ['tr', 'lin'], ['lin', 'diff.a'], ['tgt', 'diff.b'], ...e],
      { predictions: ['lin'], loss: 'loss' },
    );
    const code = gen(g);
    expect(code?.text).toContain('from sklearn.tree import DecisionTreeRegressor');
    const text = script(g);
    expect(text).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.lin = nn.Linear(4, 1, bias=True)

          def forward(self, x):
              in_ = x[:, [0]]
              tr = (torch.as_tensor(tr_tree.apply(in_.numpy())).unsqueeze(1) == tr_leaves).float()
              lin = self.lin(tr)
              return lin


      def objective(model, x, y):
          lin = model(x)
          tgt = y
          loss = F.mse_loss(lin, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
    const full = code?.text ?? '';
    expect(full.indexOf('tr_tree = ')).toBeGreaterThan(full.indexOf('y_val = '));
    expect(full.indexOf('tr_tree = ')).toBeLessThan(full.indexOf('class Model'));
  });

  test('(i) no prediction pin: parameters live in the objective, no forward', () => {
    const [n, e] = sq();
    const g = graphOf(
      [{ id: 'b', type: 'bias' }, { id: 'tgt', type: 'target' }, ...n],
      [['b', 'diff.a'], ['tgt', 'diff.b'], ...e],
      { loss: 'loss' },
    );
    const text = script(g);
    expect(text).not.toContain('def forward');
    expect(text).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.b = nn.Parameter(torch.empty(1, 1).uniform_(-1.0, 1.0))


      def objective(model, x, y):
          b = model.b.expand(x.shape[0], 1)
          tgt = y
          loss = F.mse_loss(b, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(j) the objective reads a model node that is not a prediction', () => {
    const g = mlp();
    g.nodes.push(
      { id: 'ab', type: 'abs' },
      { id: 'm2', type: 'mean' },
      { id: 'sc', type: 'scale', c: 0.5 },
      { id: 'total', type: 'add' },
    );
    g.edges.push(['a1', 'ab'], ['ab', 'm2'], ['m2', 'sc'], ['loss', 'total.a'], ['sc', 'total.b']);
    g.loss = 'total';
    const text = script(g);
    expect(text).toContain('return l2, a1');
    expect(text).toContain('l2, a1 = model(x)');
    expect(text).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.l1 = nn.Linear(1, 4, bias=True)
              self.l2 = nn.Linear(4, 1, bias=True)

          def forward(self, x):
              in_ = x[:, [0]]
              l1 = self.l1(in_)
              a1 = torch.tanh(l1)
              l2 = self.l2(a1)
              return l2, a1


      def objective(model, x, y):
          l2, a1 = model(x)
          tgt = y
          ab = torch.abs(a1)
          m2 = ab.mean()
          sc = 0.5 * m2
          loss = F.mse_loss(l2, tgt)
          total = loss + sc
          return total


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(k) ids that are Python keywords or script names get a trailing underscore', () => {
    const g = graphOf(
      [
        { id: 'x', type: 'input', columns: ['x1'] },
        { id: 'x_', type: 'linear', out: 4, bias: true },
        { id: 'class', type: 'activation', fn: 'relu' },
        { id: 'torch', type: 'linear', out: 1, bias: true },
        { id: 'y', type: 'target' },
        { id: 'model', type: 'subtract' },
        { id: 'diff', type: 'square' },
        { id: 'loss', type: 'mean' },
      ],
      [
        ['x', 'x_'],
        ['x_', 'class'],
        ['class', 'torch'],
        ['torch', 'model.a'],
        ['y', 'model.b'],
        ['model', 'diff'],
        ['diff', 'loss'],
      ],
      { predictions: ['torch'], loss: 'loss' },
    );
    const text = script(g);
    expect(text).toContain('x__ = x[:, [0]]');
    expect(text).toContain('class_ = torch.relu(');
    expect(text).toContain('self.torch_ = nn.Linear(4, 1');
    expect(text).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.x_ = nn.Linear(1, 4, bias=True)
              self.torch_ = nn.Linear(4, 1, bias=True)

          def forward(self, x):
              x__ = x[:, [0]]
              x_ = self.x_(x__)
              class_ = torch.relu(x_)
              torch_ = self.torch_(class_)
              return torch_


      def objective(model, x, y):
          torch_ = model(x)
          y_ = y
          loss = F.mse_loss(torch_, y_)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      steps = 1000

      for step in range(1, steps + 1):
          model.train()
          batch_loss = objective(model, x_train, y_train)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 100 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(l) mini-batch with Adam', () => {
    const g = mlp({ train: { batch: 4, optimizer: 'adam', lr: 0.05 } });
    const text = script(g, 200);
    expect(text).toContain('DataLoader(TensorDataset(x_train, y_train), batch_size=4');
    expect(text).toMatchInlineSnapshot(`
      "class Model(nn.Module):
          def __init__(self):
              super().__init__()
              self.l1 = nn.Linear(1, 4, bias=True)
              self.l2 = nn.Linear(4, 1, bias=True)

          def forward(self, x):
              in_ = x[:, [0]]
              l1 = self.l1(in_)
              a1 = torch.tanh(l1)
              l2 = self.l2(a1)
              return l2


      def objective(model, x, y):
          l2 = model(x)
          tgt = y
          loss = F.mse_loss(l2, tgt)
          return loss


      model = Model()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
      steps = 200
      loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=True)


      def cycle(loader):
          while True:
              yield from loader


      batches = cycle(loader)

      for step in range(1, steps + 1):
          model.train()
          xb, yb = next(batches)
          batch_loss = objective(model, xb, yb)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          if step % 20 == 0 or step == steps:
              model.eval()
              with torch.no_grad():
                  train_loss = objective(model, x_train, y_train)
                  val_loss = objective(model, x_val, y_val)
              print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")
      "
    `);
  });

  test('(m) an invalid graph gives null', () => {
    const g = mlp({ loss: null });
    expect(gen(g)).toBeNull();
    const noPin = mlp({ predictions: [] });
    noPin.edges = noPin.edges.filter((e) => e[0] !== 'l2');
    expect(gen(noPin)).toBeNull();
  });

  test('optimizers and mini-batch imports', () => {
    expect(script(mlp({ train: { optimizer: 'sgd', lr: 0.1 } }))).toContain(
      'optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n',
    );
    expect(script(mlp({ train: { optimizer: 'momentum', lr: 0.1 } }))).toContain(
      'lr=0.1, momentum=0.9)',
    );
    expect(gen(mlp())?.text).not.toContain('DataLoader');
    expect(gen(mlp({ train: { batch: 8 } }))?.text).toContain(
      'from torch.utils.data import DataLoader, TensorDataset\n',
    );
  });

  test('idioms: l1 and huber losses; literal forms when an intermediate has another reader', () => {
    const l1 = mlp();
    l1.nodes = l1.nodes.map((n) => (n.id === 'err' ? { id: 'err', type: 'abs' } : n));
    expect(script(l1)).toContain('loss = F.l1_loss(l2, tgt)');
    const hub = mlp();
    hub.nodes = hub.nodes.map((n) =>
      n.id === 'err' ? { id: 'err', type: 'huber', delta: 0.5 } : n,
    );
    expect(script(hub)).toContain('loss = F.huber_loss(l2, tgt, delta=0.5)');
    // diff is also read elsewhere: literal, piece by piece
    const shared = mlp();
    shared.nodes.push(
      { id: 'extra', type: 'abs' },
      { id: 'm2', type: 'mean' },
      { id: 'tot', type: 'add' },
    );
    shared.edges.push(['diff', 'extra'], ['extra', 'm2'], ['loss', 'tot.a'], ['m2', 'tot.b']);
    shared.loss = 'tot';
    const text = script(shared);
    expect(text).toContain('diff = l2 - tgt');
    expect(text).toContain('err = diff ** 2');
    expect(text).not.toContain('F.mse_loss');
  });

  test('literal log loss and cross-entropy without a link', () => {
    const g = graphOf(
      [
        { id: 'in', type: 'input', columns: ['x1', 'x2'] },
        { id: 'lin', type: 'linear', out: 1, bias: true },
        { id: 'p', type: 'activation', fn: 'sigmoid' },
        { id: 'tgt', type: 'target' },
        { id: 'err', type: 'log_loss' },
        { id: 'loss', type: 'mean' },
      ],
      [
        ['in', 'lin'],
        ['lin', 'p'],
        ['p', 'err.pred'],
        ['tgt', 'err.target'],
        ['err', 'loss'],
      ],
      { set: 'blobs', predictions: ['p'], loss: 'loss' },
    );
    const text = script(g);
    expect(text).toContain("err = F.binary_cross_entropy(p, tgt, reduction='none')");
    expect(text).toContain('loss = err.mean()');
  });

  test('data section: standardisation as explicit lines, folds for each literal', () => {
    const g = mlp({ data: { standardize: true } });
    const code = gen(g);
    const text = code?.text ?? '';
    expect(text).toContain('x_mean = x_train.mean(dim=0)');
    expect(text).toContain('x_std = x_train.std(dim=0, unbiased=False)');
    expect(text).toContain('x_train = (x_train - x_mean) / x_std');
    expect(code?.folds).toHaveLength(4);
    for (const f of code?.folds ?? []) {
      expect(text[f.from]).toBe('[');
      expect(text[f.to - 1]).toBe(']');
      expect(f.label).toMatch(/rows$/);
    }
    expect(text.slice(0, text.indexOf('x_train = torch.tensor'))).toBe(
      'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\ntorch.manual_seed(5)\n\n',
    );
  });

  test('categorical columns are not standardised', () => {
    const g = graphOf(
      [{ id: 'b', type: 'bias' }, { id: 'tgt', type: 'target' }, ...sq()[0]],
      [['b', 'diff.a'], ['tgt', 'diff.b'], ...sq()[1]],
      { set: 'delivery_time', predictions: ['b'], loss: 'loss', data: { standardize: true } },
    );
    expect(gen(g)?.text).toContain('x_mean[[3]] = 0.0\nx_std[[3]] = 1.0\n');
  });

  test('spans: every live node of every graph above has a span naming it or the idiom call', () => {
    expect(seen.size).toBeGreaterThan(15);
    for (const g of seen) {
      const code = gen(g);
      if (code === null) continue;
      const ds = sample(g.data);
      const a = analyse(g, ds.info, treeLeafCounts(fitTrees(g, ds)));
      for (const id of new Set([...a.order, ...a.paramNodes])) {
        const spans = code.spans.filter((s) => s.node === id);
        expect(spans.length, id).toBeGreaterThan(0);
        for (const s of spans) {
          const t = code.text.slice(s.from, s.to);
          expect(t.includes(id) || t.includes('F.'), `${id}: ${t}`).toBe(true);
          expect(t).not.toContain('\n');
        }
      }
      for (const s of code.spans) expect(a.inert).not.toContain(s.node);
    }
  });

  test('every wire-shaped test graph has spans for all live nodes', () => {
    // The other golden graphs, via their node ids: covered when a Linear has both its __init__ and forward span.
    const code = gen(mlp());
    const l1 = (code?.spans ?? []).filter((s) => s.node === 'l1');
    expect(l1).toHaveLength(2);
    const texts = l1.map((s) => code?.text.slice(s.from, s.to));
    expect(texts[0]).toBe('self.l1 = nn.Linear(1, 4, bias=True)');
    expect(texts[1]).toBe('l1 = self.l1(in_)');
  });
});

// A 2-layer LSTM + linear head, matching ActionClassifierLSTM from the original
// project (input 16, hidden 128, 2 layers, 4 classes). Weights are the original
// trained PyTorch weights, packed as float32 in this order:
// w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0, w_ih_l1, w_hh_l1, b_ih_l1, b_hh_l1, fc.weight, fc.bias

const H = 128;
const G = 4 * H; // gates: input, forget, cell, output (PyTorch order)

export class LSTMClassifier {
  constructor(buf, inputSize = 16, numClasses = 4) {
    const f = new Float32Array(buf);
    let o = 0;
    const take = (n) => { const v = f.subarray(o, o + n); o += n; return v; };
    this.layers = [];
    let inSize = inputSize;
    for (let l = 0; l < 2; l++) {
      const wih = take(G * inSize), whh = take(G * H);
      const bih = take(G), bhh = take(G);
      const b = new Float32Array(G);
      for (let i = 0; i < G; i++) b[i] = bih[i] + bhh[i];
      this.layers.push({ wih, whh, b, inSize });
      inSize = H;
    }
    this.fcw = take(numClasses * H);
    this.fcb = take(numClasses);
    this.numClasses = numClasses;
    if (o !== f.length) throw new Error(`weight size mismatch: used ${o} of ${f.length}`);
    this.gates = new Float32Array(G);
  }

  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load ${url}`);
    return new LSTMClassifier(await r.arrayBuffer());
  }

  // seq: array of T Float32Array(16). Returns logits (Float32Array(numClasses)).
  logits(seq) {
    let inputs = seq;
    const g = this.gates;
    for (const { wih, whh, b, inSize } of this.layers) {
      const h = new Float32Array(H), c = new Float32Array(H);
      const outs = [];
      for (const x of inputs) {
        for (let r = 0; r < G; r++) {
          let s = b[r];
          let o = r * inSize;
          for (let k = 0; k < inSize; k++) s += wih[o + k] * x[k];
          o = r * H;
          for (let k = 0; k < H; k++) s += whh[o + k] * h[k];
          g[r] = s;
        }
        for (let j = 0; j < H; j++) {
          const i = sigmoid(g[j]), f = sigmoid(g[H + j]);
          const cc = Math.tanh(g[2 * H + j]), oo = sigmoid(g[3 * H + j]);
          c[j] = f * c[j] + i * cc;
          h[j] = oo * Math.tanh(c[j]);
        }
        outs.push(Float32Array.from(h));
      }
      inputs = outs;
    }
    const last = inputs[inputs.length - 1];
    const out = new Float32Array(this.numClasses);
    for (let r = 0; r < this.numClasses; r++) {
      let s = this.fcb[r];
      for (let k = 0; k < H; k++) s += this.fcw[r * H + k] * last[k];
      out[r] = s;
    }
    return out;
  }

  // Returns { index, confidence, probs }
  predict(seq) {
    const z = this.logits(seq);
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const sum = e.reduce((a, v) => a + v, 0);
    const probs = e.map((v) => v / sum);
    let index = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[index]) index = i;
    return { index, confidence: probs[index], probs };
  }
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

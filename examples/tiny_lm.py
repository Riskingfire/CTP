"""A tiny character-level language model that trains on data streamed by CTProtocol.

Pure NumPy (no PyTorch / GPU needed): a byte-level embedding -> tanh hidden layer -> softmax, with
hand-written backprop and Adam. ~250k parameters. It is small on purpose: the point is to show the
full loop of *stream -> train -> save -> sample* without ever downloading the dataset.

Train (streams Tiny Shakespeare by default, ~1 MB, nothing is kept on disk afterwards):

    pip install -e .            # from the repo root, once
    python examples/tiny_lm.py train

Train on your own data (any URL / path / glob / brace range that CTProtocol understands):

    python examples/tiny_lm.py train --source "https://host/train-{000..015}.jsonl.gz" --text-field text
    python examples/tiny_lm.py train --source "my_corpus/*.txt" --format lines --epochs 5

Generate text from the saved model:

    python examples/tiny_lm.py sample --prompt "ROMEO:" --chars 400 --temperature 0.8
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ctprotocol import CTProtocolConfig, CTProtocolDataset

DEFAULT_SOURCE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
VOCAB = 256  # raw UTF-8 bytes: no tokenizer to fit, works on any text
PAD = 10  # newline, used to left-pad short prompts


# --------------------------------------------------------------------------- model
class TinyLM:
    """context of `ctx` bytes -> embedding -> tanh hidden layer -> logits for the next byte."""

    def __init__(self, ctx: int = 16, dim: int = 24, hidden: int = 384, seed: int = 0):
        self.ctx, self.dim, self.hidden = ctx, dim, hidden
        rng = np.random.default_rng(seed)
        self.p: dict[str, np.ndarray] = {
            "E": rng.normal(0, 0.1, (VOCAB, dim)),
            "W1": rng.normal(0, 1 / np.sqrt(ctx * dim), (ctx * dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, VOCAB)),
            "b2": np.zeros(VOCAB),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}  # Adam state
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.p.values())

    def forward(self, x: np.ndarray):
        p = self.p
        h0 = p["E"][x].reshape(len(x), -1)  # (B, ctx*dim)
        a1 = np.tanh(h0 @ p["W1"] + p["b1"])  # (B, hidden)
        logits = a1 @ p["W2"] + p["b2"]  # (B, VOCAB)
        return h0, a1, logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = self._softmax(self.forward(x)[2])
        return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

    def grads(self, x: np.ndarray, y: np.ndarray):
        p = self.p
        h0, a1, logits = self.forward(x)
        probs = self._softmax(logits)
        b = len(y)
        loss = float(-np.log(probs[np.arange(b), y] + 1e-12).mean())
        d_logits = probs
        d_logits[np.arange(b), y] -= 1
        d_logits /= b
        g = {"W2": a1.T @ d_logits, "b2": d_logits.sum(0)}
        d_z1 = (d_logits @ p["W2"].T) * (1 - a1**2)
        g["W1"], g["b1"] = h0.T @ d_z1, d_z1.sum(0)
        d_h0 = (d_z1 @ p["W1"].T).reshape(b, self.ctx, self.dim)
        g["E"] = np.zeros_like(p["E"])
        np.add.at(g["E"], x, d_h0)
        return loss, g

    def step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        loss, g = self.grads(x, y)
        self.t += 1
        b1, b2 = 0.9, 0.999
        for k, grad in g.items():
            np.clip(grad, -1.0, 1.0, out=grad)
            self.m[k] = b1 * self.m[k] + (1 - b1) * grad
            self.v[k] = b2 * self.v[k] + (1 - b2) * grad**2
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    # -- io / generation -----------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, ctx=self.ctx, dim=self.dim, hidden=self.hidden, **self.p)

    @classmethod
    def load(cls, path: str) -> TinyLM:
        d = np.load(path)
        model = cls(int(d["ctx"]), int(d["dim"]), int(d["hidden"]))
        for k in model.p:
            model.p[k] = d[k]
        return model

    def generate(self, prompt: str, n: int, temperature: float = 0.8, top_k: int = 40, seed: int | None = None) -> str:
        rng = np.random.default_rng(seed)
        ids = list(prompt.encode("utf-8"))
        out = bytearray(ids)
        for _ in range(n):
            window = ([PAD] * self.ctx + ids)[-self.ctx :]
            logits = self.forward(np.array([window]))[2][0] / max(temperature, 1e-4)
            if 0 < top_k < VOCAB:
                logits[logits < np.sort(logits)[-top_k]] = -np.inf
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            nxt = int(rng.choice(VOCAB, p=probs))
            ids.append(nxt)
            out.append(nxt)
        return out.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- streaming data
def windows(data: np.ndarray, ctx: int) -> tuple[np.ndarray, np.ndarray]:
    """All (context, next byte) pairs inside one chunk of bytes."""
    n = len(data) - ctx
    idx = np.arange(n)[:, None] + np.arange(ctx)[None, :]
    return data[idx], data[ctx : ctx + n]


def chunks(dataset: CTProtocolDataset, ctx: int, chunk_bytes: int):
    """Turn the record stream into fixed-size byte chunks (with `ctx` bytes of overlap so no
    context is lost at the seams). Each chunk is used once and then dropped: nothing accumulates."""
    buf = bytearray()
    for record in dataset:
        buf += (record if isinstance(record, str) else str(record)).encode("utf-8") + b"\n"
        while len(buf) >= chunk_bytes:
            yield np.frombuffer(bytes(buf[:chunk_bytes]), dtype=np.uint8).astype(np.int64)
            del buf[: chunk_bytes - ctx]
    if len(buf) > ctx + 1:
        yield np.frombuffer(bytes(buf), dtype=np.uint8).astype(np.int64)


# --------------------------------------------------------------------------- commands
def train(args: argparse.Namespace) -> None:
    model = TinyLM(args.ctx, args.dim, args.hidden, seed=args.seed)
    config = CTProtocolConfig(ahead_seconds=args.ahead_seconds, max_cache_mb=args.max_cache_mb)
    dataset = CTProtocolDataset(args.source, format=args.format, text_field=args.text_field, config=config)
    rng = np.random.default_rng(args.seed)
    print(f"model: {model.n_params:,} parameters | source: {args.source}")

    t0, seen, step = time.time(), 0, 0
    for epoch in range(args.epochs):
        lr = args.lr * (1 - 0.9 * epoch / max(args.epochs, 1))  # simple linear decay
        dataset.set_epoch(epoch)
        running: list[float] = []
        val_losses: list[float] = []
        with dataset:  # cache lives only for the duration of the pass and is deleted afterwards
            for i, data in enumerate(chunks(dataset, args.ctx, args.chunk_bytes)):
                x, y = windows(data, args.ctx)
                if i % args.val_every == args.val_every - 1:  # held-out chunk: never trained on
                    val_losses.append(model.loss(x, y))
                    continue
                order = rng.permutation(len(y))
                for s in range(0, len(order), args.batch):
                    sel = order[s : s + args.batch]
                    running.append(model.step(x[sel], y[sel], lr))
                    step += 1
                seen += len(data)
                if i % 20 == 0:
                    mb_s = seen / 1e6 / max(time.time() - t0, 1e-9)
                    print(f"epoch {epoch + 1}/{args.epochs} chunk {i:4d} | train loss {np.mean(running[-200:]):.3f} "
                          f"| {mb_s:.2f} MB/s", flush=True)
        summary = f"== epoch {epoch + 1}: train loss {np.mean(running[-500:]):.3f}"
        if val_losses:
            summary += f" | held-out loss {np.mean(val_losses):.3f} (perplexity {np.exp(np.mean(val_losses)):.1f})"
        print(summary)
        print("   stream:", dataset.stats.summary())
        model.save(args.out)
        print(f"   saved -> {args.out}")
        print("   sample:", repr(model.generate(args.prompt, 120, seed=epoch)))
    print(f"done in {time.time() - t0:.0f}s")


def sample(args: argparse.Namespace) -> None:
    model = TinyLM.load(args.out)
    print(model.generate(args.prompt, args.chars, args.temperature, args.top_k, args.seed))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="stream data and train the model")
    t.add_argument("--source", default=DEFAULT_SOURCE, help="URL / path / glob / brace range (default: Tiny Shakespeare)")
    t.add_argument("--format", default="lines", help="lines | jsonl | parquet | raw (default: lines)")
    t.add_argument("--text-field", default=None, help="field to read for jsonl / parquet records")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--ctx", type=int, default=16, help="characters of context the model sees")
    t.add_argument("--dim", type=int, default=24)
    t.add_argument("--hidden", type=int, default=384)
    t.add_argument("--chunk-bytes", type=int, default=16384, help="bytes held in memory per training chunk")
    t.add_argument("--val-every", type=int, default=20, help="hold out every Nth chunk for evaluation")
    t.add_argument("--ahead-seconds", type=float, default=30)
    t.add_argument("--max-cache-mb", type=int, default=64, help="hard cap on CTProtocol's buffer")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--prompt", default="ROMEO:", help="prompt for the sample printed after each epoch")
    t.add_argument("--out", default="tiny_lm.npz")
    t.set_defaults(fn=train)

    s = sub.add_parser("sample", help="generate text from a saved model")
    s.add_argument("--prompt", default="ROMEO:")
    s.add_argument("--chars", type=int, default=400)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--out", default="tiny_lm.npz")
    s.set_defaults(fn=sample)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

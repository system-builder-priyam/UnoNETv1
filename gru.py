"""
Model (one step, batch of B letters):
    gx = Wx[x] + bx                       (B, 3H)  input part   (one-hot x -> row lookup)
    gh = h @ Wh + bh                      (B, 3H)  hidden part
    r  = sigmoid(gx_r + gh_r)             reset gate
    z  = sigmoid(gx_z + gh_z)             update gate
    n  = tanh(gx_n + r * gh_n)            candidate state
    h' = (1 - z) * n + z * h              new hidden state (the "memory")
    logits = h' @ Wy + by                 (B, V)   scores for the next letter
Gate order inside every 3H block: [r | z | n].
"""
import numpy as np
from vocab import encode, decode, load_names, make_batch, vocab_size

V = vocab_size      #29
H = 64              #fits the UNO's flash at int8

#params
def init_params(H, V, seed=0):
    rng = np.random.default_rng(seed)
    k = 1.0 / np.sqrt(H)
    return {
        "Wx": rng.uniform(-k, k, (V, 3 * H)),   
        "Wh": rng.uniform(-k, k, (H, 3 * H)),   
        "bx": np.zeros(3 * H),
        "bh": np.zeros(3 * H),
        "Wy": rng.uniform(-k, k, (H, V)),       
        "by": np.zeros(V),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def gru_step(p, x, h):
    Hs = h.shape[1]
    gx = p["Wx"][x] + p["bx"]
    gh = h @ p["Wh"] + p["bh"]
    r = sigmoid(gx[:, :Hs] + gh[:, :Hs])
    z = sigmoid(gx[:, Hs:2 * Hs] + gh[:, Hs:2 * Hs])
    hn = gh[:, 2 * Hs:]
    n = np.tanh(gx[:, 2 * Hs:] + r * hn)
    h_new = (1 - z) * n + z * h
    return h_new, (x, h, r, z, n, hn)


def forward(p, X, Y, mask):#forward over a whole batch of names + loss
    B, T = X.shape
    Hs = p["Wh"].shape[0]
    h = np.zeros((B, Hs))
    caches, hs, probs_all = [], [], []
    total = 0.0
    n_tok = mask.sum()

    for t in range(T):
        h, cache = gru_step(p, X[:, t], h)
        logits = h @ p["Wy"] + p["by"]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
        total += -(np.log(probs[np.arange(B), Y[:, t]]) * mask[:, t]).sum()
        caches.append(cache)
        hs.append(h)
        probs_all.append(probs)

    loss = total / n_tok
    return loss, (caches, hs, probs_all)

#backpropagation through time
def backward(p, X, Y, mask, state):
    caches, hs, probs_all = state
    B, T = X.shape
    Hs = p["Wh"].shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    n_tok = mask.sum()
    dh_next = np.zeros((B, Hs))

    for t in reversed(range(T)):
        x, h_prev, r, z, n, hn = caches[t]
        h = hs[t]

        #output layer: d(loss)/d(logits) = probs - onehot(answer)
        dlogits = probs_all[t].copy()
        dlogits[np.arange(B), Y[:, t]] -= 1
        dlogits *= mask[:, t:t + 1] / n_tok
        g["Wy"] += h.T @ dlogits
        g["by"] += dlogits.sum(axis=0)

        #gradient arriving at h_t: from this step's output + from step t+1
        dh = dlogits @ p["Wy"].T + dh_next

        #h'=(1-z)*n+z*h_prev
        dn = dh * (1 - z)
        dz = dh * (h_prev - n)
        dh_prev = dh * z

        # n=tanh(an), an=gx_n+r*hn
        dan = dn * (1 - n * n)
        dr = dan * hn
        dhn = dan * r

        #r,z=sigmoid()
        dar = dr * r * (1 - r)
        daz = dz * z * (1 - z)

        dgx = np.concatenate([dar, daz, dan], axis=1)   
        dgh = np.concatenate([dar, daz, dhn], axis=1)   

        np.add.at(g["Wx"], x, dgx)                      # one-hot input to rows
        g["bx"] += dgx.sum(axis=0)
        g["Wh"] += h_prev.T @ dgh
        g["bh"] += dgh.sum(axis=0)
        dh_prev += dgh @ p["Wh"].T

        dh_next = dh_prev

    return g


#grad-check
def grad_check():
    print("gradient check (small model, H=8):")
    p = init_params(8, V, seed=1)
    for k in p:                                   # non-zero biases so every path is tested
        p[k] = p[k] + np.random.default_rng(2).normal(0, 0.1, p[k].shape)
    X, Y, mask = make_batch(["cat", "polar bear", "aye-aye"])
    loss, state = forward(p, X, Y, mask)
    g = backward(p, X, Y, mask, state)

    rng = np.random.default_rng(3)
    eps = 1e-4     
    worst = 0.0
    for k in p:
        for _ in range(10):
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][idx]
            p[k][idx] = old + eps
            lp, _ = forward(p, X, Y, mask)
            p[k][idx] = old - eps
            lm, _ = forward(p, X, Y, mask)
            p[k][idx] = old
            num = (lp - lm) / (2 * eps)
            ana = g[k][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
        print(f"  {k:2s} ok")
    print(f"  worst relative error: {worst:.2e}  ({'PASS' if worst < 1e-5 else 'FAIL'})\n")
    return worst < 1e-5


#sampling

def complete(p, prefix="", temperature=1.0, rng=None, max_len=25):
    """Feed '.' + prefix, then keep predicting until '.'. temperature=0 -> greedy."""
    rng = rng or np.random.default_rng()
    h = np.zeros((1, p["Wh"].shape[0]))
    ids = encode(prefix)                          
    for c in ids:
        h, _ = gru_step(p, np.array([c]), h)
    out = ids[1:]
    for _ in range(max_len):
        logits = (h @ p["Wy"] + p["by"])[0]
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            z = logits / temperature
            e = np.exp(z - z.max())
            nxt = int(rng.choice(V, p=e / e.sum()))
        if nxt == 0:
            break
        out.append(nxt)
        h, _ = gru_step(p, np.array([nxt]), h)
    return decode(out)


#adam used as an optimizer
def train(names, val_names, epochs=300, batch_size=64, lr=5e-3, clip=5.0, seed=0):
    p = init_params(H, V, seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}   # Adam-running mean of grads
    s = {k: np.zeros_like(v) for k, v in p.items()}   # Adam-running mean of grads^2
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    step = 0
    n_params = sum(v.size for v in p.values())
    print(f"params: {n_params}   train names: {len(names)}   val names: {len(val_names)}")

    Xv, Yv, Mv = make_batch(val_names) if val_names else (None, None, None)

    for ep in range(epochs):
        order = rng.permutation(len(names))
        ep_loss, ep_n = 0.0, 0
        for i in range(0, len(names), batch_size):
            batch = [names[j] for j in order[i:i + batch_size]]
            X, Y, mask = make_batch(batch)
            loss, state = forward(p, X, Y, mask)
            g = backward(p, X, Y, mask, state)

            #clip stops rare huge gradients from wrecking training
            norm = np.sqrt(sum((v * v).sum() for v in g.values()))
            if norm > clip:
                for k in g:
                    g[k] *= clip / norm

            #adam update
            step += 1
            lr_t = lr * (0.1 ** (ep / epochs))          
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                s[k] = b2 * s[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** step)
                sh = s[k] / (1 - b2 ** step)
                p[k] -= lr_t * mh / (np.sqrt(sh) + eps)

            ep_loss += loss * mask.sum()
            ep_n += mask.sum()

        if ep % 25 == 0 or ep == epochs - 1:
            msg = f"epoch {ep:4d}  train loss {ep_loss / ep_n:.4f}"
            if Xv is not None:
                msg += f"  val loss {forward(p, Xv, Yv, Mv)[0]:.4f}"
            print(msg)
    return p


if __name__ == "__main__":
    if not grad_check():
        raise SystemExit("gradient check failed - fix backward() first")

    names = load_names("data/animals.txt")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(names))
    VAL_FRAC = 0.0          # set to 0 to train on every name (do this for the final UNO model)
    n_val = int(len(names) * VAL_FRAC)
    val_names = [names[i] for i in order[:n_val]]
    train_names = [names[i] for i in order[n_val:]]

    p = train(train_names, val_names)
    np.savez("model_float.npz", **p)
    print("saved model_float.npz")

    rng = np.random.default_rng(1)
    print("\nrandom animals (temperature 0.8):")
    print("  " + ", ".join(complete(p, "", 0.8, rng) for _ in range(12)))

    print("\nprefix completion:  greedy  |  3 samples")
    for pre in ["ele", "kan", "pen", "cro", "sea ", "gir", "tig", "sn"]:
        best = complete(p, pre, 0)
        alts = " | ".join(complete(p, pre, 0.8, rng) for _ in range(3))
        print(f"  {pre:5s} -> {best:20s} {alts}")

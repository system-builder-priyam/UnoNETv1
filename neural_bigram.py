import numpy as np
from vocab import encode, decode, load_names, vocab_size

names = load_names("data/animals.txt")


#training data as (current letter, next letter) pairs
xs = []
ys = []

for name in names:
    ids = encode(name) + [0]
    for a, b  in zip(ids, ids[1:]):
        xs.append(a)
        ys.append(b)

xs = np.array(xs)
ys = np.array(ys)
M = len(xs)                       #number of pairs (7179)
print("pairs:", M)

#random weights + forward pass
rng = np.random.default_rng(42)
W = rng.normal(0, 0.01, size=(vocab_size, vocab_size))   #29 x 29, all near 0

def forward(W):
    logits = W[xs]                                   #(M, 29): row of W for each input letter
    logits = logits - logits.max(axis=1, keepdims=True)  #safety: avoids exp overflow, same result
    counts = np.exp(logits)                          #make everything positive
    probs = counts / counts.sum(axis=1, keepdims=True)   #sums to 1
    p_correct = probs[np.arange(M), ys]              #prob given to the true next letter
    loss = -np.log(p_correct).mean()
    return probs, loss


probs, loss = forward(W)
print("logits shape:", W[xs].shape)                 #(7179, 29)
print("row sums:", probs.sum(axis=1)[:5])           #[1. 1. 1. 1. 1.]
print("initial loss:", loss)                        

#training with gradient descent

lr = 50.0          #learning rate
steps = 300

for step in range(steps):
    #forward
    probs, loss = forward(W)

    #backward: gradient of loss w.r.t. logits = probs - onehot(answer),averaged
    G = probs.copy()
    G[np.arange(M), ys] -= 1
    G /= M

    #each example's gradient belongs to the row of W it used (row xs[k])
    dW = np.zeros_like(W)
    np.add.at(dW, xs, G)

    #update
    W -= lr * dW

    if step % 25 == 0 or step == steps - 1:
        print(f"step {step:4d}  loss {loss:.4f}")

print("final loss:", forward(W)[1], "(counting bigram: 2.564)")

#generate names from the learned 
def softmax_row(z):
    e = np.exp(z - z.max())
    return e / e.sum()


print("\nsamples:")
for _ in range(10):
    cur, out = 0, []
    while True:
        nxt = rng.choice(vocab_size, p=softmax_row(W[cur]))
        if nxt == 0:
            break
        out.append(nxt)
        cur = nxt
    print(" ", decode(out))

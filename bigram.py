from vocab import encode, decode, load_names, vocab_size
import numpy as np

names = load_names("data/animals.txt")
N = np.zeros((vocab_size, vocab_size), dtype=np.int64)

#print(encode("cat") + [0])
"""
ids = encode("cat") + [0]      # [0, 5, 3, 22, 0]  →  . c a t .

#zip() takes two (or more) lists and pairs up their items by position

for a, b in zip(ids, ids[1:]):
    print(a, b)
    """

for name in names:
    ids = encode(name) + [0] 
    for a, b in zip(ids, ids[1:]):
        N[a, b] += 1

print(N[0].sum())   # 812
print(N[0])         # how many names start with each letter

# counts to probabilities
N1 = N + 1      #no zero counts
row_sums = N1.sum(axis=1, keepdims=True)     # total of each row, shape (29, 1)
P = N1 / row_sums                            # each row now sums to 1

print(P.shape)          # (29, 29)
print(P.sum(axis=1))    # all 1.0

#generating names
cur = 0
out = []

for _ in range(10):
    while True:
        nxt = np.random.choice(vocab_size, p=P[cur]) 
        if nxt == 0:
            break
        out.append(nxt)
        cur = nxt

print(decode(out))

total = 0
count = 0
for name in names:
    ids = encode(name) + [0]
    for a, b in zip(ids, ids[1:]):
        total += (- np.log(P[a, b]))
        count += 1

loss = total/count
print(loss)
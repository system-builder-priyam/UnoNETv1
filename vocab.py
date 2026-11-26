import numpy as np

chars = list(". -abcdefghijklmnopqrstuvwxyz")
vocab_size = len(chars)
stoi = {}
itos = {}

for i in range(len(chars)):
    stoi[chars[i]] = i

for i in range(len(chars)):
    itos[i] = chars[i]

def encode(string):
    string = "." + string
    return [stoi[c] for c in string]

def decode(encoded):
    return ''.join(itos[i] for i in encoded if i != 0)

def make_batch(names):

    encoded = []
    max_len = 0
    for name in names:
        t = encode(name)
        encoded.append(t)
        if len(t) > max_len:
            max_len = len(t)

    B = len(names)
    X    = np.zeros((B, max_len), dtype=np.int64)
    Y    = np.zeros((B, max_len), dtype=np.int64)
    mask = np.zeros((B, max_len), dtype=np.float32)
  
    for i in range(B):
        ids = encoded[i]    
        L = len(ids)          

        X[i, :L] = ids                
        Y[i, :L] = ids[1:] + [0]      
        mask[i, :L] = 1       

    return X, Y, mask

def load_names(path):
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names

if __name__ == "__main__":
    X, Y, mask = make_batch(["cat", "elephant"])
    print(X)
    print(Y)
    print(mask)

    names = load_names("data/animals.txt")
    print(len(names))    # 812
    print(names[:3])     # ['aardvark', 'aardwolf', 'addax']
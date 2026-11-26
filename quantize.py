"""
Integer model (one step, letter x, int8 hidden state h):
    acc_g = WH_g @ h                          (H,)  int32 dot products, g in [r | z | n]
    rec_g = (acc_g * Mh + 2^(Sh-1)) >> Sh     Q8    recurrent part
    inp_g = (WX[x]_g * Mx + 2^(Sx-1)) >> Sx   Q8    input part
    r  = SIG(rec_r + inp_r + b_r)             0..255   (real = q / 256)
    z  = SIG(rec_z + inp_z + b_z)             0..255
    n  = TANH(inp_n + bx_n + (r * (rec_n + bh_n) + 128) >> 8)    -127..127
    h' = (n * (256 - z) + h * z + 128) >> 8   -127..127 (real = q / 127)
    logits = ((WY @ h') * My + 2^(Sy-1)) >> Sy + by               Q8
Q8 means real * 256. SIG/TANH are 257-entry lookup tables with linear interpolation.
This file is the exact spec for the Arduino C code: same integers, same shifts, same rounding.
"""

#This file is AI generated, because I did not know how to make my code UNO compatible


import os
import numpy as np
from vocab import encode, decode, load_names, chars, vocab_size
import gru

V = vocab_size      #29
LUT_N = 256         #lut segments, tables have LUT_N + 1 entries
MAX_GEN = 24        #max letters generated after the prefix
STR_LEN = 32        #fixed string size in the test vectors (prefix + MAX_GEN + '\0')

#helpers
def quant_tensor(W):
    s = np.abs(W).max() / 127.0                                 # one scale for the whole block
    q = np.clip(np.round(W / s), -127, 127).astype(np.int64)
    return q, s


def mult_shift(f):
    """write a positive real factor f as M / 2^S with M in [512, 1023]"""
    for S in range(1, 48):
        M = int(round(f * (1 << S)))
        if M >= 512:
            if M == 1024:
                M, S = 512, S - 1
            assert 512 <= M <= 1023 and S >= 1, (f, M, S)
            return M, S
    raise ValueError(f"factor too small: {f}")


def rq(acc, M, S):
    return (acc * M + (1 << (S - 1))) >> S                     # arithmetic shift, same in Python and C


def q8(x):
    return np.clip(np.round(np.asarray(x) * 256), -32767, 32767).astype(np.int64)


#lookup tables
_grid = (np.arange(LUT_N + 1) - LUT_N // 2) / 16.0                                  # -8..+8, step 1/16
SIG_LUT = np.minimum(255, np.round(256 / (1 + np.exp(-_grid)))).astype(np.int64)    # 0..255
TANH_LUT = np.round(np.tanh(_grid) * 127).astype(np.int64)                          # -127..127
EXP_LUT = np.round(65535 * np.exp(-np.arange(LUT_N + 1) / 16.0)).astype(np.int64)   # exp(-x), x in 0..16


def lut_interp(lut, pre):
    idx = np.clip(pre + 2048, 0, 4095)                         # Q8 in [-8, 8) -> 0..4095
    i = idx >> 4                                               # table entry
    f = idx & 15                                               # position between entries (0..15)
    return lut[i] + (((lut[i + 1] - lut[i]) * f + 8) >> 4)


#quantize the float params
def quantize(p):
    H = p["Wh"].shape[0]
    qp = {"H": H}
    WhT = p["Wh"].T                                            # (3H, H), one row per output unit for the C loop
    WxQ = np.zeros_like(p["Wx"], dtype=np.int64)
    WhQ = np.zeros_like(WhT, dtype=np.int64)
    qp["Mx"], qp["Sx"], qp["Mh"], qp["Sh"] = [], [], [], []
    for g in range(3):                                         # gates r, z, n each get their own scale
        blk = slice(g * H, (g + 1) * H)
        WxQ[:, blk], sx = quant_tensor(p["Wx"][:, blk])
        WhQ[blk, :], sh = quant_tensor(WhT[blk, :])
        M, S = mult_shift(sx * 256);         qp["Mx"].append(M); qp["Sx"].append(S)
        M, S = mult_shift(sh * 256 / 127);   qp["Mh"].append(M); qp["Sh"].append(S)
    qp["Wx"], qp["Wh"] = WxQ, WhQ

    WyQ, sy = quant_tensor(p["Wy"].T)                          # (V, H)
    qp["Wy"] = WyQ
    qp["My"], qp["Sy"] = mult_shift(sy * 256 / 127)

    bx, bh = p["bx"], p["bh"]
    qp["b_r"] = q8(bx[:H] + bh[:H])                            # r and z biases can be merged
    qp["b_z"] = q8(bx[H:2 * H] + bh[H:2 * H])
    qp["bx_n"] = q8(bx[2 * H:])                                # n biases stay split: bh_n sits inside r * (...)
    qp["bh_n"] = q8(bh[2 * H:])
    qp["by"] = q8(p["by"])
    return qp


#integer-only model (the spec for the C code)
MAX_SEEN = {"acc*M": 0}                                        # largest product seen, must stay < 2^31

def int_step(qp, x, h):
    H = qp["H"]
    acc = qp["Wh"] @ h                                         # exact integer dot products
    rec, inp = [], []
    for g in range(3):
        a = acc[g * H:(g + 1) * H]
        MAX_SEEN["acc*M"] = max(MAX_SEEN["acc*M"], int(np.abs(a).max()) * qp["Mh"][g])
        rec.append(rq(a, qp["Mh"][g], qp["Sh"][g]))
        inp.append(rq(qp["Wx"][x, g * H:(g + 1) * H], qp["Mx"][g], qp["Sx"][g]))

    r = lut_interp(SIG_LUT, rec[0] + inp[0] + qp["b_r"])
    z = lut_interp(SIG_LUT, rec[1] + inp[1] + qp["b_z"])
    hn = rec[2] + qp["bh_n"]
    pre_n = inp[2] + qp["bx_n"] + ((r * hn + 128) >> 8)
    n = lut_interp(TANH_LUT, pre_n)
    return (n * (256 - z) + h * z + 128) >> 8


def int_logits(qp, h):
    acc = qp["Wy"] @ h
    MAX_SEEN["acc*M"] = max(MAX_SEEN["acc*M"], int(np.abs(acc).max()) * qp["My"])
    return rq(acc, qp["My"], qp["Sy"]) + qp["by"]


def xorshift32(s):
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    return s


#integer sampling, T_q8 = temperature * 256, 0 = greedy
def int_sample(logits, T_q8, state):
    if T_q8 == 0:
        return int(np.argmax(logits)), state                   # first max, like a strict '>' loop in C
    d = logits.max() - logits                                  # >= 0
    sc = (d * 256) // T_q8                                     # >= 0, so // matches C division
    i = np.minimum(sc >> 4, LUT_N - 1)
    f = sc & 15
    w = EXP_LUT[i] - (((EXP_LUT[i] - EXP_LUT[i + 1]) * f + 8) >> 4)
    w = np.where((sc >> 4) >= LUT_N, 0, w)                     # beyond the table -> weight 0
    state = xorshift32(state)
    r = state % int(w.sum())
    k = int(np.searchsorted(np.cumsum(w), r, side="right"))    # first k with running sum > r
    return k, state


def int_complete(qp, prefix, T_q8=0, state=12345, max_len=MAX_GEN):
    h = np.zeros(qp["H"], dtype=np.int64)
    ids = encode(prefix)
    for c in ids:
        h = int_step(qp, c, h)
    out = ids[1:]
    for _ in range(max_len):
        nxt, state = int_sample(int_logits(qp, h), T_q8, state)
        if nxt == 0:
            break
        out.append(nxt)
        h = int_step(qp, nxt, h)
    return decode(out), state


#float vs int evaluation
def float_loss(p, names):
    tot, n = 0.0, 0
    for name in names:
        ids = encode(name) + [0]
        h = np.zeros((1, p["Wh"].shape[0]))
        for a, b in zip(ids, ids[1:]):
            h, _ = gru.gru_step(p, np.array([a]), h)
            z = (h @ p["Wy"] + p["by"])[0]
            z = z - z.max()
            tot += -(z[b] - np.log(np.exp(z).sum()))
            n += 1
    return tot / n


def int_loss(qp, names):
    tot, n = 0.0, 0
    for name in names:
        ids = encode(name) + [0]
        h = np.zeros(qp["H"], dtype=np.int64)
        for a, b in zip(ids, ids[1:]):
            h = int_step(qp, a, h)
            z = int_logits(qp, h) / 256.0                      # back to real numbers just to measure loss
            z = z - z.max()
            tot += -(z[b] - np.log(np.exp(z).sum()))
            n += 1
    return tot / n


#C header export
def c_array(ctype, name, arr, per_line=16):
    flat = [int(v) for v in np.asarray(arr).ravel()]
    lines = []
    for i in range(0, len(flat), per_line):
        lines.append("  " + ", ".join(str(v) for v in flat[i:i + per_line]))
    dims = "".join(f"[{d}]" for d in np.asarray(arr).shape)
    return f"const {ctype} {name}{dims} PROGMEM = {{\n" + ",\n".join(lines) + "\n};\n"


def c_strings(name, strs, width):
    for s in strs:
        assert len(s) < width, s
    body = "".join(f'  "{s}",\n' for s in strs)
    return f"const char {name}[{len(strs)}][{width}] PROGMEM = {{\n{body}}};\n"


def export_model_h(qp, path):
    H = qp["H"]
    vocab = "".join(chars)
    parts = [
        "// Generated by quantize.py - do not edit by hand.\n",
        "#pragma once\n#include <stdint.h>\n",
        "#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif\n\n",
        f"#define UNO_H {H}\n#define UNO_V {V}\n#define UNO_LUT_N {LUT_N}\n#define UNO_MAX_GEN {MAX_GEN}\n",
        f'static const char UNO_VOCAB[] = "{vocab}";\n\n',
        "// requantization: q8 = (acc * M + (1 << (S - 1))) >> S, gates r, z, n\n",
        f"static const int16_t UNO_MX[3] = {{{', '.join(map(str, qp['Mx']))}}};\n",
        f"static const uint8_t UNO_SX[3] = {{{', '.join(map(str, qp['Sx']))}}};\n",
        f"static const int16_t UNO_MH[3] = {{{', '.join(map(str, qp['Mh']))}}};\n",
        f"static const uint8_t UNO_SH[3] = {{{', '.join(map(str, qp['Sh']))}}};\n",
        f"#define UNO_MY {qp['My']}\n#define UNO_SY {qp['Sy']}\n\n",
        "// WX[letter][gate*H + j], WH[gate*H + j][i], WY[letter][i]\n",
        c_array("int8_t", "UNO_WX", qp["Wx"]),
        c_array("int8_t", "UNO_WH", qp["Wh"]),
        c_array("int8_t", "UNO_WY", qp["Wy"]),
        "\n// biases, Q8 (real * 256)\n",
        c_array("int16_t", "UNO_B_R", qp["b_r"]),
        c_array("int16_t", "UNO_B_Z", qp["b_z"]),
        c_array("int16_t", "UNO_BX_N", qp["bx_n"]),
        c_array("int16_t", "UNO_BH_N", qp["bh_n"]),
        c_array("int16_t", "UNO_BY", qp["by"]),
        "\n// lookup tables over [-8, 8) (sigmoid, tanh) and [0, 16) (exp(-x))\n",
        c_array("uint8_t", "UNO_SIG_LUT", SIG_LUT),
        c_array("int8_t", "UNO_TANH_LUT", TANH_LUT),
        c_array("uint16_t", "UNO_EXP_LUT", EXP_LUT),
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


#expected outputs the C code must reproduce bit for bit (stored in flash so the UNO can self-test)
def export_test_vectors(qp, path, prefixes, T_q8, seed, n_samples=8):
    h = np.zeros(qp["H"], dtype=np.int64)
    for c in encode("ele"):
        h = int_step(qp, c, h)
    lg = int_logits(qp, h)

    greedy = [int_complete(qp, pre, 0)[0] for pre in prefixes]
    state, samples = seed, []
    for _ in range(n_samples):                                 # one shared rng state across samples
        out, state = int_complete(qp, "", T_q8, state)
        samples.append(out)

    parts = [
        "// Generated by quantize.py - expected outputs of the integer model.\n#pragma once\n",
        "#include <stdint.h>\n#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif\n\n",
        f"#define TV_STR {STR_LEN}\n#define TV_N_GREEDY {len(prefixes)}\n#define TV_N_SAMPLES {n_samples}\n",
        f"#define TV_SEED {seed}UL\n#define TV_T_Q8 {T_q8}\n\n",
        "// hidden state and logits after feeding '.ele'\n",
        c_array("int8_t", "TV_H_ELE", h),
        c_array("int32_t", "TV_LOGITS_ELE", lg),
        "\n// greedy completions\n",
        c_strings("TV_PREFIX", prefixes, STR_LEN),
        c_strings("TV_GREEDY", greedy, STR_LEN),
        "\n// samples from '' with TV_SEED and TV_T_Q8, one after another\n",
        c_strings("TV_SAMPLES", samples, STR_LEN),
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return samples


if __name__ == "__main__":
    p = dict(np.load("model_float.npz"))
    names = load_names("data/animals.txt")
    qp = quantize(p)
    H = qp["H"]

    print("scales  Mx/Sx:", list(zip(qp["Mx"], qp["Sx"])), " Mh/Sh:", list(zip(qp["Mh"], qp["Sh"])),
          " My/Sy:", (qp["My"], qp["Sy"]))

    fl, il = float_loss(p, names), int_loss(qp, names)
    print(f"\nloss on all names   float {fl:.4f}   int8 {il:.4f}   (diff {il - fl:+.4f})")

    f_ok = sum(gru.complete(p, n[:3], 0) == n for n in names)
    i_ok = sum(int_complete(qp, n[:3], 0)[0] == n for n in names)
    agree = sum(gru.complete(p, n[:3], 0) == int_complete(qp, n[:3], 0)[0] for n in names)
    print(f"greedy from 3 letters: float {f_ok}/{len(names)}   int8 {i_ok}/{len(names)}   "
          f"same answer {agree}/{len(names)}")

    print(f"largest |acc*M| seen: {MAX_SEEN['acc*M']}  (int32 limit 2147483647)")
    assert MAX_SEEN["acc*M"] < 2**31

    print("\nprefix  float              int8")
    for pre in ["ele", "kan", "pen", "cro", "sea ", "gir", "tig", "hum", "oct"]:
        print(f"  {pre:5s} {gru.complete(p, pre, 0):18s} {int_complete(qp, pre, 0)[0]}")

    flash = qp["Wx"].size + qp["Wh"].size + qp["Wy"].size + 2 * (4 * H + V) + 2 * (LUT_N + 1) + 2 * (LUT_N + 1)
    print(f"\nweights + biases + LUTs in flash: {flash} bytes ({flash / 1024:.1f} KB of 32 KB)")

    export_model_h(qp, "firmware/unonet/model.h")
    T_q8, seed = 205, 12345                                    # temperature 0.8
    samples = export_test_vectors(qp, "firmware/unonet/test_vectors.h",
                                  ["", "ele", "kan", "pen", "cro", "sea ", "gir", "hum"], T_q8, seed)
    print("wrote firmware/unonet/model.h and firmware/unonet/test_vectors.h")
    print("int8 samples (T=0.8, seed 12345):", ", ".join(samples))

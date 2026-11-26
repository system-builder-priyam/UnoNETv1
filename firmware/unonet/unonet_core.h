/*
 * unoNET inference core - integer-only GRU, a line-by-line copy of quantize.py.
 *
 *   int_step    -> uno_step      one letter in, hidden state updated
 *   int_logits  -> uno_logits    scores for the next letter (Q8)
 *   int_sample  -> uno_sample    greedy or temperature sampling, xorshift32 rng
 *   int_complete-> uno_complete  prefix in, finished name out
 *
 * Works on the UNO (weights read from flash) and on a PC (plain C, for testing).
 * AVR note: int is 16 bits there, so every sum/product that can grow is int32_t.
 */
#pragma once
#include <stdint.h>
#include <string.h>
#include "model.h"

#ifdef __AVR__
  #include <avr/pgmspace.h>
#else
  #define pgm_read_byte(p) (*(const uint8_t *)(p))
  #define pgm_read_word(p) (*(const uint16_t *)(p))
#endif

#define RD_I8(p)  ((int8_t)pgm_read_byte(p))
#define RD_U8(p)  ((uint8_t)pgm_read_byte(p))
#define RD_I16(p) ((int16_t)pgm_read_word(p))
#define RD_U16(p) ((uint16_t)pgm_read_word(p))

//requantize: (acc * M + round) >> S
static inline int32_t uno_rq(int32_t acc, int16_t M, uint8_t S) {
    return (acc * (int32_t)M + ((int32_t)1 << (S - 1))) >> S;
}

//sigmoid / tanh lookup with linear interpolation, pre is Q8 in [-8, 8)
static inline int16_t uno_sig(int32_t pre) {
    int32_t idx = pre + 2048;
    if (idx < 0) idx = 0;
    if (idx > 4095) idx = 4095;
    int16_t i = (int16_t)(idx >> 4), f = (int16_t)(idx & 15);
    int16_t a = RD_U8(&UNO_SIG_LUT[i]), b = RD_U8(&UNO_SIG_LUT[i + 1]);
    return a + (int16_t)((((int32_t)(b - a)) * f + 8) >> 4);
}

static inline int16_t uno_tanh(int32_t pre) {
    int32_t idx = pre + 2048;
    if (idx < 0) idx = 0;
    if (idx > 4095) idx = 4095;
    int16_t i = (int16_t)(idx >> 4), f = (int16_t)(idx & 15);
    int16_t a = RD_I8(&UNO_TANH_LUT[i]), b = RD_I8(&UNO_TANH_LUT[i + 1]);
    return a + (int16_t)((((int32_t)(b - a)) * f + 8) >> 4);
}

//dot product of one weight row (in flash) with h
static inline int32_t uno_dot(const int8_t *row, const int8_t *h) {
    int32_t acc = 0;
    for (uint8_t i = 0; i < UNO_H; i++)
        acc += (int16_t)RD_I8(&row[i]) * (int16_t)h[i];      // 127*127 fits int16, sum in int32
    return acc;
}

//one GRU step: letter x in, h updated in place
static void uno_step(uint8_t x, int8_t *h) {
    int8_t h_new[UNO_H];
    for (uint8_t j = 0; j < UNO_H; j++) {
        int32_t rec_r = uno_rq(uno_dot(UNO_WH[j], h),             UNO_MH[0], UNO_SH[0]);
        int32_t rec_z = uno_rq(uno_dot(UNO_WH[UNO_H + j], h),     UNO_MH[1], UNO_SH[1]);
        int32_t rec_n = uno_rq(uno_dot(UNO_WH[2 * UNO_H + j], h), UNO_MH[2], UNO_SH[2]);

        int32_t inp_r = uno_rq(RD_I8(&UNO_WX[x][j]),             UNO_MX[0], UNO_SX[0]);
        int32_t inp_z = uno_rq(RD_I8(&UNO_WX[x][UNO_H + j]),     UNO_MX[1], UNO_SX[1]);
        int32_t inp_n = uno_rq(RD_I8(&UNO_WX[x][2 * UNO_H + j]), UNO_MX[2], UNO_SX[2]);

        int32_t r = uno_sig(rec_r + inp_r + RD_I16(&UNO_B_R[j]));        // 0..255
        int32_t z = uno_sig(rec_z + inp_z + RD_I16(&UNO_B_Z[j]));        // 0..255
        int32_t hn = rec_n + RD_I16(&UNO_BH_N[j]);
        int32_t pre_n = inp_n + RD_I16(&UNO_BX_N[j]) + ((r * hn + 128) >> 8);
        int32_t n = uno_tanh(pre_n);                                     // -127..127

        h_new[j] = (int8_t)((n * (256 - z) + (int32_t)h[j] * z + 128) >> 8);
    }
    memcpy(h, h_new, UNO_H);
}

//scores for the next letter, Q8
static void uno_logits(const int8_t *h, int32_t *logits) {
    for (uint8_t v = 0; v < UNO_V; v++)
        logits[v] = uno_rq(uno_dot(UNO_WY[v], h), UNO_MY, UNO_SY) + RD_I16(&UNO_BY[v]);
}

//rng
static uint32_t uno_rng = 12345UL;

static inline uint32_t uno_xorshift32(void) {
    uint32_t s = uno_rng;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    uno_rng = s;
    return s;
}

//pick the next letter, T_q8 = temperature * 256, 0 = greedy
static uint8_t uno_sample(const int32_t *logits, int16_t T_q8) {
    uint8_t best = 0;
    for (uint8_t v = 1; v < UNO_V; v++)
        if (logits[v] > logits[best]) best = v;                 // strict '>' keeps the first max
    if (T_q8 == 0) return best;

    uint32_t w[UNO_V], total = 0;
    for (uint8_t v = 0; v < UNO_V; v++) {
        int32_t sc = ((logits[best] - logits[v]) * 256) / T_q8; // >= 0
        int32_t i = sc >> 4;
        if (i >= UNO_LUT_N) { w[v] = 0; continue; }             // beyond the table -> weight 0
        uint32_t e0 = RD_U16(&UNO_EXP_LUT[i]), e1 = RD_U16(&UNO_EXP_LUT[i + 1]);
        w[v] = e0 - (((e0 - e1) * (uint32_t)(sc & 15) + 8) >> 4);
        total += w[v];
    }
    uint32_t r = uno_xorshift32() % total;
    uint32_t cum = 0;
    for (uint8_t v = 0; v < UNO_V; v++) {
        cum += w[v];
        if (cum > r) return v;                                  // first v with running sum > r
    }
    return UNO_V - 1;
}

//letter <-> id
static int8_t uno_char_to_id(char c) {
    if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    for (uint8_t i = 0; i < UNO_V; i++)
        if (UNO_VOCAB[i] == c) return (int8_t)i;
    return -1;                                                  // not in the vocab
}

//prefix in, finished name out (out needs room for prefix + UNO_MAX_GEN + 1)
//returns the number of letters generated after the prefix
static uint8_t uno_complete(const char *prefix, int16_t T_q8, char *out) {
    int8_t h[UNO_H];
    int32_t logits[UNO_V];
    uint8_t len = 0, gen = 0;
    memset(h, 0, UNO_H);

    uno_step(0, h);                                             // start token '.'
    for (const char *p = prefix; *p; p++) {
        int8_t id = uno_char_to_id(*p);
        if (id <= 0) continue;                                  // skip unknown letters and '.'
        uno_step((uint8_t)id, h);
        out[len++] = UNO_VOCAB[id];
    }
    while (gen < UNO_MAX_GEN) {
        uno_logits(h, logits);
        uint8_t nxt = uno_sample(logits, T_q8);
        if (nxt == 0) break;                                    // end token
        out[len++] = UNO_VOCAB[nxt];
        gen++;
        uno_step(nxt, h);
    }
    out[len] = '\0';
    return gen;
}

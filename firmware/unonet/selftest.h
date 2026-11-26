/*
 * unoNET self-test: checks the C core against test_vectors.h (made by quantize.py).
 * Every number must match exactly - if it passes, the UNO runs the same model as Python.
 * Used by the sketch (/test command) and by pc_test/test_main.c.
 */
#pragma once
#include "unonet_core.h"
#include "test_vectors.h"

#ifdef __AVR__
  #include <avr/pgmspace.h>
#else
  #define pgm_read_dword(p) (*(const uint32_t *)(p))
  #define strcpy_P(d, s) strcpy((d), (s))
#endif

typedef void (*uno_print_fn)(const char *);

//returns the number of failed checks, prints details through print()
static uint8_t uno_selftest(uno_print_fn print) {
    uint8_t fails = 0;
    uint32_t saved_rng = uno_rng;
    char pre[TV_STR], want[TV_STR], got[TV_STR];

    //1. hidden state and logits after '.ele'
    int8_t h[UNO_H];
    int32_t logits[UNO_V];
    memset(h, 0, UNO_H);
    uno_step(0, h);
    uno_step((uint8_t)uno_char_to_id('e'), h);
    uno_step((uint8_t)uno_char_to_id('l'), h);
    uno_step((uint8_t)uno_char_to_id('e'), h);
    uint8_t ok = 1;
    for (uint8_t i = 0; i < UNO_H; i++)
        if (h[i] != RD_I8(&TV_H_ELE[i])) ok = 0;
    uno_logits(h, logits);
    for (uint8_t v = 0; v < UNO_V; v++)
        if (logits[v] != (int32_t)pgm_read_dword(&TV_LOGITS_ELE[v])) ok = 0;
    print(ok ? "  PASS  hidden state + logits after '.ele'\n"
             : "  FAIL  hidden state + logits after '.ele'\n");
    fails += !ok;

    //2. greedy completions
    for (uint8_t k = 0; k < TV_N_GREEDY; k++) {
        strcpy_P(pre, TV_PREFIX[k]);
        strcpy_P(want, TV_GREEDY[k]);
        uno_complete(pre, 0, got);
        ok = strcmp(want, got) == 0;
        fails += !ok;
        print(ok ? "  PASS  greedy  '" : "  FAIL  greedy  '");
        print(pre); print("' -> "); print(got);
        if (!ok) { print("   (expected "); print(want); print(")"); }
        print("\n");
    }

    //3. sampling with a fixed seed (tests the integer softmax + rng)
    uno_rng = TV_SEED;
    for (uint8_t k = 0; k < TV_N_SAMPLES; k++) {
        strcpy_P(want, TV_SAMPLES[k]);
        uno_complete("", TV_T_Q8, got);
        ok = strcmp(want, got) == 0;
        fails += !ok;
        print(ok ? "  PASS  sample  " : "  FAIL  sample  ");
        print(got);
        if (!ok) { print("   (expected "); print(want); print(")"); }
        print("\n");
    }

    uno_rng = saved_rng;
    print(fails ? "SELF-TEST FAILED\n" : "SELF-TEST PASSED - bit-exact with quantize.py\n");
    return fails;
}

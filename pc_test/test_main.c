/*
 * unoNET PC test: compiles the exact same C core as the UNO and runs the self-test.
 *
 *   gcc -O2 -Wall -o pc_test/unonet_test pc_test/test_main.c
 *   ./pc_test/unonet_test            self-test
 *   ./pc_test/unonet_test ele kan    also complete these prefixes
 */
#include <stdio.h>
#include "../firmware/unonet/selftest.h"

static void print_str(const char *s) { fputs(s, stdout); }

int main(int argc, char **argv) {
    printf("unoNET PC self-test\n");
    uint8_t fails = uno_selftest(print_str);

    char out[64];
    for (int i = 1; i < argc; i++) {
        uno_complete(argv[i], 0, out);
        printf("%s -> %s\n", argv[i], out);
    }
    return fails ? 1 : 0;
}

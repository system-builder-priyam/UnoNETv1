/*
 * unoNET - animal name predictor running on an Arduino UNO.
 * A 20K-parameter int8 GRU language model, weights stored in flash.
 *
 * Serial Monitor: 115200 baud, line ending "Newline".
 *   ele        complete a name: best guess + 3 samples
 *   (empty)    5 random animals
 *   /t 0.8     set sampling temperature (0 = always the best guess)
 *   /test      run the bit-exact self-test
 *   /help      show commands
 */
#include "unonet_core.h"
#include "selftest.h"

#define LINE_MAX 24

static char line[LINE_MAX + 1];
static uint8_t line_len = 0;
static int16_t temp_q8 = 205;                     // temperature 0.8 * 256

static void print_str(const char *s) { Serial.print(s); }

static void print_help() {
  Serial.println(F("Type the start of an animal name and press Enter."));
  Serial.println(F("  (empty line)  5 random animals"));
  Serial.println(F("  /t 0.8        set temperature (0 = best guess only)"));
  Serial.println(F("  /test         run the self-test"));
}

//seed the rng from noise on a floating analog pin + timing jitter
static void seed_rng() {
  uint32_t s = 0;
  for (uint8_t i = 0; i < 32; i++) {
    s = (s << 1) ^ (uint32_t)analogRead(A0) ^ (uint32_t)micros();
    delay(1);
  }
  uno_rng = s ? s : 12345UL;
}

//"0.8" -> 205, "1" -> 256, "0" -> 0
static int16_t parse_temp(const char *s) {
  int32_t whole = 0, frac = 0, div = 1;
  while (*s == ' ') s++;
  while (*s >= '0' && *s <= '9') whole = whole * 10 + (*s++ - '0');
  if (*s == '.') {
    s++;
    while (*s >= '0' && *s <= '9' && div < 1000) { frac = frac * 10 + (*s++ - '0'); div *= 10; }
  }
  int32_t q = whole * 256 + (frac * 256 + div / 2) / div;
  if (q > 1024) q = 1024;                         // cap at 4.0
  return (int16_t)q;
}

static void print_temp() {
  Serial.print(F("temperature = "));
  Serial.print(temp_q8 / 256);
  Serial.print('.');
  int16_t frac = (int16_t)(((int32_t)(temp_q8 % 256) * 100 + 128) / 256);
  if (frac < 10) Serial.print('0');
  Serial.println(frac);
}

static void handle_line() {
  char out[LINE_MAX + UNO_MAX_GEN + 1];

  if (strcmp(line, "/help") == 0) { print_help(); return; }
  if (strcmp(line, "/test") == 0) {
    Serial.println(F("running self-test..."));
    uno_selftest(print_str);
    return;
  }
  if (line[0] == '/' && line[1] == 't') { temp_q8 = parse_temp(line + 2); print_temp(); return; }

  if (line_len == 0) {                             // empty line: random animals
    Serial.println(F("random animals:"));
    for (uint8_t i = 0; i < 5; i++) {
      uno_complete("", temp_q8 ? temp_q8 : 205, out);
      Serial.print(F("  "));
      Serial.println(out);
    }
    return;
  }

  Serial.print(F("> "));
  Serial.println(line);
  uint32_t t0 = millis();
  uint8_t gen = uno_complete(line, 0, out);
  uint32_t dt = millis() - t0;
  Serial.print(F("  best:  "));
  Serial.println(out);

  if (temp_q8 > 0) {
    Serial.print(F("  maybe: "));
    for (uint8_t i = 0; i < 3; i++) {
      uno_complete(line, temp_q8, out);
      if (i) Serial.print(F(" | "));
      Serial.print(out);
    }
    Serial.println();
  }
  Serial.print(F("  ("));
  Serial.print(dt);
  Serial.print(F(" ms for the best guess, "));
  Serial.print(gen ? dt / (gen + strlen(line) + 1) : dt);
  Serial.println(F(" ms per letter)"));
}

void setup() {
  Serial.begin(115200);
  seed_rng();
  Serial.println();
  Serial.println(F("=== unoNET - animal name predictor ==="));
  Serial.println(F("int8 GRU, 64 hidden units, 20K params in flash"));
  print_help();
  print_temp();
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[line_len] = '\0';
      handle_line();
      line_len = 0;
    } else if (line_len < LINE_MAX) {
      line[line_len++] = c;
    }
  }
}

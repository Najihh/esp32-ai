// Verifies int8 staging, scale alignment, ranged matvec equivalence, and
// layer/head hook dispatch - the code path the device runs, which verify.c
// (exact int4) does not reach.
//
// Staging unpacks the same nibbles and scales, so results must be bit-identical
// to the unstaged int8 path.
//
//   cc -O3 -DLLM_INT8_ACT=1 -o /tmp/sv firmware/host_verify/staging_verify.c -lm
//   /tmp/sv firmware/model/model.bin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/llm.h"

static int failures = 0;

static void check(const char *what, int ok, const char *detail) {
  printf("  %-46s %s%s%s\n", what, ok ? "ok" : "FAIL",
         detail && *detail ? "  " : "", detail ? detail : "");
  if (!ok) failures++;
}

// --- hook dispatch ----------------------------------------------------------
static int layer_hook_calls = 0, head_hook_calls = 0;

static void layer_hook(const QT *t, const float *x, float *y) {
  layer_hook_calls++;
  MATVEC(t, x, y);
}
static void head_hook(const QT *t, const float *x, float *y) {
  head_hook_calls++;
  MATVEC(t, x, y);
}

static double max_abs_diff(const float *a, const float *b, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  return m;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "firmware/model/model.bin";
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *buf = malloc(sz);
  if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 2; }
  fclose(f);

  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 2; }
  printf("model: V=%d D=%d L=%d ffn=%d ple=%d  image=%zu bytes\n\n",
         m.c.vocab, m.c.dim, m.c.n_layers, m.c.ffn, m.c.ple_dim, m.image_bytes);

  printf("image_bytes\n");
  check("equals file size", (long)m.image_bytes == sz, "");

  // --- staging arithmetic ---------------------------------------------------
  printf("\nllm_stage_int8: staged == unstaged\n");
  int D = m.c.dim, P = m.c.ple_dim;

  float *x = malloc(sizeof(float) * 4096);
  for (int i = 0; i < 4096; i++) x[i] = sinf((float)i * 0.37f) * 1.7f;

  // One tensor of each distinct shape the model contains.
  struct { const char *name; QT *t; } cases[] = {
    {"ple_model_proj [L*P, D]", &m.ple_model_proj},
    {"qkv[0]         [3D, D]",  &m.qkv[0]},
    {"attn_proj[0]   [D, D]",   &m.attn_proj[0]},
    {"gate[0]        [F, D]",   &m.gate[0]},
    {"down[0]        [D, F]",   &m.down[0]},
    {"ple_gate[0]    [P, D]",   &m.ple_gate[0]},
    {"ple_proj[0]    [D, P]",   &m.ple_proj[0]},
  };
  int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

  for (int i = 0; i < ncases; i++) {
    QT *t = cases[i].t;
    int rows = t->rows;
    float *ref = malloc(sizeof(float) * rows);
    float *got = malloc(sizeof(float) * rows);

    MATVEC(t, x, ref);                       // unstaged: packed int4 nibbles

    void *sbuf = malloc(llm_stage_int8_bytes(t));
    llm_stage_int8(t, sbuf);
    MATVEC(t, x, got);                       // staged: pre-unpacked int8

    char detail[128];
    double d = max_abs_diff(ref, got, rows);
    snprintf(detail, sizeof detail, "rows=%-5d max|diff|=%.3g", rows, d);
    check(cases[i].name, d == 0.0, detail);

    // scale array must be 4-byte aligned wherever the int8 block ends
    check("  scale8 aligned",
          ((uintptr_t)t->scale8 % sizeof(float)) == 0, "");

    free(ref); free(got);
  }

  printf("\nrow-range split == whole tensor\n");
  {
    QT *t = &m.qkv[0];                       // already staged above
    int rows = t->rows;
    float *whole = malloc(sizeof(float) * rows);
    float *split = malloc(sizeof(float) * rows);
    int8_t *xq = malloc(t->cols);
    float xs;
    quantize_act(x, t->cols, xq, &xs);
    matvec_i8_range(t, xq, xs, whole, 0, rows);
    matvec_i8_range(t, xq, xs, split, 0, rows / 2);
    matvec_i8_range(t, xq, xs, split, rows / 2, rows);
    check("halves == whole",
          max_abs_diff(whole, split, rows) == 0.0, "");
    free(whole); free(split); free(xq);
  }

  // --- hook dispatch --------------------------------------------------------
  printf("\nplatform hook dispatch\n");
  Scratch s;
  memset(&s, 0, sizeof s);
  int L = m.c.n_layers, F = m.c.ffn, S = m.c.seq_len, V = m.c.vocab;
  s.x = calloc(D, 4); s.h = calloc(F > D ? F : D, 4);
  s.qkv = calloc(3 * D, 4); s.att = calloc(D, 4);
  s.g1 = calloc(F, 4); s.g2 = calloc(P > F ? P : F, 4);
  s.ple = calloc(L * P, 4); s.tmpP = calloc(L * P, 4); s.trow = calloc(L * P, 4);
  s.logits = calloc(V, 4); s.scores = calloc(S, 4);
  s.kcache = calloc((size_t)L * S * D, 4); s.vcache = calloc((size_t)L * S * D, 4);

  float *logits_nohook = malloc(sizeof(float) * V);
  llm_forward(&m, 42, 0, &s);
  memcpy(logits_nohook, s.logits, sizeof(float) * V);
  check("no hooks: baseline forward ran", 1, "");

  m.layer_matvec = layer_hook;
  m.head_matvec  = head_hook;
  llm_forward(&m, 42, 0, &s);

  // 1 ple_model_proj + 7 per layer
  int expect_layer = 1 + 7 * L;
  char d2[64];
  snprintf(d2, sizeof d2, "%d calls, expected %d", layer_hook_calls, expect_layer);
  check("layer_matvec called for every per-position matvec",
        layer_hook_calls == expect_layer, d2);
  snprintf(d2, sizeof d2, "%d calls, expected 1", head_hook_calls);
  check("head_matvec called once", head_hook_calls == 1, d2);
  check("hooked forward == unhooked forward",
        max_abs_diff(logits_nohook, s.logits, V) == 0.0, "");

  printf("\n%s (%d failure%s)\n", failures ? "FAIL" : "PASS",
         failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}

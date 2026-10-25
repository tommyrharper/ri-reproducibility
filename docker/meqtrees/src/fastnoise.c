/* Thermal noise for simulate_point_source_ms.py, bit for bit what
 *
 *     dst += rng.normal(0.0, scale, n).astype(np.float32)
 *
 * gives for a numpy 1.26 PCG64 Generator, without the per-draw call through
 * the bitgen function pointer or the two float64 temporaries. The generator
 * is numpy's PCG64 (XSL-RR 128/64, step then output) and the sampler is its
 * 256-layer ziggurat, random_standard_normal() in distributions.c, copied
 * line for line, except that its random sign is applied without a branch;
 * normal() is loc + scale * that. dst is strided so the real and imaginary
 * halves of a complex64 array can be filled in place. state is
 * {state_hi, state_lo, inc_hi, inc_lo} and is advanced, so a second call
 * continues the stream exactly as a second rng.normal() would.
 *
 * The Python side checks this against numpy once per process and falls back
 * to numpy if they ever disagree (add_complex_noise() in
 * simulate_point_source_ms.py). */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "ziggurat_double.h"

typedef unsigned __int128 u128;

#define PCG_MULT (((u128)2549297995355413924ULL << 64) | 4865540595714422341ULL)

static inline uint64_t next64(u128 *state, u128 inc) {
  *state = *state * PCG_MULT + inc;
  uint64_t hi = (uint64_t)(*state >> 64), lo = (uint64_t)*state;
  unsigned rot = (unsigned)(hi >> 58u);
  uint64_t v = hi ^ lo;
  return (v >> rot) | (v << ((-rot) & 63u));
}

static inline double next_double(u128 *state, u128 inc) {
  return (next64(state, inc) >> 11) * (1.0 / 9007199254740992.0);
}

static inline double standard_normal(u128 *s, u128 inc) {
  for (;;) {
    uint64_t r = next64(s, inc);
    int idx = r & 0xff;
    r >>= 8;
    int sign = r & 0x1;
    uint64_t rabs = (r >> 1) & 0x000fffffffffffff;
    /* Negation by flipping the sign bit: the same double as numpy's
     * `if (sign) x = -x`, without a branch that mispredicts half the time. */
    union { double d; uint64_t u; } bits = {rabs * wi_double[idx]};
    bits.u ^= (uint64_t)sign << 63;
    double x = bits.d;
    if (rabs < ki_double[idx])
      return x;
    if (idx == 0) {
      for (;;) {
        double xx = -ziggurat_nor_inv_r * log1p(-next_double(s, inc));
        double yy = -log1p(-next_double(s, inc));
        if (yy + yy > xx * xx)
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx)
                                     : ziggurat_nor_r + xx;
      }
    } else {
      if (((fi_double[idx - 1] - fi_double[idx]) * next_double(s, inc) +
           fi_double[idx]) < exp(-0.5 * x * x))
        return x;
    }
  }
}

void add_normal_f32(uint64_t state[4], double scale, float *dst, ptrdiff_t n,
                    ptrdiff_t stride, ptrdiff_t every) {
  /* n * every draws; the first of each `every` is added, the rest are drawn
   * only to keep the stream (numpy's `normal(...)[..., 0]`). */
  u128 s = ((u128)state[0] << 64) | state[1];
  u128 inc = ((u128)state[2] << 64) | state[3];
  for (ptrdiff_t i = 0; i < n; i++) {
    dst[i * stride] += (float)(0.0 + scale * standard_normal(&s, inc));
    for (ptrdiff_t k = 1; k < every; k++)
      standard_normal(&s, inc);
  }
  state[0] = (uint64_t)(s >> 64);
  state[1] = (uint64_t)s;
}

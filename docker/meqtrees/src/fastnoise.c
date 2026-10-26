/* Thermal noise for simulate_point_source_ms.py, bit for bit what
 *
 *     dst += rng.normal(0.0, scale, n).astype(np.float32)
 *
 * gives for a numpy 1.26 PCG64 Generator, without the per-draw call through
 * the bitgen function pointer or the two float64 temporaries. The generator
 * is numpy's PCG64 (XSL-RR 128/64, step then output) and the sampler is its
 * 256-layer ziggurat, random_standard_normal() in distributions.c, copied
 * line for line, except that its random sign is applied without a branch
 * and its rejection paths are out of line (slow_normal()); normal() is
 * loc + scale * that. dst is strided so the real and imaginary
 * halves of a complex64 array can be filled in place. state is
 * {state_hi, state_lo, inc_hi, inc_lo} and is advanced, so a second call
 * continues the stream exactly as a second rng.normal() would.
 *
 * The Python side checks this against numpy once per process and falls back
 * to numpy if they ever disagree (add_complex_noise() in
 * simulate_point_source_ms.py).
 *
 * phase_ramp_c64() is the point-source predict the noise is added to, bit
 * for bit what point_source_visibilities() computes with numpy. */
#define _GNU_SOURCE /* sincos */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "ziggurat_double.h"

typedef unsigned __int128 u128;

#define PCG_MULT_HI 2549297995355413924ULL
#define PCG_MULT_LO 4865540595714422341ULL
#define PCG_MULT (((u128)PCG_MULT_HI << 64) | PCG_MULT_LO)

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

typedef struct {
  u128 s;
  double x;
} draw;

/* The ziggurat's rejection paths (~1.2% of draws), out of line: with them
 * inline, gcc kept the state on the stack and its store and reload sat on
 * the multiply chain of every draw (docs/csd3-speed.md, round 56). */
static __attribute__((noinline)) draw slow_normal(u128 s, u128 inc, int idx,
                                                  uint64_t rabs, double x) {
  for (;;) {
    if (idx == 0) {
      for (;;) {
        double xx = -ziggurat_nor_inv_r * log1p(-next_double(&s, inc));
        double yy = -log1p(-next_double(&s, inc));
        if (yy + yy > xx * xx)
          return (draw){s, ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx)
                                                : ziggurat_nor_r + xx};
      }
    } else {
      if (((fi_double[idx - 1] - fi_double[idx]) * next_double(&s, inc) +
           fi_double[idx]) < exp(-0.5 * x * x))
        return (draw){s, x};
    }
    /* Rejected: a fresh draw, as numpy's loop does. */
    uint64_t r = next64(&s, inc);
    idx = r & 0xff;
    r >>= 8;
    int sign = r & 0x1;
    rabs = (r >> 1) & 0x000fffffffffffff;
    union { double d; uint64_t u; } bits = {rabs * wi_double[idx]};
    bits.u ^= (uint64_t)sign << 63;
    x = bits.d;
    if (rabs < ki_double[idx])
      return (draw){s, x};
  }
}

static inline double fast_normal(uint64_t *hi, uint64_t *lo, uint64_t inc_hi,
                                 uint64_t inc_lo, u128 inc) {
  /* next64() on the state as two words, which gcc keeps in registers. */
  u128 p = (u128)*lo * PCG_MULT_LO + inc_lo;
  uint64_t h = (uint64_t)(p >> 64) + *hi * PCG_MULT_LO + *lo * PCG_MULT_HI + inc_hi;
  uint64_t l = (uint64_t)p;
  *hi = h;
  *lo = l;
  unsigned rot = (unsigned)(h >> 58u);
  uint64_t r = ((h ^ l) >> rot) | ((h ^ l) << ((-rot) & 63u));
  int idx = r & 0xff;
  r >>= 8;
  int sign = r & 0x1;
  uint64_t rabs = (r >> 1) & 0x000fffffffffffff;
  /* Negation by flipping the sign bit: the same double as numpy's
   * `if (sign) x = -x`, without a branch that mispredicts half the time. */
  union { double d; uint64_t u; } bits = {rabs * wi_double[idx]};
  bits.u ^= (uint64_t)sign << 63;
  if (__builtin_expect(rabs < ki_double[idx], 1))
    return bits.d;
  draw d = slow_normal(((u128)h << 64) | l, inc, idx, rabs, bits.d);
  *hi = (uint64_t)(d.s >> 64);
  *lo = (uint64_t)d.s;
  return d.x;
}

void add_normal_f32(uint64_t state[4], double scale, float *dst, ptrdiff_t n,
                    ptrdiff_t stride, ptrdiff_t every) {
  /* n * every draws; the first of each `every` is added, the rest are drawn
   * only to keep the stream (numpy's `normal(...)[..., 0]`). */
  uint64_t hi = state[0], lo = state[1], inc_hi = state[2], inc_lo = state[3];
  u128 inc = ((u128)inc_hi << 64) | inc_lo;
  /* One loop over all n * every draws, so gcc inlines one copy of
   * fast_normal() and has the registers to keep the state out of memory. */
  if (every < 1)
    every = 1;
  float *out = dst, *end = dst + (n > 0 ? n : 0) * stride;
  ptrdiff_t k = 0;
  while (out != end) {
    double x = fast_normal(&hi, &lo, inc_hi, inc_lo, inc);
    if (k == 0)
      *out += (float)(0.0 + scale * x);
    if (++k == every) {
      k = 0;
      out += stride;
    }
  }
  state[0] = hi;
  state[1] = lo;
}

void phase_ramp_c64(const double *delay, ptrdiff_t n_rows, const double *freqs,
                    ptrdiff_t n_chan, double k, double flux, float *out,
                    ptrdiff_t n_corr) {
  /* flux * np.exp(1j * k * np.outer(delay, freqs)), cast to complex64, into
   * correlations 0 and n_corr - 1 of a (rows, chans, n_corr) array. numpy's
   * complex exp is libm's cexp, which for a zero real part is sincos of the
   * imaginary part times exp(0) = 1; ~1.8x faster than numpy, which goes
   * through complex128 temporaries (docs/csd3-speed.md, round 32). */
  ptrdiff_t last = 2 * (n_corr - 1);
  for (ptrdiff_t r = 0; r < n_rows; r++) {
    for (ptrdiff_t c = 0; c < n_chan; c++) {
      double s, co;
      sincos(k * (delay[r] * freqs[c]), &s, &co);
      float *v = out + 2 * n_corr * (r * n_chan + c);
      v[0] = v[last] = (float)(flux * co);
      v[1] = v[last + 1] = (float)(flux * s);
    }
  }
}

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

/* fast_normal() for a draw nobody keeps: the same steps of the stream, but
 * the double is built only on the rejection path that needs it. */
static inline void skip_normal(uint64_t *hi, uint64_t *lo, uint64_t inc_hi,
                               uint64_t inc_lo, u128 inc) {
  u128 p = (u128)*lo * PCG_MULT_LO + inc_lo;
  uint64_t h = (uint64_t)(p >> 64) + *hi * PCG_MULT_LO + *lo * PCG_MULT_HI + inc_hi;
  uint64_t l = (uint64_t)p;
  *hi = h;
  *lo = l;
  unsigned rot = (unsigned)(h >> 58u);
  uint64_t r = ((h ^ l) >> rot) | ((h ^ l) << ((-rot) & 63u));
  int idx = r & 0xff;
  r >>= 8;
  uint64_t rabs = (r >> 1) & 0x000fffffffffffff;
  if (__builtin_expect(rabs < ki_double[idx], 1))
    return;
  union { double d; uint64_t u; } bits = {rabs * wi_double[idx]};
  bits.u ^= (uint64_t)(r & 0x1) << 63;
  draw d = slow_normal(((u128)h << 64) | l, inc, idx, rabs, bits.d);
  *hi = (uint64_t)(d.s >> 64);
  *lo = (uint64_t)d.s;
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

/* add_normal_f32(..., every=1) over n draws, n a multiple of 4, keeping only
 * the first and last of each 4: XX and YY of a 4-correlation DATA column,
 * the only two `-pol I` imaging reads. XY and YX keep their stream position
 * and get no noise, which is -15% on the noise (docs/csd3-speed.md, round 65).
 * yy is YY's offset in dst: 3 for a 4-correlation column, 1 for one that
 * holds XX and YY alone (round 66). */
static inline void xxyy_body(uint64_t state[4], double scale, float *dst,
                             ptrdiff_t n, ptrdiff_t stride, ptrdiff_t yy) {
  uint64_t hi = state[0], lo = state[1], inc_hi = state[2], inc_lo = state[3];
  u128 inc = ((u128)inc_hi << 64) | inc_lo;
  for (float *end = dst + (n > 0 ? n / 4 : 0) * (yy + 1) * stride; dst != end; dst += (yy + 1) * stride) {
    dst[0] += (float)(0.0 + scale * fast_normal(&hi, &lo, inc_hi, inc_lo, inc));
    skip_normal(&hi, &lo, inc_hi, inc_lo, inc);
    skip_normal(&hi, &lo, inc_hi, inc_lo, inc);
    dst[yy * stride] += (float)(0.0 + scale * fast_normal(&hi, &lo, inc_hi, inc_lo, inc));
  }
  state[0] = hi;
  state[1] = lo;
}

void add_normal_f32_xxyy(uint64_t state[4], double scale, float *dst,
                         ptrdiff_t n, ptrdiff_t stride) {
  xxyy_body(state, scale, dst, n, stride, 3);
}

/* The same n draws into a 2-correlation (XX, YY) column: n / 2 values. */
void add_normal_f32_xxyy2(uint64_t state[4], double scale, float *dst,
                          ptrdiff_t n, ptrdiff_t stride) {
  xxyy_body(state, scale, dst, n, stride, 1);
}

typedef double v4d __attribute__((vector_size(32)));
typedef int64_t v4i __attribute__((vector_size(32)));

/* sin and cos of four |x| < 2^19 by Cody-Waite reduction to [-pi/4, pi/4]
 * and fdlibm's kernel polynomials: within 2.3e-16 of glibc 2.28's and 2.39's
 * sincos over 2e8 draws (docs/csd3-speed.md, round 57). */
static inline void poly_sincos4(const v4d *xp, v4d *s, v4d *c) {
  v4d x = *xp;
  const v4d shift = {0x1.8p52, 0x1.8p52, 0x1.8p52, 0x1.8p52};
  v4d t = x * 6.36619772367581382433e-01 + shift; /* 2/pi */
  v4d n = t - shift;
  v4i q = (v4i)t;
  v4d r = x - n * 1.57079632673412561417e+00;
  r = r - n * 6.07710050630396597660e-11;
  r = r - n * 2.02226624871116645580e-21;
  v4d z = r * r;
  v4d sp = r + r * z * (-1.66666666666666324348e-01 + z * (8.33333333332248946124e-03
      + z * (-1.98412698298579493134e-04 + z * (2.75573137070700676789e-06
      + z * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10)))));
  v4d cp = 1.0 - 0.5 * z + z * z * (4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03
      + z * (2.48015872894767294178e-05 + z * (-2.75573143513906633035e-07
      + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11)))));
  v4i swap = -(q & 1);
  v4i si = ((v4i)cp & swap) | ((v4i)sp & ~swap), ci = ((v4i)sp & swap) | ((v4i)cp & ~swap);
  *s = (v4d)(si ^ ((q & 2) << 62));
  *c = (v4d)(ci ^ (((q + 1) & 2) << 62));
}

/* Whether every double within err of v rounds to the same float. */
static inline int rounds_alike(double v, double err) {
  return (float)(v - err) == (float)(v + err);
}

static inline __attribute__((always_inline))
void ramp_body(const double *delay, ptrdiff_t n_rows, const double *freqs,
               ptrdiff_t n_chan, double k, double flux, float *out,
               ptrdiff_t n_corr, uint64_t *state, double scale) {
  /* flux * np.exp(1j * k * np.outer(delay, freqs)), cast to complex64, into
   * correlations 0 and n_corr - 1 of a (rows, chans, n_corr) array. numpy's
   * complex exp is libm's cexp, which for a zero real part is sincos of the
   * imaginary part times exp(0) = 1, so libm's sincos is the reference.
   * poly_sincos4() does four phases at a time; where a product lies close
   * enough to a float32 rounding boundary that its error could matter, or a
   * phase is out of its range, that phase is redone with libm, so the output
   * is libm's bit for bit (docs/csd3-speed.md, rounds 32 and 57). */
  enum { BLOCK = 256 };
  ptrdiff_t last = 2 * (n_corr - 1), total = n_rows * n_chan, r = 0, c = 0;
  /* poly_sincos4()'s distance from libm with ~40x to spare, scaled by
   * flux; the products' own rounding is the relative term. The blocks keep
   * the scalar phase stores and vector loads apart, which would otherwise
   * stall on store forwarding. */
  double err = 1e-14 * fabs(flux);
  uint64_t hi = state ? state[0] : 0, lo = state ? state[1] : 0;
  uint64_t inc_hi = state ? state[2] : 0, inc_lo = state ? state[3] : 0;
  u128 inc = ((u128)inc_hi << 64) | inc_lo;
  v4d x[BLOCK / 4], re[BLOCK / 4], im[BLOCK / 4];
  for (ptrdiff_t i0 = 0; i0 < total; i0 += BLOCK) {
    ptrdiff_t m = total - i0 < BLOCK ? total - i0 : BLOCK;
    double *xs = (double *)x;
    for (ptrdiff_t j = 0; j < m; j++) {
      xs[j] = k * (delay[r] * freqs[c]);
      if (++c == n_chan) {
        c = 0;
        r++;
      }
    }
    for (ptrdiff_t j = m; j % 4; j++)
      xs[j] = 0.0;
    for (ptrdiff_t j = 0; j < (m + 3) / 4; j++) {
      v4d s, co;
      poly_sincos4(&x[j], &s, &co);
      re[j] = flux * co;
      im[j] = flux * s;
    }
    const double *res = (const double *)re, *ims = (const double *)im;
    for (ptrdiff_t j = 0; j < m; j++) {
      float *v = out + 2 * n_corr * (i0 + j);
      double a = res[j], b = ims[j];
      if (!(fabs(xs[j]) < 0x1p19 && rounds_alike(a, err + fabs(a) * 0x1p-50)
            && rounds_alike(b, err + fabs(b) * 0x1p-50))) {
        double ss, cc;
        sincos(xs[j], &ss, &cc);
        a = flux * cc;
        b = flux * ss;
      }
      v[0] = v[last] = (float)a;
      v[1] = v[last + 1] = (float)b;
      if (state) {
        /* xxyy_body()'s real half for this element, while v is in L1. */
        v[0] += (float)(0.0 + scale * fast_normal(&hi, &lo, inc_hi, inc_lo, inc));
        skip_normal(&hi, &lo, inc_hi, inc_lo, inc);
        skip_normal(&hi, &lo, inc_hi, inc_lo, inc);
        v[2] += (float)(0.0 + scale * fast_normal(&hi, &lo, inc_hi, inc_lo, inc));
      }
    }
  }
  if (state) {
    state[0] = hi;
    state[1] = lo;
  }
}

__attribute__((target_clones("avx2", "default")))
void phase_ramp_c64(const double *delay, ptrdiff_t n_rows, const double *freqs,
                    ptrdiff_t n_chan, double k, double flux, float *out,
                    ptrdiff_t n_corr) {
  ramp_body(delay, n_rows, freqs, n_chan, k, flux, out, n_corr, NULL, 0.0);
}

/* phase_ramp_c64() into 2-correlation data, then add_normal_f32_xxyy2()'s
 * real halves, in one pass: the predict is written with its noise rather
 * than written and then read back and written again, which under a full
 * node's L3 pressure is a DATA-sized trip to memory (docs/csd3-speed.md,
 * round 82). The imaginary halves still come from add_normal_f32_xxyy2(),
 * because they are the second half of the stream. */
__attribute__((target_clones("avx2", "default")))
void phase_ramp_c64_xxyy2(const double *delay, ptrdiff_t n_rows, const double *freqs,
                          ptrdiff_t n_chan, double k, double flux, float *out,
                          uint64_t state[4], double scale) {
  ramp_body(delay, n_rows, freqs, n_chan, k, flux, out, 2, state, scale);
}

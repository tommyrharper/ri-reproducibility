# The host's 65W power cap is the bottleneck

Twelve rounds of profiling have taken the per-evaluation budget apart
([docs/nested-sampling-throughput.md](nested-sampling-throughput.md)) and it is
now 84% the `wsclean` binary, of which ~75% is the major-cycle loop the search's
own hyperparameters fix. Nothing schedulable is left in software.

What *is* left is the machine. This host runs every search at a **65W package
power limit**, and that limit - not thermals, not memory bandwidth, not the
scheduler - is what sets the all-core clock the workers run at. Lifting it is
worth about **+26% evaluations per second**, it is a single write to a sysfs
file, and it needs root, which is why this is written down rather than shipped.

Everything below was measured on the current tree (`--nlive 25 --num-repeats
10 --max-ndead -1 --mpi-procs 20`, i.e. 19 workers), 29 August 2026.

## The limits, as configured

```
$ cat /sys/class/powercap/intel-rapl/intel-rapl:0/name
package-0
$ cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_name          # long_term  (PL1)
$ cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_power_limit_uw
65000000
$ cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_time_window_us
7995392
$ cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_1_name          # short_term (PL2)
$ cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_1_power_limit_uw
117000000
```

65W sustained, 117W for bursts, averaged over an 8-second window. The i5-13500
is a 65W-base part, so this is the stock configuration; 117W is the CPU's own
rated turbo power, not an overclock.

`energy_uj` is root-only on this kernel and neither `perf` nor `turbostat` is
installed, so the package power itself cannot be read from this account. The
evidence below is the clock, the temperature and the throttle counters, which
are all readable - and they agree.

## The transition is visible inside a single run

Sampling `/proc/cpuinfo` every 0.3s from before the search starts, then
aligning it against the `mtime` of every `evaluations/eval-*/metrics.json`
(seed 7, 6626 evaluations, 108s; `t` is seconds since the first evaluation
landed):

| window | evals/s | mean MHz | busiest core | package °C |
|---|---:|---:|---:|---:|
| 0-2s | 82.5 | 3834 | 4232 | 81.5 |
| 2-4s | 81.5 | 3799 | 4160 | 83.7 |
| 4-6s | 77.0 | 3360 | 3655 | 77.0 |
| 6-8s | 61.5 | 2975 | 3178 | 68.4 |
| 8-10s | 65.5 | 2935 | 3136 | 67.7 |
| 10-12s | 64.5 | 2960 | 3150 | 67.6 |
| 20-22s | 67.0 | 2937 | 3134 | 69.3 |
| 40-50s | 63.6 | 2948 | 3140 | 69.7 |
| 60-70s | 59.4 | 2938 | 3126 | 70.9 |
| 80-90s | 60.2 | 2934 | 3117 | 71.4 |

Read the first two rows against the rest:

- **Clock ratio** 3817 / 2945 = **1.30**
- **Throughput ratio** 82.0 / 65.0 = **1.26**

Throughput is clock-proportional, and the clock steps down once - at t≈5s,
which is one PL1 averaging window after the load arrives. A second run at seed
4242 reproduces it (80-84 evals/s for the first 4s, 64-67 after).

## It is not thermal

```
$ cat /sys/devices/system/cpu/cpu*/thermal_throttle/package_throttle_count
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
$ cat /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

Zero, across ten days of uptime that include every benchmark in this repo's
docs. The package sits at **68-71°C** through a steady search and touched
**83.7°C** at its hottest, inside the 117W burst. `Tjmax` for this part is
100°C. The cooler has margin; the power budget does not.

## What the cap costs, by worker count

Four searches at the same seed, sampled for 25s after 45s of warm-up so every
arm is past the burst window, killed once sampled. `busiest core` is the
maximum `cpu MHz` across the 20 hardware threads (the all-thread mean is
useless here - an idle core on this host reports ~2500 MHz, not a parked
clock).

| ranks | workers | busiest core | evals/s | ms per evaluation per worker |
|---:|---:|---:|---:|---:|
| 4 | 3 | 4532 MHz | 24.9 | 120 |
| 8 | 7 | 3700 MHz | 40.8 | 172 |
| 14 | 13 | 3329 MHz | 56.7 | 229 |
| 20 | 19 | 3111 MHz | 62.2 | 306 |

The clock falls **31%** between 3 and 19 workers at a constant 65W. This is the
same curve [the throughput doc](nested-sampling-throughput.md#the-wall-is-the-clock-not-the-memory-bus)
already reports as "the all-core clock"; what is new is that it is a *budget*,
with a documented 1.8x of headroom above it, rather than a property of the
silicon.

Note also that the scan is worth having on its own: at 19 workers a WSClean
search now runs at **62 evals/s steady** (against the 43-45 the same table
measured before iterations 10-12), and the last four ranks are still worth
+9.7% - so `--mpi-procs 20` remains right for WSClean, and the RAM those ranks
cost is still better spent on `--nlive` for a memory-capped R2D2 search.

## Raising it

One command, as root:

```sh
# 95W rather than the full 117W: the 83.7°C above was reached in a 4-second
# burst, which does not reach steady-state temperature. Step it up and watch.
echo 95000000 | sudo tee /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_power_limit_uw
```

Then verify, during a search:

```sh
# clock: should sit well above the 2940 MHz below
awk '/cpu MHz/{if($4>m)m=$4} END{print m}' /proc/cpuinfo
# temperature: keep it under ~90°C
cat /sys/class/thermal/thermal_zone1/temp        # x86_pkg_temp, milli-°C
# throttling: must stay at zero
cat /sys/devices/system/cpu/cpu*/thermal_throttle/package_throttle_count
```

Revert with `echo 65000000 | sudo tee ...`. The setting does not survive a
reboot, and it applies to the whole host, so on a machine several sessions
share it is a decision for whoever owns the box, not for a run.

Expected: **+26% evaluations per second at the full 117W** - that is the
measured burst-window ratio, not an extrapolation - and proportionally less at
95W. On the `--nlive 500 --num-repeats 25` run this repo is aiming at
(~196,000 evaluations, ~60 minutes at 62 evals/s), that is ~13 minutes.

If the temperature does climb into the 90s, the honest reading is that this
host is cooled for a 65W part and the win needs a better cooler rather than a
bigger number.

## Corrections to what was written before

- **"The first four seconds of any measurement lie by 20%"**
  ([throughput doc](nested-sampling-throughput.md#the-first-four-seconds-of-any-measurement-lie-by-20))
  is right about the effect and wrong about the cause. It is not the clock
  "settling"; it is PL2 (117W) giving way to PL1 (65W) after the 8-second
  averaging window fills. The practical rule is unchanged and now has a number
  attached to it: **any measurement arm shorter than ~10 seconds of sustained
  load is measuring the 117W machine**, and the burst recharges during the idle
  gap between arms, so back-to-back sweeps get one per arm.
- **"stock behaviour for an i5-13500 ... all-core sustained clock is around
  3GHz"** ([same doc](nested-sampling-throughput.md#the-wall-is-the-clock-not-the-memory-bus))
  is true but incomplete. 3GHz is what 65W buys at 19 workers. The part will do
  ~3.8GHz all-core at its own rated 117W, and the 1.59x clock component of the
  2.21x concurrency inflation splits into ~1.22x of all-core-versus-single-core
  turbo (not recoverable) and ~1.30x of PL1 (recoverable).

## Run-level costs, measured and closed off

While looking for something that grows with run size, the end-of-run and
resume paths were timed against a synthetic 196,000-evaluation `evaluations/`
directory - the size the target `--nlive 500 --num-repeats 25` run reaches:

| | at 196,000 evaluations |
|---|---:|
| `adopt_completed_evaluations` (every rank, every restart) | 7.8s |
| `load_evaluations_from_dir` (rank 0, once) | 7.6s |
| `write_json_atomic` of the 880MB `summary.json` | 9.5s |

25 seconds on a 60-minute run, and the two that matter are paid once. There is
nothing here worth changing; recorded so the next iteration does not measure it
again.

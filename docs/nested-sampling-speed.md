# Making a nested-sampling search faster: the index

Current operating points on this host, from `./ri bench`:

| search | evaluations/second | per-worker memory | ranks x threads |
|---|---:|---:|---|
| WSClean | **109** | 34.5 MB | 20 x 1 |
| R2D2 | **0.91** | 3.47 GB | 15 x 2 |

For the 9-parameter space on CSD3, see [csd3-speed.md](csd3-speed.md).

Thirty-three profiling rounds got there. Values in the tables below are
per-round baselines, except the two figures above; pre-round-18 rates are
historical and roughly half the current rate.

## Round 33: R2D2 is 2.1x faster, WSClean is at its floor

R2D2 went from **0.436 to 0.913 evaluations/second** on the default benchmark
preset - a **2.1x** speedup - with per-worker memory unchanged at 3.47 GB, so
the rank budget a host can afford is unchanged too. Ten changes contribute; the
four that carry it are all in how a checkpoint reaches the U-Net, which the
phase profiler identified as **98.3%** of logged image time.

WSClean did not move, and the honest reading of thirty-odd controls is that it
should not have: it entered this round at ~110 evaluations/second after
thirty-two rounds of work, and every lever tried since is either below this
host's ~5% run-to-run variance or not result-preserving. What that variance is,
and the probes it swallowed, are in [what was measured and rejected](#what-was-measured-and-rejected).

The measurement machinery is the other half of the round. Group medians replaced
means, the ledger records memory and idle fraction, arms alternate run by run
instead of running as blocks, and the group key now includes the image ids - so
a rebuild between two rows can no longer make them look like one comparison.
Without those, several of the R2D2 results below would have been unreadable.

## What shipped

Cumulative. Rounds 1-32 first, then this round.

| change | where it lives | measured worth |
|---|---|---|
| PolyChord's asynchronous MPI mode by default | `polychord_wsclean.py` | -27-33% wall clock per dead point ([throughput](nested-sampling-throughput.md)) |
| WSClean built for the host CPU, `x86-64-v3` default | `WSCLEAN_TARGET_CPU` | +8.6% evals/s |
| MeqTrees no longer predicts a constant | `point_source_forest.py` | +16-20% evals/s |
| `WEIGHT`/`SIGMA` columns no longer written | `simulate_point_source_ms.py` | +3.8% evals/s |
| Measurement Set built in a shared host tmpfs | `NS_SCRATCH_DIR` | -1.2 ms/eval ([I/O placement](nested-sampling-io-placement.md)) |
| Cross-device Measurement Set copy removed | `scratch_root_for()` | -6.4 ms/eval, ~+2.2% evals/s |
| Six unread MS subtables dropped | `simulate_point_source_ms.py` | +14.9% evals/s ([MS open](nested-sampling-ms-open.md)) |
| `FEED` dropped too | same | -3.2% on the `wsclean` binary |
| Antenna names cached in-process | `patches/0001` | +10.5% evals/s ([patches](nested-sampling-wsclean-patches.md)) |
| `wsclean-zygote` fork server | `docker/wsclean/src/zygote.cpp` | +8.4% evals/s ([zygote](nested-sampling-wsclean-zygote.md)) |
| `-data-column DATA` | `polychord_wsclean.py` | -1.0% on the binary ([cost model](nested-sampling-cost-model.md)) |
| Dead work cut from the simulate stage | `simulate_point_source_ms.py` | -20% of that stage ([simulate](nested-sampling-simulate-stage.md)) |
| One parent-MS open shared five ways | `patches/0003` | -3.7% on the binary ([shared open](nested-sampling-shared-ms-open.md)) |
| FFTW planner warmed in the zygote parent | `zygote.cpp` | -6.9% on the binary ([FFTW](nested-sampling-fftw-planner.md)) |
| `schaapcommon` FFTW plan cache | `patches/0004` | -2.7% on the binary ([gridder floor](nested-sampling-gridder-floor.md)) |
| Measurement Set read in row blocks | `patches/0005` | -2.2% on the binary ([row blocks](nested-sampling-row-blocks.md)) |
| w-gridding turned off | `patches/0006` | -29% on the binary, at 2.3e-5 of the image peak ([gridder floor](nested-sampling-gridder-floor.md)) |
| ducc0 gridder index reused across passes | `patches/0007` | -7.4% on the binary solo, 8-10% in 38-rank CSD3 searches, byte-identical ([csd3-speed](csd3-speed.md), round 23) |
| Reordered part files mapped, not read row by row | `patches/0008` | -6.0% on the binary solo, ~10% in 38-rank CSD3 searches, byte-identical ([csd3-speed](csd3-speed.md), round 24) |
| ducc0 `scanData` fast path for uniform weights | `patches/0009` | -2.5% on the binary solo, ~4.5% in 38-rank CSD3 searches, byte-identical ([csd3-speed](csd3-speed.md), round 25) |
| Thermal noise drawn in C, numpy's PCG64 stream bit for bit | `docker/meqtrees/src/fastnoise.c` | -26% (-28ms) on simulate in 38-rank CSD3 searches, chains identical ([csd3-speed](csd3-speed.md), round 26) |
| INTERVAL/EXPOSURE set on the template before `addrows`; antenna table unpacked only for makems | `simulate_point_source_ms.py` | -7% (-6 to -8ms) on simulate in 38-rank CSD3 searches, chains identical ([csd3-speed](csd3-speed.md), round 27) |
| WSClean zygote runs with glibc malloc tunables (no mmap/trim of large blocks) | `run-nested-sampling.sh` | -2.9% WSClean wall at 38-way in a controlled driver (kernel time 29 -> 23ms); ~-1% in-run, unresolved; chains identical ([csd3-speed](csd3-speed.md), round 28) |
| Each reordered file mapped once per evaluation, not once per pass | `patches/0010` | -2.2% (-4.6ms) on WSClean in 38-rank CSD3 searches (8 runs, ABBA+BAAB, no overlap); faults halved solo; byte-identical, chains identical ([csd3-speed](csd3-speed.md), round 30) |
| Simulate's fill writes in `extend_template()`'s table open (no reopen, no UVW/CHAN_FREQ/FLAG read-back) | `simulate_point_source_ms.py` | -6.3% (-6.4ms) on simulate in 38-rank CSD3 searches (8 runs, no overlap); MSs byte-identical, chains identical ([csd3-speed](csd3-speed.md), round 31) |
| Point-source phase ramp in C, numpy's complex `exp` bit for bit | `docker/meqtrees/src/fastnoise.c` | -17% (-10.5ms) on simulate in 38-rank CSD3 searches (8 runs, makems misses trimmed, no overlap); chains identical ([csd3-speed](csd3-speed.md), round 32) |
| One-timestep template re-laid with 32KB StandardStMan buckets (makems' 256B made ANTENNA1/2 writes walk ~1000 buckets) | `simulate_point_source_ms.py` | -6.1% (-5.7ms) on simulate in 38-rank CSD3 searches (8 runs, makems misses trimmed, no overlap); every column equal, chains identical ([csd3-speed](csd3-speed.md), round 33) |
| Meta records copied at their fixed size (no store-forwarding stall per row) | `patches/0011` | -1.0% (-1.6ms) on WSClean at 37-way in a controlled driver (two 16-arm interleaved sets, 7 of 8 ABBA blocks); in-run unresolved; byte-identical ([csd3-speed](csd3-speed.md), round 35) |
| Unseen observations built once at the search's longest observation, not once per new longest draw | `simulate_point_source_ms.py`, `common.py` | +88% evals/s with a warm observation cache (simulate 157 -> 59ms) and +21% cold, 38-rank CSD3 WSClean searches; log(Z) and chains identical in a synchronous pair ([csd3-speed](csd3-speed.md), round 36) |
| Ranks stop claiming failed-artefact slots once all 20 are taken | `common.py` | ~36ms less rank main-thread time per failed evaluation with 36 ranks failing at once (~15% of a WSClean evaluation in a failure storm); stage timing only, scoring untouched ([csd3-speed](csd3-speed.md), round 38) |
| (diagnostic) PolyChord's root measured with a sleeping toy likelihood: keeps up with ~6,500 evals/s at nlive 4000/20k dead, ~50x a WSClean search's demand; its only cost is a ~2-3s pause per `update_files` write (~0.1-0.2% of a real run) | none | no change ([csd3-speed](csd3-speed.md), round 40) |
| WSClean zygote's malloc thresholds raised from 1GB to 2^62 (WSClean sizes per-pass buffers from RAM, 47-110GB virtual on CSD3, so they were still mmapped per pass) | `run-nested-sampling.sh` | -3% (-3.5 to -4.5ms) on WSClean at 37-way in a controlled driver (two 8-arm interleaved sets, 8 of 8 blocks); faults per request 4.0k -> 1.6k; byte-identical, chains identical in a synchronous pair ([csd3-speed](csd3-speed.md), round 41) |
| Host-built `fastnoise.so` bound into a `meqtrees.sif` that lacks it (the shipped one predates rounds 26/32, so production drew noise with numpy) | `start-sidecars.sh` | warm simulate -32ms (-37 to -41%), worker ms/eval -11 to -12% in two raced 18-rank pairs; chains identical in a synchronous pair ([csd3-speed](csd3-speed.md), round 42) |
| WSClean rebuilt with patches 0001-0011 on a compute node and layered over the shipped `wsclean.sif` (built 4 Sep with 0001-0006 only, so rounds 23-35 never reached production) | `patch-wsclean-sif.sh`, `./ri images patch-wsclean` | WSClean binary -16 to -27%, warm worker ms/eval -14 to -23% in three raced 18-rank pairs; log(Z) and chains identical in a synchronous pair; takes effect once run on the cluster ([csd3-speed](csd3-speed.md), round 43) |
| Template's TIME and TIME_CENTROID moved from makems' IncrementalStMan into its (32KB) StandardStMan | `simulate_point_source_ms.py` | solo extension 11.5 -> 8.5ms; in-run simulate -4.5 to -6.6ms, worker ms/eval -1.5 to -5.6% in three raced 18-rank pairs, WSClean binary unchanged; log(Z) and chains identical in a synchronous pair ([csd3-speed](csd3-speed.md), round 45) |
| A rank's top/bottom-20 extremes so far held in tmpfs `.held/` and flushed to Lustre every 600s and at the end, instead of published to Lustre and stripped on eviction (9-23% of evaluations at 24-72 ranks) | `common.py` | +6.5 to +30% whole-run evals/s in four raced 18-rank WSClean pairs with both trees on `/rds`, matching the no-publish bound; worker ms/eval unchanged; log(Z) and chains identical in a synchronous pair ([csd3-speed](csd3-speed.md), round 47) |
| (diagnostic) Same code in a `/rds` tree and a `/dev/shm` tree, raced | - | No gap left: 61.8/62.9 vs 61.9/59.0 evals/s, middle-of-run rates equal within 1.4%, workers 97-99% busy mid-run (round 39: 84-87%); round 45's ~20% Lustre penalty is gone after round 47 ([csd3-speed](csd3-speed.md), round 50) |
| A run publishes only the held extremes it keeps; prune lists each eval dir once | `common.py` `finished_run_records()` | A production search (36 ranks, to convergence) spent 52.7s of 389.5s after its last evaluation; now 37.0s (-30%), same kept images ([csd3-speed](csd3-speed.md), round 51) |
| Rank 0 gets the records over MPI; prune skips dirs that hold only a record | `common.py` `finished_run_records()`, `prune_run_artefacts()` | Post-run record read 7.5s to 1.0s and prune 9.5s to 5.3s in a production search (36 ranks); found the record writer lagging up to ~3 min behind scoring ([csd3-speed](csd3-speed.md), round 52) |
| (rejected) Records written by a helper process over a pipe instead of a thread | - | Drain 13-18s to 0s and lag no longer grows, but -11% evals/s raced at 18 ranks (58.5/62.3 vs 68.4/67.9): the bounded pipe makes ranks wait on Lustre. The writer is slow because `mkdir` in a live `evaluations/` takes 130-600ms, not mainly the GIL ([csd3-speed](csd3-speed.md), round 53) |
| Record-writer thread unpinned from its rank's CPU | 0 evals/s (in noise) | `mkdir` in a live `evaluations/` 180-240ms to 0.14-9ms; record lag median ~40s to ~0s; end-of-run drain 7-9s to 0s. The writer held the directory's lock while waiting for a core its rank's worker was saturating ([csd3-speed](csd3-speed.md), round 54) |
| Meta record fields read straight from the mapping (0011's fixed-size copy still left a store-forwarding stall on every row) | `patches/0012` | -0.75% on WSClean at 37-way in a controlled driver (4 of 4 ABBA blocks), -1.3% solo; `CacheMeta` 2.9% -> 0.9% of the binary; byte-identical. In-run unresolved ([csd3-speed](csd3-speed.md), round 55) |
| Noise generator's PCG state kept in registers (gcc 8 had spilled it to the stack every draw) | `docker/meqtrees/src/fastnoise.c` | noise -17% (MS fill) / -30% (R2D2 `--mat-only`), 30/30 interleaved pairs; simulate -1.5ms (-4.9%) in a raced 16-rank pair; chains byte-identical. Reaches cluster runs without a rebuild, through the host-built copy ([csd3-speed](csd3-speed.md), round 56) |
| Phase ramp's `sincos` vectorised (4-wide polynomial; libm redoes any phase whose float32 rounding its error could change) | `docker/meqtrees/src/fastnoise.c` | phase ramp -63% (43.2 -> 15.9ms at 351000x4, 30/30 pairs, byte-identical); simulate -2ms (-4.3%/-4.9%) in two warm raced 16-rank pairs; chains byte-identical. No rebuild needed ([csd3-speed](csd3-speed.md), round 57) |
| Reordered rows buffered in memory and written in 1MB chunks (three small `ostream` writes and two `vector` allocations per row) | `patches/0013` | reorder phase -19.5% solo (6/6 ABBA blocks), -18% at 36-way; WSClean -3.3% solo, -2.0%/-3.0% at 36-way (30/36 adjacent pairs); binary mean -2.8%/-2.6% in two raced 16-rank pairs; reordered files and chains byte-identical. Needs a `wsclean.sif` rebuild ([csd3-speed](csd3-speed.md), round 58) |
| Time-offset `std::map` lookups skipped per row outside facets | `patches/0014` | "loading data in memory" -5.3% and WSClean -0.35% solo (6/6 ABBA blocks); FITS data byte-identical. A `thread_local` wavelength cache tried with it was +9.9% and dropped. Needs a `wsclean.sif` rebuild ([csd3-speed](csd3-speed.md), round 59) |
| FFTW warm-up sized from the request's `-size` (it was hard-coded to the 128-pixel sizes, so at today's 32 every evaluation re-planned 32/36/40) | `zygote.cpp` | planning 5.0 -> 0.3 ms/eval; WSClean -5.9% solo (6/6 ABBA blocks), -3.6/-3.7% binary mean in two raced 16-rank searches; FITS data and chains byte-identical. Needs a `wsclean.sif` rebuild ([csd3-speed](csd3-speed.md), round 60) |
| Simulator argparse parser built once per worker instead of per request | `simulate_point_source_ms.py` | -0.41 ms per simulate request (7.51 -> 7.11 ms on a small MS, 30/30 interleaved pairs); MS byte-identical. Reaches cluster runs with no rebuild ([csd3-speed](csd3-speed.md), round 61, which also has the WSClean binary cost model) |
| One-timestep template copied from bytes the worker read once, not `shutil.copytree` (39 files, each an open, `sendfile` and `copystat`) | `simulate_point_source_ms.py` | -1.25 ms solo simulate (17.69 -> 16.46 ms over 48 real draws, 30/30 interleaved rounds), -1.30 ms at 16 copies at once (160/160); 768 of 768 MSs byte-identical. Reaches cluster runs with no rebuild ([csd3-speed](csd3-speed.md), round 62) |
| (rejected) Noise generator's PCG64 stream drawn in 4 jump-ahead lanes, or with an AVX-512 ziggurat | - | Both bit-identical; lanes 4.05 -> 3.98 ns a normal (flat), AVX-512 6.1 ns (slower). Each 128-bit step's three multiplies are already the bound ([csd3-speed](csd3-speed.md), round 63) |
| XY/YX noise drawn (stream kept) but not computed or stored; `-pol I` reads XX and YY only | `docker/meqtrees/src/fastnoise.c`, `simulate_point_source_ms.py` | noise -15% solo; in-run simulate -1.9 ms (4 raced 17-rank pairs, both launch orders, all four lower); chains byte-identical. Kept MSs' XY/YX now carry no noise. No rebuild needed ([csd3-speed](csd3-speed.md), round 65) |
| Simulated MS holds XX and YY only (template cut to 2 correlations; noise still drawn as the 4-correlation stream) | `simulate_point_source_ms.py` `_write_template()`, `fastnoise.c` `add_normal_f32_xxyy2` | WSClean -2.4% solo (24 FITS byte-identical); in-run simulate -2.6 to -3.4 ms in 3 raced 17-rank pairs (both launch orders), binary -1.1% and evals/s +3.2% on the same draws; chains byte-identical. No rebuild needed ([csd3-speed](csd3-speed.md), round 66) |
| (rejected) Imaging-weight u/v divisions computed first in vectorisable 16-channel blocks (candidate patch 0015) | - | Byte-identical but +1.3 to +1.7% slower on the three largest MSs (0/6 ABBA blocks faster): at 1-8 channels per row the cost is per-row overhead, not division ([csd3-speed](csd3-speed.md), round 67) |
| Each row's uvw and final weight kept from the first inversion and reused on every later major cycle (patch 0015) | `docker/wsclean/patches/0015-reuse-inversion-weights.patch` | per-cycle load 33 -> 15 ms on the largest MS; WSClean -6.2 to -7.3% solo on the three largest MSs (24/24 ABBA blocks each), 1440 FITS byte-identical; same-draws 17-rank search: binary -7.4%, evals/s +5.2%, all chains and log(Z) byte-identical ([csd3-speed](csd3-speed.md), round 68) |
| Residual inversions read data and model in place from the reordered mappings, into a buffer kept across passes (patch 0016) | `docker/wsclean/patches/0016-read-residual-rows-in-blocks.patch` | residual-pass load 14 -> 5.7 ms on the largest MS; WSClean -2.9 to -4.6% solo on the three largest MSs (24/24 ABBA blocks each), 1440 FITS byte-identical; same-draws 17-rank searches in both launch orders: binary -2.3%/-2.6%, all chains and log(Z) byte-identical ([csd3-speed](csd3-speed.md), round 69) |
| Search warns when `wsclean.sif` was not built from this tree's WSClean patches (production's 21 Sep SIF stops at 0006, so 0007-0016 never reach a run until `./ri images patch-wsclean`) | `defaults.sh` `ns_warn_stale_wsclean_sif()` | no timing change; makes the largest open cluster speed-up visible. Simulate re-profiled: 17.8 ms solo steady state, noise 5.0 ms near its floor ([csd3-speed](csd3-speed.md), round 70) |
| Shipped 21 Sep `wsclean.sif` (0001-0006) against this tree's 0001-0016 (`./ri images patch-wsclean`), and bench image ids that tell them apart | `defaults.sh` `ns_image_id()` | raced 17-rank same-draws searches, both launch orders: binary 193 -> 126 ms (-34.6%), **evals/s 26.4 -> 37.4 (+42%)**, chains byte-identical; the upgrade itself is still the user's to run ([csd3-speed](csd3-speed.md), round 71) |
| Production-shaped 36-rank search on 0001-0016: 233 evals/s steady, 174 whole run; the idle 24% is the last chain alone. WSClean `-j N` for it rejected (changes every FITS, PSF included; nondeterministic) | none | diagnostic; MeqTrees now 18% of evaluation time ([csd3-speed](csd3-speed.md), round 72) |
| ducc0's scan skips the `\|w\|` min/max it never reads with w-gridding off (patch 0017) | `docker/wsclean/patches/0017-skip-unused-w-range-scan.patch` | WSClean -1.9 to -2.3% solo on the three largest MSs (21-24/24 ABBA blocks), 1440 FITS byte-identical; same-draws 17-rank searches in both launch orders: binary -0.8%/-2.0%, evals/s +1.5%/+1.6%, chains and log(Z) byte-identical ([csd3-speed](csd3-speed.md), round 74) |
| WSClean 0001-0017 built with GCC profile-guided optimisation, trained on five real-draw MSs: rejected | none | +0.9 to +1.9% slower solo on the four larger MSs (2-5/24 ABBA blocks faster), and no FITS data block byte-identical (1.8e-7 of the image peak, most likely because FMA contraction follows inlining) ([csd3-speed](csd3-speed.md), round 75) |
| Why simulate is ~1.5x slower in-run than solo (28 against 18 ms) | none | diagnostic: L3 capacity, not locks, syscalls or clock. 34 simulate-only copies: +55% wall, user +51%, IPC 2.99 to 1.85, LLC miss rate 24% to 72%; a pure-compute loop does not slow down. The lever left is a smaller per-call working set ([csd3-speed](csd3-speed.md), round 76) |
| The makems observation cache stored as a pickle, not an `.npz` | `simulate_point_source_ms.py` | simulate -4.0% in-run (30.49 to 29.28 ms, 29 pinned copies, 4/4 ABBA pairs); `np.load` 0.8 ms against `pickle.load` 0.05 ms; arrays bit-identical ([csd3-speed](csd3-speed.md), round 78) |
| FLAG no longer written: casacore's zero-filled tiles already read False | `simulate_point_source_ms.py` | simulate -0.7% in-run (23.79 to 23.62 ms, 34 pinned copies, 4/4 ABBA pairs); every column identical ([csd3-speed](csd3-speed.md), round 79) |
| Phase ramp output allocated with `np.empty`: the C ramp writes every element of 1- and 2-correlation data | `simulate_point_source_ms.py` | simulate -1.9% in-run (23.58 to 23.14 ms, 34 pinned copies, 4/4 ABBA pairs); all 48 MSs column-identical ([csd3-speed](csd3-speed.md), round 80) |
| Chains still in flight end once PolyChord has finished: its root has written every output before it waits for them, and keeps only their `nlike` (`PolyChordFinish`) | `common.py`, both drivers | async PolyChord wall -13.8% / -8.9% in raced 17-rank 600-dead pairs, both launch orders (a fixed ~20-40 s tail per run); sync pair byte-identical ([csd3-speed](csd3-speed.md), round 81) |
| Phase ramp written with its XX/YY real noise in one pass (`phase_ramp_c64_xxyy2`) | `fastnoise.c`, `simulate_point_source_ms.py` | simulate -1.5% in-run (23.10 to 22.76 ms, 34 pinned copies, 4/4 ABBA pairs); all 48 MSs column-identical ([csd3-speed](csd3-speed.md), round 82) |
| (diagnostic) WSClean binary profiled over 48 real draws, not the largest MS | - | ducc0 46% of the binary (kernels 40%); nothing else above ~2.3%, so no non-kernel WSClean patch is worth more than ~1% of an evaluation ([csd3-speed](csd3-speed.md), round 83) |
| (diagnostic) icelake against sapphire, same 48-draw WSClean + simulate workload | - | A sapphire core is 13% faster solo but 6% slower fully loaded; half a sapphire node (54 ranks) does 307 evals/s against half an icelake node's 218 (+41%), ~5% fewer per allocated core ([csd3-speed](csd3-speed.md), round 84) |
| (diagnostic) Kernel time of a forked WSClean request | - | 8.2ms solo, 13.7ms of a 101.5ms request at 36-way (half a fresh process's); only ~1.4k faults (~3-5ms), a quarter of them re-mapped library text that `fork` does not copy; the rest was guessed as syscalls/fork/exit, but round 87 found it is mostly faults ([csd3-speed](csd3-speed.md), round 85) |
| (diagnostic, rejected) The R2D2 U-Net re-profiled on torch 2.13 | - | 45% of a forward is im2col + MKL `sgemm` (every conv at <= 8x8), 30% oneDNN, 10% instance norms; `jit.freeze` no faster, MKL-packed weights and `MKL_ENABLE_INSTRUCTIONS=AVX2` faster only with changed bits (AVX2: -2.3% at 36 copies, 6.8e-6) ([csd3-speed](csd3-speed.md), round 86) |
| (diagnostic, rejected) What a forked WSClean request's kernel time is | - | At 36-way ~9.5 of 13.8ms is page faults (~1.4k at ~6.6us), ~2ms syscalls (mostly `read`), ~2ms exit, 0.23ms `fork`; THP off (+3%) and populating library text after `fork` (+11-17%) are slower ([csd3-speed](csd3-speed.md), round 87) |
| (diagnostic, rejected) The zygote parent's heap reshaped before `fork` | - | The parent's heap is 64MB of VMA but ~3.5MB resident, one 2MB huge page plus ~1.5MB of small pages at its unaligned head; `malloc_trim(0)` before `fork` and a 2MB-aligned heap start both left a solo request at 77.0-77.4ms against 76.8-77.5ms ([csd3-speed](csd3-speed.md), round 88) |
| (diagnostic) Rounds 72-88 together, production-shaped (36 ranks, nlive 150, 900 dead) | - | 250.5 evals/s whole run and 258 steady on a 0001-0017 SIF, against round 72's 174/233; no single-rank tail; simulate is 17.5% of an evaluation; the shipped 0001-0006 SIF manages 180/184, so patching it (`./ri images patch-wsclean`) is still worth +40% ([csd3-speed](csd3-speed.md), round 89) |
| FIELD, OBSERVATION and SPECTRAL_WINDOW values spliced into the template copy at offsets found (and verified byte for byte) once per template, instead of three casacore opens | `simulate_point_source_ms.py` | simulate -4.2% in-run (23.21 to 22.23 ms, 34 pinned copies, 4/4 ABBA pairs), -6.4% solo; all 48 MSs column-identical ([csd3-speed](csd3-speed.md), round 90) |
| (diagnostic) A search warns when its expected file count passes the free project quota; simulate's casacore writes re-timed | `run-config.sh` `ns_warn_file_quota()` | no timing change. A converged run leaves ~2.1 files per evaluation (150k for 71k evaluations at nlive 150), so at 250 evals/s the ~650k files left fill in ~20 minutes of evaluations; nlive ~1000 cannot finish in an empty quota. Simulate's main-table writes are ~2.5 ms (DATA 0.76, UVW 0.51, four scalar columns ~0.24 each) and a bigger SSM cache does not move them ([csd3-speed](csd3-speed.md), round 91) |
| One record file per evaluation (`eval-*.json`), a dir only for what it keeps; every reader takes both layouts | `common.py`, health, runs, progress bar, resume | files per evaluation 2.06 -> 1.07 in raced 17-rank WSClean A/Bs, log(Z) identical in the synchronous pass, throughput unchanged within noise; a converged run needs ~510 files per unit of nlive instead of ~970 ([csd3-speed](csd3-speed.md), round 92) |
| A run's records packed into `evaluations/records.jsonl` at finish and at resume (rank 0, unlinks off its pinned CPU); `prune-run-artefacts.py` packs older finished runs | `common.py`, health, runs, progress bar | a finished run keeps ~0.09 files per evaluation instead of ~1.07 (117 for 1,337); a resume carries only its last segment as files; reading a run's records back 2.2 -> 0.94 s per 20k; throughput untouched ([csd3-speed](csd3-speed.md), round 93) |
| A WSClean search inside a Slurm job uses a `wsclean.sif` patched to the tree: built once (~2 minutes) into `$RI_WORK_DIR/.cache/ri` when the SIF lags, reused after; `NS_PATCH_WSCLEAN=0` keeps the stock one | `defaults.sh` `ns_use_patched_wsclean_sif()` | 16-rank sync search on the shipped 0001-0006 SIF: 20.1-20.5 -> 30.6 evals/s (+50%), binary 178 -> 112 ms, chains byte-identical; first run pays ~1m54s of build ([csd3-speed](csd3-speed.md), round 94) |
| The 1 in `NS_KEEP_DETAIL_EVERY` evaluations kept whole moves from tmpfs scratch to Lustre on the rank's record-writer thread, not its main thread | `common.py` `publish_evaluation_scratch(defer=)` | 36-rank production-shaped search: worker time between evaluations 117.5/109.6 -> 70.2/73.0 worker-s (-1.3% of worker time), gaps over 100 ms 281/242 -> 69/76; kept dirs and log(Z) unchanged ([csd3-speed](csd3-speed.md), round 95) |
| The R2D2 pool parent's checkpoint pages are interleaved across the NUMA nodes holding the job's CPUs (`R2D2_INTERLEAVE_WEIGHTS=0` turns it off) | `r2d2_serve.py` `interleaved_pages()` | whole icelake node, 76 x 1: 41.8 -> 46.6/46.5 evals/s (+11%), warm request 1705 -> 1474/1533 ms; neutral on a 32/6 half node ([csd3-speed](csd3-speed.md), round 99) |
| (rejected) One R2D2 checkpoint copy per NUMA node, each worker pinned to its node | - | node-local weights 11% faster in a 38-copy pinned probe, but async 38 x 1 searches 5-6% slower per request (1116/1117 -> 1170/1188 ms); pinning costs more than locality ([round 98](csd3-speed.md)) |
| (rejected) MKL-packed weights for the U-Net's 2x2 convs, inside the real forward | - | 5% slower solo, 11% slower at 38 copies (58.8 -> 65.4 ms), 4e-6 off; THP already backs 78% of checkpoint memory; nothing outside the U-Net is over 5% of a request ([round 100](csd3-speed.md)) |
| (robustness) A rank reconnecting to a pool that is gone gives up after `NS_POOL_GONE_SECONDS` (30s), not `NS_POOL_CONNECT_SECONDS` (300s); `sidecar_restore` no longer runs in a subshell, so the pool it starts is tracked and stopped | `common.py`, `progress-bar.sh` | pool loss to finished run >300s -> 39-41s; `./ri self-check self-heal` passes again ([round 101](csd3-speed.md)) |
| (diagnostic) Synchronous seeded R2D2 searches checked for reproducibility | - | byte-identical chains at 8 x 1, 38 x 1 and 8 x 4 (round 98's mismatch was async runs); the thread count alone moves objectives by up to 1.8e-5, so chain md5 gates arithmetic-preserving R2D2 A/Bs but not thread-count ones ([round 102](csd3-speed.md)) |
| (diagnostic) R2D2 on `sapphire` against `icelake`, half node each, one rank per CPU | - | 54 against 30 evals/s (1.8x per half node, ~+20% per core; request ~0.92 against ~1.16 s), log(Z)s agree; the real U-Net forward under full load is 33.7 against 61.3 ms at 38 copies. The opposite of WSClean's round 84 result, so `--partition sapphire` is the faster R2D2 run ([round 103](csd3-speed.md)) |
| `NS_R2D2_MAX_RANKS` 56 -> 112: one R2D2 rank per CPU on whole nodes, where the shipped default ran 56 x 2 | +19% icelake, +21% sapphire (whole node) | icelake 39.7 -> 47.2 evals/s (76 x 1), sapphire 70.0 -> 84.6 (112 x 1), ABAB at 1000 dead, log(Z)s agree ([round 105](csd3-speed.md)) |
| One R2D2 checkpoint copy per NUMA node on balanced multi-socket jobs, each worker pinned to its node (workers split in proportion to each node's CPUs; `R2D2_WEIGHTS_PER_NODE`, default auto) | `r2d2_serve.py` `load_checkpoints()`, `worker_nodes()` | whole sapphire node 86.8 -> 103.5 evals/s (+19%), request 1258 -> 1056 ms, two ABAB pairs, log(Z)s agree; whole icelake node 46.9 -> 56.7 evals/s (+21%), request 1582 -> 1304 ms ([rounds 106-107](csd3-speed.md)) |
| (diagnostic) Half-node R2D2 placement: 56 CPUs on one socket against 28 + 28, same exclusive sapphire node | - | one socket 54.5, 28/28 with one copy 64.3, 28/28 with per-socket copies 74.9 evals/s (+37% overall), log(Z)s agree; `NS_SLURM_SOCKETS=2` asks a sized job for two sockets, checked end to end in a real submitted search ([rounds 111-113](csd3-speed.md)) |
| (diagnostic) Half-node WSClean placement: 56 ranks on one socket against 28 + 28, same exclusive sapphire node | - | no effect: 462 against 475 evals/s over 6 arms each (+2.6%, arm-to-arm spread ~±10%), log(Z)s agree; WSClean is not bound on memory bandwidth the way the U-Net is, so `NS_SLURM_SOCKETS=2` stays R2D2 advice ([round 114](csd3-speed.md)) |
| (diagnostic) The 22 Sep production R2D2 search (24 x 1 sapphire, nlive 150, 15 repeats, same seed and SIFs) rerun to convergence on the current tree | - | 6h20m -> 41 min (7.0x), 4.12 -> 26.6 evals/s, simulate 1,936 -> 32 ms (39% -> 3.8% of worker time), idle 11.5% -> 0.05%, files ~225k -> 6.6k, log(Z)s agree (0.7 sigma) ([round 115](csd3-speed.md)) |
| `./ri health` reads a run's records on 32 threads instead of one at a time (`evaluation_scan`) | `nested-sampling-health.py` | not a search speed-up: a cold scan of a 10-11k-record run fell from 382 s to 8.9 s (34 -> 0.9 ms a record), so health answers on large live runs again; output unchanged ([round 116](csd3-speed.md)) |
| (diagnostic) CUDA MPS on one A100, P copies of the 32x32 U-Net forward | - | the card saturates at ~600 forwards/s without MPS (contexts time-slice) and ~1,680 with it (2.8x, two ABAB passes), i.e. U-Net capacity for ~93 R2D2 evals/s, about a whole sapphire node; end to end in round 119 ([round 117](csd3-speed.md)) |
| GPU searches start a CUDA MPS daemon so the R2D2 workers share the card (`NS_CUDA_MPS`, default on) | `start-sidecars.sh` `sidecar_cuda_mps_flags` | 16 ranks on one A100: 15.8 -> 41.8 evals/s (2.65x, three ABAB pairs plus a warm-up), R2D2 request ~900 -> ~300 ms, log(Z)s agree; beats 24 sapphire CPUs (26.6) ([round 119](csd3-speed.md)) |
| (rejected) One GPU copy of R2D2's weights shared with the workers over CUDA IPC | - | GPU memory at 16 ranks 58.5 -> 14.7 GB, but 16 ranks no faster (40.6 -> 41.7 evals/s, noise) and 32 ranks slower (38.8): the 32 cores per A100 are the cap, not GPU memory ([round 120](csd3-speed.md)) |
| (diagnostic) The R2D2 U-Net forward replayed from a captured CUDA graph | - | Bit-identical to eager; solo 3.06 -> 1.29 ms a forward (2.4x), but under MPS the 16- and 24-copy plateau stays at ~1,680 forwards/s (1,679/1,689 at 16), so the plateau is GPU work, not launch overhead; without MPS 593 -> 705 at 16 copies. Not shipped: worth an end-to-end A/B only below the plateau ([round 122](csd3-speed.md)) |
| GPU R2D2 workers wait for the U-Net on a blocking CUDA event instead of spin-waiting in the copy back (`R2D2_CUDA_BLOCKING_SYNC`, default on) | `r2d2_serve.py` `patch_cuda_network()` | 16 ranks on one A100 with MPS: 37.5 -> 39.8 evals/s (+6%, three ABAB pairs, no overlap), CPU-only simulate ~48 -> ~43 ms, log(Z)s agree ([round 123](csd3-speed.md)) |
| A GPU search runs 19 R2D2 ranks per A100 by default, not 16 (`NS_R2D2_CUDA_MAX_RANKS`, the most the card's memory holds) | `defaults.toml` | one A100 with MPS and blocking sync: 41.1 -> 43.9 evals/s (+7%, three pairs, every 19-rank arm above every 16-rank one), log(Z)s agree ([round 124](csd3-speed.md)) |
| (rejected) TF32, `cudnn.benchmark` or channels_last for the GPU U-Net | - | On one A100 under MPS at 19 copies, production's float32 forward is fastest (~1,716 forwards/s); TF32 -3% and 1.7e-3 off, `cudnn.benchmark` -23% and non-deterministic across workers, channels_last -39% ([round 125](csd3-speed.md)) |
| (rejected) 1 or 3 CPU threads per GPU R2D2 rank instead of the derived 2 (`R2D2_OMP_THREADS`) | - | one A100, 19 ranks, MPS and blocking sync: 2 threads beat 1 in all three pairs (43.7 vs 41.5 evals/s) and 3 in two of three (42.0 vs 40.6) ([round 126](csd3-speed.md)) |
| (rejected) batching U-Net forwards across requests on the GPU (a gathering server) | - | one A100, float32: batched forwards are bit-identical to batch 1 up to B=32, but B=19 gives only 2,326 forwards/s against 1,740 for 19 MPS copies (1.33x, not round 117's TF32 4,762); a 19-rank search uses ~790/s and is bound by its 32 CPU cores ([round 127](csd3-speed.md)) |
| (rejected) the GPU U-Net replayed from per-iteration CUDA graphs in a search (`R2D2_CUDA_GRAPHS`) | - | one A100, 19 ranks, MPS and blocking sync: 64 of 64 FITS byte-identical and a solo request 13% faster, but the search ran 43.3 evals/s with graphs against 44.7 eager (graphs lost 2 of 3 pairs), so the GPU, not CPU launches, is what binds ([round 130](csd3-speed.md)) |
| (rejected) mimalloc, jemalloc or tcmalloc preloaded into `wsclean-zygote` instead of glibc | - | 36 pinned zygotes on icelake, 48 real draws: glibc + production tunables 154 ms a request; mimalloc 162-166, jemalloc 167 (solo tcmalloc 170 against 111); FITS byte-identical ([csd3-speed](csd3-speed.md), round 131) |
| (rejected) bfloat16 U-Net forwards on CPU (AMX on sapphire) | - | 56 pinned copies on sapphire, 32x32: autocast 75 against 32 ms a forward; weights cast once 19 against 23 ms solo but 33.8 against 32.7 at 56 copies; output off by 1-12% of the peak ([csd3-speed](csd3-speed.md), round 134) |
| (diagnostic) more threads for the end-of-run record unlink; records sharded over subdirectories | - | one flat `evaluations/` on Lustre caps one node at ~2k record creates/s and ~4-7k unlinks/s whatever the thread count (4-128); 16 subdirectories give ~10k creates/s and ~35-45k unlinks/s. Production (<=~1k records/s a node) is under the cap, so nothing changed ([csd3-speed](csd3-speed.md), round 133) |
| (diagnostic, one fix) rounds 90-134 together for WSClean, raced against round 89's tree; `.launcher.pid` renamed into place | `start-sidecars.sh` | 18+18 ranks raced on one patched SIF, launch order alternated: +4.5% evals/s (117.8 to 123.1, at the noise floor), simulate -13% (25.6 to 22.2 ms) in every pair. Launch position alone was worth ~9%. Fixed a torn `.launcher.pid` read that let one search's reaper kill another's starting pool ([csd3-speed](csd3-speed.md), round 135) |
| (diagnostic) batched CPU U-Net forwards at one rank per CPU | - | 36 icelake CPUs, P workers x T threads with batch T: 2 x 18 does +64% images/s, 4 x 9 +47%, over 36 x 1. Threads alone lose 52%. Batch 2 is 20% slower than batch 1. Rows differ by ~1e-6 of the peak, so it is not bit-identical. It needs a lockstep worker for requests at the same R2D2 iteration ([csd3-speed](csd3-speed.md), round 136) |
| (rejected) lockstep R2D2 requests with batched CPU forwards | -24% to -45% | Real requests in lockstep in one worker, each iteration's forward batched (9 x 4 B 4 to 2 x 18 B 18 on 36 icelake CPUs). Prune thins the batches to 3.0/6.5/12.8, and the per-request work behind the GIL takes 36-47% of the wall. Images are within 3e-6 of the peak. Only a cross-rank forward server could win, at most ~+30% ([csd3-speed](csd3-speed.md), round 137) |
| (rejected) cross-rank U-Net forward server for CPU R2D2 | -52% to -70% | R ranks on their own CPUs send each forward to one server on S CPUs, which batches by checkpoint (27+9, 36+9 shared, 32+4, 18+18 on 36 icelake CPUs). The server is the bottleneck in every arm: ~290 forwards/s on 9 CPUs at batch 6-10, against production's ~670 on 36. Images are within 3e-6 of the peak. CPU batching is closed in every form ([csd3-speed](csd3-speed.md), round 139) |
| `R2D2_RUNTIME=onnx` (CPU default since round 145; `torch` opts out): ONNX Runtime for the CPU U-Net forward | +33-47% R2D2 CPU evals/s | The float32 U-Net exported per checkpoint and run by onnxruntime 1.23 (1 thread, `ORT_ENABLE_ALL`, sessions shared by fork): 26 against 39.6 ms a forward, and 42.0 against 31.7 evals/s in ABAB async searches on a half icelake node, with log(Z) within errors. Not bit-identical (1.5e-5 of the peak at worst), so GPU runs and pre-round-141 resumes keep torch; onnxruntime is pip-installed into the work-dir cache. Without ORT's CPU arena it uses ~9 GB less PSS at 38 ranks for the same speed and bit-identical forwards, and the rank budget gives it 1 GB a rank (`NS_R2D2_ONNX_MB_PER_RANK`) Run to convergence on two seeds it matches torch's log(Z) (within 0.25 sigma) and posterior at +44% evals/s ([csd3-speed](csd3-speed.md), rounds 140-145) |
| ONNX Runtime sessions built once per NUMA node, beside round 106's per-node checkpoints (`enter_node()` swaps a worker's node's set in) | `r2d2_serve.py` `patch_onnx_network()` | whole icelake node, 76 x 1 onnx: 59.2 -> 80.5 evals/s (+36%), request 1197 -> 857 ms, ABBA, log(Z)s agree; the shipped single set sat on one socket and ran a whole node at torch's rate ([round 149](csd3-speed.md)) |
| Round 149 validated on a whole sapphire node (production shape) | - | 112 x 1 onnx: 83.2 -> 116.9 evals/s (+40%), +22% over torch (95.7); the pre-149 single ORT set made onnx slower than torch there ([round 150](csd3-speed.md)) |
| A request's four FITS images written from a cached header and its config parsed by libyaml (`R2D2_FAST_IO`) | `r2d2_serve.py` `patch_request_io()` | byte-identical; 6.2 -> 0.2 ms a request solo (~1.4%); ABAB -1.3% a request, inside noise, log(Z)s agree ([round 151](csd3-speed.md)) |
| Diagnostic: one 36-rank WSClean search against two 18-rank searches raced on the same CPUs | - | single 253 evals/s, +11% over the pair; nothing per search limits WSClean at 36 ranks. A whole sapphire node runs 737-749 evals/s at 112 ranks; there the raced pairs lost their first search to the other's startup reaper ([round 152](csd3-speed.md), fixed in round 154) |
| A request's two FINUFFT type-1 NUFFTs without the bin sort (`R2D2_NUFFT_SORT`, 0 by default) | `r2d2_serve.py` `nufft_sort()` | -16 to -21% a call solo and under 36-copy load (kernel 17.7 -> 14.6 ms, adjoint 13.1 -> 10.4 ms loaded), ~6 ms a request; images within 1.2e-7 of the peak; ABAB inside noise, log(Z)s agree ([round 153](csd3-speed.md)) |
| Startup reaper skips a pool worker that has exec'd instead of reading it as a pool with no run (a raced search killed the other's live pool) | `rank-budget.sh` `_ns_pool_row()` | correctness, not speed: a fake starting pool was falsely reaped on 4/300 passes before, 0/300 after; raced A/B arms in a batch job no longer lose one ([round 154](csd3-speed.md)) |
| (rejected) OpenVINO 2026.4 instead of ONNX Runtime for the CPU U-Net | - | one 32x32 forward, f32: solo 15.5 ms against ORT's 17.0, but 28.3 against 24.4 (+16%) with 36 forked children, the pool's shape ([round 155](csd3-speed.md)) |
| (rejected) `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` for the ORT U-Net's weights | - | THP `always` already backs 88% of the 25 sessions' 3.5 GB; 24.57 against 24.66 ms a forward with 36 forked children (ABABAB, noise) ([round 156](csd3-speed.md)) |
| (rejected) 2-3 ORT intra-op threads per R2D2 worker (18 x 2, 12 x 3) | - | -2% and -14% forwards per node against 36 x 1, outputs bit-identical. Fork-shared sessions also give 11% from shared L3 when workers align on a checkpoint, of which ~5% remains over a staggered pool ([round 157](csd3-speed.md)) |
| (diagnostic) R2D2 worker CPU under load after rounds 148-153 | - | ORT 78.4% (was 67.8%), FINUFFT 8.2%, torch 4.8%, numpy 3.5%; no single item outside ORT above 4%, so per-request work outside the U-Net is closed; 46.4 evals/s against 44.8 for the same probe in round 146 ([round 158](csd3-speed.md)) |
| (final status) stop condition met: rounds 155-158 each under 2% end to end | - | R2D2 CPU is 78% U-Net forward, memory-bound, every runtime/kernel/threading/batching option closed; levers left are a different network, GPUs, whole-node job shapes ([round 160](csd3-speed.md)) |
| (pending) 2-GPU R2D2 search: round 128's `R2D2_CUDA_GPUS` ported, job 39368806 queued | - | not yet measured ([round 161](csd3-speed.md)) |
| (rejected) ORT session options: fixed 32x32 input shapes (free-dimension override) and flush-to-zero (`session.set_denormal_as_zero`) | - | 36 forked children over a staggered N1..N25 walk, ABCD x3: fixed shapes 27.6 against 26.9 ms a forward (+2.5%, every rep), FTZ 26.9 (noise); all outputs bit-identical ([round 162](csd3-speed.md)) |
| cfitsio and casacore init moved to the zygote parent | `zygote.cpp` | -2.39 ms/eval ([warm-up](nested-sampling-process-warm-up.md)) |

### Round 33: R2D2

Read the column as a chain: each row is measured against the row above it, on
the default benchmark preset at 8 ranks unless it says otherwise.

| change | where it lives | measured worth |
|---|---|---|
| R2D2 checkpoints loaded once and shared by forked workers | `r2d2_serve.py` | Removes a 25-checkpoint `torch.load` per evaluation; first checkpoint-backed control 0.4372 eval/s |
| Model-load garbage collection removed | `patches/skip-model-load-gc.patch` | 25 checkpoint swaps 1.05 s to 0.235 s; **0.4372 to 0.5622 eval/s (+28.6%)** |
| Checkpoint weights assigned by reference, not copied | `patches/assign-checkpoint-weights.patch` | **0.5622 to 0.5948 eval/s (+5.8%)**, three controlled runs |
| Per-rank thread share rounds up instead of down | `run-nested-sampling-r2d2.sh` | **0.6198 to 0.7249 eval/s (+16.9%)** |
| Automatic threads follow the memory-clamped rank count | `run-nested-sampling-r2d2.sh` | 15 ranks x 2 threads instead of 20 x 1: **0.7374 to 0.9092 eval/s (+23.3%)** |
| The unread final residual is no longer computed | `patches/remove-final-residual.patch` | Removes one forward and one adjoint transform per evaluation |
| Checkpoint key normalisation cached | `r2d2_serve.py` | 4.36 to 0.01 us per repeated 24-key lookup |
| FINUFFT plans reused across sequential evaluations | `r2d2_serve.py` | 128x128 plan setup 0.86 ms fresh to 0.091 ms retargeted (9.5x) |
| Deterministic MKLDNN kernels | `r2d2_serve.py` | 128x128 U-Net 65.0-70.6 to 59.7-60.3 ms/forward (~8%), bit-identical output |
| `torch.inference_mode` on the inference paths | `r2d2_serve.py` | 8.589 to 8.467 ms/forward (1.4%) |
| `.mat` conversion skips zlib compression | `ms_to_r2d2_mat.py` | 8.59 to 0.39 ms for 12k visibilities |
| `.mat` output follows the MS onto tmpfs scratch | `polychord_r2d2.py` | Removes the conversion's cross-device write |
| `.mat` conversion avoids broadcast temporaries | `ms_to_r2d2_mat.py` | 1.13 to 1.09 ms for 1404 visibilities; weight expansion 1.802 to 1.657 us (8.0%) |

The three `patches/` entries are `git apply`ed into the R2D2 tree at image build
time, the same mechanism WSClean's patches use, and `scripts/build.sh` now hashes
`docker/r2d2/patches` into the image's build-input label so editing one rebuilds.

### Round 33: WSClean and the harness

Individually small and, apart from the metrics stage, none of them is separable
from host variance end to end. They are here because they are strictly less
work for an identical result.

| change | where it lives | measured worth |
|---|---|---|
| `sigma_res` FITS files load only when the objective uses `sigma_res` | `polychord_wsclean.py`, `polychord_r2d2.py` | Metrics stage 1.03 to 0.46 ms/evaluation |
| The dirty image is not written when the objective ignores it | `polychord_wsclean.py` | One fewer FITS write per evaluation |
| NaN-free image RMS uses a BLAS dot product | `common.py` | 25.2 to 7.8 us per RMS call |
| Off-source metric masks are cached per (shape, source) | `common.py` | 25.2 to 3.6 us per mask |
| RMS and relative-L2 share one residual norm and one buffer | `common.py` | Removes a full-image copy per evaluation |
| Row-block copy metadata cached | `patches/0005` follow-up | 71.06 to 71.90 eval/s over six controls; inside variance |
| Progress scans classify both counts in one GNU `find` walk | `progress-bar.sh` | 0.232 to 0.073 s for 20 scans of a 635-evaluation run; BSD `find` keeps the two-pass fallback |

### Round 33: the benchmark and profiler

The reason the R2D2 numbers above are trustworthy and the WSClean ones are
honest about being noise.

| change | where it lives | why |
|---|---|---|
| Groups report a median with an IQR-based error bar | `bench.py` | One long-tail run used to move a three-repeat mean by more than most effects are worth |
| `--interleave SETTING A B` | `bench.py`, `ri` | Arms alternate run by run, so both meet the same host. Sequential arms give consistent ~4% false positives here |
| The ledger records peak memory, busy wall and idle fraction | `bench.py` | A 42.8 eval/s outlier turned out to be 69.0% idle at a *normal* 101.1 ms/evaluation image cost - a scheduler stall, not a regression |
| The group key includes the three image ids | `run-config.sh`, `bench.py` | The images bake the code, so two rows at one commit could otherwise compare different binaries |
| `--timeout` stops the whole process tree | `bench.py` | MeqTrees can hang; a hung probe used to cost the session |
| `throughput` and `production` presets | `defaults.toml` | `production` is the real workload: `nlive 150`, `num-repeats 15`, no `max-ndead` |
| `./ri profile --r2d2-phases` | `profile-nested-sampling-run.py` | Splits model update from residual computation. This is what identified inference as 98.3% of R2D2 image time |
| Phase tables report a median and a p90 | `profile-nested-sampling-run.py` | One slow evaluation no longer chooses the next optimization target |
| `run.env` is parsed with `shlex` | `bench.py` | A metric expression with spaces used to truncate its group key |
| `./ri bench \| head` exits cleanly | `bench.py` | `BrokenPipeError` is normal report consumption |

Two rounds bought run *size* rather than speed - see
[disk footprint](nested-sampling-disk-footprint.md) and
[run scaling](nested-sampling-run-scaling.md):

| change | measured worth |
|---|---|
| `sim.ms` deleted as the evaluation is scored | evaluation directory 3.4x smaller |
| WSClean's four unread FITS images pruned | another 3.94x - 393.6 KB to 99.9 KB, so this host's ceiling went from 477k evaluations to 1.88M |
| A resume keeps each adopted objective, not each record | 62 GB down to 5.4 GB, which is the difference between the target run finishing and being OOM-killed |
| Progress-bar redraw backed off to 9x its own cost | 44% of a core down to ~12%, and no longer growing with the run |

## What is priced but deliberately not taken

| lever | worth | why not |
|---|---|---|
| `-mgain 0.9` | +15-20% evals/s | Not result-preserving for `peak_flux_abs_error_jy` or `sigma_res`, and it is the experiment definition every archived run was scored under. Now `./ri search --mgain 0.9` and `./ri bench run wsclean --interleave NS_WSCLEAN_MGAIN 0.8 0.9`; default still 0.8. [clean loop](nested-sampling-clean-loop.md) |
| Raising the 65W RAPL package limit | ~+26% evals/s | **Closed.** Docker here is rootless, so `--privileged` still maps to an unprivileged user and cannot write it. [power limit](nested-sampling-power-limit.md) |
| R2D2 at four OpenMP threads | +5-6% evals/s at 8 ranks | Fastest measured explicit setting on this 20-CPU host, but it oversubscribes: the automatic `ceil(cpus / ranks)` has to be right on smaller-rank and larger hosts too. `--omp-threads 4` still asks for it |

## What was measured and rejected

Nothing here changed the tree. Each line is the shortest form of a probe that
was run, so it is not run again.

**This host's resolution.** Fifteen WSClean control repeats at one commit
measured 109.0 +/- 1.5 evaluations/second over a 102.1-115.9 range, and a
three-repeat control spans ~5% routinely. Anything smaller than that needs an
interleaved probe or a direct stage timing, not an end-to-end A/B.

**WSClean rank scaling** (`--interleave NS_MPI_PROCS`, three repeats each,
34.0-34.9 MB peak worker memory at every point, so no rank count buys memory):

| ranks | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 | 16 | 18 | 19 | **20** | 21 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evals/s | 15.9 | 17.3 | 32.2 | 45.7 | 55.8 | 67.7 | 71.8 | 74.6 | 87.4 | 89.1 | 96.0 | 107.8 | 109.3 | 113.4 | **109** | 104.7 |

Throughput saturates at ~18-20 ranks and 21 is 2.8% slower. 20 stays the
default; 18, 19 and 20 are statistically indistinguishable, and none of the
lower counts trades throughput for memory.

**R2D2 thread scaling** at 8 ranks, 3.47 GB per worker at every point:
2 threads 0.657, 3 (automatic) 0.725, **4 0.813**, 5 0.818, 6 0.787, 7 0.796,
8 0.770, 9 0.755, 10 0.740, 11 0.720 evaluations/second. Beyond four, extra
oversubscription costs model-update time and returns no memory.

**R2D2 rank/thread packing.** 16 x 2 measured 0.9077 against 8 x 4 at 0.8133
(+11.6%), and 16 x 2 against 16 x 1 at 0.7374 (+23.3%). 15 x 2 and 16 x 2 tie
at 0.911, so 15 is the default: it saves 3.47 GB for nothing.

**R2D2 inference alternatives, all closed.** TorchScript tracing 1.313 s
against 1.324 s eager, and `optimize_for_inference` cannot execute this U-Net
at all (`Currently Mkldnn tensor does not support view`). Disabling MKLDNN is a
36% regression. Denormal flushing is 6.65% slower. `OMP_PROC_BIND=spread` is
2.2x slower and `close` is identical to the unset default. A larger oneDNN
primitive cache is 0.3%. `torch.compile` needs a `g++` the runtime image does
not have. PyTorch eval mode measured 0.8103 against 0.8126 and its patch was
removed. Removing legacy weight-norm reparameterisation: 114.96 against
115.08 ms/forward.

**R2D2 swap machinery is finished.** All 25 checkpoint swaps now take 11.0 ms,
about 0.44 ms/evaluation, under 0.2% of the model-update phase. The 512x512
`_model_prev` clone it might have replaced costs 0.221 ms. Model inference is
the only R2D2 target left. On CSD3, caching one net per checkpoint so the swap
becomes a lookup was measured and gained nothing (0.331 against 0.336 s a
request; [csd3-speed](csd3-speed.md), round 97).

**WSClean `-march=native`** measured 111.6 against 114.8 evaluations/second for
matched `x86-64-v3`, with one 36.9 eval/s outlier in the native arm. Not
retained; `./ri bench run wsclean --native` still builds it.

**WSClean `-log-time`** costs nothing measurable: 130.08 ms against a 128.95 ms
baseline and a 126.66 ms null in a 756-command shuffled replay, and 109.4
against 115.8 evaluations/second in an interleaved three-repeat probe - both
inside the null. It stays on, because `--phases` and `--over-time` need it.

## Where the remaining time goes

At 126 evals/s and 143 ms/evaluation, 5.3% is unaccounted; a
`--mpi-procs 1` control attributes 1.9% to harness Python. **No idle time
remains:** PolyChord costs 3-7 us per likelihood call at every `nlive`.

59% of an evaluation is ducc0's gridding and degridding, and
[the gridder floor](nested-sampling-gridder-floor.md) prices that as
arithmetic. The pass count is the only handle on it, which is what `-mgain`
moves. See [the evaluation floor](nested-sampling-evaluation-floor.md) for the
current budget and the six further avenues closed with numbers.

## Reading your own run

Every run already carries these profiles:

```bash
./ri profile <run>              # per-stage budget and worker utilisation
./ri profile <run> --phases     # inside the wsclean binary, from its own -log-time output
./ri profile <run> --over-time  # why evaluations/second falls as a run goes deeper
```

Throughput falls because nested sampling reaches longer observations and more
channels: `70.7 ms + 4.58 us x visibilities` per evaluation. See [the cost
model](nested-sampling-cost-model.md).

With the checkpoint archive supplied through external `CHECKPOINTS_DIR`, an
interleaved four-thread R2D2 rank probe measured **0.6934 eval/s** at 5 ranks
and **0.7983 eval/s** at 8 ranks, both at **3.47 GB** peak worker memory. The
two-repeat sample is directionally clear but below production-strength size;
8 ranks remains the setting to test next. The latest phase profile attributes
**6568 ms** per evaluation to 25 model updates and **126 ms** to residual
computation, so U-Net inference remains the dominant optimization target.

## Keeping what is on this page

The table above is history: each row was measured once, against the commit
before it. `./ri bench` is the standing version - what this commit costs on
this machine now, with an error bar that narrows as repeats accumulate, so the
next change is measured rather than argued about and a regression shows up as
a column rather than a surprise. See
[benchmarks](nested-sampling-benchmarks.md).

## The rest of the pages

Chronological; each page starts where the last stopped.

| page | what it settles |
|---|---|
| [benchmarks](nested-sampling-benchmarks.md) | `./ri bench`: throughput per commit, machine and settings, and the three measurement traps its preset is pinned against - asynchronous draw variation, the cold first run, and a window too short to mean anything. |
| [throughput](nested-sampling-throughput.md) | The running log of iterations 1-14: idle ranks, the asynchronous-MPI switch, rank 0, the AVX2 build, the constant predict, the two constant columns, the scratch tmpfs. Long, and the older figures in it are superseded. |
| [run scaling](nested-sampling-run-scaling.md) | What a bigger run costs. A 10x `--nlive` costs nothing per evaluation; the ceiling on this host is ~850k evaluations, set by rank 0's memory while it builds `summary.json`. |
| [power limit](nested-sampling-power-limit.md) | The host's 65W RAPL cap, why it is not thermal, and the 8-second averaging window behind the burst-clock measurement trap. |
| [evaluation budget](nested-sampling-evaluation-budget.md) | The first production-concurrency decomposition, and the flag screen that closed WSClean's own knobs. |
| [I/O placement](nested-sampling-io-placement.md) | Where an evaluation's 2.4 MB goes, and why A/B arms must run *simultaneously* - sequential ones give consistent ~4% false positives on this host. |
| [clean loop](nested-sampling-clean-loop.md) | `-mgain`, and its exact science cost across all eight metrics on 600 real evaluations. |
| [MS open](nested-sampling-ms-open.md) | casacore re-attaching every subtable on each of the ~16 parent-MS opens, and why the avenue is closed from the simulator's side. |
| [WSClean patches](nested-sampling-wsclean-patches.md) | How `docker/wsclean/patches/` works - `git apply` in the Dockerfile, hashed into `ri.build-inputs`, and a `WSCLEAN_GIT_TAG` bump that breaks a patch fails the build on purpose. |
| [zygote](nested-sampling-wsclean-zygote.md) | The fork server: 27 ms of every 163 ms process ran before `main()`. Also why an archived run's `image_binary` column is ~5 ms low against a run made since. |
| [phase profile](nested-sampling-phase-profile.md) | `-log-time` by default, and `./ri profile --phases`. Read it on *medians*. |
| [cost model](nested-sampling-cost-model.md) | Why throughput falls through a run, and `--over-time`. |
| [simulate stage](nested-sampling-simulate-stage.md) | The MeqTrees half, and the confirmation that nothing measurable is left in it. |
| [shared MS open](nested-sampling-shared-ms-open.md) | `patches/0003`, counted with a 30-line `LD_PRELOAD` shim over `open64` - no rebuild, no profiler. |
| [FFTW planner](nested-sampling-fftw-planner.md) | 63 transform plans built and thrown away per process. Note the four warmed sizes derive from `DEFAULT_IMAGE_DIM = 128`; a stale list costs the speedup, not a result. |
| [gridder floor](nested-sampling-gridder-floor.md) | ducc0's own timer tree, `patches/0004`, and the -29% that `patches/0006` takes by turning w-gridding off. Two arms' FITS files never byte-compare equal - the header carries the `-name` path, so compare data blocks. |
| [row blocks](nested-sampling-row-blocks.md) | `patches/0005`: 29484 casacore column reads an evaluation down to 27. |
| [process warm-up](nested-sampling-process-warm-up.md) | The 5 ms before the first visibility - and the unsigned-arithmetic bug that makes WSClean fit its restoring beam on a quarter of the box it means to use. A real failure mode, not result-preserving to fix. |
| [disk footprint](nested-sampling-disk-footprint.md) | Bytes rather than clock. A run is `~17 x nlive x num_repeats` evaluations, and `evaluations/` needs no sharding. |
| [evaluation floor](nested-sampling-evaluation-floor.md) | The current budget, `--mgain` as a flag, and the last six avenues closed. |

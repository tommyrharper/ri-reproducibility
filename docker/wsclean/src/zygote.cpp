// wsclean-zygote - a fork server for wsclean.
//
// Why this exists: at the concurrency a nested-sampling search runs at, ~27ms
// of every ~163ms `wsclean` process is spent before main() gets control. That
// is the C++ static initialisation of casacore and the 72 other shared objects
// wsclean links (`LD_PRELOAD=libcasa_ms.so.9 /bin/true` alone costs 11ms), and
// it is identical work every time. The search runs ~70 of these a second.
//
// So pay it once. This process links the same wsclean-lib the `wsclean` binary
// does, and after its own static initialisers have run it reads one request per
// line from stdin and forks a child per request. The child inherits the fully
// initialised address space and calls the same CommandLine::Parse/Run pair
// main() calls, so it images exactly what `wsclean` would have imaged.
//
// Protocol - one request line in, one reply line out, tab separated:
//   request: <cwd> \t <stdout path> \t <stderr path> \t <arg> ...
//   reply:   <exit status> \t <wall seconds> \t <peak rss bytes>
// The reply carries what `/usr/bin/time -v` used to be spawned to report, from
// wait4()'s rusage, so the caller also stops paying for that fork+exec. A child
// killed by a signal reports 128+signal, as a shell would. Paths and arguments
// therefore may not contain a tab or a newline; the caller checks that.
//
// See docs/nested-sampling-wsclean-zygote.md.

#include "commandline.h"
#include "wsclean.h"

#include <aocommon/checkblas.h>
#include <aocommon/logger.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>

#include <fftw3.h>
#include <fitsio.h>

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, '\t')) fields.push_back(field);
  return fields;
}

// fork() only copies the calling thread, so a second thread holding a lock at
// fork time deadlocks the child. Nothing wsclean links starts one from a static
// initialiser today - checked here rather than assumed, because the failure is
// a search that hangs rather than one that stops.
void RefuseIfThreaded() {
  std::ifstream status("/proc/self/status");
  std::string key;
  int threads = 1;
  while (status >> key) {
    if (key == "Threads:") {
      status >> threads;
      break;
    }
  }
  if (threads != 1) {
    std::cerr << "FATAL: wsclean-zygote started with " << threads
              << " threads; fork() is not safe here\n";
    std::exit(1);
  }
}

void Redirect(const std::string& path, int fd, int flags) {
  const int opened = open(path.c_str(), flags, 0644);
  if (opened < 0) _exit(126);
  if (dup2(opened, fd) < 0) _exit(126);
  if (opened != fd) close(opened);
}

[[noreturn]] void RunChild(const std::vector<std::string>& fields) {
  if (chdir(fields[0].c_str()) != 0) _exit(126);
  Redirect("/dev/null", 0, O_RDONLY);
  Redirect(fields[1], 1, O_WRONLY | O_CREAT | O_TRUNC);
  Redirect(fields[2], 2, O_WRONLY | O_CREAT | O_TRUNC);

  std::vector<const char*> argv;
  for (size_t i = 3; i < fields.size(); ++i) argv.push_back(fields[i].c_str());

  int status = 0;
  try {
    // Scoped so WSClean's destructor runs - it is what removes the reordered
    // temp files - before _exit() skips the global destructors and the unmapping
    // of 73 shared objects, which are pure teardown cost.
    wsclean::WSClean wsclean;
    if (wsclean::CommandLine::Parse(wsclean, static_cast<int>(argv.size()),
                                    argv.data(), false))
      wsclean::CommandLine::Run(wsclean);
  } catch (std::exception& e) {
    // Byte for byte what main.cpp prints, including its exit status of -1.
    aocommon::Logger::Error << "+ + + + + + + + + + + + + + + + + + +\n"
                            << "+ An exception occured:\n";
    std::istringstream iss(e.what());
    for (std::string line; std::getline(iss, line);)
      aocommon::Logger::Error << "+ >>> " << line << "\n";
    aocommon::Logger::Error << "+ + + + + + + + + + + + + + + + + + +\n";
    status = 255;
  }
  std::cout.flush();
  std::cerr.flush();
  _exit(status);
}

// FFTW builds a per-transform-size plan the first time it is asked for one, and
// wsclean asks 63 times per evaluation while never keeping one: schaapcommon's
// Convolve() creates and destroys four 1-D plans on every call (radler runs one
// per major cycle) and its Resampler constructor creates two 2-D ones per
// gridding and degridding pass. Counted with an LD_PRELOAD shim over the
// fftwf_plan_* entry points, that is 6.3ms of a ~56ms serial process, of which
// 4.4ms is the once-per-size build and 1.9ms the repeats.
//
// The once-per-size half is process-global state a forked child inherits, so
// the parent pays it here and no evaluation pays it again. Sizes, for this
// search's fixed `-size 128 128` and its baseline-derived `-scale`:
//   128  the image itself
//   142  radler's deconvolution convolution, even(ceil(1.1 x 128))
//   156  the padded image
//   108  the gridder's chosen inversion size ("using optimal: 108 x 108" in
//        every one of a 6641-evaluation run's logs)
// A size that stops being used costs this warm-up and nothing else; one that
// starts being used is simply not warmed, so this can only lose the speedup,
// never a result. See docs/nested-sampling-fftw-planner.md.
void WarmFftwPlanner() {
  for (const int n : {108, 128, 142, 156}) {
    fftwf_destroy_plan(fftwf_plan_dft_r2c_1d(n, nullptr, nullptr, FFTW_ESTIMATE));
    fftwf_destroy_plan(
        fftwf_plan_dft_1d(n, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE));
    fftwf_destroy_plan(
        fftwf_plan_dft_1d(n, nullptr, nullptr, FFTW_BACKWARD, FFTW_ESTIMATE));
    fftwf_destroy_plan(fftwf_plan_dft_c2r_1d(n, nullptr, nullptr, FFTW_ESTIMATE));

    // Resampler plans against fftw-allocated buffers, and FFTW keys what it
    // learns on their alignment, so the warm-up has to allocate too.
    float* image = fftwf_alloc_real(static_cast<size_t>(n) * n);
    fftwf_complex* spectrum =
        fftwf_alloc_complex(static_cast<size_t>(n) * (n / 2 + 1));
    fftwf_destroy_plan(
        fftwf_plan_dft_r2c_2d(n, n, image, spectrum, FFTW_ESTIMATE));
    fftwf_destroy_plan(
        fftwf_plan_dft_c2r_2d(n, n, spectrum, image, FFTW_ESTIMATE));
    fftwf_free(spectrum);
    fftwf_free(image);
  }
}

// The other two pieces of process-global state a child builds for itself.
//
// cfitsio's one-time initialisation is 0.47 ms and needs nothing from the
// request, so it is paid at start-up. casacore's is ~0.94 ms and needs a real
// Measurement Set to open - opening a plain casacore::Table warms less than
// half of it and creating a throwaway one costs 26 ms - so it is paid on the
// first request, out of the set that request names. Both are ordinary lazy
// initialisation of shared library state, so a child that inherits it images
// exactly what it would have imaged. A path that turns out not to be a
// Measurement Set costs this warm-up and nothing else: the child opens it
// again and reports the error itself.
//
// See docs/nested-sampling-process-warm-up.md.
void WarmCasacore(const std::string& measurement_set) {
  try {
    const casacore::MeasurementSet ms(measurement_set);
  } catch (const std::exception&) {
  }
}

double MonotonicSeconds() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

}  // namespace

int main() {
  check_openblas_multithreading();
  RefuseIfThreaded();
  WarmFftwPlanner();
  fits_init_cfitsio();

  bool casacore_warmed = false;
  for (std::string line; std::getline(std::cin, line);) {
    const std::vector<std::string> fields = SplitTabs(line);
    if (fields.size() < 4) {
      std::cout << "126\t0\t0" << std::endl;
      continue;
    }
    if (!casacore_warmed) {
      // wsclean's own argument order: the input Measurement Set is last.
      WarmCasacore(fields.back());
      // Re-checked rather than assumed, for the same reason it is checked at
      // start-up: this is the one warm-up that runs library code with a
      // thread pool in it, and it happens between the check and the fork.
      RefuseIfThreaded();
      casacore_warmed = true;
    }
    // Nothing buffered may survive into the child, or it is written twice.
    std::cout.flush();
    std::cerr.flush();
    const double started = MonotonicSeconds();
    const pid_t child = fork();
    if (child == 0) RunChild(fields);
    if (child < 0) {
      std::cout << "126\t0\t0" << std::endl;
      continue;
    }
    int status = 0;
    rusage usage{};
    while (wait4(child, &status, 0, &usage) < 0 && errno == EINTR) {
    }
    const int code = WIFSIGNALED(status) ? 128 + WTERMSIG(status)
                                         : WEXITSTATUS(status);
    std::cout << code << '\t' << (MonotonicSeconds() - started) << '\t'
              << static_cast<long long>(usage.ru_maxrss) * 1024 << std::endl;
  }
  return 0;
}

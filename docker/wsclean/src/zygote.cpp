// Fork an initialized WSClean child per evaluation. Protocol details:
// docs/nested-sampling-wsclean-zygote.md.

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

// fork() copies only the calling thread; a held lock would hang the child.
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
    // Scope destructor removes reordered temp files before _exit skips globals.
    wsclean::WSClean wsclean;
    if (wsclean::CommandLine::Parse(wsclean, static_cast<int>(argv.size()),
                                    argv.data(), false))
      wsclean::CommandLine::Run(wsclean);
  } catch (std::exception& e) {
    // Match main.cpp's exception output and failure status.
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

// Warm process-global FFTW plans inherited by forked evaluations. Sizes match
// this search's fixed image and scale; see docs/nested-sampling-fftw-planner.md.
void WarmFftwPlanner() {
  for (const int n : {108, 128, 142, 156}) {
    fftwf_destroy_plan(fftwf_plan_dft_r2c_1d(n, nullptr, nullptr, FFTW_ESTIMATE));
    fftwf_destroy_plan(
        fftwf_plan_dft_1d(n, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE));
    fftwf_destroy_plan(
        fftwf_plan_dft_1d(n, nullptr, nullptr, FFTW_BACKWARD, FFTW_ESTIMATE));
    fftwf_destroy_plan(fftwf_plan_dft_c2r_1d(n, nullptr, nullptr, FFTW_ESTIMATE));

    // FFTW keys resampler plans by buffer alignment, so allocate matching buffers.
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

// Warm casacore once against the first request's Measurement Set.
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
      WarmCasacore(fields.back());
      // Casacore warm-up may create threads between the initial check and fork.
      RefuseIfThreaded();
      casacore_warmed = true;
    }
    // Avoid duplicated buffered output after fork.
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

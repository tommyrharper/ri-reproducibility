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

double MonotonicSeconds() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

}  // namespace

int main() {
  check_openblas_multithreading();
  RefuseIfThreaded();

  for (std::string line; std::getline(std::cin, line);) {
    const std::vector<std::string> fields = SplitTabs(line);
    if (fields.size() < 4) {
      std::cout << "126\t0\t0" << std::endl;
      continue;
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

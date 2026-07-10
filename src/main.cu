#include <iostream>

#include "pi/method/quadrature/riemann.cuh"

int main() {
  unsigned long long intervals = 1'000'000'000ULL;
  pi::method::Riemann method(intervals);

  pi::core::PiResult result = method.compute();

  std::cout << "Algorithm: Riemann Sum\n";
  std::cout << "Result:    " << result.digits << "\n";
  std::cout << "Time:      " << result.compute_time_ms << " ms\n";
  std::cout << "Steps:     " << result.iterations << "\n";

  return 0;
}

#include <iomanip>
#include <iostream>
#include <string>

#include "pi/method/quadrature/riemann.cuh"
#include "pi/method/series/leibniz.cuh"
#include "pi/method/series/nilakantha.cuh"
#include "pi/method/stochastic/monte_carlo.cuh"

void print_results(const std::string& name, pi::core::BaseMethod& method) {
  std::cout << "Algorithm: " << name << "\n";

  pi::core::PiResult result = method.compute();

  const double TRUE_PI = 3.14159265358979323846;  // from Wikipedia
  double error = std::abs(result.value - TRUE_PI);

  std::cout << std::fixed << std::setprecision(15);
  std::cout << "Result:   " << result.value << "\n";
  std::cout << "Error:    " << error << "\n";
  std::cout << std::setprecision(3);
  std::cout << "Time:     " << result.compute_time_ms << " ms\n";
  std::cout << "Steps:    " << result.iterations << "\n";
  std::cout << "--------------------------------------\n";
}

int main() {
  unsigned long long intervals = 1'000'000'000ULL;

  pi::method::Riemann riemann(intervals);
  print_results("Riemann", riemann);

  pi::method::MonteCarlo monte_carlo(intervals);
  print_results("Monte Carlo", monte_carlo);

  pi::method::Leibniz leibniz(intervals);
  print_results("Leibniz", leibniz);

  pi::method::Nilakantha nilakantha(intervals);
  print_results("Nilakantha", nilakantha);

  return 0;
}

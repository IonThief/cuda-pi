#include <chrono>
#include <iomanip>
#include <iostream>

#include "pi_reduce.cuh"

int main(int argc, char* argv[]) {
  unsigned long long num_intervals = 100'000'000;

  if (argc > 1) {
    num_intervals = std::stoull(argv[1]);
  }

  std::cout << "Calculating Pi using " << num_intervals << " intervals..."
            << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();
  double pi = cuda_pi::calculate(num_intervals);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end_time - start_time;

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Calculated Pi: " << pi << std::endl;
  std::cout << "Time taken   : " << duration.count() << " ms" << std::endl;

  return 0;
}

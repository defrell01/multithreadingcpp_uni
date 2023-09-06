#include <iostream>
#include <chrono>

unsigned long long rowSum(unsigned long long start, unsigned long long end) {
    unsigned long long sum = 0;
    for (unsigned long long n = start; n <= end; ++n) {
        sum += n;
    }
    return sum;
}

int main() {
    unsigned long long start = 0;
    unsigned long long end = 10000000000ULL; 

    auto start_time = std::chrono::steady_clock::now();

    unsigned long long res = rowSum(start, end);

    auto end_time = std::chrono::steady_clock::now();

    std::cout << res << '\n';

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Время выполнения программы составило " << elapsed_time << " миллисекунд\n";

    return 0;
}

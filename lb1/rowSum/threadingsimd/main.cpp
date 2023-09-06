#include <thread>
#include <iostream>
#include <chrono>
#include <vector>
#include <immintrin.h> // Include AVX2 intrinsics header

void rowSum(std::vector<unsigned long long> &res, unsigned long long start, unsigned long long end) {
    unsigned long long tmp = 0;

    while (start <= end) {
        tmp += start;
        start++;
    }

    res.emplace_back(tmp);
}

unsigned long long sumVectorAVX2(const std::vector<unsigned long long> &v) {
    unsigned long long sum = 0;

    // Calculate the sum using AVX2
    __m256i sumVector = _mm256_setzero_si256(); // Initialize a 256-bit vector to zero

    for (size_t i = 0; i < v.size(); i += 4) {
        __m256i data = _mm256_loadu_si256((__m256i*)&v[i]); // Load 256 bits of data from the vector
        sumVector = _mm256_add_epi64(sumVector, data); // Add the data to the sum vector
    }

    // Extract the sum from the AVX2 vector
    unsigned long long sumArray[4];
    _mm256_storeu_si256((__m256i*)sumArray, sumVector);

    // Sum the four values from the AVX2 vector
    for (int i = 0; i < 4; i++) {
        sum += sumArray[i];
    }

    return sum;
}

int main() {
    unsigned long long n = 10000000000;

    std::vector<unsigned long long> results;

    std::vector<std::thread> threads;

    unsigned int threadsNum = std::thread::hardware_concurrency();

    for (unsigned long long i = 0; i < threadsNum; i++) {
        unsigned long long start = i * (n / threadsNum);
        unsigned long long end = (i == threadsNum - 1) ? n : (i + 1) * (n / threadsNum) - 1;
        threads.emplace_back(rowSum, std::ref(results), start, end);
    }

    auto start_time = std::chrono::steady_clock::now();

    for (std::thread &thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::steady_clock::now();

    // Sum the results using AVX2
    unsigned long long res = sumVectorAVX2(results);

    std::cout << res << '\n';

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Время выполнения программы с разделением на " << threadsNum << " потоков составило " << elapsed_time << " миллисекунд\n";

    return 0;
}

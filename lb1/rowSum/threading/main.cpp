#include <thread>
#include <iostream>
#include <chrono>
#include <vector>

void rowSum(std::vector<unsigned long long> &res, unsigned long long start, unsigned long long end) {
    unsigned long long tmp = 0;

    while (start <= end) {
        tmp += start; 
        start++;
    }

    res.emplace_back(tmp);
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

    auto start = std::chrono::steady_clock::now();

    for (std::thread &thread : threads) {
        thread.join();
    }

    unsigned long long res = 0; 

    for (auto &n : results) {
        res += n;
    }

    std::cout << res << '\n';

    auto end = std::chrono::steady_clock::now();

    std::cout << "Время выполнения программы с разделением на " << threadsNum << " потоков составило " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " миллисекунд\n";

    return 0; 
}

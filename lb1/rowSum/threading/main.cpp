#include <thread>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <vector>
#include <mutex>

std::mutex mtx;

void rowSum(unsigned long long &n, unsigned long long start, unsigned long long end) {
    
    unsigned long long partial_sum = 0;
    
    for (unsigned long long i = start; i <= end; ++i) {
        if (i > 0) {
            if (i % 2)
                partial_sum += ((i + 1) >> 1) * i;
            else
                partial_sum += (i >> 1) * (i + 1);
        } 
        else if (i == 0) {
            partial_sum += 1;
        } 
        else {
            unsigned long long temp = -i;
            if (temp % 2)
                partial_sum += ((temp + 1) >> 1) * temp;
            else
                partial_sum += (temp >> 1) * (temp + 1);
            partial_sum = 1 - partial_sum;
        }
    }

    // Lock the mutex to safely update the shared 'n' variable
    std::lock_guard<std::mutex> lock(mtx);
    n += partial_sum;
}


int main() {
    
    unsigned long long n = 10000000000;

    unsigned int threadsNum = 6;

    std::vector<std::thread> threads;
    
    

    for (unsigned long long i = 0; i < threadsNum; i++) {
        unsigned long long start = i * (n / threadsNum);
        unsigned long long end = (i == threadsNum - 1) ? n : (i + 1) * (n / threadsNum) - 1;
        threads.emplace_back(rowSum, std::ref(n), start, end);
    }

    auto start = std::chrono::steady_clock::now();

    for (std::thread &thread : threads) {
        thread.join();
    }

    std::cout << n << '\n';

    auto end = std::chrono::steady_clock::now();

    std::cout << "Время выполнения программы с разделением на 6 потоков составило " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << " наносекунд\n";

}
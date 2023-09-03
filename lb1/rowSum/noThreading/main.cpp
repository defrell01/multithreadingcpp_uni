#include <thread>
#include <iostream>
#include <chrono>
#include <unistd.h>

void rowSum(unsigned long long &n) {
    
    if (n > 0)
    {
        if (n%2)
            n = ((n+1) >> 1)*n;
        else
            n = (n >> 1)*(n+1);
    }
    else if (n==0) n = 1;
    else
    {
        n = -n;
        if (n%2)
            n = ((n+1) >> 1)*n;
        else
            n = (n >> 1)*(n+1);
        n = 1 - n;
    }
    
}


int main() {
    
    unsigned long long n = 10000000000;
    
    auto start = std::chrono::steady_clock::now();

    rowSum(n);

    std::cout << n << '\n';

    auto end = std::chrono::steady_clock::now();

    std::cout << "Время выполнения программы составило " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << " наносекунд\n";

}
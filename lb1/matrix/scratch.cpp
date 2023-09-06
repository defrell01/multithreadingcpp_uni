#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <vector>

using namespace std;
using namespace std::chrono;

const int SIZE = 1024;
const unsigned int threadsNum = std::thread::hardware_concurrency();

void printMatrix(int16_t** matrix)
{
    // Оставляем функцию без изменений
}

int16_t** generateMatrix(bool empty)
{
    // Оставляем функцию без изменений
}

int16_t** transpose(int16_t** matrix)
{
    // Оставляем функцию без изменений
}

void multVectorBlock(int16_t** A, int16_t** B, int16_t** C, int startRow, int endRow)
{
    for (int i = startRow; i < endRow; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            __m256i c_line = _mm256_setzero_si256();

            for (int k = 0; k < SIZE; k += 16)
            {
                __m256i a_line = _mm256_loadu_si256((__m256i*)&A[i][k]);
                __m256i b_line = _mm256_loadu_si256((__m256i*)&B[j][k]);
                __m256i mult = _mm256_mullo_epi16(a_line, b_line);

                c_line = _mm256_add_epi16(c_line, mult);
            }

            int16_t temp[16];
            _mm256_storeu_si256((__m256i*)temp, c_line);

            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7] +
                      temp[8] + temp[9] + temp[10] + temp[11] + temp[12] + temp[13] + temp[14] + temp[15];
        }
    }
}

int main()
{
    srand(time(NULL)); // Seed the random number generator

    cout << "Generating matrices" << endl;
    int16_t** fMatrix = generateMatrix(false);
    int16_t** sMatrix = generateMatrix(false);
    cout << endl;

    int16_t** fRes = generateMatrix(true);
    int16_t** sRes = generateMatrix(true);

    cout << "Scalar multiplication benchmark:" << endl;
    auto startS = high_resolution_clock::now();
    fRes = multScalar(fMatrix, sMatrix, fRes);
    auto stopS = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stopS - startS);
    cout << "Execution time is " << duration.count() << " s " << endl;

    cout << "-------------------------------------------------------------------" << endl;
    cout << "Vector multiplication benchmark:" << endl;
    auto startV = high_resolution_clock::now();
    int16_t** transposed = transpose(sMatrix);

    vector<thread> threads;
    vector<int16_t**> results(threadsNum);

    for (unsigned int i = 0; i < threadsNum; ++i)
    {
        int startRow = i * SIZE / threadsNum;
        int endRow = (i + 1) * SIZE / threadsNum;
        results[i] = generateMatrix(true);
        threads.emplace_back(multVectorBlock, fMatrix, transposed, results[i], startRow, endRow);
    }

    for (unsigned int i = 0; i < threadsNum; ++i)
    {
        threads[i].join();
    }

    // Объединяем результаты подсчета в sRes
    for (unsigned int i = 0; i < threadsNum; ++i)
    {
        for (int row = i * SIZE / threadsNum; row < (i + 1) * SIZE / threadsNum; ++row)
        {
            for (int col = 0; col < SIZE; ++col)
            {
                sRes[row][col] = results[i][row][col];
            }
        }
    }

    auto stopV = high_resolution_clock::now();
    auto durationV = duration_cast<seconds>(stopV - startV);
    cout << "Execution time is " << durationV.count() << " s " << endl;

    cout << "Check if matrices are equal: ";
    cout << (isEq(fRes, sRes) ? "Yes" : "No") << endl;

    return 0;
}

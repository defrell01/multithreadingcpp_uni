#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <vector>
#include <cstdint>


const int SIZE = 1024;

void printMatrix(int16_t** matrix)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << '\n';
    }
}

int16_t** generateMatrix(bool empty)
{
    int16_t** matrix = new int16_t* [SIZE];

    for (int i = 0; i < SIZE; ++i)
    {
        matrix[i] = new int16_t[SIZE];
        for (int j = 0; j < SIZE; ++j)
        {
            if (!empty)
            {
                matrix[i][j] = rand() % 32767;
            }
            else
            {
                matrix[i][j] = 0;
            }
        }
    }
    return matrix;
}

int16_t** transpose(int16_t** matrix)
{
    int16_t** res = new int16_t* [SIZE];
    for (int i = 0; i < SIZE; ++i)
    {
        res[i] = new int16_t[SIZE];
        for (int j = 0; j < SIZE; ++j)
        {
            res[i][j] = matrix[j][i];
        }
    }
    return res;
}

int16_t** multScalar(int16_t** A, int16_t** B, int16_t** res)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            res[i][j] = 0;
            for (int k = 0; k < SIZE; ++k)
            {
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return res;
}

int16_t** multVector(int16_t** A, int16_t** B, int16_t** C)
{
    for (int i = 0; i < SIZE; ++i)
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
    return C;
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

void multVectorBlockScalar(int16_t** A, int16_t** B, int16_t** C, int startRow, int endRow)
{
    for (int i = startRow; i < endRow; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            int16_t sum = 0;

            for (int k = 0; k < SIZE; ++k)
            {
                sum += A[i][k] * B[j][k];
            }

            C[i][j] = sum;
        }
    }
}


bool isEq(int16_t** A, int16_t** B)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            if (A[i][j] != B[i][j]) return false;
        }
    }
    return true;
}

int main()
{

    const unsigned int threadsNum = std::thread::hardware_concurrency();
    srand(time(NULL));

    std::cout << "Генерация матриц" << '\n';
    int16_t** fMatrix = generateMatrix(false);
    int16_t** sMatrix = generateMatrix(false);
    std::cout << '\n';

    int16_t** fRes = generateMatrix(true);
    int16_t** sRes = generateMatrix(true);

    std::cout << "Скалярное перемножение:" << '\n';
    auto startS = std::chrono::high_resolution_clock::now();
    fRes = multScalar(fMatrix, sMatrix, fRes);
    auto stopS = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopS - startS);
    std::cout << "Время исполнения " << duration.count() << " миллисекунд " << '\n';

    std::cout << "-------------------------------------------------------------------" << '\n';

    std::cout << "Векторное перемножение" << '\n';
    std::vector<std::thread> threads;
    std::vector<int16_t**> results(threadsNum);
    int16_t** transposed = transpose(sMatrix);
    
    for (unsigned int i = 0; i < threadsNum; i++) {
        int startRow = i * SIZE / threadsNum;
        int endRow = (i + 1) * SIZE / threadsNum;
        results[i] = generateMatrix(true);
        threads.emplace_back(multVectorBlockScalar, fMatrix, transposed, results[i], startRow, endRow);
    }

    for (std::thread &thread : threads) {
        thread.join();
    }
    auto startV = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < threadsNum; ++i)
    {
        for (int row = i * SIZE / threadsNum; row < (i + 1) * SIZE / threadsNum; ++row)
        {
            for (int col = 0; col < SIZE; ++col)
            {
                sRes[row][col] = results[i][row][col];
            }
        }
    }
    
    

    auto stopV = std::chrono::high_resolution_clock::now();
    
    auto durationV = std::chrono::duration_cast<std::chrono::microseconds>(stopV - startV);
    
    std::cout << "Время исполнения " << durationV.count() << " микросекунд \n";

    std::cout << "Проверка, равны ли матрицы ";
    std::cout << (isEq(fRes, sRes) ? "Да" : "Нет") << '\n';

    return 0;
}

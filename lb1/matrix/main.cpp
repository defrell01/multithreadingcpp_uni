#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace std::chrono;

const int SIZE = 1024;

void printMatrix(int16_t** matrix)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
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
                matrix[i][j] = rand() % 32767; // Generate random int16 values
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
    sRes = multVector(fMatrix, transposed, sRes);
    auto stopV = high_resolution_clock::now();
    auto durationV = duration_cast<seconds>(stopV - startV);
    cout << "Execution time is " << durationV.count() << " s " << endl;

    cout << "Check if matrices are equal: ";
    cout << (isEq(fRes, sRes) ? "Yes" : "No") << endl;

    return 0;
}

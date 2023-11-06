#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <openssl/md5.h>
#include <string>

// Функция для перебора паролей
std::string generate_next_password(std::string current_password)
{
    for (int i = current_password.length() - 1; i >= 0; i--)
    {
        if (current_password[i] == '9')
        {
            current_password[i] = 'a';
            return current_password;
        }
        else if (current_password[i] == 'z')
        {
            current_password[i] = 'A';
            return current_password;
        }
        else if (current_password[i] == 'Z')
        {
            current_password[i] = '0';
        }
        else
        {
            current_password[i]++;
            return current_password;
        }
    }
    return ""; // Все возможные пароли перебраны
}

std::string calculate_md5(const std::string &input)
{
    unsigned char md5_hash[MD5_DIGEST_LENGTH];
    MD5((const unsigned char *)input.c_str(), input.length(), md5_hash);

    char md5_str[33];
    for (int i = 0; i < 16; i++)
    {
        sprintf(&md5_str[i * 2], "%02x", (unsigned int)md5_hash[i]);
    }

    return std::string(md5_str);
}

int main(int argc, char **argv)
{
    std::string password;

    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int password_length = 4; // Задайте длину пароля

    auto start = std::chrono::high_resolution_clock::now();

    std::string target_md5 =
        "a9bf98000f304b48bedde63af949f5bd"; // Замените на целевой хэш

    std::string current_password = "";
    for (int i = 0; i < password_length; i++)
    {
        current_password += '0'; // Начинаем с минимального значения
    }

    long long iterations = 0;

    bool password_found = false;

    while (!password_found)
    {
        std::string md5 = calculate_md5(current_password);
        iterations++;

        if (md5 == target_md5)
        {
            password = current_password;
            password_found = true;
        }

        // Распределение работы между процессами MPI
        if (world_rank == 0)
        {
            // Процесс с рангом 0 генерирует следующий пароль
            current_password = generate_next_password(current_password);
        }

        // Рассылка текущего пароля всем процессам
        MPI_Bcast(&current_password[0], password_length, MPI_CHAR, 0,
                  MPI_COMM_WORLD);

        if (password_found)
        {
            // Сигнализируем о завершении работы другим процессам
            int finished = 1;
            MPI_Bcast(&finished, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    if (world_rank == 0)
    {
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        std::cout << "Password found: " << password
                  << "\nTime elapsed: " << duration.count() << " ms\n";
    }

    MPI_Finalize();
    return 0;
}
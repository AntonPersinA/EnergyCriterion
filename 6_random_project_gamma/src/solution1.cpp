#include "solution1.h"


template <typename T>
void print(const T& s)
{
    std::cout << s << "\n";
}


template <typename T>
void print_iter(const T& s)
{
    for (auto i : s) {
        std::cout << i << " ";
    }
    std::cout << "\n";
}


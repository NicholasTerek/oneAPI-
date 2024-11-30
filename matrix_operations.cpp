#include "oneapi/mkl.hpp"
#include <iostream>
#include <vector>

int main() {
    constexpr size_t size = 3;
    std::vector<float> matrix(size * size, 1.0f); // Simulate a matrix with all elements 1.0
    std::vector<float> result(size * size, 0.0f);

    // Compute the matrix-vector product
    oneapi::mkl::blas::row_major::gemv(
        oneapi::mkl::blas::row_major::layout::row_major, size, size, 1.0, matrix.data(), size,
        matrix.data(), 1, 0.0, result.data(), 1);

    std::cout << "Processed Matrix:" << std::endl;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            std::cout << result[i * size + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

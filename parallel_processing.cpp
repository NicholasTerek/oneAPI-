// parallel_processing.cpp
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>

using namespace sycl;

int main() {
    constexpr size_t data_size = 1024;
    std::vector<float> bacteria_count(data_size, 0.0f);
    std::vector<float> virus_rna_level(data_size, 0.0f);

    // Populate sample data
    for (size_t i = 0; i < data_size; ++i) {
        bacteria_count[i] = i % 100 + 1; // Simulate bacteria count
        virus_rna_level[i] = (i % 50) / 10.0f; // Simulate RNA levels
    }

    // Buffer setup
    buffer<float> bacteria_buf(bacteria_count.data(), range<1>(data_size));
    buffer<float> virus_buf(virus_rna_level.data(), range<1>(data_size));

    queue q;

    // Submit a parallel task to process the data
    q.submit([&](handler &h) {
        auto bacteria = bacteria_buf.get_access<access::mode::read_write>(h);
        auto virus = virus_buf.get_access<access::mode::read_write>(h);

        h.parallel_for(range<1>(data_size), [=](id<1> idx) {
            // Simple processing logic
            bacteria[idx] = bacteria[idx] * 2; // Amplify bacteria signal
            virus[idx] = virus[idx] + 1;      // Increment RNA levels
        });
    }).wait();

    std::cout << "Processed Data (first 10):" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Bacteria: " << bacteria_count[i] << ", RNA: " << virus_rna_level[i] << std::endl;
    }

    return 0;
}

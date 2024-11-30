#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace sycl;

int main() {
    constexpr size_t data_size = 1024;
    std::vector<float> bacteria_count(data_size);
    std::vector<float> virus_rna_level(data_size);

    // Generate random data
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> bacteria_dist(100, 1000);
    std::uniform_real_distribution<float> rna_dist(0.1, 10.0);

    for (size_t i = 0; i < data_size; ++i) {
        bacteria_count[i] = bacteria_dist(rng);
        virus_rna_level[i] = rna_dist(rng);
    }

    // Process data with DPC++
    buffer<float> bacteria_buf(bacteria_count.data(), range<1>(data_size));
    buffer<float> virus_buf(virus_rna_level.data(), range<1>(data_size));

    queue q;
    q.submit([&](handler &h) {
        auto bacteria = bacteria_buf.get_access<access::mode::read_write>(h);
        auto virus = virus_buf.get_access<access::mode::read_write>(h);

        h.parallel_for(range<1>(data_size), [=](id<1> idx) {
            bacteria[idx] *= 2;  // Amplify bacteria signal
            virus[idx] += 1;     // Increment RNA levels
        });
    }).wait();

    // Display processed data
    std::cout << "Processed Data (First 10 Samples):\n";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Bacteria: " << bacteria_count[i] << ", RNA: " << virus_rna_level[i] << "\n";
    }

    return 0;
}

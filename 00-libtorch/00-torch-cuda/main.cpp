#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "CUDA available: " << torch::cuda::is_available() << "\n";
    std::cout << "CUDA devices:   " << torch::cuda::device_count() << "\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    auto a = torch::randn({2048, 2048}, device);
    auto b = torch::randn({2048, 2048}, device);
    auto c = torch::matmul(a, b);
    torch::cuda::synchronize();

    std::cout << "Result device: " << c.device() << "\n";
    std::cout << "Sum: " << c.sum().item<float>() << "\n";
    return 0;
}
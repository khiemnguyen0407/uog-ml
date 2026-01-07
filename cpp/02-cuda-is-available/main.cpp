#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a 2x3 random tensor
    torch::Tensor tensor = torch::rand({2, 3});
    
    std::cout << "Hello from LibTorch!\n";
    std::cout << "Generated Tensor:\n" << tensor << std::endl;

    // Check if CUDA is available (useful for cluster work)
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Moving tensor to GPU...\n";
        tensor = tensor.to(torch::kCUDA);
        std::cout << "GPU Tensor:\n" << tensor << std::endl;
    } else {
        std::cout << "CUDA not available, staying on CPU.\n";
    }

    return 0;
}
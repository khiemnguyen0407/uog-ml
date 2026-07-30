// #include "ATen/core/grad_mode.h"
// #include "ATen/ops/randn.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/grad_mode.h"
#include "c10/core/TensorOptions.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/functional/loss.h"
#include <cstdint>
// #include <exception>
#include <string>
#include <torch/torch.h>
#include <iostream>

namespace F = torch::nn::functional;

void print_banner(const std::string& message)
{
    std::cout << std::string(30, '=') << std::endl;
    std::cout << message << std::endl;
    std::cout << std::string(30, '=') << std::endl;
}
int main()
{
    
    // Create tensors with gradient tracking
    torch::Tensor x = torch::tensor({2.0}, torch::requires_grad());
    torch::Tensor w = torch::tensor({3.0}, torch::requires_grad());
    torch::Tensor b = torch::tensor({4.0}, torch::requires_grad());

    // Forward pass: y = w * x + b
    torch::Tensor y = w * x + b;
    y.backward();

    // Graidents of y with respect to its tensor nodes:
    std::cout << "dy/dx = " << x.grad() << std::endl;
    std::cout << "dy/dw = " << w.grad() << std::endl;
    std::cout << "dy/db = " << b.grad() << std::endl;

    // Zero gradients before next iteration
    w.mutable_grad().zero_();
    b.mutable_grad().zero_();

    // Wrap operations in torch::NoGradGuard guard; to disable gradient tracking
    // just like with torch.nograd():
    // {
    //     torch::NoGradGuard no_grad_guard;

    //     torch::Tensor z = w * x + b;    
    //     // We cannot do z.backward()
    //     try {
    //         z.backward();
    //     }
    //     catch (std::exception& exc){
    //         exc.what();
    //         std::cerr << "z.backward() does not work due to no grad tracking.\n\n";

    //     }
    // }
    

    // Linear Regression from scratch by using autograd
    print_banner("Linear Regression from Scratch");

    // Hyperparameters
    const int64_t input_size {3};
    const int64_t output_size {2};
    const int64_t num_samples {100};

    const int64_t epochs {10000};
    const double learning_rate  {0.01};
    const double noise_amplitude {1e-2};

    const int64_t num_prints {10};
    const int64_t print_period = static_cast<int64_t>( static_cast<double>(epochs) / num_prints);

    // Create synthetic data: y = W x + b + noise
    torch::Tensor X = torch::randn({num_samples, input_size});
    torch::Tensor W_true = torch::tensor(
        {{1.0, 2.0, 0.5}, 
        {2.0, 1.0, 0.5}});

    torch::Tensor bias_true = torch::tensor({0.5, 1.5});
    torch::Tensor y_true = X.matmul(W_true.t()) + bias_true 
        + noise_amplitude * torch::randn({num_samples, output_size});

    std::cout << "W true:\n" << W_true << std::endl;
    std::cout << "bias true:\n" << bias_true << std::endl;

    
    // Initialize learnable parameters
    torch::Tensor weights = torch::randn({output_size, input_size},
        torch::requires_grad());
    torch::Tensor bias = torch::zeros({output_size}, torch::requires_grad());

    // Training loop
    for (std::int64_t epoch {0}; epoch < epochs; ++epoch)
    {   
        // Forward pass
        auto predict = X.matmul(weights.t()) + bias;
        auto loss = F::mse_loss(predict, y_true);

        // Backward pass
        loss.backward();
        
        // Manual parameter update (no optimizer used here!)
        {
            torch::NoGradGuard no_grad;
            weights -= learning_rate * weights.grad();
            bias -= learning_rate * bias.grad();
        }

        // Zero gradients before the next backward pass
        weights.mutable_grad().zero_();
        bias.mutable_grad().zero_();


        if (epoch % print_period == 0)
        {
            std::cout << "Epoch [" << epoch << "/" << epochs
                      << "], Loss: " << loss.item<double>() << std::endl;
        }
    }   

    // Print the learnable parameters on the console
    std::cout << "Learned weights: \n" << weights << std::endl;
    std::cout << "Learned bias:\n" << bias << std::endl;


    return 0;

}

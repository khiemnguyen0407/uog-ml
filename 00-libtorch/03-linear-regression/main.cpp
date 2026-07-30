#include "ATen/core/TensorBody.h"
#include "ATen/ops/tensor.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/functional/loss.h"
#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/optim/sgd.h"
#include <cstdint>
#include <memory>
#include <torch/torch.h>
#include <iostream>

// Define model

namespace nn = torch::nn;
namespace F = torch::nn::functional;

// we cannot use class because parameters will be hidden as private
struct LinearRegression : nn::Module    
{
  
public:

    nn::Linear linear{nullptr};

    // Constructor
    LinearRegression(int64_t in_features, int64_t out_features)
    {
        linear = register_module("linear", 
            nn::Linear(in_features, out_features));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return linear->forward(x);
    }
};

int main(int argv, char** argc)
{   
    const double noise_amplitude {0.01};

    const int8_t input_features {3};
    const int8_t output_features {2};
    const int64_t n_samples { 1000 };
    
    const int64_t epochs {10000};
    const int64_t print_every {10};
    const int64_t print_period 
        = static_cast<int64_t>(static_cast<double>(epochs) / print_every);
    
    const torch::Tensor W_true = torch::tensor(
        {{1.0, 2.0, 0.5}, 
        {2.0, 1.0, 0.5}});

    const torch::Tensor b_true = torch::tensor({0.5, 1.5});

    torch::Tensor X = torch::randn({n_samples, input_features});
    torch::Tensor y_true = X.matmul(W_true.t()) + b_true 
        + noise_amplitude * torch::randn({n_samples, output_features});

    // Model, loss and optimizer

    auto model = std::make_shared<LinearRegression>(input_features, output_features);
    auto optimizer = torch::optim::Adam(model->parameters(), 0.01);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        optimizer.zero_grad();
        auto predict = model->forward(X);
        auto loss = F::mse_loss(predict, y_true);

        loss.backward();
        optimizer.step();

        if (epoch % print_period == 0)
        {
            std::cout << "Epoch [" << epoch << "/" << epochs << "] | "
            << "Loss = " << loss.item<double>() << std::endl;
        }
    }

    // Inspect parameters of the model
    for (const auto& p : model->named_parameters())
    {
        std::cout << p.key() << ":\n";
        std::cout << p.value() << std::endl;
    }

    return 0;
}
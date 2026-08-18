#include "ATen/core/TensorBody.h"
#include "ATen/core/grad_mode.h"
#include "torch/nn/init.h"
#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/pimpl.h"
#include <cstdint>
#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

namespace algo {
    constexpr double pi = 3.14159265358979323846;
    constexpr double material_C = 10.0;
    constexpr double k = 50.0;
    constexpr double prescribed_traction = 1.0;
    
    constexpr int64_t n_subdomains = 4;
    constexpr int64_t hidden_width = 32;
    constexpr int64_t n_quad_points = 1024;
    constexpr int64_t n_lbfgs_steps = 50;
    constexpr int64_t lbfgs_iterations_per_step = 20;

}   // namespace for algorithmic constants

/*
 * Local sine-activate neural networks
 */
struct LocalNetworkImpl : torch::nn::Module
{
    LocalNetworkImpl(int64_t input_dimension, int64_t hidden_dimension, int64_t output_dimension)
    {
        input_layer = register_module("layer_01", torch::nn::Linear(input_dimension, hidden_dimension));
        hidden_layer_01 = register_module("layer_02", torch::nn::Linear(hidden_dimension, hidden_dimension));
        hidden_layer_02 = register_module("layer_03", torch::nn::Linear(hidden_dimension, hidden_dimension));
        output_layer = register_module("layer_04", torch::nn::Linear(hidden_dimension, output_dimension));
    }

    torch::Tensor forward(const torch::Tensor& local_coordinate) {
        auto value = torch::sin(input_layer->forward(local_coordinate));
        value = torch::sin(hidden_layer_01->forward(value));
        value = torch::sin(hidden_layer_02->forward(value));

        return output_layer->forward(value);
    }

private:

    void initialize_paramters()
    {
        torch::NoGradGuard no_grad;
        for (auto& parameter_pair : this->named_parameters()) {
            auto& parameter = parameter_pair.value();

            if (parameter.dim() >= 2)
            {
                torch::nn::init::xavier_uniform_(parameter);
            } else {
                parameter.zero_();
            }
        }
    }

    torch::nn::Linear input_layer {nullptr};
    torch::nn::Linear hidden_layer_01 {nullptr};
    torch::nn::Linear hidden_layer_02 {nullptr};
    torch::nn::Linear output_layer {nullptr};
};

TORCH_MODULE(LocalNetwork);

int main()
{
    return 0;
}
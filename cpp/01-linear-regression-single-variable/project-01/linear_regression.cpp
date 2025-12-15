#include <torch/torch.h>
#include <iostream>

// Define the Simple Linear Model
// Inherit from torch::nn::Module
struct LinearRegression : torch::nn::Module
{
    LinearRegression()
    {
        // Initialize the single linear layer: y = W*x + b
        // The input dimension is 1 (for x), and the output dimension is 1 (for y)
        linear = register_module("linear", torch::nn::Linear(1, 1));
    }

    // Implement the forward pass: this is how the data flows through the model
    torch::Tensor forward(torch::Tensor x)
    {
        return linear(x);
    }

    // Module container for the linear layer
    torch::nn::Linear linear{nullptr};
};

int main()
{
    std::cout << "Starting Linear Regression Training...\n";

    // --- 1. Data Generation and Preparation ---

    // Define the true parameters for the data: y = 1.5 * x + 2.5
    const double TRUE_WEIGHT = 1.5;
    const double TRUE_BIAS = 2.5;
    const int NUM_SAMPLES = 10000;
    std::cout << "Number of samples = " << NUM_SAMPLES << "\n\n";
    // Generate x-values (100 values between 0 and 1, reshaped to [100, 1])
    auto x_data = torch::rand({NUM_SAMPLES, 1});

    // Generate y-values with some noise: y = TRUE_WEIGHT * x + TRUE_BIAS + noise
    auto y_data = TRUE_WEIGHT * x_data + TRUE_BIAS;

    // Add a small amount of Gaussian noise to make it realistic
    auto noise = torch::randn({NUM_SAMPLES, 1}) * 0.1;
    y_data += noise;

    // --- 2. Model, Loss, and Optimizer Setup ---

    // Instantiate the model and move it to the device (CPU in this case)
    auto model = std::make_shared<LinearRegression>();
    std::cout << model << "\n\n";
    // Define the loss function: Mean Squared Error (MSE)
    auto loss_function = torch::nn::MSELoss();

    // Define the optimizer: Stochastic Gradient Descent (SGD)
    // We pass the model's parameters and a learning rate (LR)
    auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(0.01));

    // --- 3. Training Loop ---
    const int NUM_EPOCHS = 500;

    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch)
    {
        // 1. Zero the gradients from the last step
        optimizer.zero_grad();

        // 2. Forward pass: compute the predicted y-values
        auto predictions = model->forward(x_data);

        // 3. Compute loss
        auto loss = loss_function(predictions, y_data);
        // std::cout << loss;
        // 4. Backward pass: compute gradients of the loss w.r.t model parameters
        loss.backward();

        // 5. Update weights: Optimizer takes a step
        optimizer.step();

        // Print progress every 50 epochs
        if (epoch % 50 == 0)
        {
            std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << "\n";
        }
    }

    // --- 4. Evaluation and Results ---

    std::cout << "\n--- Training Complete ---\n";

    // Extract the final learned parameters (Weight and Bias)
    auto weight = model->linear->weight.data().item<float>();
    auto bias = model->linear->bias.data().item<float>();

    std::cout << "True function: y = " << TRUE_WEIGHT << "x + " << TRUE_BIAS << "\n";
    std::cout << "Learned function: y = " << weight << "x + " << bias << "\n";


    torch::Device device(torch::kCPU);
    torch::Tensor tensor = torch::zeros({2, 2});
    std::cout << tensor << std::endl;


    if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! " << std::endl;
    device = torch::kCUDA;
    }

    torch::Tensor test_gpu_tensor = tensor.to(device);

    std::cout << test_gpu_tensor << std::endl;
    return 0;
}
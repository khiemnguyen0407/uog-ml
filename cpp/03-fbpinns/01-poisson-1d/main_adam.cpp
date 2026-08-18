#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double pi = 3.14159265358979323846;

constexpr double material_c = 10.0;
constexpr double wave_number_k = 50.0;
constexpr double prescribed_traction = 1.0;

constexpr int64_t number_of_subdomains = 4;
constexpr int64_t hidden_width = 32;
constexpr int64_t number_of_quadrature_points = 64;

constexpr int64_t number_of_adam_epochs = 1000;
constexpr double adam_learning_rate = 1.0e-3;
constexpr int64_t print_frequency = 50;

}  // namespace


// =============================================================================
// Local neural network
// =============================================================================

struct LocalNetworkImpl : torch::nn::Module {
    LocalNetworkImpl(
        int64_t input_dimension,
        int64_t hidden_dimension,
        int64_t output_dimension
    ) {
        first_layer = register_module(
            "first_layer",
            torch::nn::Linear(
                input_dimension,
                hidden_dimension
            )
        );

        second_layer = register_module(
            "second_layer",
            torch::nn::Linear(
                hidden_dimension,
                hidden_dimension
            )
        );

        third_layer = register_module(
            "third_layer",
            torch::nn::Linear(
                hidden_dimension,
                hidden_dimension
            )
        );

        output_layer = register_module(
            "output_layer",
            torch::nn::Linear(
                hidden_dimension,
                output_dimension
            )
        );

        initialise_parameters();
    }

    torch::Tensor forward(
        const torch::Tensor& local_coordinate
    ) {
        auto value = torch::sin(
            first_layer->forward(local_coordinate)
        );

        value = torch::sin(
            second_layer->forward(value)
        );

        value = torch::sin(
            third_layer->forward(value)
        );

        return output_layer->forward(value);
    }

private:
    void initialise_parameters() {
        torch::NoGradGuard no_grad;

        for (auto& parameter_pair : named_parameters()) {
            auto& parameter = parameter_pair.value();

            if (parameter.dim() >= 2) {
                torch::nn::init::xavier_uniform_(parameter);
            } else {
                parameter.zero_();
            }
        }
    }

    torch::nn::Linear first_layer{nullptr};
    torch::nn::Linear second_layer{nullptr};
    torch::nn::Linear third_layer{nullptr};
    torch::nn::Linear output_layer{nullptr};
};

TORCH_MODULE(LocalNetwork);


// =============================================================================
// Four-subdomain FBPINN
// =============================================================================

struct FbpinnImpl : torch::nn::Module {
    FbpinnImpl() {
        /*
         * Four overlapping subdomain centres in the transformed
         * coordinate z in [0,1].
         */
        const std::vector<double> subdomain_centres{
            0.125,
            0.375,
            0.625,
            0.875
        };

        /*
         * Every subdomain has total width 0.5.
         *
         * The nominal supports are:
         *
         * Subdomain 1: [-0.125, 0.375]
         * Subdomain 2: [ 0.125, 0.625]
         * Subdomain 3: [ 0.375, 0.875]
         * Subdomain 4: [ 0.625, 1.125]
         */
        const std::vector<double> subdomain_half_widths{
            0.25,
            0.25,
            0.25,
            0.25
        };

        /*
         * Create and register one neural network per subdomain.
         */
        for (
            int64_t subdomain_index = 0;
            subdomain_index < number_of_subdomains;
            ++subdomain_index
        ) {
            auto local_network = LocalNetwork(
                1,
                hidden_width,
                1
            );

            register_module(
                "local_network_"
                    + std::to_string(subdomain_index),
                local_network
            );

            local_networks.push_back(local_network);
        }

        const auto buffer_options = torch::TensorOptions()
            .dtype(torch::kFloat64);

        /*
         * Registering these as buffers means that they are included
         * when the model is transferred to a different device.
         */
        centres = register_buffer(
            "centres",
            torch::tensor(
                subdomain_centres,
                buffer_options
            ).reshape({1, number_of_subdomains})
        );

        half_widths = register_buffer(
            "half_widths",
            torch::tensor(
                subdomain_half_widths,
                buffer_options
            ).reshape({1, number_of_subdomains})
        );
    }

    torch::Tensor forward(
        const torch::Tensor& physical_coordinate
    ) {
        /*
         * Define the transformed coordinate
         *
         *     z(x) = x(2 - x).
         *
         * Its important properties are
         *
         *     z(0)  = 0,
         *     z'(1) = 0.
         */
        auto transformed_coordinate =
            physical_coordinate
            * (2.0 - physical_coordinate);

        /*
         * Evaluate the four-subdomain finite-basis approximation.
         */
        auto finite_basis_value =
            evaluate_finite_basis(
                transformed_coordinate
            );

        /*
         * Hard-constrained solution:
         *
         *     u(x) = 1 + x/C + z(x) F(z(x)).
         *
         * At x = 0:
         *
         *     u(0) = 1.
         *
         * At x = 1:
         *
         *     z'(1) = 0,
         *
         * so
         *
         *     u'(1) = 1/C,
         *
         * and therefore
         *
         *     C u'(1) = 1.
         */
        auto constrained_solution =
            1.0
            + physical_coordinate / material_c
            + transformed_coordinate
                * finite_basis_value;

        return constrained_solution;
    }

private:
    torch::Tensor evaluate_finite_basis(
        const torch::Tensor& transformed_coordinate
    ) {
        const int64_t batch_size =
            transformed_coordinate.size(0);

        /*
         * transformed_coordinate has shape [N,1].
         *
         * expanded_coordinate has shape [N,4].
         */
        auto expanded_coordinate =
            transformed_coordinate.expand(
                {
                    batch_size,
                    number_of_subdomains
                }
            );

        /*
         * Normalised distance from every point to each
         * subdomain centre:
         *
         *     d_i = (z - c_i)/h_i.
         */
        auto normalised_distance =
            (
                expanded_coordinate - centres
            ) / half_widths;

        auto absolute_distance =
            torch::abs(normalised_distance);

        /*
         * Compact cosine-squared windows:
         *
         *                 cos^2(pi d_i/2), if |d_i| < 1
         *     w_i(z) =
         *                 0,               otherwise
         *
         * The cosine-squared window and its first derivative
         * are zero at the boundary of its support.
         *
         * This is sufficient for the energy formulation because
         * the energy requires only the first spatial derivative.
         */
        auto raw_windows = torch::where(
            absolute_distance < 1.0,
            torch::cos(
                0.5 * pi * normalised_distance
            ).pow(2),
            torch::zeros_like(normalised_distance)
        );

        /*
         * Normalise the windows so that
         *
         *     sum_i phi_i(z) = 1.
         */
        auto window_sum = raw_windows.sum(
            1,
            true
        );

        auto normalised_windows =
            raw_windows
            / window_sum.clamp_min(1.0e-14);

        auto assembled_value =
            torch::zeros_like(
                transformed_coordinate
            );

        for (
            int64_t subdomain_index = 0;
            subdomain_index < number_of_subdomains;
            ++subdomain_index
        ) {
            auto centre_i = centres.index(
                {
                    0,
                    subdomain_index
                }
            );

            auto half_width_i = half_widths.index(
                {
                    0,
                    subdomain_index
                }
            );

            /*
             * Map the transformed global coordinate to the
             * local coordinate of subdomain i:
             *
             *     xi_i = (z - c_i)/h_i.
             */
            auto local_coordinate =
                (
                    transformed_coordinate - centre_i
                ) / half_width_i;

            auto local_value =
                local_networks
                    .at(subdomain_index)
                    ->forward(local_coordinate);

            /*
             * Extract window i while retaining the [N,1] shape.
             */
            auto window_i =
                normalised_windows.index(
                    {
                        torch::indexing::Slice(),
                        torch::indexing::Slice(
                            subdomain_index,
                            subdomain_index + 1
                        )
                    }
                );

            assembled_value =
                assembled_value
                + window_i * local_value;
        }

        return assembled_value;
    }

    std::vector<LocalNetwork> local_networks;

    torch::Tensor centres;
    torch::Tensor half_widths;
};

TORCH_MODULE(Fbpinn);


// =============================================================================
// Source function
// =============================================================================

torch::Tensor source_function(
    const torch::Tensor& coordinate
) {
    /*
     *     f(x) = sin(2 pi k x)
     *
     * with k = 50.
     */
    return torch::sin(
        2.0
        * pi
        * wave_number_k
        * coordinate
    );
}


// =============================================================================
// Exact analytical solution
// =============================================================================

torch::Tensor exact_solution(
    const torch::Tensor& coordinate
) {
    const double angular_frequency =
        2.0 * pi * wave_number_k;

    /*
     * Strong problem:
     *
     *     -C u'' = sin(omega x),
     *
     * with
     *
     *     u(0)    = 1,
     *     C u'(1) = 1.
     *
     * The analytical solution is
     *
     *     u(x) =
     *         1
     *         + A x
     *         + sin(omega x)/(C omega^2),
     *
     * where
     *
     *     A = [1 - cos(omega)/omega]/C.
     */
    const double linear_coefficient =
        (
            prescribed_traction
            - std::cos(angular_frequency)
                / angular_frequency
        ) / material_c;

    return 1.0
        + linear_coefficient * coordinate
        + torch::sin(
            angular_frequency * coordinate
        ) / (
            material_c
            * angular_frequency
            * angular_frequency
        );
}


// =============================================================================
// Spatial derivative using LibTorch autograd
// =============================================================================

torch::Tensor spatial_derivative(
    const torch::Tensor& solution,
    const torch::Tensor& coordinate,
    bool create_graph
) {
    /*
     * Explicit vectors are used here instead of initializer lists.
     * This is more compatible across different LibTorch versions.
     */
    std::vector<torch::Tensor> outputs{
        solution
    };

    std::vector<torch::Tensor> inputs{
        coordinate
    };

    std::vector<torch::Tensor> gradient_outputs{
        torch::ones_like(solution)
    };

    /*
     * The fourth argument is retain_graph.
     * The fifth argument is create_graph.
     * The sixth argument is allow_unused.
     *
     * During training, create_graph must be true because the energy
     * depends on du/dx and must subsequently be differentiated with
     * respect to the neural-network parameters.
     */
    auto gradients = torch::autograd::grad(
        outputs,
        inputs,
        gradient_outputs,
        create_graph,
        create_graph,
        false
    );

    return gradients.at(0);
}


// =============================================================================
// Total potential-energy functional
// =============================================================================

torch::Tensor potential_energy(
    Fbpinn& model,
    const torch::Tensor& quadrature_coordinates
) {
    /*
     * Create a leaf coordinate tensor that records gradients.
     */
    auto coordinate =
        quadrature_coordinates
            .detach()
            .clone()
            .set_requires_grad(true);

    /*
     * Evaluate the hard-constrained FBPINN solution.
     */
    auto solution =
        model->forward(coordinate);

    /*
     * Calculate du/dx using LibTorch autograd.
     */
    auto solution_gradient =
        spatial_derivative(
            solution,
            coordinate,
            true
        );

    auto forcing =
        source_function(coordinate);

    /*
     * Potential-energy density:
     *
     *     psi(x)
     *       = 0.5 C [u'(x)]^2
     *         - f(x)u(x).
     */
    auto energy_density =
        0.5
        * material_c
        * solution_gradient.pow(2)
        - forcing * solution;

    /*
     * Midpoint quadrature on a unit-length domain:
     *
     *     integral_0^1 psi dx
     *
     * is approximated by
     *
     *     mean(psi).
     */
    auto domain_potential =
        energy_density.mean();

    /*
     * Evaluate u(1) for the external boundary work.
     *
     * The right-coordinate tensor does not need input gradients here
     * because u(1) itself, rather than u'(1), appears in the energy.
     * Gradient propagation to the network parameters is still retained.
     */
    auto right_coordinate = torch::ones(
        {1, 1},
        quadrature_coordinates.options()
    );

    auto right_solution =
        model->forward(right_coordinate);

    /*
     * Boundary potential:
     *
     *     -t u(1),
     *
     * where t = 1.
     */
    auto boundary_potential =
        -prescribed_traction
        * right_solution.mean();

    /*
     * Total potential:
     *
     *     Pi = integral psi dx - t u(1).
     */
    return domain_potential
        + boundary_potential;
}


// =============================================================================
// Boundary-condition verification
// =============================================================================

void verify_boundary_conditions(
    Fbpinn& model,
    const torch::TensorOptions& tensor_options
) {
    double left_value = 0.0;

    /*
     * Verify u(0) = 1.
     */
    {
        torch::NoGradGuard no_grad;

        auto left_coordinate = torch::zeros(
            {1, 1},
            tensor_options
        );

        auto left_solution =
            model->forward(left_coordinate);

        left_value =
            left_solution.item<double>();
    }

    /*
     * Verify C u'(1) = 1.
     *
     * Input gradient tracking must be enabled here.
     */
    auto right_coordinate = torch::ones(
        {1, 1},
        tensor_options
    ).set_requires_grad(true);

    auto right_solution =
        model->forward(right_coordinate);

    auto right_gradient =
        spatial_derivative(
            right_solution,
            right_coordinate,
            false
        );

    const double right_flux =
        material_c
        * right_gradient.item<double>();

    std::cout
        << "\nBoundary-condition verification\n"
        << "-------------------------------\n"
        << "u(0)       = "
        << left_value
        << "\n"
        << "C u'(1)    = "
        << right_flux
        << "\n";
}


// =============================================================================
// Solution-error evaluation
// =============================================================================

void evaluate_model(
    Fbpinn& model,
    const torch::TensorOptions& tensor_options
) {
    torch::NoGradGuard no_grad;

    auto evaluation_coordinate =
        torch::linspace(
            0.0,
            1.0,
            1001,
            tensor_options
        ).reshape({-1, 1});

    auto predicted_solution =
        model->forward(
            evaluation_coordinate
        );

    auto analytical_solution =
        exact_solution(
            evaluation_coordinate
        );

    auto error =
        predicted_solution
        - analytical_solution;

    auto relative_l2_error =
        torch::sqrt(
            torch::mean(error.pow(2))
        ) / torch::sqrt(
            torch::mean(
                analytical_solution.pow(2)
            )
        );

    auto maximum_absolute_error =
        torch::max(
            torch::abs(error)
        );

    /*
     * Also evaluate the high-frequency correction after removing
     * the dominant affine part.
     *
     * This is more informative because the oscillatory displacement
     * has an amplitude of approximately 1e-6.
     */
    const double angular_frequency =
        2.0 * pi * wave_number_k;

    const double linear_coefficient =
        (
            prescribed_traction
            - std::cos(angular_frequency)
                / angular_frequency
        ) / material_c;

    auto affine_solution =
        1.0
        + linear_coefficient
            * evaluation_coordinate;

    auto predicted_oscillatory_part =
        predicted_solution
        - affine_solution;

    auto exact_oscillatory_part =
        analytical_solution
        - affine_solution;

    auto oscillatory_error =
        predicted_oscillatory_part
        - exact_oscillatory_part;

    auto oscillatory_absolute_l2_error =
        torch::sqrt(
            torch::mean(
                oscillatory_error.pow(2)
            )
        );

    auto exact_oscillatory_l2_norm =
        torch::sqrt(
            torch::mean(
                exact_oscillatory_part.pow(2)
            )
        );

    auto oscillatory_relative_l2_error =
        oscillatory_absolute_l2_error
        / exact_oscillatory_l2_norm.clamp_min(
            1.0e-16
        );

    std::cout
        << "\nSolution errors\n"
        << "---------------\n"
        << "Full solution relative L2 error:       "
        << relative_l2_error.item<double>()
        << "\n"
        << "Full solution maximum absolute error:  "
        << maximum_absolute_error.item<double>()
        << "\n"
        << "Oscillatory relative L2 error:         "
        << oscillatory_relative_l2_error.item<double>()
        << "\n"
        << "Oscillatory absolute L2 error:         "
        << oscillatory_absolute_l2_error.item<double>()
        << "\n";
}


// =============================================================================
// Main program
// =============================================================================

int main() {
    /*
     * Fixed seed for repeatability.
     */
    torch::manual_seed(1234);

    std::cout
        << std::scientific
        << std::setprecision(10);

    /*
     * Double precision is used because:
     *
     * 1. the forcing contains 50 cycles;
     * 2. the energy contains spatial derivatives;
     * 3. the oscillatory solution amplitude is approximately 1e-6.
     */
    const auto tensor_options =
        torch::TensorOptions()
            .dtype(torch::kFloat64)
            .device(torch::kCPU);

    /*
     * Create the FBPINN.
     */
    auto model = Fbpinn();

    /*
     * The model parameters and registered buffers must use the
     * same precision and device as the input tensors.
     */
    model->to(torch::kFloat64);
    model->to(torch::kCPU);
    model->train();

    /*
     * Fixed midpoint quadrature:
     *
     *     x_j = (j + 1/2)/N,
     *
     * where
     *
     *     j = 0, ..., N-1.
     */
    auto quadrature_indices =
        torch::arange(
            number_of_quadrature_points,
            tensor_options
        ).reshape({-1, 1});

    auto quadrature_coordinates =
        (
            quadrature_indices + 0.5
        ) / static_cast<double>(
            number_of_quadrature_points
        );

    /*
     * Adam optimiser.
     *
     * Unlike L-BFGS, Adam does not require a closure.
     */
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(
            adam_learning_rate
        )
    );

    std::cout
        << "Training four-subdomain FBPINN with Adam\n"
        << "-----------------------------------------\n"
        << "Material C:               "
        << material_c
        << "\n"
        << "Wave number k:            "
        << wave_number_k
        << "\n"
        << "Quadrature points:        "
        << number_of_quadrature_points
        << "\n"
        << "Number of subdomains:     "
        << number_of_subdomains
        << "\n"
        << "Hidden width:             "
        << hidden_width
        << "\n"
        << "Adam learning rate:       "
        << adam_learning_rate
        << "\n"
        << "Number of Adam epochs:    "
        << number_of_adam_epochs
        << "\n\n";

    /*
     * Adam training loop.
     */
    for (
        int64_t epoch = 1;
        epoch <= number_of_adam_epochs;
        ++epoch
    ) {
        /*
         * Clear gradients accumulated during the previous epoch.
         */
        optimizer.zero_grad();

        /*
         * Calculate the discretised total potential energy.
         */
        auto energy =
            potential_energy(
                model,
                quadrature_coordinates
            );

        /*
         * Compute the gradient of the energy with respect to all
         * registered local-network parameters.
         */
        energy.backward();

        /*
         * Optional gradient clipping can improve robustness if a
         * particular initialisation produces unusually large gradients.
         */
        torch::nn::utils::clip_grad_norm_(
            model->parameters(),
            100.0
        );

        /*
         * Update all local-network parameters.
         */
        optimizer.step();

        if (
            epoch == 1
            || epoch % print_frequency == 0
            || epoch == number_of_adam_epochs
        ) {
            const double energy_value =
                energy.detach().item<double>();

            std::cout
                << "Epoch "
                << std::setw(6)
                << epoch
                << " | potential = "
                << energy_value
                << "\n";
        }
    }

    /*
     * Evaluation mode does not disable autograd. It only changes the
     * behaviour of modules such as dropout and batch normalisation,
     * neither of which is used here.
     */
    model->eval();

    verify_boundary_conditions(
        model,
        tensor_options
    );

    evaluate_model(
        model,
        tensor_options
    );

    /*
     * Save the trained parameters.
     */
    torch::save(
        model,
        "fbpinn_poisson_k50_adam.pt"
    );

    std::cout
        << "\nSaved model to "
        << "fbpinn_poisson_k50_adam.pt\n";

    return 0;
}
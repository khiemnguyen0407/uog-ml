import math

import matplotlib.pyplot as plt
import torch
from torch import nn


# =============================================================================
# Configuration
# =============================================================================

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

pi = math.pi
wave_number_k = 50.0

number_of_subdomains = 4
hidden_width = 16
number_of_quadrature_points = 64
print_frequency = 50

number_of_lbfgs_steps = 500
lbfgs_iterations_per_step = 20

# Boundary conditions:
#
#     u(0) = 0
#     u'(1) = 10
#
left_displacement = 0.0
right_derivative = 10.0

# At x = 1:
#
#     C(1) = 2.5 + sin(100*pi) = 2.5
#
# Therefore, the equivalent natural traction is
#
#     C(1)u'(1) = 2.5*10 = 25.
#
right_traction = 25.0


# =============================================================================
# Local subdomain neural network
# =============================================================================

class LocalNetwork(nn.Module):
    def __init__(self, hidden_dimension):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(1, hidden_dimension),
                nn.Linear(hidden_dimension, hidden_dimension),
                nn.Linear(hidden_dimension, 1),
            ]
        )

        self.initialise_parameters()

    def initialise_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, local_coordinate):
        value = local_coordinate

        for layer in self.layers[:-1]:
            value = torch.sin(layer(value))
            
        output = self.layers[-1](value)
        return output


# =============================================================================
# Four-subdomain FBPINN
# =============================================================================

class Fbpinn(nn.Module):
    def __init__(self):
        super().__init__()

        self.local_networks = nn.ModuleList(
            [
                LocalNetwork(hidden_width)
                for _ in range(number_of_subdomains)
            ]
        )

        # Four overlapping subdomains:
        #
        #     [-0.125, 0.375]
        #     [ 0.125, 0.625]
        #     [ 0.375, 0.875]
        #     [ 0.625, 1.125]
        #
        self.register_buffer(
            "subdomain_centres",
            torch.tensor(
                [
                    0.125,
                    0.375,
                    0.625,
                    0.875,
                ]
            ).reshape(1, number_of_subdomains),
        )

        self.register_buffer(
            "subdomain_half_widths",
            torch.tensor(
                [
                    0.25,
                    0.25,
                    0.25,
                    0.25,
                ]
            ).reshape(1, number_of_subdomains),
        )

    def evaluate_windows(self, transformed_coordinate):
        """
        Evaluate compact cosine-squared window functions.

        For subdomain i:

            d_i = (z - centre_i)/half_width_i

        and

            w_i = cos^2(pi*d_i/2),  if |d_i| < 1,
                = 0,                otherwise.

        The raw windows are normalised to form a partition of unity.
        """

        normalised_distance = (
            transformed_coordinate
            - self.subdomain_centres
        ) / self.subdomain_half_widths

        raw_windows = torch.where(
            torch.abs(normalised_distance) < 1.0,
            torch.cos(
                0.5 * pi * normalised_distance
            ).pow(2),
            torch.zeros_like(normalised_distance),
        )

        window_sum = raw_windows.sum(
            dim=1,
            keepdim=True,
        )

        normalised_windows = (
            raw_windows
            / window_sum.clamp_min(1.0e-14)
        )

        return normalised_windows

    def evaluate_finite_basis(
        self,
        transformed_coordinate,
    ):
        """
        Construct the finite-basis approximation

            F(z) = sum_i phi_i(z) v_i(xi_i),

        where

            xi_i = (z - centre_i)/half_width_i.
        """

        normalised_windows = self.evaluate_windows(
            transformed_coordinate
        )

        assembled_value = torch.zeros_like(
            transformed_coordinate
        )

        for (subdomain_index, local_network) in enumerate(self.local_networks):
            centre_i = self.subdomain_centres[:, subdomain_index:subdomain_index + 1]

            half_width_i = self.subdomain_half_widths[
                :,
                subdomain_index:subdomain_index + 1,
            ]

            local_coordinate = (
                transformed_coordinate - centre_i
            ) / half_width_i

            local_value = local_network(
                local_coordinate
            )

            window_i = normalised_windows[
                :,
                subdomain_index:subdomain_index + 1,
            ]

            assembled_value = (
                assembled_value
                + window_i * local_value
            )

        return assembled_value

    def forward(self, physical_coordinate):
        """
        Hard-constrained FBPINN trial function.

        Define

            z(x) = x(2 - x).

        The relevant properties are

            z(0) = 0,
            z'(1) = 0.

        The trial function is

            u_theta(x)
                = 10x + z(x) F_theta(z(x)).

        Consequently,

            u_theta(0) = 0

        and

            u_theta'(1) = 10

        for every value of the network parameters.
        """

        transformed_coordinate = (
            physical_coordinate
            * (2.0 - physical_coordinate)
        )

        finite_basis_value = (
            self.evaluate_finite_basis(
                transformed_coordinate
            )
        )

        constrained_solution = (
            right_derivative * physical_coordinate
            + transformed_coordinate
            * finite_basis_value
        )

        return constrained_solution


# =============================================================================
# Material coefficient
# =============================================================================

def material_coefficient(coordinate):
    """
    Spatially varying coefficient:

        C(x) = 5/2 + sin(100*pi*x)

             = 2.5 + sin(2*pi*k*x),

    where k = 50.
    """

    return (
        2.5
        + torch.sin(
            2.0
            * pi
            * wave_number_k
            * coordinate
        )
    )


# =============================================================================
# Automatic differentiation
# =============================================================================

def spatial_derivative(
    solution,
    coordinate,
    create_graph=False,
):
    """
    Calculate du/dx using PyTorch automatic differentiation.
    """

    return torch.autograd.grad(
        outputs=solution,
        inputs=coordinate,
        grad_outputs=torch.ones_like(solution),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]


# =============================================================================
# Potential-energy functional
# =============================================================================

def calculate_potential_energy(
    model,
    quadrature_coordinates,
):
    """
    Calculate the total potential energy

        Pi[u]
            = integral_0^1
                0.5*C(x)*[u'(x)]^2 dx
              - t*u(1),

    where

        t = C(1)u'(1) = 25.

    The external body force is zero.
    """

    coordinate = (
        quadrature_coordinates
        .detach()
        .clone()
        .requires_grad_(True)
    )

    solution = model(coordinate)

    solution_gradient = spatial_derivative(
        solution,
        coordinate,
        create_graph=True,
    )

    coefficient = material_coefficient(
        coordinate
    )

    energy_density = (
        0.5
        * coefficient
        * solution_gradient.pow(2)
    )

    # Composite midpoint quadrature:
    #
    #     integral_0^1 psi(x) dx
    #
    # is approximated by mean(psi) because the
    # length of the domain is one.
    domain_potential = energy_density.mean()

    right_coordinate = torch.ones(
        (1, 1),
        dtype=torch.get_default_dtype(),
        device=device,
    )

    right_solution = model(
        right_coordinate
    )

    boundary_potential = (
        -right_traction
        * right_solution.squeeze()
    )

    total_potential = (
        domain_potential
        + boundary_potential
    )

    return total_potential


# =============================================================================
# Exact reference solution
# =============================================================================

def exact_solution(evaluation_coordinate):
    """
    The governing equation is

        -d/dx[C(x)u'(x)] = 0.

    Therefore,

        C(x)u'(x) = constant.

    Using C(1)u'(1) = 25 gives

        u'(x) = 25/C(x).

    Since u(0) = 0,

        u(x) = 25*integral_0^x 1/C(s) ds.

    This routine calculates the integral numerically using the
    cumulative trapezoidal rule on the evaluation grid.
    """

    coefficient = material_coefficient(
        evaluation_coordinate
    )

    inverse_coefficient = (
        1.0 / coefficient
    )

    x_values = evaluation_coordinate.reshape(-1)
    integrand_values = inverse_coefficient.reshape(-1)

    coordinate_increments = (
        x_values[1:]
        - x_values[:-1]
    )

    trapezoidal_increments = (
        0.5
        * (
            integrand_values[1:]
            + integrand_values[:-1]
        )
        * coordinate_increments
    )

    cumulative_integral = torch.cat(
        [
            torch.zeros(
                1,
                dtype=evaluation_coordinate.dtype,
                device=evaluation_coordinate.device,
            ),
            torch.cumsum(
                trapezoidal_increments,
                dim=0,
            ),
        ]
    )

    return (
        right_traction
        * cumulative_integral.reshape(-1, 1)
    )


# =============================================================================
# Boundary-condition verification
# =============================================================================

def verify_boundary_conditions(model):
    left_coordinate = torch.zeros(
        (1, 1),
        dtype=torch.get_default_dtype(),
        device=device,
    )

    with torch.no_grad():
        left_solution = model(
            left_coordinate
        )

    right_coordinate = torch.ones(
        (1, 1),
        dtype=torch.get_default_dtype(),
        device=device,
        requires_grad=True,
    )

    right_solution = model(
        right_coordinate
    )

    right_gradient = spatial_derivative(
        right_solution,
        right_coordinate,
        create_graph=False,
    )

    coefficient_at_right = material_coefficient(
        right_coordinate
    )

    right_flux = (
        coefficient_at_right
        * right_gradient
    )

    print()
    print("Boundary-condition verification")
    print("--------------------------------")
    print(
        f"u(0)       = "
        f"{left_solution.item():.12e}"
    )
    print(
        f"u'(1)      = "
        f"{right_gradient.item():.12e}"
    )
    print(
        f"C(1)u'(1) = "
        f"{right_flux.item():.12e}"
    )


# =============================================================================
# Solution evaluation and plotting
# =============================================================================

def evaluate_and_plot(
    model,
    energy_history,
):
    evaluation_coordinate = torch.linspace(
        0.0,
        1.0,
        20001,
        dtype=torch.get_default_dtype(),
        device=device,
    ).reshape(-1, 1)

    with torch.no_grad():
        predicted_solution = model(
            evaluation_coordinate
        )

        reference_solution = exact_solution(
            evaluation_coordinate
        )

    error = (
        predicted_solution
        - reference_solution
    )

    relative_l2_error = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(
            reference_solution
        )
    )

    maximum_absolute_error = torch.max(
        torch.abs(error)
    )

    print()
    print("Solution errors")
    print("---------------")
    print(
        f"Relative L2 error:       "
        f"{relative_l2_error.item():.12e}"
    )
    print(
        f"Maximum absolute error: "
        f"{maximum_absolute_error.item():.12e}"
    )

    x_values = (
        evaluation_coordinate
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    predicted_values = (
        predicted_solution
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    reference_values = (
        reference_solution
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
    )

    axes[0].plot(
        x_values,
        reference_values,
        label="Exact reference solution",
        linewidth=2.0,
    )

    axes[0].plot(
        x_values,
        predicted_values,
        "--",
        label="FBPINN solution",
        linewidth=1.5,
    )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].set_title(
        "Variable-coefficient 1D Poisson problem"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        range(
            1,
            len(energy_history) + 1,
        ),
        energy_history,
        linewidth=1.5,
    )

    axes[1].set_xlabel(
        "L-BFGS outer step"
    )
    axes[1].set_ylabel(
        "Potential energy"
    )
    axes[1].set_title(
        "Optimisation history"
    )
    axes[1].grid(True)

    figure.tight_layout()

    figure_filename = (
        "fbpinn_variable_coefficient_results.png"
    )

    figure.savefig(
        figure_filename,
        dpi=300,
        bbox_inches="tight",
    )

    print()
    print(
        f"Saved plot to {figure_filename}"
    )

    plt.show()


# =============================================================================
# Main training routine
# =============================================================================

print(
    "Training variable-coefficient "
    "four-subdomain FBPINN"
)
print(
    "-----------------------------------------------"
)
print(f"Device:                     {device}")
print(f"Wave number k:              {wave_number_k}")
print(
    f"Number of subdomains:       "
    f"{number_of_subdomains}"
)
print(
    f"Hidden width:               "
    f"{hidden_width}"
)
print(
    f"Quadrature points:          "
    f"{number_of_quadrature_points}"
)
print(
    f"Print frequency:            "
    f"{print_frequency}"
)
print(
    f"Right-end derivative:       "
    f"{right_derivative}"
)
print(
    f"Equivalent right traction: "
    f"{right_traction}"
)
print()

model = Fbpinn().to(device)

# Fixed composite midpoint quadrature:
#
#     x_j = (j + 0.5)/N
#
quadrature_indices = torch.arange(
    number_of_quadrature_points,
    dtype=torch.get_default_dtype(),
    device=device,
).reshape(-1, 1)

quadrature_coordinates = (
    quadrature_indices + 0.5
) / number_of_quadrature_points

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=lbfgs_iterations_per_step,
    max_eval=25,
    tolerance_grad=1.0e-10,
    tolerance_change=1.0e-12,
    history_size=100,
    line_search_fn="strong_wolfe",
)

global closure_evaluations
closure_evaluations = 0
energy_history = []

for outer_step in range(
    1,
    number_of_lbfgs_steps + 1,
):
    def closure():
        global closure_evaluations

        optimizer.zero_grad(
            set_to_none=True
        )

        energy = calculate_potential_energy(
            model,
            quadrature_coordinates,
        )

        energy.backward()

        closure_evaluations += 1

        return energy

    optimizer.step(closure)

    current_energy = (
        calculate_potential_energy(
            model,
            quadrature_coordinates,
        )
        .detach()
        .item()
    )

    energy_history.append(
        current_energy
    )

    if (
        outer_step == 1
        or outer_step % print_frequency == 0
        or outer_step == number_of_lbfgs_steps
    ):
        print(
            f"L-BFGS step {outer_step:5d}"
            f" | potential = "
            f"{current_energy:.12e}"
            f" | closure evaluations = "
            f"{closure_evaluations}"
        )

model.eval()

verify_boundary_conditions(model)

evaluate_and_plot(
    model,
    energy_history,
)

model_filename = (
    "fbpinn_variable_coefficient_lbfgs.pt"
)

torch.save(
    model.state_dict(),
    model_filename,
)

print(
    f"Saved model parameters to "
    f"{model_filename}"
)
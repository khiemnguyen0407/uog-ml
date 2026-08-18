import math
import matplotlib.pyplot as plt
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pi = math.pi
wave_number_k = 50.0
number_of_subdomains = 50
hidden_width = 16
subdomain_overlap_factor = 1.0
number_of_quadrature_points = number_of_subdomains * 20
print_frequency = 50
number_of_adam_epochs = 5000
adam_learning_rate = 1.0e-3
right_derivative = 10.0
right_traction = 25.0


class LocalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1, hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.Linear(hidden_width, 1),
        ])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, local_coordinate):
        value = local_coordinate
        for layer in self.layers[:-1]:
            value = torch.sin(layer(value))
        return self.layers[-1](value)


class Fbpinn(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_networks = nn.ModuleList([
            LocalNetwork() for _ in range(number_of_subdomains)
        ])
        # Uniformly distribute the subdomains over [0, 1].
        # For N subdomains, the centre spacing is 1/N.
        # overlap_factor = 1.0 gives half_width = 1/N, so adjacent
        # supports overlap over half of their total width.
        subdomain_spacing = 1.0 / number_of_subdomains

        subdomain_centres = (
            torch.arange(number_of_subdomains, dtype=torch.get_default_dtype())
            + 0.5
        ) * subdomain_spacing

        subdomain_half_widths = torch.full(
            (number_of_subdomains,),
            subdomain_overlap_factor * subdomain_spacing,
            dtype=torch.get_default_dtype(),
        )

        self.register_buffer(
            "subdomain_centres",
            subdomain_centres.reshape(1, number_of_subdomains),
        )
        self.register_buffer(
            "subdomain_half_widths",
            subdomain_half_widths.reshape(1, number_of_subdomains),
        )

    def evaluate_finite_basis(self, transformed_coordinate):
        distance = (
            transformed_coordinate - self.subdomain_centres
        ) / self.subdomain_half_widths
        raw_windows = torch.where(
            torch.abs(distance) < 1.0,
            torch.cos(0.5 * pi * distance).pow(2),
            torch.zeros_like(distance),
        )
        windows = raw_windows / raw_windows.sum(dim=1, keepdim=True).clamp_min(1e-14)
        output = torch.zeros_like(transformed_coordinate)

        for index, local_network in enumerate(self.local_networks):
            centre = self.subdomain_centres[:, index:index + 1]
            half_width = self.subdomain_half_widths[:, index:index + 1]
            local_coordinate = (transformed_coordinate - centre) / half_width
            output = output + windows[:, index:index + 1] * local_network(local_coordinate)
        return output

    def forward(self, physical_coordinate):
        # Evaluate the local networks in the physical x-coordinate.
        # This keeps uniformly spaced subdomains aligned with the periodic
        # coefficient C(x), instead of compressing them near x = 1.
        finite_basis_value = self.evaluate_finite_basis(physical_coordinate)

        # D(x) = x(2-x) satisfies D(0)=0 and D'(1)=0, while D(1)=1.
        # Therefore u(0)=0 and u'(1)=10 are imposed exactly without
        # incorrectly fixing the value u(1).
        distance_function = physical_coordinate * (2.0 - physical_coordinate)
        return (
            right_derivative * physical_coordinate
            + distance_function * finite_basis_value
        )


def material_coefficient(coordinate):
    return 2.5 + torch.sin(2.0 * pi * wave_number_k * coordinate)


def spatial_derivative(solution, coordinate, create_graph=False):
    return torch.autograd.grad(
        solution,
        coordinate,
        grad_outputs=torch.ones_like(solution),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]


def potential_energy(model, quadrature_coordinates):
    coordinate = quadrature_coordinates.detach().clone().requires_grad_(True)
    solution = model(coordinate)
    solution_gradient = spatial_derivative(solution, coordinate, create_graph=True)
    energy_density = 0.5 * material_coefficient(coordinate) * solution_gradient.pow(2)
    right_coordinate = torch.ones((1, 1), device=device)
    return energy_density.mean() - right_traction * model(right_coordinate).squeeze()


def exact_solution(coordinate):
    inverse_coefficient = 1.0 / material_coefficient(coordinate)
    x = coordinate.reshape(-1)
    y = inverse_coefficient.reshape(-1)
    increments = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
    integral = torch.cat([
        torch.zeros(1, dtype=coordinate.dtype, device=coordinate.device),
        torch.cumsum(increments, dim=0),
    ])
    return right_traction * integral.reshape(-1, 1)


def verify_boundary_conditions(model):
    with torch.no_grad():
        left_value = model(torch.zeros((1, 1), device=device)).item()
    right_coordinate = torch.ones((1, 1), device=device, requires_grad=True)
    right_gradient = spatial_derivative(model(right_coordinate), right_coordinate).item()
    print("\nBoundary conditions")
    print(f"u(0)  = {left_value:.12e}")
    print(f"u'(1) = {right_gradient:.12e}")


def plot_results(model, energy_history):
    coordinate = torch.linspace(0.0, 1.0, 20001, device=device).reshape(-1, 1)
    with torch.no_grad():
        predicted = model(coordinate)
        exact = exact_solution(coordinate)
    error = predicted - exact
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(exact)
    print(f"Relative L2 error = {relative_l2.item():.12e}")

    x = coordinate.cpu().numpy().reshape(-1)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(x, exact.cpu().numpy(), label="Exact solution", linewidth=2)
    axes[0].plot(x, predicted.cpu().numpy(), "--", label="FBPINN solution")
    axes[0].set(xlabel="x", ylabel="u(x)", title="Exact versus FBPINN solution")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(range(1, len(energy_history) + 1), energy_history)
    axes[1].set(xlabel="Adam epoch", ylabel="Potential energy", title="Training history")
    axes[1].grid(True)
    figure.tight_layout()
    figure.savefig("fbpinn_variable_coefficient_adam_results.png", dpi=300)
    plt.show()


def main():
    model = Fbpinn().to(device)
    indices = torch.arange(number_of_quadrature_points, device=device).reshape(-1, 1)
    quadrature_coordinates = (indices + 0.5) / number_of_quadrature_points
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_learning_rate)
    energy_history = []

    for epoch in range(1, number_of_adam_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        energy = potential_energy(model, quadrature_coordinates)
        energy.backward()
        optimizer.step()
        energy_history.append(energy.detach().item())

        if epoch == 1 or epoch % print_frequency == 0:
            print(f"Epoch {epoch:6d} | potential = {energy.item():.12e}")

    model.eval()
    verify_boundary_conditions(model)
    plot_results(model, energy_history)
    torch.save(model.state_dict(), "fbpinn_variable_coefficient_adam.pt")


if __name__ == "__main__":
    main()

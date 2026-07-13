#include <torch/torch.h>
#include <iostream>
#include <utility>

struct BayesianLinearRegression {
    torch::Tensor m0;      // prior mean, shape [p, 1]
    torch::Tensor S0;      // prior covariance, shape [p, p]
    torch::Tensor mN;      // posterior mean, shape [p, 1]
    torch::Tensor SN;      // posterior covariance, shape [p, p]
    double sigma2;         // known noise variance

    BayesianLinearRegression(const torch::Tensor& prior_mean,
                             const torch::Tensor& prior_cov,
                             double noise_var)
        : m0(prior_mean.clone()),
          S0(prior_cov.clone()),
          sigma2(noise_var) {}

    void fit(const torch::Tensor& X, const torch::Tensor& y) {
        // X: [N, p]
        // y: [N, 1]

        torch::Tensor S0_inv = torch::linalg_inv(S0);
        torch::Tensor XtX = X.transpose(0, 1).matmul(X);
        torch::Tensor Xty = X.transpose(0, 1).matmul(y);

        torch::Tensor SN_inv = S0_inv + (1.0 / sigma2) * XtX;
        SN = torch::linalg_inv(SN_inv);

        torch::Tensor rhs = S0_inv.matmul(m0) + (1.0 / sigma2) * Xty;
        mN = SN.matmul(rhs);
    }

    std::pair<torch::Tensor, torch::Tensor> predict(const torch::Tensor& Xs) const {
        // Xs: [M, p]
        // returns mean [M,1], variance [M,1]
        torch::Tensor mean = Xs.matmul(mN);  // [M,1]

        // diag(Xs SN Xs^T) + sigma2
        torch::Tensor tmp = Xs.matmul(SN);                 // [M,p]
        torch::Tensor var = (tmp * Xs).sum(1, true) + sigma2; // [M,1]

        return {mean, var};
    }

    torch::Tensor sample_weights(int64_t n_samples) const {
        // samples beta ~ N(mN, SN), returns [n_samples, p]
        int64_t p = mN.size(0);
        torch::Tensor L = torch::linalg_cholesky(SN);      // [p,p]
        torch::Tensor eps = torch::randn({n_samples, p}, mN.options());
        torch::Tensor samples = eps.matmul(L.transpose(0,1)) + mN.transpose(0,1);
        return samples;
    }
};

int main() {
    torch::manual_seed(0);

    // Example data: y = 2 + 3x + noise
    int64_t N = 50;
    torch::Tensor x = torch::linspace(-1.0, 1.0, N).reshape({N, 1});
    torch::Tensor ones = torch::ones({N, 1});
    torch::Tensor X = torch::cat({ones, x}, 1);  // [N,2], intercept + slope

    torch::Tensor true_beta = torch::tensor({{2.0}, {3.0}});
    double sigma2 = 0.04; // noise variance = 0.2^2

    torch::Tensor y = X.matmul(true_beta) + 0.2 * torch::randn({N, 1});

    // Prior: beta ~ N(0, 10 I)
    int64_t p = X.size(1);
    torch::Tensor m0 = torch::zeros({p, 1});
    torch::Tensor S0 = 10.0 * torch::eye(p);

    BayesianLinearRegression blr(m0, S0, sigma2);
    blr.fit(X, y);

    std::cout << "Posterior mean mN:\n" << blr.mN << "\n";
    std::cout << "Posterior covariance SN:\n" << blr.SN << "\n";

    auto pred = blr.predict(X);
    torch::Tensor mean = pred.first;
    torch::Tensor var  = pred.second;

    std::cout << "First 5 predictive means:\n" << mean.slice(0, 0, 5) << "\n";
    std::cout << "First 5 predictive variances:\n" << var.slice(0, 0, 5) << "\n";

    torch::Tensor beta_samples = blr.sample_weights(5);
    std::cout << "Posterior beta samples:\n" << beta_samples << "\n";

    return 0;
}
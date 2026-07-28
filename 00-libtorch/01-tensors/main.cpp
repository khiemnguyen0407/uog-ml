
// #include "ATen/core/TensorBody.h"
// #include "ATen/ops/arange.h"
// #include "ATen/ops/rand.h"
// #include "ATen/ops/zeros.h"
// #include "torch/csrc/autograd/generated/variable_factories.h"
// #include "torch/types.h"
#include "ATen/TensorIndexing.h"
// #include "c10/core/TensorOptions.h"
#include <torch/torch.h>
#include <iostream>

int main()
{
    // Creation of tensor
    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0});
    torch::Tensor b = torch::zeros({2, 3});
    torch::Tensor c = torch::rand({2, 3});
    torch::Tensor d = torch::arange(0, 10, torch::kFloat32);    // dtype specified

    std::cout << "a = \n" << a << std::endl;
    std::cout << "b = \n" << b << std::endl;
    std::cout << "c = \n" << c << std::endl;
    // std::cout << "d = \n" << d << std::endl;

    // Tensor operations
    torch::Tensor x = torch::rand({3, 4});
    torch::Tensor y = torch::rand({3, 4});

    // Arithmetic
    torch::Tensor z1 = x + y;
    torch::Tensor z2 = x * y;   // element-wise 
    torch::Tensor z3 = x.matmul(y.t()); // matrix multiplication

    // in-place operation (trailing underscore)
    x.add_(y);

    // Reductions
    auto s = x.sum();
    auto m = x.mean();
    auto mx = x.max();      // returns tuple (values, indices)

    // Accessing scalar value
    double double_val = s.item<double>();
    float float_val = s.item<float>();

    std::cout << "double_val = " << double_val << std::endl;
    std::cout << "float_val =  " << float_val << std::endl;

    // Indexing and slicing

    auto row = x[0];    // first row
    auto slice = x.slice(0, 0, 2);  // Rows 0 to 1 (dim, start, end)
    auto sub = x.index({torch::indexing::Slice(0, 2), 
        torch::indexing::Slice(1, torch::indexing::None)});
    
    // Reshaping
    auto flat = x.view({-1});
    auto reshaped = x.reshape({2, 6});

    std::cout << "x[0] = " << row << std::endl;
    std::cout << "slice of x = " << slice << std::endl;

    std::cout << "sub = " << sub << std::endl;

    std::cout << "flat = " << flat << std::endl;
    std::cout << "reshape = " << reshaped << std::endl;

    return 0;
}
# uog-ml

>This repository collects all the teaching materials from me at University of Glasgow for two courses
>
> - **ENG4200**: *Introduction to Artificial Intelligence and Machine Learning 4*
> - **ENG5337**: *Advanced Artificial Intelligence and Machine Learning 5*

**Note**  

- I taught **ENG4200** in two academic years 2023-2024 and 2024-2025. Since the academic year 2025-2026, it have been taught by my colleagues. So the materials they provided can be different from mine. 
- Katy and I have been teaching **ENG5337** since the academic year 2024-2025.

The teaching materials are pretty much inspired by Andrew Ng's lectures 

- Coursera
- Stanford Lectures [Link](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

## Implementation in Python

| Directory                                 | What is it for                          |
| :-----------------------------------------| :---------------------------------------|
| `01-linear-regression-single-variable`    | Linear regression of single variable    |
| `02-linear-regression-multiple-variables` | Linear regression of multiple variables |
| `03-logistic-regression`                  | Logistic regression                     |
| `04-multilayer-perceptrons`               | Conventional feed forward neural networks for regression problems, including continuous regression and classification problems|
| `05-convolution-neural-networks`          | Convolution neural networks for computer vision problems |
| `06-recurrent-neural-networks`            | Recurrent neural networks for sequence prediction problems |

## Libtorch 
- Implementation of machine learning frameworks using `Libtorch` is under construction. Although PyTorch is implemented as Pythonic as possible, the core implementation is actually in C++. To run files in directory `cpp`, please follow the instruction by official website [cppdocs](https://docs.pytorch.org/cppdocs/index.html). First, ou need to [install C++ distributions of PyTorch](https://docs.pytorch.org/cppdocs/installing.html), which is `Libtorch`.
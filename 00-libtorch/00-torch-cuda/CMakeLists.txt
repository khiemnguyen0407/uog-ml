cmake_minimum_required(VERSION 3.28)


project(torch_test LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBTORCH "$ENV{HOME}/opt/libtorch" CACHE PATH "Path to libtorch")

add_executable(torch_test main.cpp)

target_include_directories(torch_test PRIVATE
    ${LIBTORCH}/include
    ${LIBTORCH}/include/torch/csrc/api/include)

target_link_directories(torch_test PRIVATE ${LIBTORCH}/lib)

target_link_libraries(torch_test PRIVATE
    -Wl,--no-as-needed
    torch torch_cuda torch_cpu c10 c10_cuda)

set_target_properties(torch_test PROPERTIES
    BUILD_RPATH "${LIBTORCH}/lib")
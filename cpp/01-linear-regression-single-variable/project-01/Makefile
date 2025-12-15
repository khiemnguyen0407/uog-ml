# --- Configuration Variables ---
# Makefile does not work at this point -- Use cmake instead please
#=================================================================


# 1. Define the absolute path to your LibTorch 12.8 installation
LIBTORCH_ROOT := /home/cafesua/libtorch/libtorch_cuda128

# 2. Define your source file, target executable, and C++ standard
TARGET := linear-regression-cuda
SOURCE := linear_regression.cpp

CXX_STD := -std=c++17

# --- Compiler and Linker Flags ---

# Include paths for LibTorch headers
INCLUDES := -I$(LIBTORCH_ROOT)/include \
            -I$(LIBTORCH_ROOT)/include/torch/csrc/api/include

# Library paths and runtime path (RPATH) for linking
LDFLAGS := -L$(LIBTORCH_ROOT)/lib \
           -Wl,-rpath,$(LIBTORCH_ROOT)/lib

# Define the LibTorch libraries needed for CUDA execution
# Include CPU core, Autograd, and Global dependencies for a complete build
LIBS := -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -ltorch_global_deps -lshm
# C++ ABI Flag (Crucial for PyTorch)
ABI_FLAG := -D_GLIBCXX_USE_CXX11_ABI=0

# --- Recipe to Build the Target ---

# Define the compiler command (using GNU g++)
CXX := g++

# Default target: builds the executable
$(TARGET): $(SOURCE)
	@echo "--- Compiling $(TARGET) with LibTorch CUDA 12.8 ---"
	$(CXX) $(CXX_STD) $(ABI_FLAG) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "--- Build successful! Run with: ./$(TARGET) ---"

# --- Utility Targets ---

# Clean target: removes the compiled executable
.PHONY: clean
clean:
	rm -f $(TARGET)
	@echo "Cleaned project directory."

# Run target: executes the compiled program
.PHONY: run
run: $(TARGET)
	@echo "--- Running $(TARGET) ---"
	./$(TARGET)
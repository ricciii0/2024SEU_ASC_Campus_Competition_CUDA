# Makefile
NVCC = nvcc
CXXFLAGS = -O2
TARGET = mnist_cuda

all: $(TARGET)

$(TARGET): mnist_cuda.cu
	$(NVCC) $(CXXFLAGS) -o $(TARGET) mnist_cuda.cu

clean:
	rm -f $(TARGET)

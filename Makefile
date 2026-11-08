# int8 GEMM on the GPU: build, test, benchmark, plot
#
#   make            build build/gemm_bench
#   make test       correctness tests on edge-case shapes (needs a GPU)
#   make bench      benchmark every kernel at 512..4096 -> results/
#   make plots      charts from results/ -> results/*.png
#   make clean
#
# ARCH defaults to the GPU in this machine; override e.g. ARCH=sm_75 (T4).

NVCC  ?= nvcc
ARCH  ?= native
FLAGS := -O3 -std=c++17 -lineinfo -arch=$(ARCH)
BUILD := build

.PHONY: all test bench plots clean

all: $(BUILD)/gemm_bench

$(BUILD)/gemm_bench: src/gemm_bench.cu src/common.cuh $(wildcard src/kernels/*.cuh) | $(BUILD)
	$(NVCC) $(FLAGS) -o $@ src/gemm_bench.cu -lcublas

test: $(BUILD)/gemm_bench
	./$(BUILD)/gemm_bench --test

bench: $(BUILD)/gemm_bench
	./$(BUILD)/gemm_bench --sizes 512,1024,2048,4096 --out results

plots:
	python3 scripts/plot_results.py results

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

NVCC := nvcc

TEMP_NVCC := $(shell which nvcc)
CUDA_HOME :=  $(shell echo $(TEMP_NVCC) | rev |  cut -d'/' -f3- | rev)

# internal flags
SM      	?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')
NVCCFLAGS   := -std=c++17 -O3 -gencode arch=compute_$(SM),code=sm_$(SM) --compiler-options="-O2 -pipe -Wall -g" --generate-line-info
LDFLAGS     := -L$(CUDA_HOME)/lib64 -lcuda -lnvidia-ml
NAME 		:= shmem-stream
PREFIX		:=

$(PREFIX)cuda-$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f cuda-$(NAME)


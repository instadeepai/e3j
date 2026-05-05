# Copyright (c) 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################
# RULES:
# - make test
# - make bin/test_scatter_add
# - make bin/test_tensor_product
# - ...
# - make e3j_ops (/!\ build with skbuild + cmake instead!)
#
# See also:
# [1]: https://makefiletutorial.com/
# [2]: https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html
# [3]: https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable
#
###########################################################

SHELL=/bin/bash

#### G++ ##################################################

# CXX => implicit C++ compilation rules [1]
CXX=g++
CXX_FLAGS=
CXX_LIBS=

#### NVCC ##################################################

# NVCC compiler options:
# -g -lineinfo will relate cuda source to SASS
NVCC=nvcc

# Compute capability of the local GPU, e.g. "89" for sm_89.
GPU_ARCH ?= $(shell __nvcc_device_query)

# Fall back to a default set of architectures when no GPU is detected.
ifeq ($(strip $(GPU_ARCH)),)
GPU_ARCH := 80 86
$(info No GPU detected, building for sm_$(GPU_ARCH))
else
$(info Building for sm_$(GPU_ARCH))
endif

GENCODE_FLAGS=$(foreach arch,$(GPU_ARCH),\
	--generate-code=arch=compute_$(arch),code=[compute_$(arch),sm_$(arch)])

NVCC_FLAGS=--threads 4 -Xcompiler -w -lineinfo -ldl\
	--expt-relaxed-constexpr -O3 -DNDEBUG -Xcompiler -O3\
	$(GENCODE_FLAGS)\
	-Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden
NVCC_LIBS=

# CUDA directories:
CUDA_ROOT_DIR=/usr/local/cuda
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

##### Python bindings #####################################

PYBIND_INC=$(shell python3 -c 'import pybind11; print(pybind11.get_include())')
PYTHON_EXE=$(shell python3 -c 'import sys; print(sys.executable)')

# Was necessary to find <Python.h> locally too:
# see SO#21530577
# => apt install python-dev | libpython3.10-dev
PYTHON_INC=$(shell ${PYTHON_EXE} -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
PYTHON_FLAGS=$(shell ${PYTHON_EXE} -c 'import sysconfig; print(sysconfig.get_config_var("CFLAGS"))')
PYTHON_SUFFIX=$(shell ${PYTHON_EXE} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

#### Project paths #######################################

UV_GROUPS = --group cuda13_local --group exp
SRC_DIR = lib/e3j_ops
OBJ_DIR = bin
INC_DIR = $(SRC_DIR)

# Target executable name:
EXE = bin/test_scatter_add

CU_OBJ = \
	bin/fill.cu.o\
	bin/scatter_add.cu.o\
	bin/convolution.cu.o\
	bin/tensor_product_bwd.cu.o\
	bin/tensor_product.cu.o\

FFI_OBJ = bin/e3j_ops.cpp.o

CU_TEST = \
	test_scatter_add\
	test_tensor_product\
	test_tensor_product_bwd\
	test_convolution

TENSOR_PRODUCT_OBJ = \
	bin/tensor_product.cu.o

#	bin/tensor_product_leading.cu.o \
#	bin/tensor_product_trailing.cu.o \

TENSOR_PRODUCT_BWD_OBJ = \
	bin/tensor_product.cu.o \
	bin/tensor_product_bwd.cu.o

CONVOLUTION_OBJ = \
	bin/convolution.cu.o


#### RULES ###############################################

.PHONY: cutest pytest test_mosaic_tpu test clean uv e3j_ops docs

#=== Static libraries builds ===

bin/tensor_product.a: $(TENSOR_PRODUCT_OBJ)
	nvcc --lib -o $@ $^

bin/tensor_product_bwd.a: $(TENSOR_PRODUCT_BWD_OBJ)
	nvcc --lib -o $@ $^

bin/convolution.a: $(CONVOLUTION_OBJ)
	nvcc --lib -o $@ $^

bin/tensor_product_%.cu.o: $(SRC_DIR)/cuda/tensor_product/%_channels.cuh
	nvcc $(NVCC_FLAGS) -x cu -c $^ -o $@ -I $(INC_DIR) $(CUDA_INC_DIR)

bin/tensor_product_bwd_%.cu.o: $(SRC_DIR)/cuda/tensor_product_bwd/%_channels.cuh
	nvcc $(NVCC_FLAGS) -x cu -c $^ -o $@ -I $(INC_DIR) $(CUDA_INC_DIR)

#=== Objects and test builds ===

bin/%.cu.o: $(SRC_DIR)/cuda/%.cu $(SRC_DIR)/cuda/%.cuh
	nvcc $(NVCC_FLAGS) -c $< -o $@ -I $(INC_DIR) $(CUDA_INC_DIR)

bin/test_tensor_product:\
	$(SRC_DIR)/tests/test_tensor_product.cpp bin/tensor_product.a
	g++ $^ -o $@ -I $(INC_DIR) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

bin/test_tensor_product_bwd:\
	$(SRC_DIR)/tests/test_tensor_product_bwd.cpp bin/tensor_product_bwd.a
	g++ $^ -o $@ -I $(INC_DIR) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

bin/test_convolution:\
	$(SRC_DIR)/tests/test_convolution.cpp bin/convolution.a
	g++ $^ -o $@ -I $(INC_DIR) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

bin/test_tensor_product_bwd:\
	$(SRC_DIR)/tests/test_tensor_product_bwd.cpp bin/tensor_product_bwd.cu.o bin/tensor_product.a

	g++ $^ -o $@ -I $(INC_DIR) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

bin/test_%: bin/%.cu.o $(SRC_DIR)/tests/test_%.cpp
	g++ $^ -o $@ -I $(INC_DIR) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

test_%: bin/test_%
	$^

#=== Test suites ====

cutest: $(CU_TEST)

pytest:
	if [[ -n "$(k)" ]]; then\
		uv run pytest -v -m "e3j_ops" -k $(k) tests/test_ops;\
	else\
		uv run pytest -v -m "e3j_ops" tests/test_ops/test_tensor_product_op.py;\
	fi

test_mosaic_tpu:
	uv run pytest -v -m "mosaic_tpu" tests/

test: $(CU_TEST) pytest

#=== Package build ===

clean:
	rm bin/* || echo pass
	rm -r docs/_build || echo pass

docs:
	uv run sphinx-build -b html docs/ docs/_build/

# Force re-build of cuda bindings with uv and CMake

uv:
	uv sync --group cuda13_local --extra ops\
		--reinstall-package e3j_ops\
		--reinstall-package e3j

# Build the python bindings `e3j_ops.xxx.so` [2]

e3j_ops: $(CU_OBJ)
	# === 0. Check paths and includes ===
	PYBIND_INC=$(PYBIND_INC)
	PYTHON_EXE=$(PYTHON_EXE)
	PYTHON_SUFFIX=$(PYTHON_SUFFIX)
	# === 2. Compile C++ source to object files ===
	# g++ -c *.cpp
	c++ -I/usr/local/cuda/include $(PYTHON_FLAGS)\
		-O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects\
		-I $(INC_DIR) -I $(PYBIND_INC) -I $(PYTHON_INC)\
		-c $(SRC_DIR)/ffi/e3j_ops.cpp\
		-o bin/e3j_ops.cpp.o
	# === 3. Link to 'cpython' shared object binary ===
	# g++ *.o -o e3j_ops.cpython-310-x86_64-linux-gnu.so
	c++ -fPIC -O3 -DNDEBUG -O3 -flto -shared\
		bin/e3j_ops.cpp.o $(CU_OBJ)\
		-L/usr/local/cuda/lib64  -lcudadevrt -lcudart_static -lrt -lpthread -ldl\
		-o bin/e3j_ops$(PYTHON_SUFFIX)
	# === 4. strip "removes symbols from object files" ===
	strip bin/e3j_ops$(PYTHON_SUFFIX)

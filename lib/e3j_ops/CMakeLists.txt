cmake_minimum_required(VERSION 3.15...3.27)

project(e3j
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 4 -Xcompiler -w \
    -ldl --expt-relaxed-constexpr -O3 -DNDEBUG -O3 \
    -gencode=arch=compute_80,code=sm_80\
    -gencode=arch=compute_86,code=sm_86\
    -gencode=arch=compute_89,code=sm_89\
    -gencode=arch=compute_90,code=sm_90\
    -gencode=arch=compute_90a,code=sm_90a\
    -gencode=arch=compute_100,code=sm_100\
    -gencode=arch=compute_100,code=compute_100\
  -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto -fno-fat-lto-objects")


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(/usr/local/cuda/include)


Python_add_library(e3j_ops MODULE WITH_SOABI
    ffi/e3j_ops.cpp
    cuda/scatter_add.cu
    cuda/tensor_product.cu
    cuda/tensor_product_bwd.cu
    cuda/convolution.cu
    cuda/convolution_bwd.cu
    cuda/fill.cu)

target_link_libraries(e3j_ops PRIVATE
    pybind11::headers
    cudadevrt
    cudart_static
    rt
    pthread
    dl
    cudart)


target_compile_options(e3j_ops PRIVATE -w)

install(TARGETS e3j_ops DESTINATION .)

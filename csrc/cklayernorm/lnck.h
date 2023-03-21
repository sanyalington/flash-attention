#pragma once

#include <vector>
#include <iostream>
#include <ATen/hip/HIPGeneratorImpl.h>


struct lnck_fprop_params {
    int rows;
    int cols;
    void* x_ptr;
    void* gamma_ptr;
    void* beta_ptr;
    void* z_ptr;
    float epsilon;
};

template<typename Kernel_params>
struct Launch_params{
    Launch_params(hipDeviceProp_t * props_,
                  hipStream_t stream_)
        : props(props_)
        , stream(stream_) {
    }

    hipDeviceProp_t * props;
    hipStream_t stream;
    Kernel_params params;
};

int oldmain();

void modmain(Launch_params<lnck_fprop_params> &launch_params);

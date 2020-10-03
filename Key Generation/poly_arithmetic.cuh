#pragma once

#ifndef POLY_ARITHMETIC
#define POLY_ARITHMETIC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "uint128.h"
#include "ntt.cuh"

__global__ void barrett(unsigned long long[], const unsigned long long[], unsigned long long, unsigned long long, int);

__global__ void poly_add(unsigned long long[], const unsigned long long[], unsigned long long);

__global__ void poly_sub(unsigned long long[], const unsigned long long[], unsigned long long);

__host__ void poly_add_device(unsigned long long*, const unsigned long long*, unsigned N, cudaStream_t&, unsigned long long);

__host__ void poly_sub_device(unsigned long long*, const unsigned long long*, unsigned N, cudaStream_t&, unsigned long long);

unsigned long long* full_poly_mul(unsigned long long*, unsigned long long*, unsigned long long*, unsigned long long*, unsigned, cudaStream_t&, cudaStream_t&, unsigned long long, unsigned long long, int, unsigned long long*, unsigned long long*);

void full_poly_mul_device(unsigned long long*, unsigned long long*, unsigned, cudaStream_t&, cudaStream_t&, unsigned long long, unsigned long long, int, unsigned long long*);

#endif 
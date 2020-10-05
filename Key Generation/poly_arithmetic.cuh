#pragma once

#ifndef POLY_ARITHMETIC
#define POLY_ARITHMETIC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "uint128.h"
#include "ntt.cuh"

__global__ void barrett(unsigned long long[], const unsigned long long[], unsigned long long, unsigned long long, int);

__global__ void barrett_int(unsigned long long a[], const unsigned long long b, unsigned long long q, unsigned long long mu, int qbit);

__global__ void fast_convert_array_kernel_t(unsigned long long** input_poly, unsigned long long** result_poly, unsigned long long t, unsigned long long** base_change_matrix_device, unsigned q_amount);

__global__ void fast_convert_array_kernel_gamma(unsigned long long** input_poly, unsigned long long** result_poly, unsigned long long gamma, unsigned long long** base_change_matrix_device, unsigned q_amount, int gamma_bit_length, unsigned long long mu_gamma);

__host__ void fast_convert_array_kernels(unsigned long long** input_poly, unsigned long long** result_poly, unsigned long long t, unsigned long long** base_change_matrix_device, unsigned q_amount, unsigned long long gamma, int gamma_bit_length, unsigned long long mu_gamma, cudaStream_t& stream1, cudaStream_t& stream2, unsigned N);

__global__ void poly_add(unsigned long long[], const unsigned long long[], unsigned long long);

__global__ void poly_sub(unsigned long long[], const unsigned long long[], unsigned long long);

__host__ void poly_add_device(unsigned long long*, const unsigned long long*, unsigned N, cudaStream_t&, unsigned long long);

__host__ void poly_sub_device(unsigned long long*, const unsigned long long*, unsigned N, cudaStream_t&, unsigned long long);

unsigned long long* full_poly_mul(unsigned long long*, unsigned long long*, unsigned long long*, unsigned long long*, unsigned, cudaStream_t&, cudaStream_t&, unsigned long long, unsigned long long, int, unsigned long long*, unsigned long long*);

__host__ void half_poly_mul_device(unsigned long long*, unsigned long long*, unsigned, cudaStream_t&, unsigned long long, unsigned long long, int, unsigned long long*, unsigned long long*);

void full_poly_mul_device(unsigned long long*, unsigned long long*, unsigned, cudaStream_t&, cudaStream_t&, unsigned long long, unsigned long long, int, unsigned long long*);

__host__ void poly_mul_int(unsigned long long*, const unsigned long long, unsigned, cudaStream_t&, unsigned long long, unsigned long long, int);

#endif 
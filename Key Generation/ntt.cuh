#pragma once

#ifndef NTT
#define NTT

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "uint128.h"

template<unsigned l, unsigned N>
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);

template<unsigned l, unsigned N>
__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

template<unsigned l, unsigned N>
__global__ void CTBasedNTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);

template<unsigned l, unsigned N>
__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

__host__ void forwardNTTdouble(unsigned long long* device_a, unsigned long long* device_b, unsigned N, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers);

__host__ void forwardNTT(unsigned long long* device_a, unsigned N, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers);

__host__ void inverseNTT(unsigned long long* device_a, unsigned N, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psiinv_powers);

// --------------------------------------------------------------------------------------------------------------------------------------------------------
// explicit template instantiations
// all permutations are required for the program to compile

// N = 2048
template __global__ void CTBasedNTTInnerSingle<1, 2048>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInnerSingle<1, 2048>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// N = 4096
template __global__ void CTBasedNTTInnerSingle<1, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<2, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// N = 8192
template __global__ void CTBasedNTTInner<1, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<2, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<4, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// N = 16384
template __global__ void CTBasedNTTInner<1, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInner<2, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<4, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<4, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<8, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// N = 32768
template __global__ void CTBasedNTTInner<1, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInner<2, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInner<4, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<8, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<4, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<8, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<16, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// --------------------------------------------------------------------------------------------------------------------------------------------------------

#endif
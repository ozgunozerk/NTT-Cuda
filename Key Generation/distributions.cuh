#ifndef __DISTRIBUTIONS__
#define __DISTRIBUTIONS__

#include <inttypes.h>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "salsa_common.h"

__global__ void VecCrypt(unsigned char* A, unsigned int N, uint64_t nblocks, uint64_t p_nonce, int blks_per_chunk);

__global__ void convert_gaussian(unsigned* in, unsigned long long* out, unsigned long long q);

__global__ void convert_range(unsigned long long* in, unsigned long long* out, unsigned long long q);

__global__ void convert_ternary(unsigned char* in, unsigned long long* out, unsigned long long q);

void generate_random(unsigned char* a, unsigned n, cudaStream_t& stream);

void generate_random_default(unsigned char* a, unsigned n);

void gaussian_dist(unsigned* in, unsigned long long* out, unsigned n, cudaStream_t& stream, unsigned long long q);

void uniform_dist(unsigned long long* in, unsigned long long* out, unsigned n, cudaStream_t& stream, unsigned long long q);

void ternary_dist(unsigned char* in, unsigned long long* out, unsigned n, cudaStream_t& stream, unsigned long long q);

#endif
#ifndef __DISTRIBUTIONS__
#define __DISTRIBUTIONS__

#include <inttypes.h>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "salsa_common.h"

__global__ void VecCrypt(unsigned char*, unsigned int, uint64_t, uint64_t, int);

__global__ void convert_gaussian(unsigned*, unsigned long long*, unsigned long long);

__global__ void convert_range(unsigned long long*, unsigned long long*, unsigned long long);

__global__ void convert_ternary(unsigned char*, unsigned long long*, unsigned long long);

void generate_random(unsigned char*, unsigned, cudaStream_t&, int length);

void gaussian_dist(unsigned*, unsigned long long*, unsigned, cudaStream_t&, unsigned long long);

void uniform_dist(unsigned long long*, unsigned long long*, unsigned, cudaStream_t&, unsigned long long);

void ternary_dist(unsigned char*, unsigned long long*, unsigned, cudaStream_t&, unsigned long long);

#endif
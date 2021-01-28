#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "uint128.h"

__constant__ unsigned long long q_cons[16];
__constant__ unsigned q_bit_cons[16];
__constant__ unsigned long long mu_cons[16];
__constant__ unsigned long long inv_q_last_mod_q_cons[16];
__constant__ unsigned long long inv_punctured_q_cons[16];
__constant__ unsigned long long prod_t_gamma_mod_q_cons[16];

// --------------------------------------------------------------------------------------------------------------------------------------------------------
// declarations for templated ntt functions

template<unsigned l, unsigned n>  // single kernel NTT
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);

template<unsigned l, unsigned n>  // single kernel INTT
__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

template<unsigned l, unsigned n>  // multi kernel NTT
__global__ void CTBasedNTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);

template<unsigned l, unsigned n>  // multi kernel INTT
__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

template<unsigned l, unsigned n>  // single kernel NTT batch
__global__ void CTBasedNTTInnerSingle_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division);

template<unsigned l, unsigned n>  // single kernel INTT batch
__global__ void GSBasedINTTInnerSingle_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

template<unsigned l, unsigned n>  // multi kernel omg are you still reading this
__global__ void CTBasedNTTInner_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division);

template<unsigned l, unsigned n>  // i'm not gonna write this one, figure this out on your own
__global__ void GSBasedINTTInner_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// --------------------------------------------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void singleBarrett(uint128_t& a, unsigned long long& q, unsigned long long& mu, int& qbit)
{
    uint128_t rx;

    rx = a >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(a, rx);

    if (a.low >= q)
        a.low -= q;

}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[])
{
    register int local_tid = threadIdx.x; 

    extern __shared__ unsigned long long shared_array[];  // declaration of shared_array

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copying to shared from global
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
    }

#pragma unroll
    for (int length = l; length < n; length *= 2)
    {  // for loops are required since we are handling all the remaining iterations in this kernel
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = target_result;

            first_target_value += q * (first_target_value < second_target_value);

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copy back to global from shared
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
    }

}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[])
{
    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];  // declaration of shared_array

    register unsigned long long q2 = (q + 1) >> 1;

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copying to shared from global
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
    }

    __syncthreads();

#pragma unroll
    for (int length = (n / 2); length >= l; length /= 2)
    {  // for loops are required since we are handling all the remaining iterations in this kernel
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

            first_target_value += q * (first_target_value < second_target_value);

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);

            register unsigned long long temp_storage_low = temp_storage.low;

            shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copy back to global from shared
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
    }
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[])
{
    // no shared memory - handling only one iteration in here

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psi = psi_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;
}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[])
{
    // no shared memory - handling only one iteration in here

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psiinv = psiinv_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    register unsigned long long q2 = (q + 1) >> 1;

    target_result = (target_result >> 1) + q2 * (target_result & 1);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);

    register unsigned long long temp_storage_low = temp_storage.low;

    temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

    a[target_index + step] = temp_storage_low;
}

__host__ void forwardNTTdouble(unsigned long long* device_a, unsigned long long* device_b, unsigned n, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{  // performs 2 NTT operations together, check the forwardNTT function below for detailed comments
    if (n == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 8192)
    {
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
}

__host__ void forwardNTT(unsigned long long* device_a, unsigned n, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{  // performs single NTT operation
    if (n == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // second iter (mult-kernel)

        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // third iter (multi-kernel)

        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // second iter (mult-kernel)

        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 8192)
    {
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // all iterations, single-kernel
    }
    else if (n == 2048)
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // all iterations, single-kernel
    }
}

__host__ void inverseNTT(unsigned long long* device_a, unsigned n, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psiinv_powers)
{
    if (n == 32768)
    {
        GSBasedINTTInnerSingle<16, 32768> << <16, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till last 4 iterations
        
        GSBasedINTTInner<8, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 4th iter
        GSBasedINTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 3rd iter
        GSBasedINTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 2nd iter
        GSBasedINTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter
    }
    else if (n == 16384)
    {
        GSBasedINTTInnerSingle<8, 16384> << <8, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till last 3 iterations

        GSBasedINTTInner<4, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 3rd iter
        GSBasedINTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 2nd iter
        GSBasedINTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter
    }
    else if (n == 8192)
    {
        GSBasedINTTInnerSingle<4, 8192> << <4, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till last 2 iterations

        GSBasedINTTInner<2, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 2nd iter
        GSBasedINTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last last iter
    }
    else if (n == 4096)
    {
        GSBasedINTTInnerSingle<2, 4096> << <2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till the last iteration

        GSBasedINTTInner<1, 4096> << <4096 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter
    }
    else if (n == 2048)
    {
        GSBasedINTTInnerSingle<1, 2048> << <1, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel for all iterations
    }
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInnerSingle_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_cons[index];
    unsigned long long mu = mu_cons[index];
    int qbit = q_bit_cons[index];

    register int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];
    }

#pragma unroll
    for (int length = l; length < n; length *= 2)
    {
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step + index * n];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = target_result;

            first_target_value += q * (first_target_value < second_target_value);

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
    }

}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInnerSingle_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_cons[index];
    unsigned long long mu = mu_cons[index];
    int qbit = q_bit_cons[index];

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

    register unsigned long long q2 = (q + 1) >> 1;

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];
    }

    __syncthreads();

#pragma unroll
    for (int length = (n / 2); length >= l; length /= 2)
    {
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step + index * n];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

            first_target_value += q * (first_target_value < second_target_value);

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);

            register unsigned long long temp_storage_low = temp_storage.low;

            shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
    }
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInner_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_cons[index];
    unsigned long long mu = mu_cons[index];
    int qbit = q_bit_cons[index];

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.y * n;

    register unsigned long long psi = psi_powers[length + psi_step + index * n];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;
}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInner_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_cons[index];
    unsigned long long mu = mu_cons[index];
    int qbit = q_bit_cons[index];

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.y * n;

    register unsigned long long psiinv = psiinv_powers[length + psi_step + index * n];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    register unsigned long long q2 = (q + 1) >> 1;

    target_result = (target_result >> 1) + q2 * (target_result & 1);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);

    register unsigned long long temp_storage_low = temp_storage.low;

    temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

    a[target_index + step] = temp_storage_low;
}

__host__ void forwardNTT_batch(unsigned long long* device_a, unsigned n, unsigned long long* psi_powers, unsigned num, unsigned division)
{
    if (n == 32768)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        CTBasedNTTInner_batch<1, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInner_batch<2, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInner_batch<4, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInnerSingle_batch<8, 32768> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
    else if (n == 16384)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(4, num);
        CTBasedNTTInner_batch<1, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInner_batch<2, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInnerSingle_batch<4, 16384> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
    else if (n == 8192)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(2, num);
        CTBasedNTTInner_batch<1, 8192> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInnerSingle_batch<2, 8192> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
    else if (n == 4096)
    {
        dim3 single_dim(1, num);
        CTBasedNTTInnerSingle_batch<1, 4096> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
    else if (n == 2048)
    {
        dim3 single_dim(1, num);
        CTBasedNTTInnerSingle_batch<1, 2048> << <single_dim, 1024, 2048 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
}

__host__ void inverseNTT_batch(unsigned long long* device_a, unsigned n, unsigned long long* psiinv_powers, unsigned num, unsigned division)
{
    if (n == 32768)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(16, num);
        GSBasedINTTInnerSingle_batch<16, 32768> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<8, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<4, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<2, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<1, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 16384)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        GSBasedINTTInnerSingle_batch<8, 16384> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<4, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<2, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<1, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 8192)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(4, num);
        GSBasedINTTInnerSingle_batch<4, 8192> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<2, 8192> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<1, 8192> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 4096)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(2, num);
        GSBasedINTTInnerSingle_batch<2, 4096> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<1, 4096> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 2048)
    {
        dim3 single_dim(1, num);
        GSBasedINTTInnerSingle_batch<1, 2048> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------
// explicit template instantiations
// all these are required for the program to compile

// n = 2048
template __global__ void CTBasedNTTInnerSingle<1, 2048>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInnerSingle<1, 2048>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 4096
template __global__ void CTBasedNTTInnerSingle<1, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<2, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 8192
template __global__ void CTBasedNTTInner<1, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<2, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<4, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 16384
template __global__ void CTBasedNTTInner<1, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInner<2, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<4, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<4, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<8, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 32768
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
// explicit template instantiations for batch ntt
// all these are required for the program to compile

// n = 2048
template __global__ void CTBasedNTTInnerSingle_batch<1, 2048>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<1, 2048>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 4096
template __global__ void CTBasedNTTInnerSingle_batch<1, 4096>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 4096>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<2, 4096>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 8192
template __global__ void CTBasedNTTInner_batch<1, 8192>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInnerSingle_batch<2, 8192>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 8192>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<2, 8192>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<4, 8192>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 16384
template __global__ void CTBasedNTTInner_batch<1, 16384>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInner_batch<2, 16384>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInnerSingle_batch<4, 16384>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<2, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<4, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<8, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 32768
template __global__ void CTBasedNTTInner_batch<1, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInner_batch<2, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInner_batch<4, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInnerSingle_batch<8, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<2, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<4, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<8, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<16, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// --------------------------------------------------------------------------------------------------------------------------------------------------------
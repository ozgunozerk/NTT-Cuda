#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// --------------------------------------------------------------------------------------------------------------------------------------------------------
// declarations for templated ntt functions

__global__ void barrett_30bit(unsigned a[], const unsigned b[], unsigned q, unsigned mu, int qbit)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    unsigned ra = a[i];
    unsigned rb = b[i];

    unsigned long long rc, rx;

    rc = (unsigned long long)ra * rb;

    rx = rc >> (qbit - 2);

    rx *= mu;

    rx >>= qbit + 2;

    rx *= q;

    rc -= rx;

    if (rc < q)
        a[i] = rc;
    else
        a[i] = rc - q;
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInnerSingle(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInnerSingle(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInner(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInner(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);

// --------------------------------------------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void singleBarrett(unsigned long long& a, unsigned& q, unsigned& mu, int& qbit)
{
    unsigned long long rx;

    rx = a >> (qbit - 2);

    rx *= mu;

    rx >>= qbit + 2;

    rx *= q;

    a -= rx;

    if (a >= q)
        a -= q;
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInnerSingle(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[])
{
    int local_tid = threadIdx.x;

    extern __shared__ unsigned shared_array[];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
    }

#pragma unroll
    for (int length = l; length < n; length *= 2)
    {
        int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {

            int global_tid = local_tid + iteration_num * 1024;
            int psi_step = global_tid / step;
            int target_index = psi_step * step * 2 + global_tid % step;;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            unsigned psi = psi_powers[length + psi_step];

            unsigned first_target_value = shared_array[target_index];
            unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            temp_storage *= psi;

            singleBarrett(temp_storage, q, mu, qbit);
            unsigned second_target_value = temp_storage;

            unsigned target_result = first_target_value + second_target_value;

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
        int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
    }

}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInnerSingle(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[])
{
    int local_tid = threadIdx.x;

    extern __shared__ unsigned shared_array[];

    unsigned q2 = (q + 1) >> 1;

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
    }

    __syncthreads();

#pragma unroll
    for (int length = (n / 2); length >= l; length /= 2)
    {
        int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {
            int global_tid = local_tid + iteration_num * 1024;
            int psi_step = global_tid / step;
            int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            unsigned psiinv = psiinv_powers[length + psi_step];

            unsigned first_target_value = shared_array[target_index];
            unsigned second_target_value = shared_array[target_index + step];

            unsigned target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

            first_target_value += q * (first_target_value < second_target_value);

            unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage, q, mu, qbit);

            unsigned temp_storage_low = temp_storage;

            shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
    }
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInner(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[])
{
    int length = l;

    int global_tid = blockIdx.x * 1024 + threadIdx.x;
    int step = (n / length) / 2;
    int psi_step = global_tid / step;
    int target_index = psi_step * step * 2 + global_tid % step;

    unsigned psi = psi_powers[length + psi_step];

    unsigned first_target_value = a[target_index];
    unsigned long long temp_storage = a[target_index + step];

    temp_storage *= psi;

    singleBarrett(temp_storage, q, mu, qbit);
    unsigned second_target_value = temp_storage;

    unsigned target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;
}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInner(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[])
{
    int length = l;

    int global_tid = blockIdx.x * 1024 + threadIdx.x;
    int step = (n / length) / 2;
    int psi_step = global_tid / step;
    int target_index = psi_step * step * 2 + global_tid % step;

    unsigned psiinv = psiinv_powers[length + psi_step];

    unsigned first_target_value = a[target_index];
    unsigned second_target_value = a[target_index + step];

    unsigned target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    unsigned q2 = (q + 1) >> 1;

    target_result = (target_result >> 1) + q2 * (target_result & 1);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    unsigned long long temp_storage = first_target_value - second_target_value;

    temp_storage *= psiinv;

    singleBarrett(temp_storage, q, mu, qbit);

    unsigned temp_storage_low = temp_storage;

    temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

    a[target_index + step] = temp_storage_low;
}

__host__ void forwardNTTdouble(unsigned* device_a, unsigned* device_b, unsigned n, cudaStream_t& stream1, cudaStream_t& stream2, unsigned q, unsigned mu, int bit_length, unsigned* psi_powers)
{
    if (n == 65536)
    {
        CTBasedNTTInner<1, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 65536> << <65536 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 65536> << <65536 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<4, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<4, 65536> << <65536 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<8, 65536> << <8, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<8, 65536> << <8, 1024, 8192 * sizeof(unsigned), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<4, 32768> << <4, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<4, 32768> << <4, 1024, 8192 * sizeof(unsigned), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<2, 16384> << <2, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<2, 16384> << <2, 1024, 8192 * sizeof(unsigned), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 8192)
    {
        CTBasedNTTInnerSingle<1, 8192> << <1, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 8192> << <1, 1024, 8192 * sizeof(unsigned), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
}

__host__ void forwardNTT(unsigned* device_a, unsigned n, cudaStream_t& stream1, unsigned q, unsigned mu, int bit_length, unsigned* psi_powers)
{
    if (n == 65536)
    {
        CTBasedNTTInner<1, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<4, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<8, 65536> << <8, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (n == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<4, 32768> << <8, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (n == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<2, 16384> << <4, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (n == 8192)
    {
        CTBasedNTTInnerSingle<1, 8192> << <2, 1024, 8192 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (n == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (n == 2048)
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
}

__host__ void inverseNTT(unsigned* device_a, unsigned n, cudaStream_t& stream1, unsigned q, unsigned mu, int bit_length, unsigned* psiinv_powers)
{
    if (n == 65536)
    {
        GSBasedINTTInnerSingle<16, 65536> << <16, 1024, 4096 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<8, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<4, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<2, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<1, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (n == 32768)
    {
        GSBasedINTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (n == 16384)
    {
        GSBasedINTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (n == 8192)
    {
        GSBasedINTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (n == 4096)
    {
        GSBasedINTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (n == 2048)
    {
        GSBasedINTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned), stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------
// explicit template instantiations
// all permutations are required for the program to compile

// n = 2048
template __global__ void CTBasedNTTInnerSingle<1, 2048>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void GSBasedINTTInnerSingle<1, 2048>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);

// n = 4096
template __global__ void CTBasedNTTInnerSingle<1, 4096>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void GSBasedINTTInnerSingle<1, 4096>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);

// n = 8192
template __global__ void CTBasedNTTInnerSingle<1, 8192>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void GSBasedINTTInner<1, 8192>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<2, 8192>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);

// n = 16384
template __global__ void CTBasedNTTInner<1, 16384>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<2, 16384>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void GSBasedINTTInner<1, 16384>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 16384>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<4, 16384>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);

// n = 32768
template __global__ void CTBasedNTTInner<1, 32768>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void CTBasedNTTInner<2, 32768>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<4, 32768>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void GSBasedINTTInner<1, 32768>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 32768>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInner<4, 32768>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<8, 32768>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);

// n = 65536
template __global__ void CTBasedNTTInner<1, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void CTBasedNTTInner<2, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void CTBasedNTTInner<4, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<8, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[]);
template __global__ void GSBasedINTTInner<1, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInner<4, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInner<8, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<16, 65536>(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[]);
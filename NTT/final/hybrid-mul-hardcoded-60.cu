#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <sstream>
using std::cout;
using std::endl;

#include "helper.h"
#include "parameter.h"

#define N 1024 * 32
#define nttBlockSize 1024
#define blockSize 256
#define blockCount N / blockSize
#define check 0
#define numPoly 8

int size_array = sizeof(unsigned long long) * N;
int size = sizeof(unsigned long long);

__constant__ unsigned long long q_;
__constant__ unsigned long long ninv_;

__constant__ unsigned long long mu_;
__constant__ int bit_length_;


__device__ __forceinline__ void mul64(unsigned long long& a, unsigned long long& b, uint128_t& c)
{
	uint4 res;

	asm("{\n\t"
		"mul.lo.u32      %3, %5, %7;    \n\t"
		"mul.hi.u32      %2, %5, %7;    \n\t" //alow * blow
		"mad.lo.cc.u32   %2, %4, %7, %2;\n\t"
		"madc.hi.u32     %1, %4, %7,  0;\n\t" //ahigh * blow
		"mad.lo.cc.u32   %2, %5, %6, %2;\n\t"
		"madc.hi.cc.u32  %1, %5, %6, %1;\n\t" //alow * bhigh
		"madc.hi.u32     %0, %4, %6,  0;\n\t"
		"mad.lo.cc.u32   %1, %4, %6, %1;\n\t" //ahigh * bhigh
		"addc.u32        %0, %0, 0;     \n\t" //add final carry
		"}"
		: "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
		: "r"((unsigned)(a >> 32)), "r"((unsigned)a), "r"((unsigned)(b >> 32)), "r"((unsigned)b));

	c.high = ((unsigned long long)res.x << 32) + res.y;
	c.low = ((unsigned long long)res.z << 32) + res.w;;
}


__global__ void mulBarrett(unsigned long long a[])
{

    for (int i = 1; i < numPoly; i <<= 1)
    {
        for (int j = 0; j < (numPoly - i); j += (i * 2))
        {
            register int target_index = blockIdx.x * nttBlockSize + threadIdx.x + j * N;
            register int bit_length = bit_length_;

            register unsigned long long ra = a[target_index];
            register unsigned long long rb = a[target_index + i * N];

            uint128_t rc, rx;

            mul64(ra, rb, rc);

            rx = rc >> (bit_length - 2);

            mul64(rx.low, mu_, rx);

            uint128_t::shiftr(rx, bit_length + 2);

            mul64(rx.low, q_, rx);

            rc -= rx;

            if (rc < q_)
            {
                a[target_index] = rc.low;
            }
            else
            {
                a[target_index] = rc.low - q_;
            }

        }
    }

}

__global__ void barrett(unsigned long long a[], unsigned long long b[])
{

    register int i = blockIdx.x * blockSize + threadIdx.x;
    register int bit_length = bit_length_;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (bit_length - 2);

    mul64(rx.low, mu_, rx);

    uint128_t::shiftr(rx, bit_length + 2);

    mul64(rx.low, q_, rx);

    rc -= rx;

    if (rc < q_)
    {
        a[i] = rc.low;
    }
    else
    {
        a[i] = rc.low - q_;
    }

}

__device__ __forceinline__ void singleBarrett(uint128_t& a)
{
    register int bit_length = bit_length_;

    uint128_t rx;

    rx = a >> (bit_length - 2);

    mul64(rx.low, mu_, rx);

    uint128_t::shiftr(rx, bit_length + 2);

    mul64(rx.low, q_, rx);

    a -= rx;

    if (a >= q_)
        a -= q_;

}

__global__ void CTBasedNTTInnerSingleL1(unsigned long long a[], unsigned long long psi_powers[])
{
    int l = 1;

    register int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

    #pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    #pragma unroll
    for (int length = l; length < N; length *= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index;
            target_index = psi_step * step * 2 + global_tid % step + blockIdx.x;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            shared_array[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q_;

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

    #pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }

}

__global__ void CTBasedNTTInnerSingleL2(unsigned long long a[], unsigned long long psi_powers[])
{
    int l = 2;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[4096];

    #pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    #pragma unroll
    for (int length = l; length < N; length *= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index;
            target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            shared_array[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q_;

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void CTBasedNTTInnerSingleL4(unsigned long long a[], unsigned long long psi_powers[])
{
    int l = 4;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[4096];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    #pragma unroll
    for (int length = l; length < N; length *= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index;
            target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            shared_array[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q_;

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void CTBasedNTTInnerSingleL8(unsigned long long a[], unsigned long long psi_powers[])
{
    int l = 8;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[4096];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

#pragma unroll
    for (int length = l; length < N; length *= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index;
            target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            shared_array[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q_;

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void GSBasedINTTInnerSingleL1(unsigned long long a[], unsigned long long psiinv_powers[])
{
    unsigned l = 1;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    __syncthreads();

    #pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned long long q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                shared_array[target_index] = (target_result >> 1) + q2;
            else
                shared_array[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage);

            register unsigned long long temp_storage_low = temp_storage.low;
            if (temp_storage_low & 1)
                shared_array[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                shared_array[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void GSBasedINTTInnerSingleL2(unsigned long long a[], unsigned long long psiinv_powers[])
{
    unsigned l = 2;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    __syncthreads();

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned long long q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                shared_array[target_index] = (target_result >> 1) + q2;
            else
                shared_array[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage);

            register unsigned long long temp_storage_low = temp_storage.low;
            if (temp_storage_low & 1)
                shared_array[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                shared_array[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void GSBasedINTTInnerSingleL4(unsigned long long a[], unsigned long long psiinv_powers[])
{
    unsigned l = 4;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    __syncthreads();

    #pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned long long q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                shared_array[target_index] = (target_result >> 1) + q2;
            else
                shared_array[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage);

            register unsigned long long temp_storage_low = temp_storage.low;
            if (temp_storage_low & 1)
                shared_array[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                shared_array[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void GSBasedINTTInnerSingleL8(unsigned long long a[], unsigned long long psiinv_powers[])
{
    unsigned l = 8;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    __syncthreads();

    #pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned long long q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                shared_array[target_index] = (target_result >> 1) + q2;
            else
                shared_array[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage);

            register unsigned long long temp_storage_low = temp_storage.low;
            if (temp_storage_low & 1)
                shared_array[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                shared_array[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void GSBasedINTTInnerSingleL16(unsigned long long a[], unsigned long long psiinv_powers[])
{
    unsigned l = 16;

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    __syncthreads();

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned long long q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                shared_array[target_index] = (target_result >> 1) + q2;
            else
                shared_array[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage);

            register unsigned long long temp_storage_low = temp_storage.low;
            if (temp_storage_low & 1)
                shared_array[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                shared_array[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

__global__ void CTBasedNTTInnerLength1(unsigned long long a[], unsigned long long psi_powers[])
{
    int length = 1;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psi = psi_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_resulttarget_result = first_target_value + second_target_value;

    if (target_resulttarget_result >= q_)
        target_resulttarget_result -= q_;

    a[target_index] = target_resulttarget_result;

    if (first_target_value < second_target_value)
        first_target_value += q_;

    a[target_index + step] = first_target_value - second_target_value;
}

__global__ void CTBasedNTTInnerLength2(unsigned long long a[], unsigned long long psi_powers[])
{
    int length = 2;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psi = psi_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_resulttarget_result = first_target_value + second_target_value;

    if (target_resulttarget_result >= q_)
        target_resulttarget_result -= q_;

    a[target_index] = target_resulttarget_result;

    if (first_target_value < second_target_value)
        first_target_value += q_;

    a[target_index + step] = first_target_value - second_target_value;
}

__global__ void CTBasedNTTInnerLength4(unsigned long long a[], unsigned long long psi_powers[])
{
    int length = 4;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psi = psi_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_resulttarget_result = first_target_value + second_target_value;

    if (target_resulttarget_result >= q_)
        target_resulttarget_result -= q_;

    a[target_index] = target_resulttarget_result;

    if (first_target_value < second_target_value)
        first_target_value += q_;

    a[target_index + step] = first_target_value - second_target_value;
}

__global__ void GSBasedINTTInnerLength1(unsigned long long a[], unsigned long long psiinv_powers[])
{
    int length = 1;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psiinv = psiinv_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned long long q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage);

    register unsigned long long temp_storage_low = temp_storage.low;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

__global__ void GSBasedINTTInnerLength2(unsigned long long a[], unsigned long long psiinv_powers[])
{
    int length = 2;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psiinv = psiinv_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned long long q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage);

    register unsigned long long temp_storage_low = temp_storage.low;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

__global__ void GSBasedINTTInnerLength4(unsigned long long a[], unsigned long long psiinv_powers[])
{
    int length = 4;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psiinv = psiinv_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned long long q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage);

    register unsigned long long temp_storage_low = temp_storage.low;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

__global__ void GSBasedINTTInnerLength8(unsigned long long a[], unsigned long long psiinv_powers[])
{
    int length = 8;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psiinv = psiinv_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned long long q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage);

    register unsigned long long temp_storage_low = temp_storage.low;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

__global__ void emptyKernel()
{
    return;
}

void printArray(unsigned long long a[])
{
    std::string sss = "[";
    cout << sss;

    for (int i = 0; i < (N - 1); i++)
    {

        cout << a[i] << ", ";
    }

    cout << a[N - 1];

    cout << "]\n\n" << endl;
}

__global__ void GSBasedINTTMerged(unsigned long long a[], unsigned long long psiinv[])
{
#pragma unroll
    for (int m = (N / 2); m > 0; m /= 2)
    {
        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
        GSBasedINTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
        GSBasedINTTInnerLength4 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
        GSBasedINTTInnerLength8 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);

        GSBasedINTTInnerSingleL1 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
        GSBasedINTTInnerSingleL2 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
        GSBasedINTTInnerSingleL4 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
        GSBasedINTTInnerSingleL8 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
        GSBasedINTTInnerSingleL16 << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv);
    }
}

int main()
{
    cudaSetDevice(2);

    emptyKernel << <1, 1, 0, 0 >> > ();

    unsigned long long q, psi, psiinv, ninv;
    unsigned int q_bit;

    getParams(q, psi, psiinv, ninv, q_bit, N);

    unsigned long long psiTable[N];
    unsigned long long psiinvTable[N];
    fillTablePsi128(psi, q, psiinv, psiTable, psiinvTable, N); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    unsigned long long* d_psi;
    unsigned long long* d_psiinv;
    cudaMalloc(&d_psi, size_array);
    cudaMalloc(&d_psiinv, size_array);

    cudaMemcpy(d_psi, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(d_psiinv, psiinvTable, size_array, cudaMemcpyHostToDevice);

    cout << "n = " << N << endl;
    cout << "q = " << q << endl;
    cout << "Psi = " << psi << endl;
    cout << "Psi Inverse = " << psiinv << endl;

    //copy q and n inverse to gpu
    cudaMemcpyToSymbol(q_, &q, size);
    cudaMemcpyToSymbol(ninv_, &ninv, size);

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    uint128_t mu1 = uint128_t::exp2(2 * bit_length);
    mu1 = mu1 / q;
    unsigned long long mu = mu1.low;

    //copy barrett parameters to device
    cudaMemcpyToSymbol(mu_, &mu, size);
    cudaMemcpyToSymbol(bit_length_, &bit_length, size);

    //--------------------------------------------------------------------------

    unsigned long long* all_a;
    cudaMallocHost(&all_a, size_array * numPoly);
    randomArray128(all_a, N * numPoly, q); //fill array with random numbers between 0 and q - 1

    unsigned long long* d_a;
    cudaMalloc(&d_a, size_array * numPoly);

    unsigned long long* refc;
    if (check)
        refc = refPolyMul128(all_a, all_a + N, q, N);

    cudaStream_t ntt_streams[numPoly];
    for (int i = 0; i < numPoly; i++)
    {
        cudaStreamCreate(&ntt_streams[i]);
    }

    //copy random arrays to gpu
    
    for (int i = 0; i < numPoly; i++)
    {
        cudaMemcpyAsync(d_a + N * i, all_a + N * i, size_array, cudaMemcpyHostToDevice, ntt_streams[i]);
    }

    for (int i = 0; i < numPoly; i++)
    {
        if (N == 32768)
        {
            CTBasedNTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerLength4 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerSingleL8 << <8, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);
        }
        else if (N == 16384)
        {
            CTBasedNTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerSingleL4 << <4, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);
        }
        else if (N == 8192)
        {
            CTBasedNTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerSingleL2 << <2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);
        }
        else
        {
            CTBasedNTTInnerSingleL1 << <1, nttBlockSize, min(N, 4096) * sizeof(unsigned long long), ntt_streams[i] >> > (d_a + N * i, d_psi);
        }
    }
    
    mulBarrett<<<N / nttBlockSize, nttBlockSize, 0, 0 >>>(d_a);

    if (N == 32768)
    {
        GSBasedINTTInnerSingleL16 << <16, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);

        GSBasedINTTInnerLength8 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength4 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    else if (N == 16384)
    {
        GSBasedINTTInnerSingleL8 << <8, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);

        GSBasedINTTInnerLength4 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    else if (N == 8192)
    {
        GSBasedINTTInnerSingleL4 << <4, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);

        GSBasedINTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    else if (N == 4096)
    {
        GSBasedINTTInnerSingleL2 << <2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);

        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    else
    {
        GSBasedINTTInnerSingleL1 << <1, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    
    cudaMemcpyAsync(all_a, d_a, size_array, cudaMemcpyDeviceToHost, 0);    

    cudaDeviceSynchronize();
    
    if (check) //check the correctness of results
    {
        for (int i = 0; i < N; i++)
        {
            if (all_a[i] != refc[i])
            {
                cout << "error" << endl;
                cout << i << "   " << all_a[i] << "   " << refc[i] << endl;
            }
                
        }

        free(refc);
    }

    cudaFreeHost(all_a);

    return 0;
}

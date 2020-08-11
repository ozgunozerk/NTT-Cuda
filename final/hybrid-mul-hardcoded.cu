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
#define numPoly 4

int size_array = sizeof(unsigned) * N;
int size = sizeof(unsigned);

__constant__ unsigned q_;
__constant__ unsigned ninv_;

__constant__ unsigned mu_;
__constant__ int bit_length_;

__global__ void mulBarrett(unsigned a[])
{

    for (int i = 1; i < numPoly; i <<= 1)
    {
        for (int j = 0; j < (numPoly - i); j += (i * 2))
        {
            register int target_index = blockIdx.x * nttBlockSize + threadIdx.x + j * N;
            register int bit_length = bit_length_;

            register unsigned ra = a[target_index];
            register unsigned rb = a[target_index + i * N];

            unsigned long long rc, rx;

            rc = (unsigned long long)ra * rb;

            rx = rc >> (bit_length - 2);

            rx *= mu_;

            rx >>= (bit_length + 2);

            rx *= q_;

            rc -= rx;

            if (rc < q_)
            {
                a[target_index] = rc;
            }
            else
            {
                a[target_index] = rc - q_;
            }
        }
    }

}

__global__ void barrett(unsigned a[], unsigned b[])
{

    register int i = blockIdx.x * blockSize + threadIdx.x;
    register int bit_length = bit_length_;

    register unsigned ra = a[i];
    register unsigned rb = b[i];

    unsigned long long rc, rx;

    rc = (unsigned long long)ra * rb;

    rx = rc >> (bit_length - 2);

    rx *= mu_;

    rx >>= (bit_length + 2);

    rx *= q_;

    rc -= rx;

    if (rc < q_)
    {
        a[i] = rc;
    }
    else
    {
        a[i] = rc - q_;
    }
    
}

__device__ __forceinline__ void singleBarrett(unsigned long long& a)
{
    register int bit_length = bit_length_;

    unsigned long long rx;

    rx = a >> (bit_length - 2);

    rx *= mu_;

    rx >>= (bit_length + 2);

    rx *= q_;

    a -= rx;

    if (a >= q_)
        a -= q_;
 
}

__global__ void CTBasedNTTInnerSingleL8(unsigned a[], unsigned psi_powers[])
{
    unsigned int l = 8;

    register int local_tid = threadIdx.x;

    __shared__ unsigned shared_array[8192];

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
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psi = psi_powers[length + psi_step];

            register unsigned first_target_value = shared_array[target_index];
            register unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            temp_storage *= psi;

            singleBarrett(temp_storage);
            register unsigned second_target_value = temp_storage;

            register unsigned target_result = first_target_value + second_target_value;

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

__global__ void CTBasedNTTInnerSingleL4(unsigned a[], unsigned psi_powers[])
{
    unsigned int l = 4;

    register int local_tid = threadIdx.x;

    __shared__ unsigned shared_array[8192];

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
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psi = psi_powers[length + psi_step];

            register unsigned first_target_value = shared_array[target_index];
            register unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            temp_storage *= psi;

            singleBarrett(temp_storage);
            register unsigned second_target_value = temp_storage;

            register unsigned target_result = first_target_value + second_target_value;

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

__global__ void CTBasedNTTInnerSingleL2(unsigned a[], unsigned psi_powers[])
{
    unsigned int l = 2;

    register int local_tid = threadIdx.x;

    __shared__ unsigned shared_array[8192];

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
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psi = psi_powers[length + psi_step];

            register unsigned first_target_value = shared_array[target_index];
            register unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            temp_storage *= psi;

            singleBarrett(temp_storage);
            register unsigned second_target_value = temp_storage;

            register unsigned target_result = first_target_value + second_target_value;

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

__global__ void CTBasedNTTInnerSingleL1(unsigned a[], unsigned psi_powers[])
{
    unsigned int l = 1;

    register int local_tid = threadIdx.x;

    extern __shared__ unsigned shared_array[];

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
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psi = psi_powers[length + psi_step];

            register unsigned first_target_value = shared_array[target_index];
            register unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            temp_storage *= psi;

            singleBarrett(temp_storage);
            register unsigned second_target_value = temp_storage;

            register unsigned target_result = first_target_value + second_target_value;

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

__global__ void GSBasedINTTInnerSingleL16(unsigned a[], unsigned psiinv_powers[]) //NOT SHARED. FOR SOME REASON DOES NOT YIELD CORRECT RESULTS WITH IF SHARED VERSION IS USED. WHY???
{

    unsigned l = 16;

    register int local_tid = threadIdx.x;

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.x * (N / l);

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psiinv = psiinv_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned second_target_value = a[target_index + step];

            register unsigned target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                a[target_index] = (target_result >> 1) + q2;
            else
                a[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage);

            register unsigned temp_storage_low = temp_storage;
            if (temp_storage_low & 1)
                a[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                a[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

}

__global__ void GSBasedINTTInnerSingleL8(unsigned a[], unsigned psiinv_powers[]) //NOT SHARED. FOR SOME REASON DOES NOT YIELD CORRECT RESULTS WITH IF SHARED VERSION IS USED. WHY???
{

    unsigned l = 8;

    register int local_tid = threadIdx.x;

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.x * (N / l);

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psiinv = psiinv_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned second_target_value = a[target_index + step];

            register unsigned target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                a[target_index] = (target_result >> 1) + q2;
            else
                a[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage);

            register unsigned temp_storage_low = temp_storage;
            if (temp_storage_low & 1)
                a[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                a[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

}

__global__ void GSBasedINTTInnerSingleL4(unsigned a[], unsigned psiinv_powers[]) //NOT SHARED. FOR SOME REASON DOES NOT YIELD CORRECT RESULTS WITH IF SHARED VERSION IS USED. WHY???
{

    unsigned l = 4;

    register int local_tid = threadIdx.x;

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.x * (N / l);

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psiinv = psiinv_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned second_target_value = a[target_index + step];

            register unsigned target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                a[target_index] = (target_result >> 1) + q2;
            else
                a[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage);

            register unsigned temp_storage_low = temp_storage;
            if (temp_storage_low & 1)
                a[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                a[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

}

__global__ void GSBasedINTTInnerSingleL2(unsigned a[], unsigned psiinv_powers[]) //NOT SHARED. FOR SOME REASON DOES NOT YIELD CORRECT RESULTS WITH IF SHARED VERSION IS USED. WHY???
{

    unsigned l = 2;

    register int local_tid = threadIdx.x;

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.x * (N / l);

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psiinv = psiinv_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned second_target_value = a[target_index + step];

            register unsigned target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                a[target_index] = (target_result >> 1) + q2;
            else
                a[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage);

            register unsigned temp_storage_low = temp_storage;
            if (temp_storage_low & 1)
                a[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                a[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

}

__global__ void GSBasedINTTInnerSingleL1(unsigned a[], unsigned psiinv_powers[]) //NOT SHARED. FOR SOME REASON DOES NOT YIELD CORRECT RESULTS WITH IF SHARED VERSION IS USED. WHY???
{

    unsigned l = 1;

    register int local_tid = threadIdx.x;

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.x * (N / l);

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned psiinv = psiinv_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned second_target_value = a[target_index + step];

            register unsigned target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            register unsigned q2 = (q_ + 1) >> 1;

            if (target_result & 1)
                a[target_index] = (target_result >> 1) + q2;
            else
                a[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage);

            register unsigned temp_storage_low = temp_storage;
            if (temp_storage_low & 1)
                a[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                a[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

}

__global__ void CTBasedNTTInnerLength1(unsigned a[], unsigned psi_powers[])
{
    int length = 1;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned psi = psi_powers[length + psi_step];

    register unsigned first_target_value = a[target_index];
    register unsigned long long temp_storage = a[target_index + step];

    temp_storage *= psi;

    singleBarrett(temp_storage);
    register unsigned second_target_value = temp_storage;

    register unsigned target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    a[target_index] = target_result;

    if (first_target_value < second_target_value)
        first_target_value += q_;

    a[target_index + step] = first_target_value - second_target_value;
}

__global__ void CTBasedNTTInnerLength2(unsigned a[], unsigned psi_powers[])
{
    int length = 2;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned psi = psi_powers[length + psi_step];

    register unsigned first_target_value = a[target_index];
    register unsigned long long temp_storage = a[target_index + step];

    temp_storage *= psi;

    singleBarrett(temp_storage);
    register unsigned second_target_value = temp_storage;

    register unsigned target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    a[target_index] = target_result;

    if (first_target_value < second_target_value)
        first_target_value += q_;

    a[target_index + step] = first_target_value - second_target_value;
}

__global__ void CTBasedNTTInnerLength4(unsigned a[], unsigned psi_powers[])
{
    int length = 4;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned psi = psi_powers[length + psi_step];

    register unsigned first_target_value = a[target_index];
    register unsigned long long temp_storage = a[target_index + step];

    temp_storage *= psi;

    singleBarrett(temp_storage);
    register unsigned second_target_value = temp_storage;

    register unsigned target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    a[target_index] = target_result;

    if (first_target_value < second_target_value)
        first_target_value += q_;

    a[target_index + step] = first_target_value - second_target_value;
}

__global__ void GSBasedINTTInnerLength1(unsigned a[], unsigned psiinv_powers[])
{
    int length = 1;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned psiinv = psiinv_powers[length + psi_step];

    register unsigned first_target_value = a[target_index];
    register unsigned second_target_value = a[target_index + step];

    register unsigned target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register unsigned long long temp_storage = first_target_value - second_target_value;

    temp_storage *= psiinv;

    singleBarrett(temp_storage);

    register unsigned temp_storage_low = temp_storage;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

__global__ void GSBasedINTTInnerLength2(unsigned a[], unsigned psiinv_powers[])
{
    int length = 2;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned psiinv = psiinv_powers[length + psi_step];

    register unsigned first_target_value = a[target_index];
    register unsigned second_target_value = a[target_index + step];

    register unsigned target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register unsigned long long temp_storage = first_target_value - second_target_value;

    temp_storage *= psiinv;

    singleBarrett(temp_storage);

    register unsigned temp_storage_low = temp_storage;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

__global__ void GSBasedINTTInnerLength4(unsigned a[], unsigned psiinv_powers[])
{
    int length = 4;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned psiinv = psiinv_powers[length + psi_step];

    register unsigned first_target_value = a[target_index];
    register unsigned second_target_value = a[target_index + step];

    register unsigned target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register unsigned long long temp_storage = first_target_value - second_target_value;

    temp_storage *= psiinv;

    singleBarrett(temp_storage);

    register unsigned temp_storage_low = temp_storage;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

__global__ void GSBasedINTTInnerLength8(unsigned a[], unsigned psiinv_powers[])
{
    int length = 8;

    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned psiinv = psiinv_powers[length + psi_step];

    register unsigned first_target_value = a[target_index];
    register unsigned second_target_value = a[target_index + step];

    register unsigned target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    register unsigned q2 = (q_ + 1) >> 1;

    if (target_result & 1)
        a[target_index] = (target_result >> 1) + q2;
    else
        a[target_index] = (target_result >> 1);

    if (first_target_value < second_target_value)
        first_target_value += q_;

    register unsigned long long temp_storage = first_target_value - second_target_value;

    temp_storage *= psiinv;

    singleBarrett(temp_storage);

    register unsigned temp_storage_low = temp_storage;
    if (temp_storage_low & 1)
        a[target_index + step] = (temp_storage_low >> 1) + q2;
    else
        a[target_index + step] = (temp_storage_low >> 1);
}

void printArray(unsigned a[])
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

__global__ void emptyKernel()
{
    return;
}

int main()
{
    cudaSetDevice(2);

    emptyKernel << <1, 1, 0, 0 >> > ();

    unsigned q, psi, psiinv, ninv;
    unsigned int q_bit;

    getParams30(q, psi, psiinv, ninv, q_bit, N);

    unsigned psiTable[N];
    unsigned psiinvTable[N];
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, N); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    unsigned* d_psi;
    unsigned* d_psiinv;
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
    unsigned long long mu1 = powl(2, 2 * bit_length);
    mu1 = mu1 / q;
    unsigned mu = mu1;

    //copy barrett parameters to device
    cudaMemcpyToSymbol(mu_, &mu, size);
    cudaMemcpyToSymbol(bit_length_, &bit_length, size);

    //---------------------------------------------------------------------------------------------------------------------------------------------------------//

    unsigned* all_a;
    cudaMallocHost(&all_a, size_array * numPoly);
    randomArray64(all_a, N * numPoly, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned* d_a;
    cudaMalloc(&d_a, size_array * numPoly);

    unsigned* refc;
    if (check)
        refc = refPolyMul64(all_a, all_a + N, q, N);

    cudaStream_t ntt_streams[numPoly];
    for (int i = 0; i < numPoly; i++)
    {
        cudaStreamCreate(&ntt_streams[i]);
    }

    
    int inttlength = 1 << ((int)log2((float)N) - 12);

    //copy random arrays to gpu
    
    for (int i = 0; i < numPoly; i++)
    {
        cudaMemcpyAsync(d_a + N * i, all_a + N * i, size_array, cudaMemcpyHostToDevice, ntt_streams[i]);
    }

    for (int i = 0; i < numPoly; i++)
    {
        if (N == 65536)
        {
            CTBasedNTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerLength4 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerSingleL8 << <8, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);
        }
        else if (N == 32768)
        {
            CTBasedNTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerSingleL4 << <4, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);
        }
        else if (N == 16384)
        {
            CTBasedNTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);

            CTBasedNTTInnerSingleL2 << <2, nttBlockSize, 0, ntt_streams[i] >> > (d_a + N * i, d_psi);
        }
        else
        {
            CTBasedNTTInnerSingleL1 << <1, nttBlockSize, std::min(N, 8192) * sizeof(unsigned), ntt_streams[i] >> > (d_a + N * i, d_psi);
        }

    }

    mulBarrett<<< N / nttBlockSize, nttBlockSize, 0, 0>>>(d_a);

    if (N == 65536)
    {
        GSBasedINTTInnerSingleL16 << <16, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);

        GSBasedINTTInnerLength8 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength4 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    else if (N == 32768)
    {
        GSBasedINTTInnerSingleL8 << <8, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);

        GSBasedINTTInnerLength4 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    else if (N == 16384)
    {
        GSBasedINTTInnerSingleL4 << <4, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);

        GSBasedINTTInnerLength2 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
        GSBasedINTTInnerLength1 << <N / nttBlockSize / 2, nttBlockSize, 0, 0 >> > (d_a, d_psiinv);
    }
    else if (N == 8192)
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
using std::cout;
using std::endl;

#include "helper.h"
#include "parameter.h"
#include "uint128.h"

#define N 1024 * 2
#define nttBlockSize 1024
#define blockSize 256
#define blockCount N / blockSize
#define check 0

int size_array = sizeof(unsigned long long) * N;
int size = sizeof(unsigned long long);

__constant__ unsigned long long q_;
__constant__ unsigned long long ninv_;

__constant__ unsigned long long mu_;
__constant__ int bit_length_;
__constant__ unsigned long long q2_;


__global__ void barrett(unsigned long long a[], unsigned long long b[])
{

    register int i = blockIdx.x * blockSize + threadIdx.x;
    register int bit_length = bit_length_;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    rc.low = ra * rb;
    rc.high = __umul64hi(ra, rb);

    rx = rc >> (bit_length - 2);

    rx.high = __umul64hi(rx.low, mu_);
    rx.low = rx.low * mu_;

    uint128_t::shiftr(rx, bit_length + 2);

    rx.high = __umul64hi(rx.low, q_);
    rx.low = rx.low * q_;

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

__global__ void inverseNTTbarrett(unsigned long long a[])
{
    register int i = blockIdx.x * blockSize + threadIdx.x;
    register int bit_length = bit_length_;

    register unsigned long long ra = a[i];
    uint128_t rc, rx;
    
    rc.low = ra * ninv_;
    rc.high = __umul64hi(ra, ninv_);

    rx = rc >> (bit_length - 2);

    rx.high = __umul64hi(rx.low, mu_);
    rx.low = rx.low * mu_;

    uint128_t::shiftr(rx, bit_length + 2);

    rx.high = __umul64hi(rx.low, q_);
    rx.low = rx.low * q_;

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

__device__ __forceinline__ inline void singleBarrett(uint128_t& a)
{
    register int bit_length = bit_length_;

    uint128_t rx;

    rx = a >> (bit_length - 2);

    rx.high = __umul64hi(rx.low, mu_);
    rx.low = rx.low * mu_;

    uint128_t::shiftr(rx, bit_length + 2);

    rx.high = __umul64hi(rx.low, q_);
    rx.low = rx.low * q_;

    a -= rx;

    if (a >= q_)
        a -= q_;
 
}

__global__ void CTBasedNTTInner(unsigned long long a[], unsigned long long psi_powers[], int length)
{
    register int global_tid = blockIdx.x * nttBlockSize + threadIdx.x;
    register int step = (__fdividef(__fdividef(N, length)), 2);
    register int step_group_amount = __fdividef(global_tid, step);
    register int target_index = step_group_amount * step * 2 + global_tid % step;

    register unsigned long long psi = psi_powers[length + step_group_amount];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    temp_storage.high = __umul64hi(temp_storage.low, psi);
    temp_storage.low = temp_storage.low * psi;

    singleBarrett(temp_storage);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_result = first_target_value + second_target_value;

    if (target_result >= q_)
        target_result -= q_;

    a[target_index] = target_result;

    if (first_target_value < second_target_value)
        first_target_value += q_;

    a[target_index + step] = first_target_value - second_target_value;
}

/*void CTBasedNTTMerged(unsigned long long a[], unsigned long long psi[], cudaStream_t stream)
{
    #pragma unroll
    for (int m = 1; m < N; m *= 2)
    {
        CTBasedNTTInner << <N / nttBlockSize / 2, nttBlockSize, 1, stream >> > (a, psi, m);
    }
}*/

__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long psiinv[], int m)
{
    register int k = blockIdx.x * nttBlockSize + threadIdx.x;
    register int t = (N / m) / 2;
    register int i = k / t;
    register int j = i * t * 2 + k % t;

    register unsigned long long s = psiinv[m + i];

    register unsigned long long u = a[j];
    register unsigned long long v = a[j + t];

    register unsigned long long uu = u + v;

    if (uu >= q_)
        uu -= q_;

    a[j] = uu;

    if (u < v)
        u += q_;

    register uint128_t ut = u - v;

    ut.high = __umul64hi(ut.low, s);
    ut.low = ut.low * s;

    singleBarrett(ut);
    a[j + t] = ut.low;
}

__global__ void INTTOptimisation(unsigned long long a[])
{
    register int global_idx = blockIdx.x * blockSize + threadIdx.x;

    register int ra = a[global_idx];

    if (ra & 1)
        a[global_idx] = (ra >> 1) + q2_;
    else
        a[global_idx] = (ra >> 1);
}

__global__ void GSBasedINTTMerged(unsigned long long a[], unsigned long long psiinv[])
{
    #pragma unroll
    for (int m = (N / 2); m > 0; m /= 2)
    {
        GSBasedINTTInner << <N / nttBlockSize / 2, nttBlockSize >> > (a, psiinv, m);
    }
}

int main()
{
    unsigned long long q, psi, psiinv, ninv;
    unsigned int q_bit;

    getParams(q, psi, psiinv, ninv, q_bit, N);

    unsigned long long psiTable[N];
    unsigned long long psiinvTable[N];
    fillTablePsi128(psi, q, psiinv, psiTable, psiinvTable, N); //gel psi psi

    unsigned long long q2 = (q + 1) >> 1;

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
    cudaMemcpyToSymbol(q2_, &q2, size);

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    uint128_t mu1 = uint128_t::exp2(2 * bit_length);
    mu1 = mu1 / q;
    unsigned long long mu = mu1.low;

    //copy barrett parameters to device
    cudaMemcpyToSymbol(mu_, &mu, size);
    cudaMemcpyToSymbol(bit_length_, &bit_length, size);

    cudaDeviceSynchronize();

    unsigned long long* a = (unsigned long long*)malloc(sizeof(unsigned long long) * N);
    randomArray128(a, N, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned long long* b = (unsigned long long*)malloc(sizeof(unsigned long long) * N);
    randomArray128(b, N, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned long long* refc;
    if (check)
        refc = refPolyMul128(a, b, q, N);

    cudaDeviceSynchronize();

    float total_time = 0;

    float time;
    cudaEvent_t startf, stopf;

    cudaEventCreate(&startf);
    cudaEventCreate(&stopf);
    cudaEventRecord(startf, 0);

    //copy random arrays to gpu
    unsigned long long* d_a;
    cudaMalloc(&d_a, size_array);
    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);
    unsigned long long* d_b;
    cudaMalloc(&d_b, size_array);
    cudaMemcpy(d_b, b, size_array, cudaMemcpyHostToDevice);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    printf("\nTime taken to copy input polynomials to device:  %f ms \n", time);
    total_time += time;

    cudaDeviceSynchronize();

    cudaEventRecord(startf, 0);

    #pragma unroll
    for (int m = 1; m < N; m *= 2)
    {
        CTBasedNTTInner << <N / nttBlockSize / 2, nttBlockSize>> > (d_b, d_psi, m);
    }

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    printf("\nTime taken to perform 1st forward NTT:  %f ms \n", time);
    total_time += time;

    cudaEventRecord(startf, 0);

    #pragma unroll
    for (int m = 1; m < N; m *= 2)
    {
        CTBasedNTTInner << <N / nttBlockSize / 2, nttBlockSize>> > (d_a, d_psi, m);
    }

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    printf("\nTime taken to perform 2nd forward NTT:  %f ms \n", time);
    total_time += time;

    cudaStream_t rest;
    cudaStreamCreate(&rest);

    cudaEventRecord(startf, 0);

    barrett<<<blockCount, blockSize>>>(d_a, d_b);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    printf("\nTime taken to perform coefficient multiplication with barrett:  %f ms \n", time);
    total_time += time;


    cudaEventRecord(startf, 0);

    for (int m = (N / 2); m > 0; m /= 2)
    {
        GSBasedINTTInner << <N / nttBlockSize / 2, nttBlockSize>> > (d_a, d_psiinv, m);
    }

    inverseNTTbarrett << <blockCount, blockSize, 0, 0>> > (d_a);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    printf("\nTime taken to perform inverse NTT:  %f ms \n", time);
    total_time += time;


    cudaEventRecord(startf, 0);

    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    cudaEventDestroy(startf); cudaEventDestroy(stopf);

    printf("\nTime taken to copy results back to host:  %f ms \n", time);
    total_time += time;

    printf("\nTotal execution time:  %f ms \n", total_time);
    
    if (check) //check the correctness of results
    {
        for (int i = 0; i < N; i++)
        {
            if (a[i] != refc[i])
            {
                cout << "error" << endl;

                cout << i << "   " << a[i] << "   " << refc[i] << endl;
            }
                
        }

        free(refc);
    }

    free(a); free(b);

    return 0;
}
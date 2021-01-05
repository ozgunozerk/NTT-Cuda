#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
using std::cout;
using std::endl;

#include "helper.h"
#include "parameter.h"
#include "uint128.h"

#define N 1024 * 4
#define nttBlockSize 1024
#define blockSize 256
#define blockCount N / blockSize
#define check 1

int size_array = sizeof(unsigned long long) * N;
int size = sizeof(unsigned long long);

__constant__ unsigned long long q_;
__constant__ unsigned long long ninv_;

__constant__ unsigned long long mu_;
__constant__ int bit_length_;

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

__device__ __forceinline__ void singleBarrett(uint128_t& a)
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

__device__ __forceinline__ void singleBarrettInv(unsigned long long& a)
{
    register uint128_t aa;

    aa.low = a * ninv_;
    aa.high = __umul64hi(a, ninv_);

    singleBarrett(aa);

    a = aa.low;
}



__global__ void CTBasedNTTInner(unsigned long long global_array[], unsigned long long psi_powers[])
{

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[N];

    #pragma unroll
    for (int iteration_num = 0; iteration_num < __fdividef(N, nttBlockSize); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = global_array[global_tid];
    }

    __syncthreads();

    #pragma unroll
    for (int length = 1; length < N; length *= 2)
    {
        register int step = __fdividef(__fdividef(N, length), 2);

        #pragma unroll
        for (int iteration_num = 0; iteration_num < __fdividef(__fdividef(N, nttBlockSize), 2); iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * nttBlockSize;

            register int psi_step = __fdividef(global_tid, step);
            register int target_index = psi_step * step * 2 + global_tid % step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            temp_storage.high = __umul64hi(temp_storage.low, psi);
            temp_storage.low = temp_storage.low * psi;

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

    for (int iteration_num = 0; iteration_num < __fdividef(N, nttBlockSize); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        global_array[global_tid] = shared_array[global_tid];
    }
}

__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long psiinv_powers[])
{
    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[N];

    #pragma unroll
    for (int iteration_num = 0; iteration_num < __fdividef(N, nttBlockSize); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        shared_array[global_tid] = a[global_tid];
    }

    __syncthreads();

    #pragma unroll
    for (int length = __fdividef(N, 2); length > 0; length = __fdividef(length, 2))
    {
        register int step = __fdividef(__fdividef(N, length), 2);

        #pragma unroll
        for (int iteration_num = 0; iteration_num < __fdividef(__fdividef(N, nttBlockSize), 2); iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;

            register int psi_step = __fdividef(global_tid, step);;
            register int target_index = psi_step * step * 2 + global_tid % step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            shared_array[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register uint128_t temp_storage = first_target_value - second_target_value;

            temp_storage.high = __umul64hi(temp_storage.low, psiinv);
            temp_storage.low = temp_storage.low * psiinv;

            singleBarrett(temp_storage);
            shared_array[target_index + step] = temp_storage.low;
        }

        __syncthreads();
    }
    #pragma unroll
    for (int iteration_num = 0; iteration_num < __fdividef(N, nttBlockSize); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * nttBlockSize;
        a[global_tid] = shared_array[global_tid];
    }
    
}

__global__ void GSBasedINTTMerged(unsigned long long a[], unsigned long long psiinv[])
{
    #pragma unroll
    for (int m = __fdividef(N, 2); m > 0; m = __fdividef(m, 2))
    {
        GSBasedINTTInner <<< __fdividef(__fdividef(N, nttBlockSize), 2), nttBlockSize >>> (a, psiinv);
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

    cudaDeviceSynchronize();

    unsigned long long* a;
    cudaMallocHost(&a, sizeof(unsigned long long) * N);
    randomArray128(a, N, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned long long* b;
    cudaMallocHost(&b, sizeof(unsigned long long) * N);
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

    cudaStream_t ntt1, ntt2;
    cudaStreamCreate(&ntt1);
    cudaStreamCreate(&ntt2);

    cudaEventRecord(startf, 0);

    //copy random arrays to gpu
    unsigned long long* d_a;
    cudaMallocHost(&d_a, size_array);
    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, ntt1);
    unsigned long long* d_b;
    cudaMallocHost(&d_b, size_array);
    cudaMemcpyAsync(d_b, b, size_array, cudaMemcpyHostToDevice, ntt2);

    CTBasedNTTInner << <1, nttBlockSize, 0, ntt1 >> > (d_a, d_psi);
    CTBasedNTTInner << <1, nttBlockSize, 0, ntt2 >> > (d_b, d_psi);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    printf("\nTime taken to copy input polynomials to device and perform forwards NTTs:  %f ms \n", time);
    total_time += time;

    cudaDeviceSynchronize();

    cudaStreamDestroy(ntt1); cudaStreamDestroy(ntt2);

    cudaDeviceSynchronize();

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

    GSBasedINTTInner << <1, nttBlockSize>> > (d_a, d_psiinv);

    inverseNTTbarrett << <blockCount, blockSize>> > (d_a);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    printf("\nTime taken to perform inverse NTT:  %f ms \n", time);
    total_time += time;


    cudaEventRecord(startf, 0);

    cudaMemcpyAsync(a, d_a, size_array, cudaMemcpyDeviceToHost, 0);

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

    cudaFreeHost(a); cudaFreeHost(b);

    return 0;
}

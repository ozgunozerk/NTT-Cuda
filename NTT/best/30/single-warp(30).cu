#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <sstream>
using std::cout;
using std::endl;

#include "helper.h"
#include "parameter.h"

#define N 1024 * 2
#define nttBlockSize 1024
#define blockSize 256
#define blockCount N / blockSize
#define check 0

int size_array = sizeof(unsigned) * N;
int size = sizeof(unsigned);

__constant__ unsigned q_;
__constant__ unsigned ninv_;

__constant__ unsigned mu_;
__constant__ int bit_length_;

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

__global__ void inverseNTTbarrett(unsigned a[])
{
    register int i = blockIdx.x * blockSize + threadIdx.x;
    register int bit_length = bit_length_;

    register unsigned ra = a[i];
    unsigned long long rc, rx;

    rc = (unsigned long long)ra * ninv_;

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

__global__ void CTBasedNTTInner(unsigned a[], unsigned psi_powers[])
{
    register int local_tid = threadIdx.x;

    register int z = psi_powers[local_tid % 32];

    #pragma unroll
    for (int length = 1; length < 17; length *= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < N / nttBlockSize / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;

            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            //register unsigned psi = __shfl_sync(0xffffffff, z, (1 << length) - 1 + (global_tid / (N / (1 << length))));
            register unsigned psi = __shfl_sync(0xffffffff, z, length + psi_step);

            //register unsigned psi = psi_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned long long temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

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

        __syncthreads();
    }

    #pragma unroll
    for (int length = 32; length < N; length *= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * nttBlockSize;

            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            register unsigned psi = psi_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned long long temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

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

        __syncthreads();
    }

}

/*void CTBasedNTTMerged(unsigned long long a[], unsigned long long psi[], cudaStream_t stream)
{
    #pragma unroll
    for (int m = 1; m < N; m *= 2)
    {
        CTBasedNTTInner << <N / nttBlockSize / 2, nttBlockSize, 1, stream >> > (a, psi, m);
    }
}*/

__global__ void GSBasedINTTInner(unsigned a[], unsigned psiinv_powers[])
{
    register int local_tid = threadIdx.x;

    register int z = psiinv_powers[local_tid % 32];

    #pragma unroll
    for (int length = (N / 2); length > 16; length /= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;

            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            register unsigned psiinv = psiinv_powers[length + psi_step];

            register unsigned first_target_value = a[target_index];
            register unsigned second_target_value = a[target_index + step];

            register unsigned target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            a[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage);
            a[target_index + step] = temp_storage;
        }

        __syncthreads();
    }
    #pragma unroll
    for (int length = 16; length > 0; length /= 2)
    {
        register int step = (N / length) / 2;

        #pragma unroll
        for (int iteration_num = 0; iteration_num < (N / nttBlockSize) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;

            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            //register unsigned psiinv = psiinv_powers[length + psi_step];
            register unsigned psiinv = __shfl_sync(0xffffffff, z, length + psi_step);

            register unsigned first_target_value = a[target_index];
            register unsigned second_target_value = a[target_index + step];

            register unsigned target_result = first_target_value + second_target_value;

            if (target_result >= q_)
                target_result -= q_;

            a[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q_;

            register unsigned long long temp_storage = first_target_value - second_target_value;

            temp_storage *= psiinv;

            singleBarrett(temp_storage);
            a[target_index + step] = temp_storage;
        }

        __syncthreads();
    }
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

int main()
{
    cudaSetDevice(2);

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

    unsigned* a;
    cudaMallocHost(&a, sizeof(unsigned) * N);
    randomArray64(a, N, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned* b;
    cudaMallocHost(&b, sizeof(unsigned) * N);
    randomArray64(b, N, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned* d_a;
    cudaMalloc(&d_a, size_array);
    unsigned* d_b;
    cudaMalloc(&d_b, size_array);

    unsigned* refc;
    if (check)
        refc = refPolyMul64(a, b, q, N);

    cudaStream_t ntt1, ntt2;
    cudaStreamCreate(&ntt1);
    cudaStreamCreate(&ntt2);

    //copy random arrays to gpu
    
    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, ntt1);
    cudaMemcpyAsync(d_b, b, size_array, cudaMemcpyHostToDevice, ntt2);

    CTBasedNTTInner << <1, nttBlockSize, 0, ntt1 >> > (d_a, d_psi);
    CTBasedNTTInner << <1, nttBlockSize, 0, ntt2 >> > (d_b, d_psi);

    barrett<<<blockCount, blockSize, 0, 0>>>(d_a, d_b);    

    GSBasedINTTInner << <1, nttBlockSize, 0, 0>> > (d_a, d_psiinv);
 
    inverseNTTbarrett << <blockCount, blockSize, 0, 0>> > (d_a);
    
    cudaMemcpyAsync(a, d_a, size_array, cudaMemcpyDeviceToHost, ntt1);    

    cudaDeviceSynchronize();

    cudaStreamDestroy(ntt1); cudaStreamDestroy(ntt2);
    
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

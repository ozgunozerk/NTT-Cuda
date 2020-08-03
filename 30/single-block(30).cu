#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <sstream>
using std::cout;
using std::endl;

#include "helper.h"
#include "parameter.h"

#define N 1024 * 4
#define nttBlockSize 1024
#define blockSize 256
#define blockCount N / blockSize
#define check 1

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

    #pragma unroll
    for (int length = __fdividef(N, 2); length > 0; length = __fdividef(length, 2))
    {
        register int step = __fdividef(__fdividef(N, length), 2);

        #pragma unroll
        for (int iteration_num = 0; iteration_num < __fdividef(__fdividef(N, nttBlockSize), 2); iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * nttBlockSize;
            register int psi_step = __fdividef(global_tid, step);
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

    cudaDeviceSynchronize();

    unsigned* a = (unsigned*)malloc(sizeof(unsigned) * N);
    randomArray64(a, N, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned* b = (unsigned*)malloc(sizeof(unsigned) * N);
    randomArray64(b, N, q - 1); //fill array with random numbers between 0 and q - 1

    unsigned* refc;
    if (check)
        refc = refPolyMul64(a, b, q, N);

    cudaDeviceSynchronize();

    float total_time = 0;

    float time;
    cudaEvent_t startf, stopf;

    cudaEventCreate(&startf);
    cudaEventCreate(&stopf);
    cudaEventRecord(startf, 0);

    //copy random arrays to gpu
    unsigned* d_a;
    cudaMalloc(&d_a, size_array);
    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);
    unsigned* d_b;
    cudaMalloc(&d_b, size_array);
    cudaMemcpy(d_b, b, size_array, cudaMemcpyHostToDevice);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    printf("\nTime taken to copy input polynomials to device:  %f ms \n", time);
    total_time += time;

    cudaDeviceSynchronize();

    cudaStream_t ntt1, ntt2;
    cudaStreamCreate(&ntt1);
    cudaStreamCreate(&ntt2);

    cudaEventRecord(startf, 0);

    CTBasedNTTInner << <1, nttBlockSize, 0, 0 >> > (d_a, d_psi);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    printf("\nTime taken to perform 1st forward NTT:  %f ms \n", time);
    total_time += time;

    cudaEventRecord(startf, 0);

    CTBasedNTTInner << <1, nttBlockSize, 0, 0 >> > (d_b, d_psi);

    cudaEventRecord(stopf, 0);
    cudaEventSynchronize(stopf);
    cudaEventElapsedTime(&time, startf, stopf);

    cudaDeviceSynchronize();

    cudaStreamDestroy(ntt1); cudaStreamDestroy(ntt2);

    printf("\nTime taken to perform 2nd forward NTT:  %f ms \n", time);
    total_time += time;

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
                //cout << "error" << endl;
                cout << "i:   " << i << ", Computed result:   " << a[i] << ", Reference result:   " << refc[i] << endl;
            }
                
        }

        free(refc);
    }

    free(a); free(b);

    return 0;
}

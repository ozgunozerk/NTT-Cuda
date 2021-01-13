#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

#include "helper.h"
#include "parameter.h"
#include "ntt_60bit.cuh"
#include "poly_arithmetic.cuh"

#define check 0

int main()
{
    unsigned N = 1024 * 2;

    int size_array = sizeof(unsigned long long) * N;
    int size = sizeof(unsigned long long);

    unsigned long long q, psi, psiinv, ninv;
    unsigned int q_bit;

    getParams(q, psi, psiinv, ninv, q_bit, N);

    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi128(psi, q, psiinv, psiTable, psiinvTable, N); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    unsigned long long* psi_powers, * psiinv_powers;

    cudaMalloc(&psi_powers, size_array);
    cudaMalloc(&psiinv_powers, size_array);

    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);

    cout << "n = " << N << endl;
    cout << "q = " << q << endl;
    cout << "Psi = " << psi << endl;
    cout << "Psi Inverse = " << psiinv << endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    uint128_t mu1 = uint128_t::exp2(bit_length * 2);
    unsigned long long mu = (mu1 / q).low;

    unsigned long long* a;
    cudaMallocHost(&a, sizeof(unsigned long long) * N);
    randomArray128(a, N, q); //fill array with random numbers between 0 and q - 1

    unsigned long long* b;
    cudaMallocHost(&b, sizeof(unsigned long long) * N);
    randomArray128(b, N, q); //fill array with random numbers between 0 and q - 1

    unsigned long long* d_a;
    cudaMalloc(&d_a, size_array);
    unsigned long long* d_b;
    cudaMalloc(&d_b, size_array);

    unsigned long long* refc;
    if (check)
        refc = refPolyMul128(a, b, q, N);

    cudaStream_t ntt1, ntt2;
    cudaStreamCreate(&ntt1);
    cudaStreamCreate(&ntt2);

    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, ntt1);
    cudaMemcpyAsync(d_b, b, size_array, cudaMemcpyHostToDevice, ntt2);

    forwardNTTdouble(d_a, d_b, N, ntt1, ntt2, q, mu, bit_length, psi_powers);
    barrett << <N / 256, 256 >> > (d_a, d_b, q, mu, bit_length);
    inverseNTT(d_a, N, ntt1, q, mu, bit_length, psiinv_powers);

    cudaMemcpyAsync(a, d_a, size_array, cudaMemcpyDeviceToHost, 0);

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



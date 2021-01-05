#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

#include "helper.h"
#include "parameter.h"
#include "ntt_30bit.cuh"

#define check 1

int main()
{
    unsigned N = 1024 * 64;

    int size_array = sizeof(unsigned) * N;
    int size = sizeof(unsigned);

    unsigned q, psi, psiinv, ninv;
    unsigned int q_bit;

    getParams30(q, psi, psiinv, ninv, q_bit, N);

    unsigned* psiTable = (unsigned*)malloc(size_array);
    unsigned* psiinvTable = (unsigned*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, N); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    unsigned* psi_powers, * psiinv_powers;

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
    double mu1 = powl(2, 2 * bit_length);
    unsigned mu = mu1 / q;

    unsigned* a;
    cudaMallocHost(&a, sizeof(unsigned) * N);
    randomArray64(a, N, q); //fill array with random numbers between 0 and q - 1

    unsigned* b;
    cudaMallocHost(&b, sizeof(unsigned) * N);
    randomArray64(b, N, q); //fill array with random numbers between 0 and q - 1

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

    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, ntt1);
    cudaMemcpyAsync(d_b, b, size_array, cudaMemcpyHostToDevice, ntt2);

    forwardNTTdouble(d_a, d_b, N, ntt1, ntt2, q, mu, bit_length, psi_powers);
    barrett_30bit << <N / 256, 256 >> > (d_a, d_b, q, mu, bit_length);
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



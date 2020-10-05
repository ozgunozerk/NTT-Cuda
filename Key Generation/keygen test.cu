#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

#include "helper.h"
#include "parameter.h"

#include "poly_arithmetic.cuh"
#include "distributions.cuh"

#define check 0

int main2()
{
    int n = 1024 * 4;

    vector<unsigned long long> q = { 68719403009, 68719230977, 137438822401 };
    vector<unsigned long long> psi_roots = { 24250113, 29008497, 8625844 };
    vector<unsigned> q_bit_lengths = { 36, 36, 37 };
    unsigned q_amount = q.size();

    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * q_amount);
    for (int i = 0; i < q_amount; i++)
        cudaStreamCreate(&streams[i]);
    
    unsigned char* in;
    cudaMalloc(&in, (sizeof(char) + sizeof(unsigned) + sizeof(unsigned long long)) * q_amount * n);

    unsigned long long** secret_key = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&secret_key[i], sizeof(unsigned long long) * n);
    }
    
    unsigned long long*** public_key = (unsigned long long***)malloc(sizeof(unsigned long long**) * 2);
    public_key[0] = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    public_key[1] = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < q_amount; j++)
        {
            cudaMalloc(&public_key[i][j], sizeof(unsigned long long) * n);
        }
    }
    unsigned long long** temp = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&temp[i], sizeof(unsigned long long) * q_amount * n);
    }

    generate_random(in, (sizeof(char) + sizeof(unsigned) + sizeof(unsigned long long)) * q_amount * n, streams[0]);

    for (int i = 0; i < q_amount; i++)
    {
        ternary_dist(in + i * n, secret_key[i], n, streams[i], q[i]);
    }

    unsigned long long* output;
    cudaMallocHost(&output, sizeof(unsigned long long) * n);
    cudaMemcpyAsync(output, secret_key[2], sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost, streams[2]);

    cudaDeviceSynchronize();

    /*int c = 0, v = 0, b = 0;
    for (int i = 0; i < n; i++)
    {
        if (output[i] == (q[2] - 1))
            c++;
        else if (output[i] == 0)
            v++;
        else if (output[i] == 1)
            b++;
    }

    printf("%d, %d, %d\n", c, v, b);*/


    return 0;
}

int main3()
{
    //unsigned N = atoi(argv[1]);
    unsigned N = 1024 * 8;

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
    uint128_t mu1 = uint128_t::exp2(2 * bit_length);
    mu1 = mu1 / q;

    unsigned long long mu = mu1.low;

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

    /*cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, ntt1);
    cudaMemcpyAsync(d_b, b, size_array, cudaMemcpyHostToDevice, ntt2);

    full_poly_mul_device(d_a, d_b, N, ntt1, ntt2, q, mu, bit_length, psi_powers, psiinv_powers);

    cudaMemcpyAsync(a, d_a, size_array, cudaMemcpyDeviceToHost, 0);*/

    unsigned long long* result = full_poly_mul(a, b, d_a, d_b, N, ntt1, ntt2, q, mu, bit_length, psi_powers, psiinv_powers);

    cudaDeviceSynchronize();

    cudaStreamDestroy(ntt1); cudaStreamDestroy(ntt2);

    if (check) //check the correctness of results
    {
        for (int i = 0; i < N; i++)
        {
            if (result[i] != refc[i])
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

/*int main()
{
    int n = 1024 * 4;

    unsigned long long q = 68719403009;

    unsigned char* in;
    unsigned long long* out;
    unsigned long long* host;

    cudaMallocHost(&host, n * sizeof(unsigned long long));

    cudaMalloc(&in, n * sizeof(char)); cudaMalloc(&out, n * sizeof(unsigned long long));

    cudaStream_t stream; cudaStreamCreate(&stream);

    generate_random(in, n, stream, 1);
    uniform_dist((unsigned long long*)in, out, n, stream, q);

    cudaMemcpyAsync(host, out, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);

    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++)
    {
        printf("%llu\n", host[i]);
    }

    return 0;
}*/



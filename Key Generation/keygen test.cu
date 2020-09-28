#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

#include "helper.h"
#include "parameter.h"

//#include "poly_arithmetic.cuh"
#include "distributions.cuh"

#define check 0

int main()
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
}

/*int main()
{
    int n = 1024 * 4;

    vector<unsigned long long> q = { 68719403009, 68719230977, 137438822401 };
    cudaStream_t streams[3];
    unsigned q_amount = q.size();

    unsigned char** in = (unsigned char**)malloc(sizeof(unsigned char*) * q_amount);
    unsigned long long** out = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&in[i], sizeof(unsigned char) * n);
        cudaMalloc(&out[i], sizeof(unsigned long long) * n);
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        ternary_dist(in[i], out[i], n, streams[i], q[i]);
    }

    unsigned long long* host;
    cudaMallocHost(&host, sizeof(unsigned long long) * n);
    cudaMemcpyAsync(host, out[2], sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost, streams[0]);

    cudaDeviceSynchronize();

    unsigned c = 0, v = 0, b = 0;
    for (int i = 0; i < n; i++)
    {
        if (host[i] == (q[2] - 1))
            c++;
        if (host[i] == 0)
            v++;
        if (host[i] == 1)
            b++;
    }

    printf("q - 1 count: %d, 0 count: %d, 1 count: %d\n", c, v, b);

    return 0;
}*/

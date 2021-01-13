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

__global__ void ternary_test(unsigned char* in, int* out)
{
    int i = threadIdx.x + blockIdx.x * convertBlockSize;

    register float d = (float)in[i];

    d /= (255.0f / 3);

    if (d >= 2)
        out[i] = 1;
    else if (d >= 1)
        out[i] = 0;
    else
        out[i] = -1;
}

__global__ void ternary_test2(unsigned char* in, int* out)
{
    int i = threadIdx.x + blockIdx.x * convertBlockSize;

    register float d = (float)in[i];

    d /= (255.0f / 3);

    out[i] = int(d) - 1;
}

int main()
{
    int n = 1024 * 333333;

    unsigned char* in;
    int* out;
    int* host;

    cudaMallocHost(&host, n * sizeof(int));

    cudaMalloc(&in, n * sizeof(char)); cudaMalloc(&out, n * sizeof(int));

    generate_random_default(in, n);
    ternary_test2<<<n/convertBlockSize, convertBlockSize>>>(in, out);

    cudaMemcpy(host, out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    int sum0 = 0, sum1 = 0, sum2 = 0;

    for (int i = 0; i < n; i++)
    {
        if (host[i] == -1)
            sum0++;
        if (host[i] == 0)
            sum1++;
        if (host[i] == 1)
            sum2++;
    }

    printf("Number of -1 generated: %d\n", sum0);
    printf("Number of 0 generated: %d\n", sum1);
    printf("Number of 1 generated: %d\n", sum2);

    return 0;
}



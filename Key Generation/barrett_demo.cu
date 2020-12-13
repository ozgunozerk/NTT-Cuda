#include <iostream>
using std::cout;
using std::endl;

#include "helper.h"
#include "poly_arithmetic.cuh"

#define n 1024 * 32
#define nthreads 256

void main()
{
	unsigned long long q = 36028797017456641; //55-bit prime
	unsigned q_bit_length = log2(q) + 1; //calculate bit-length of q
	unsigned long long mu; //precomputed factor

	uint128_t mu1 = uint128_t::exp2(2 * q_bit_length);
	mu1 = mu1 / q;
	mu = mu1.low; //mu = 2^(2 * q_bit_length) / q

	unsigned long long a[n], b[n]; //allocate memory for a and b on CPU

	randomArray128(a, n, q); //fill a and b with random numbers from 0 to q - 1
	randomArray128(b, n, q);

	unsigned long long* device_a, * device_b;

	cudaMalloc(&device_a, sizeof(unsigned long long) * n); //allocate memory for a and b on GPU
	cudaMalloc(&device_b, sizeof(unsigned long long) * n);

	cudaMemcpy(device_a, a, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice); //copy the randomly generated arrays from CPU to GPU
	cudaMemcpy(device_b, b, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);

	barrett<<<n / nthreads, nthreads >>>(device_a, device_b, q, mu, q_bit_length); //compute a[i] * b[i] % q

	unsigned long long check[n];
	cudaMemcpy(check, device_a, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost); //copy results back to CPU

	//cudaDeviceSynchronize(); //wait for the GPU operations to complete

	bool correct = 1; //check the correctness of the results

	for (int i = 0; i < n; i++)
	{
		uint128_t mul = host64x2(a[i], b[i]);
		mul = mul % q;
		unsigned long long result = mul.low;

		if (result != check[i])
		{
			correct = 0;
			break;
		}
	}

	if (correct)
		cout << "Correct" << endl;
	else
		cout << "Wrong" << endl;
}
#include "poly_arithmetic.cuh"

__global__ void barrett(unsigned long long a[], const unsigned long long b[], unsigned long long q, unsigned long long mu, int qbit)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);

    a[i] = rc.low;
}

__global__ void poly_add(unsigned long long a[], const unsigned long long b[], unsigned long long q)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i] + b[i];

    if (ra > (q << 2))
        ra -= (q << 1);

    a[i] = ra;
}

__global__ void poly_sub(unsigned long long a[], const unsigned long long b[], unsigned long long q)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    if (ra < rb)
        ra += (q << 1);

    a[i] = ra;
}

__host__ unsigned long long* full_poly_mul(unsigned long long* host_a, unsigned long long* host_b, unsigned long long* device_a, unsigned long long* device_b, unsigned N, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{
    size_t array_size = sizeof(unsigned long long) * N;
    unsigned long long* result = (unsigned long long*)malloc(array_size);

    cudaMemcpyAsync(device_a, host_a, array_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(device_b, host_b, array_size, cudaMemcpyHostToDevice, stream2);

    forwardNTTdouble(device_a, device_b, N, stream1, stream2, q, mu, bit_length, psi_powers);

    barrett << <N / 256, 256, 0, stream2 >> > (device_a, device_b, q, mu, bit_length);

    cudaMemcpyAsync(result, device_a, array_size, cudaMemcpyDeviceToHost, stream2);

    return result;
}

__host__ void full_poly_mul_device(unsigned long long* device_a, unsigned long long* device_b, unsigned N, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{
    forwardNTTdouble(device_a, device_b, N, stream1, stream2, q, mu, bit_length, psi_powers);

    barrett << <N / 256, 256, 0, stream2 >> > (device_a, device_b, q, mu, bit_length);
}

__host__ void half_poly_mul_device(unsigned long long* device_a, unsigned long long* device_b, unsigned N, cudaStream_t& stream, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{
    forwardNTT(device_b, N, stream, q, mu, bit_length, psi_powers);

    barrett << <N / 256, 256, 0, stream >> > (device_a, device_b, q, mu, bit_length);
}

__host__ void poly_add_device(unsigned long long* device_a, const unsigned long long* device_b, unsigned N, cudaStream_t& stream, unsigned long long q)
{
    poly_add << <N / 256, 256, 0, stream >> > (device_a, device_b, q);
}

__host__ void poly_sub_device(unsigned long long* device_a, const unsigned long long* device_b, unsigned N, cudaStream_t& stream, unsigned long long q)
{
    poly_sub << <N / 256, 256, 0, stream >> > (device_a, device_b, q);
}

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "uint128.h"
#include "ntt_60bit.cuh"

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

    if (rc.low < q)
        a[i] = rc.low;
    else
        a[i] = rc.low - q;
}

__global__ void barrett_batch(unsigned long long a[], const unsigned long long b[], unsigned n, unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_cons[index];
    unsigned long long mu = mu_cons[index];
    int qbit = q_bit_cons[index];

    register int i = blockIdx.x * 256 + threadIdx.x + blockIdx.y * n;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);

    if (rc.low < q)
        a[i] = rc.low;
    else
        a[i] = rc.low - q;
}

__global__ void barrett_batch_3param(unsigned long long c[], unsigned long long a[], const unsigned long long b[], unsigned n, unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_cons[index];
    unsigned long long mu = mu_cons[index];
    int qbit = q_bit_cons[index];

    register int i = blockIdx.x * 256 + threadIdx.x + blockIdx.y * n;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);

    if (rc.low < q)
        c[i] = rc.low;
    else
        c[i] = rc.low - q;
}

__global__ void barrett_int(unsigned long long a[], const unsigned long long b, unsigned long long q, unsigned long long mu, int qbit)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b;

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);

    if (rc.low < q)
        a[i] = rc.low;
    else
        a[i] = rc.low - q;

}

__global__ void mod_t(unsigned long long a[], const unsigned long long b, unsigned long long t)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b;

    uint128_t rc;

    mul64(ra, rb, rc);

    register unsigned mask = t - 1;
    a[i] = rc.low & mask;
}

__global__ void poly_add(unsigned long long a[], const unsigned long long b[], unsigned long long q)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i] + b[i];

    if (ra > q)
        ra -= q;

    a[i] = ra;
}

__global__ void poly_add_integer(unsigned long long a[], unsigned long long b, unsigned long long q)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i] + b;

    if (ra > q)
        ra -= q;

    a[i] = ra;
}

__global__ void poly_sub(unsigned long long a[], const unsigned long long b[], unsigned long long q)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    if (ra < rb)
        ra += q;

    a[i] = ra;
}

__global__ void divide_and_round_q_last_inplace_loop(unsigned long long* input_poly, unsigned long long* rns_poly_minus1, unsigned long long base_q_i,
    unsigned long long half_mod, unsigned long long inv_q_last_mod_q_i, unsigned long long base_q_i_mu, int base_q_i_qbit)
{
    register int i = blockIdx.x * 256 + threadIdx.x;  // get an ID to each thread

    unsigned long long temp_poly_i = rns_poly_minus1[i] % base_q_i;
    // get the last polynomials respective index with = rns_poly_minus1[i] 
    // get the the base q of the polynomial (one of other polynomials other than the last one) = base_q_i 
    // store the result in a variable = temp_poly_i

    if (temp_poly_i < half_mod)  // mod operation for safe substraction on line 139
        temp_poly_i += base_q_i;

    temp_poly_i -= half_mod; // substract half_modulus from the index of last polynomial

    if (input_poly[i] < temp_poly_i)  // now we gonna substract that computed value from other polynomials
        input_poly[i] += base_q_i;  // so we have to ensure substraction safety (underflow)

    input_poly[i] -= temp_poly_i;
    // substract the last_polynomials respective calculated value = temp_poly_i
    // from the respective index of the other polynomial = input_poly[i]

    uint128_t mult;
    mul64(input_poly[i], inv_q_last_mod_q_i, mult);
    // multiply the input_poly[i] with:
    // inverse of last polynomials q
    // to the mod of respective polynomials q
    // which is: inv_q_last_mod_q_i 
    // :)

    singleBarrett(mult, base_q_i, base_q_i_mu, base_q_i_qbit);
    // we might have fucked up, so apply mod again

    input_poly[i] = mult.low;  // store the result in the given input_polynomial
}

// c0[0] is given as the resulting poly because it is unused now
__global__ void fast_convert_array_kernel_t(unsigned long long* input_poly, unsigned long long* result_poly, unsigned long long t, unsigned long long* base_change_matrix_device, unsigned q_amount, unsigned n)
{
    register int k = blockIdx.x * 256 + threadIdx.x;

    result_poly[k] = 0;
    unsigned mask = t - 1;
    unsigned long long tmp;
#pragma unroll
    for (int i = 0; i < q_amount; i++)
    {
        tmp = input_poly[k + i * n] * base_change_matrix_device[i];
        tmp = tmp & mask; //taking mod t (works since t is a power of 2)

        result_poly[k] += tmp;
        //printf("i = %d, value = %llu\n", i, base_change_matrix_device[i]);
    }
    result_poly[k] = result_poly[k] & mask;
}

// c0[1] is now given as the resulting poly
__global__ void fast_convert_array_kernel_gamma(unsigned long long* input_poly, unsigned long long* result_poly, unsigned long long gamma, unsigned long long* base_change_matrix_device, unsigned q_amount, int gamma_bit_length, unsigned long long mu_gamma, unsigned n)
{
    register int k = blockIdx.x * 256 + threadIdx.x;

    result_poly[k + n] = 0;
    uint128_t tmp;
#pragma unroll
    for (int i = 0; i < q_amount; i++)
    {
        mul64(input_poly[k + i * n], base_change_matrix_device[i + q_amount], tmp);
        singleBarrett(tmp, gamma, mu_gamma, gamma_bit_length);
        result_poly[k + n] = (result_poly[k + n] + tmp.low) % gamma;
    }
    result_poly[k + n] = result_poly[k + n] % gamma;
}

__global__ void dec_round_kernel(unsigned long long* input_poly, unsigned long long* result_poly, unsigned long long t, unsigned long long gamma, unsigned long long gamma_div_2, unsigned n)
{
    register int i = blockIdx.x * 256 + threadIdx.x;
    register unsigned long long mask = t - 1;

    if (input_poly[i + n] > gamma_div_2)  // poly(g) >= (g >> 1) ?? buraya kesin bakalim, ferhat >= demis, biz > yapmisiz
        result_poly[i] = (input_poly[i] + (gamma - input_poly[i + n])) & mask;  // extra stuff: add gamma to the polynomial
    else
        result_poly[i] = (input_poly[i] - input_poly[i + n]) & mask;  // regular stuff: don't add gamma to the polynomial, rest is same

}

__host__ void dec_round(unsigned long long* input_poly, unsigned long long* result_poly, unsigned long long t, unsigned long long gamma, unsigned long long gamma_div_2, unsigned n, cudaStream_t& stream)
{
    dec_round_kernel << <n / 256, 256, 0, stream >> > (input_poly, result_poly, t, gamma, gamma_div_2, n);
}

__host__ void fast_convert_array_kernels(unsigned long long* input_poly, unsigned long long* result_poly, unsigned long long t, unsigned long long* base_change_matrix_device, unsigned q_amount, unsigned long long gamma, int gamma_bit_length, unsigned long long mu_gamma, cudaStream_t& stream1, cudaStream_t& stream2, unsigned n)
{
    fast_convert_array_kernel_t << <n / 256, 256, 0, 0 >> > (input_poly, result_poly, t, base_change_matrix_device, q_amount, n);

    fast_convert_array_kernel_gamma << <n / 256, 256, 0, stream2 >> > (input_poly, result_poly, gamma, base_change_matrix_device, q_amount, gamma_bit_length, mu_gamma, n);
}

__host__ unsigned long long* full_poly_mul(unsigned long long* host_a, unsigned long long* host_b, unsigned long long* device_a, unsigned long long* device_b, unsigned n, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers, unsigned long long* psiinv_powers)
{
    size_t array_size = sizeof(unsigned long long) * n;
    unsigned long long* result = (unsigned long long*)malloc(array_size);

    cudaMemcpyAsync(device_a, host_a, array_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(device_b, host_b, array_size, cudaMemcpyHostToDevice, stream2);

    forwardNTTdouble(device_a, device_b, n, stream1, stream2, q, mu, bit_length, psi_powers);

    barrett << <n / 256, 256, 0, stream2 >> > (device_a, device_b, q, mu, bit_length);

    inverseNTT(device_a, n, stream2, q, mu, bit_length, psiinv_powers);

    cudaMemcpyAsync(result, device_a, array_size, cudaMemcpyDeviceToHost, stream2);

    return result;
}

__host__ void full_poly_mul_device(unsigned long long* device_a, unsigned long long* device_b, unsigned n, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{
    forwardNTTdouble(device_a, device_b, n, stream1, stream2, q, mu, bit_length, psi_powers);

    barrett << <n / 256, 256, 0, stream2 >> > (device_a, device_b, q, mu, bit_length);
}

__host__ void half_poly_mul_device(unsigned long long* device_a, unsigned long long* device_b, unsigned n, cudaStream_t& stream, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers, unsigned long long* psiinv_powers)
{
    forwardNTT(device_a, n, stream, q, mu, bit_length, psi_powers);

    barrett << <n / 256, 256, 0, stream >> > (device_a, device_b, q, mu, bit_length);

    inverseNTT(device_a, n, stream, q, mu, bit_length, psiinv_powers);
}

__host__ void poly_add_device(unsigned long long* device_a, const unsigned long long* device_b, unsigned n, cudaStream_t& stream, unsigned long long q)
{
    poly_add << <n / 256, 256, 0, stream >> > (device_a, device_b, q);
}

__host__ void poly_mul_int(unsigned long long* device_a, const unsigned long long b, unsigned n, cudaStream_t& stream, unsigned long long q, unsigned long long mu, int bit_length)
{
    barrett_int << <n / 256, 256, 0, stream >> > (device_a, b, q, mu, bit_length);
}

__host__ void poly_mul_int_t(unsigned long long* device_a, const unsigned long long b, unsigned n, cudaStream_t& stream, unsigned long long t)
{
    mod_t << <n / 256, 256, 0, stream >> > (device_a, b, t);
}

__host__ void poly_sub_device(unsigned long long* device_a, const unsigned long long* device_b, unsigned n, cudaStream_t& stream, unsigned long long q)
{
    poly_sub << <n / 256, 256, 0, stream >> > (device_a, device_b, q);
}

__global__ void poly_negate(unsigned long long* a, unsigned long long q)
{
    register int i = blockIdx.x * 256 + threadIdx.x;

    a[i] = q - a[i];
    a[i] *= a[i] != q; // transform a[i] to 0 if a[i] is equal to q
}

__host__ void poly_negate_device(unsigned long long* device_a, unsigned n, cudaStream_t& stream, unsigned long long q)
{
    poly_negate << <n / 256, 256, 0, stream >> > (device_a, q);
}

__host__ void poly_add_integer_device(unsigned long long* device_a, unsigned long long b, unsigned n, cudaStream_t& stream, unsigned long long q)
{
    poly_add_integer << <n / 256, 256, 0, stream >> > (device_a, b, q);
}

__host__ void poly_add_integer_device_default(unsigned long long* device_a, unsigned long long b, unsigned n, unsigned long long q)
{
    poly_add_integer << <n / 256, 256, 0, 0 >> > (device_a, b, q);
}
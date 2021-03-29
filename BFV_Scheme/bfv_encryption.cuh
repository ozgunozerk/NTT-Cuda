#pragma once

#include <vector>
#include <cstdio>
#include <stdio.h>
#include <cuda.h>
using std::vector;

#include "ntt_60bit.cuh"
#include "uint128.h"
#include "salsa_common.h"
#include "distributions.cuh"
#include "poly_arithmetic.cuh"

#define small_block 128

__global__ void convert_ternary_gaussian_x2(unsigned char* in, unsigned long long* out_t1, unsigned long long* out_e1, unsigned n, int q_amount)
{
    int i = blockIdx.x * convertBlockSize + threadIdx.x;

    //ternary

    float d = (float)in[i % n];

    d /= (255.0f / 3);

    /*if (d >= 2)
        out_t1[i] = 1;
    else if (d >= 1)
        out_t1[i] = 0;
    else
        out_t1[i] = q_cons[i / n] - 1;*/

    int b = int(d) - 1;
    out_t1[i] = (b < 0) * q_cons[i / n] + b;

    /*if (d >= 2)
        out_t1[i + n * q_amount] = 1;
    else if (d >= 1)
        out_t1[i + n * q_amount] = 0;
    else
        out_t1[i + n * q_amount] = q_cons[i / n] - 1;*/

    b = int(d) - 1;
    out_t1[i + n * q_amount] = (b < 0) * q_cons[i / n] + b;

    // gaussian

    unsigned* in_u = (unsigned*)(in + n);
    d = in_u[i % n];

    d /= 4294967295;

    if (d == 0)
        d += 1.192092896e-07F;
    else if (d == 1)
        d -= 1.192092896e-07F;

    d = normcdfinvf(d);

    d = d * (float)dstdev + dmean;

    if (d > 19.2)
    {
        d = 19.2;
    }
    else if (d < -19.2)
    {
        d = -19.2;
    }

    int dd = (int)d;

    if (dd < 0)
        out_e1[i] = q_cons[i / n] + dd;
    else
        out_e1[i] = dd;

    in_u = (unsigned*)(in + n * 5);
    d = in_u[i % n];

    d /= 4294967295;

    if (d == 0)
        d += 1.192092896e-07F;
    else if (d == 1)
        d -= 1.192092896e-07F;

    d = normcdfinvf(d);

    d = d * (float)dstdev + dmean;

    if (d > 19.2)
    {
        d = 19.2;
    }
    else if (d < -19.2)
    {
        d = -19.2;
    }

    dd = (int)d;

    if (dd < 0)
        out_e1[i + n * q_amount] = q_cons[i / n] + dd;
    else
        out_e1[i + n * q_amount] = dd;

}

__global__ void divide_and_round_q_last_inplace_add_x2(unsigned long long* c, unsigned n, int q_amount)
{
    unsigned long long last_modulus = q_cons[q_amount - 1];  // get the last q from the array
    unsigned long long half_last_modulus = last_modulus >> 1;  // divide it by 2

    int i = blockIdx.x * small_block + threadIdx.x;

    unsigned long long ra = c[n * (q_amount - 1) + i % n + (n * q_amount) * (i >= n)];
    ra += half_last_modulus;

    if (ra >= last_modulus)
        ra -= last_modulus;

    c[n * (q_amount - 1) + i % n + (n * q_amount) * (i >= n)] = ra;
}

__global__ void divide_and_round_q_last_inplace_loop_xq(unsigned long long* c, unsigned q_amount, unsigned n)
{
    int i = blockIdx.x * small_block + threadIdx.x;
    int i_i = i % n;

    unsigned long long last_modulus = q_cons[q_amount - 1];  // get the last q from the array
    unsigned long long half_last_modulus = last_modulus >> 1;  // divide it by 2

    unsigned index = (i % (n * (q_amount - 1))) / n;

    unsigned long long q = q_cons[index];
    unsigned long long mu = mu_cons[index];
    int q_bit = q_bit_cons[index];
    unsigned long long half_mod = half_last_modulus % q;

    unsigned long long inv_q_last_mod_q = inv_q_last_mod_q_cons[index];

    unsigned second_half = i >= (n * (q_amount - 1));
    unsigned division = (i - n * second_half * (q_amount - 1)) / n;
    unsigned long long* rns_poly_minus1 = c + second_half * (n * q_amount) + n * (q_amount - 1);
    unsigned long long* input_poly = c + second_half * (n * q_amount) + n * division;

    unsigned long long temp_poly_i = rns_poly_minus1[i_i] % q;
    // get the last polynomials respective index with = rns_poly_minus1[i] 
    // get the the base q of the polynomial (one of other polynomials other than the last one) = base_q_i 
    // store the result in a variable = temp_poly_i

    if (temp_poly_i < half_mod)  // mod operation for safe substraction on line 139
        temp_poly_i += q;

    temp_poly_i -= half_mod; // substract half_modulus from the index of last polynomial

    if (input_poly[i_i] < temp_poly_i)  // now we gonna substract that computed value from other polynomials
        input_poly[i_i] += q;  // so we have to ensure substraction safety (underflow)

    input_poly[i_i] -= temp_poly_i;
    // substract the last_polynomials respective calculated value = temp_poly_i
    // from the respective index of the other polynomial = input_poly[i]

    uint128_t mult;
    mul64(input_poly[i_i], inv_q_last_mod_q, mult);
    // multiply the input_poly[i] with:
    // inverse of last polynomials q
    // to the mod of respective polynomials q
    // which is: inv_q_last_mod_q_i 
    // :)

    singleBarrett(mult, q, mu, q_bit);
    // we might have fucked up, so apply mod again

    input_poly[i_i] = mult.low;  // store the result in the given input_polynomial
}

__global__ void poly_add_xq(unsigned long long* c, unsigned long long* e, unsigned n, int q_amount)
{
    int i = blockIdx.x * small_block + threadIdx.x;
    int j = blockIdx.y;

    unsigned long long ra = c[i + (n * q_amount) * blockIdx.y] + e[i + (n * q_amount) * blockIdx.y];

    if (ra > q_cons[i / n])
        ra -= q_cons[i / n];

    c[i + (n * q_amount) * blockIdx.y] = ra;
}

__global__ void weird_m_stuff(unsigned long long* m_poly, unsigned long long* c0, unsigned long long t, unsigned long long* qi_div_t_rns_array_device,
    unsigned long long* q_array_device, unsigned q_amount, unsigned n) // q_mod_t is taken as 1
{
    register int j = blockIdx.x * 256 + threadIdx.x;

    // second reminder: q mod t is assumed 1, multiplying the polynomial with that value becomes unnecessary. 
    // if that's not the case though, include that operation in right here
    unsigned long long numerator = m_poly[j] + ((t + 1) >> 1);  // add t to 1, then divide it by 2, add that value to each element of the polynomial (only for m bits)
    unsigned long long fix = numerator / t;  // divide that value with t, (we assume t is a perfect power of 2, so we can apply shift instead of division with log2t)

#pragma unroll
    for (int i = 0; i < q_amount - 1; i++)  // for every polynomial, except the last one
    {
        c0[j + i * n] = (c0[j + i * n] + ((m_poly[j] * qi_div_t_rns_array_device[i]) + fix)) % q_array_device[i];
        // c0 is a flattened array, we need i*n for reaching the polynomial we want to, then [j] for reaching to the respective element in the polynomial
        // qi_div_t_rns_array_device is storing the rns representation of each qi, divided by t,
        // to be more specific, it's storing the result of base_q value, divided by t
        // but its represented as rns, so its value is computed by, [base_q / t] mod of each qi 
        // qi's are the q's in the base_q
    }
}


	/*
	ENCRYPTION:
	C is the ciphertext. It is the concetanation of c0 and c1. They both have (n * q_amount) items, so, c has (n * q_amount * 2) elements.
	However, in the end, last q's in the RNS system are dropped => c0 and c1 having the size of (n * (q_amount-1) ). Although, relocating space would require extra work and
	is inefficient, we do not resize the array c. Last parts of the c0 and c1 are simply ignored, which are c[8192:12288] and c[20480:24576].
	This is approach is also followed in Decryption (ignoring the irrelevant parts). Read Decryption's comments for more detail
	*/
void encryption_rns(unsigned long long* c, unsigned long long* public_key, unsigned char* in, unsigned long long** u, unsigned long long* e, unsigned n,
    cudaStream_t streams[], unsigned long long* q, vector<unsigned> q_bit_lengths,
    vector<unsigned long long> mu_array, vector<unsigned long long> inv_q_last_mod_q, unsigned long long* psi_table_device, unsigned long long* psiinv_table_device,
    unsigned long long* m_poly_device, unsigned long long* qi_div_t_rns_array_device, unsigned long long* q_array_device, unsigned t, int q_amount)
{
    generate_random_default(in, sizeof(char) * n + sizeof(unsigned) * n * 2);  // default is for default stream: this is for synchronization issues
    // otherwise ternary distributions may run before this function, which is UNACCEPTABLE

    /*for (int i = 0; i < q_amount; i++)
    {
        ternary_dist(in, c + i * n, n, streams[i], q[i]);  // generate ternary dist poly directly into c0 and c1. c0 = c1,
        ternary_dist(in, c + i * n + q_amount * n, n, streams[i], q[i]);  // its represented by 'u'
        // for ease of operations and memory allocation, we have generated 2 of them (identical), since we override some stuff in polynomial multiplication.
    }

    for (int i = 0; i < q_amount; i++)
    {
        gaussian_dist((unsigned*)(in + n), e + i * n, n, streams[i], q[i]);  // this is again for generation ternary distribution, although it's name is gaussian
        // e0

        gaussian_dist((unsigned*)(in + n + n * 5), e + i * n + n * q_amount, n, streams[i], q[i]);  // i was joking this is for gaussian
        // e1
    }*/

    convert_ternary_gaussian_x2<<< q_amount * n / convertBlockSize, convertBlockSize, 0, 0 >>>(in, c, e, n, q_amount);

    /*for (int i = 0; i < q_amount; i++)
    {
        // multiply each public key with 'u'(c0 and c1). Remember that c0 and c1 are identical
        half_poly_mul_device(c + i * n, public_key + i * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n, psiinv_table_device + i * n);
        half_poly_mul_device(c + i * n + q_amount * n, public_key + i * n + q_amount * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n, psiinv_table_device + i * n);
    }*/

    /*unsigned long long arr[4096];
    for (int i = 0; i < q_amount; i++)
    {
        cudaMemcpy(arr, c + i * 4096, 8 * 4096, cudaMemcpyDeviceToHost);
        print_array(arr);
    }
    for (int i = 0; i < q_amount; i++)
    {
        cudaMemcpy(arr, c + 4096 * 3 + i * 4096, 8 * 4096, cudaMemcpyDeviceToHost);
        print_array(arr);
    }*/

    forwardNTT_batch(c, n, psi_table_device, q_amount * 2, q_amount);
    dim3 barrett_dim(n / 256, q_amount * 2);
    barrett_batch<<< barrett_dim, 256, 0, 0 >>>(c, public_key, n, q_amount);
    inverseNTT_batch(c, n, psiinv_table_device, q_amount * 2, q_amount);

    /*for (int i = 0; i < q_amount; i++)
    {
        poly_add_device(c + i * n, e + i * n, n, streams[i], q[i]);  // add e0 to publickey[0]
        poly_add_device(c + i * n + q_amount * n, e + i * n + n * q_amount, n, streams[i], q[i]);  // add e1 to publickey[1]
    }*/

    dim3 add_xq_dim(n * q_amount / small_block, 2);
    poly_add_xq<<< add_xq_dim, small_block, 0, 0 >> > (c, e, n, q_amount);

    divide_and_round_q_last_inplace_add_x2<<< n * 2 / small_block, small_block, 0, 0 >>>(c, n, q_amount);

    //divide_and_round_q_last_inplace(c, n, streams, q, q_bit_lengths, mu_array, inv_q_last_mod_q, q_amount);  // do that complicated stuff for each public key
    //divide_and_round_q_last_inplace(c + q_amount * n, n, streams, q, q_bit_lengths, mu_array, inv_q_last_mod_q, q_amount);

    divide_and_round_q_last_inplace_loop_xq<<< n * 2 * (q_amount - 1) / small_block, small_block, 0, 0 >>>(c, q_amount, n);

    weird_m_stuff<<< n / 256, 256, 0, 0 >>>(m_poly_device, c, t, qi_div_t_rns_array_device, q_array_device, q_amount, n);  // look at the comments in the function
}
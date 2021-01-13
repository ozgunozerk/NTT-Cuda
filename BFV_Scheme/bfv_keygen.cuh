#pragma once

#include <vector>
using std::vector;

#include "ntt_60bit.cuh"
#include "uint128.h"
#include "salsa_common.h"
#include "distributions.cuh"
#include "poly_arithmetic.cuh"

#define small_block 128

__global__ void ternary_dist_xq(unsigned char* in, unsigned long long* secret_key, unsigned n, unsigned q_amount)
{
    int i = blockIdx.x * convertBlockSize + threadIdx.x;

    float d = (float)in[i % n];

    d /= (255.0f / 3);

    /*if (d >= 2)
        secret_key[i] = 1;
    else if (d >= 1)
        secret_key[i] = 0;
    else
        secret_key[i] = q_cons[i / n] - 1;*/

    int b = int(d) - 1;
    secret_key[i] = (b < 0) * q_cons[i / n] + b;
}

__global__ void uniform_dist_xq(unsigned char* in, unsigned long long* public_key, unsigned n, unsigned q_amount)
{
    int i = blockIdx.x * convertBlockSize + threadIdx.x;

    unsigned long long* inl = (unsigned long long*)in;
    register double d = (double)inl[i];

    d /= UINT64_MAX;

    d *= (double)(q_cons[i / n] - 1);

    public_key[i] = (unsigned long long)d;
}

__global__ void gaussian_dist_xq(unsigned char* in, unsigned long long* temp, unsigned n, unsigned q_amount)
{
    int i = blockIdx.x * convertBlockSize + threadIdx.x;

    float d = ((unsigned*)(in))[i % n];

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
        temp[i] = q_cons[i / n] + dd;
    else
        temp[i] = dd;
}

__global__ void poly_add_negate_xq(unsigned long long* a, unsigned long long* b, unsigned n, unsigned q_amount)
{
    int i = blockIdx.x * small_block + threadIdx.x;
    unsigned long long q = q_cons[i / n];

    unsigned long long ra = a[i] + b[i];

    if (ra >= q)
        ra -= q;

    ra = q - ra;
    a[i] = ra * (ra != q); // transform a[i] to 0 if a[i] is equal to q
}

void keygen_rns(unsigned char in[], int q_amount, unsigned long long* q, unsigned n, unsigned long long* secret_key, unsigned long long* public_key,
    cudaStream_t* streams, unsigned long long* temp, vector<unsigned long long> mu_array, vector<unsigned> q_bit_lengths,
    unsigned long long* psi_table_device, unsigned long long* psiinv_table_device)
{
    generate_random_default(in, (sizeof(unsigned char) + sizeof(unsigned long long)) * q_amount * n + sizeof(unsigned) * n);

    // convert random bytes to ternary distribution
    // use same byte sequence for each element of the secret key
    /*for (int i = 0; i < q_amount; i++)
    {
        ternary_dist(in, secret_key + i * n, n, streams[i], q[i]);
    }

    // convert random bytes to uniform distribution
    // use different byte sequences for each q
    for (int i = 0; i < q_amount; i++)
    {
        uniform_dist((unsigned long long*)(in + n + i * n * sizeof(unsigned long long)), public_key + i * n + q_amount * n, n, streams[i], q[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        gaussian_dist((unsigned*)(in + n + q_amount * n * sizeof(unsigned long long)), temp + i * n, n, streams[i], q[i]);
    }*/

    ternary_dist_xq << < q_amount * n / convertBlockSize, convertBlockSize, 0, 0 >> > (in, secret_key, n, q_amount);
    uniform_dist_xq << < q_amount * n / convertBlockSize, convertBlockSize, 0, 0 >> > (in + n, public_key + q_amount * n, n, q_amount);
    gaussian_dist_xq << < q_amount * n / convertBlockSize, convertBlockSize, 0, 0 >> > (in + n + q_amount * n * sizeof(unsigned long long), temp, n, q_amount);

    /*for (int i = 0; i < q_amount; i++)
    {
        forwardNTT(secret_key + i * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n);
    }*/

    forwardNTT_batch(secret_key, n, psi_table_device, q_amount, q_amount);

    dim3 barrett_dim(n / 256, q_amount);
    barrett_batch_3param << <barrett_dim, 256, 0, 0 >> > (public_key, public_key + q_amount * n, secret_key, n, q_amount);
    inverseNTT_batch(public_key, n, psiinv_table_device, q_amount, q_amount);
    //poly_add_negate_xq << <q_amount * n / small_block, small_block, 0, 0 >> > (public_key, temp, n, q_amount);

    /*for (int i = 0; i < q_amount; i++)
    {
        barrett_3param << <n / 256, 256, 0, streams[i] >> > (public_key + i * n, public_key + i * n + q_amount * n, secret_key + i * n, q[i], mu_array[i], q_bit_lengths[i]);
        inverseNTT(public_key + i * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psiinv_table_device + i * n);
        //poly_add_device(public_key + i * n, temp + i * n, n, streams[i], q[i]);
        //poly_negate_device(public_key + i * n, n, streams[i], q[i]);
    }*/

    poly_add_negate_xq << <q_amount * n / small_block, small_block, 0, 0 >> > (public_key, temp, n, q_amount);
    forwardNTT_batch(public_key, n, psi_table_device, q_amount, q_amount);

    /*for (int i = 0; i < q_amount; i++)
    {
        forwardNTT(public_key + i * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n);
    }*/
}
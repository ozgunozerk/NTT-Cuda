#pragma once

//#include <inttypes.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "salsa_common.h"

__constant__ unsigned char MY_ALIGN(sizeof(uint32_t)) key[XSALSA20_CRYPTO_KEYBYTES * THREADS_PER_BLOCK];
__constant__ unsigned char MY_ALIGN(sizeof(uint32_t)) sigma[16];

const unsigned char hsigma[17] = "expand 32-byte k";
unsigned char h_nonce[XSALSA20_CRYPTO_NONCEBYTES];

__host__ __device__ static inline uint32_t
rotate(uint32_t u, int c)
{
    return (u << c) | (u >> (32 - c));
}

__host__ __device__ static inline uint32_t
load_littleendian(const unsigned char* x)
{
    return
        (uint32_t)(x[0]) \
        | (((uint32_t)(x[1])) << 8) \
        | (((uint32_t)(x[2])) << 16) \
        | (((uint32_t)(x[3])) << 24)
        ;
}

__host__ static inline uint32_t
load_littleendian64(const unsigned char* x)
{
    return
        (uint64_t)(x[0]) \
        | (((uint64_t)(x[1])) << 8) \
        | (((uint64_t)(x[2])) << 16) \
        | (((uint64_t)(x[3])) << 24) \
        | (((uint64_t)(x[4])) << 32) \
        | (((uint64_t)(x[5])) << 40) \
        | (((uint64_t)(x[6])) << 48) \
        | (((uint64_t)(x[7])) << 56)
        ;
}

__global__ void VecCrypt(unsigned char* A, unsigned int N, uint64_t nblocks, uint64_t p_nonce, int blks_per_chunk)
{
    uint64_t i = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;

    if (i < N) {
        int k, tot;
        uint32_t* mem;
        uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
        uint32_t j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15;
        uint64_t blockno, adj_blockno;

        blockno = i * blks_per_chunk;
        tot = (nblocks - blockno > blks_per_chunk) ? blks_per_chunk : (nblocks - blockno);

        for (k = 0; k < tot; k++) {
            j0 = x0 = load_littleendian(sigma + 0);
            j1 = x1 = load_littleendian(key + 0);
            j2 = x2 = load_littleendian(key + 4);
            j3 = x3 = load_littleendian(key + 8);
            j4 = x4 = load_littleendian(key + 12);
            j5 = x5 = load_littleendian(sigma + 4);

            adj_blockno = blockno;
            j6 = x6 = p_nonce;
            j7 = x7 = p_nonce >> 32;
            j8 = x8 = adj_blockno;
            j9 = x9 = adj_blockno >> 32;

            j10 = x10 = load_littleendian(sigma + 8);
            j11 = x11 = load_littleendian(key + 16);
            j12 = x12 = load_littleendian(key + 20);
            j13 = x13 = load_littleendian(key + 24);
            j14 = x14 = load_littleendian(key + 28);
            j15 = x15 = load_littleendian(sigma + 12);

            for (i = ROUNDS; i > 0; i -= 2) {
                x4 ^= rotate(x0 + x12, 7);
                x8 ^= rotate(x4 + x0, 9);
                x12 ^= rotate(x8 + x4, 13);
                x0 ^= rotate(x12 + x8, 18);
                x9 ^= rotate(x5 + x1, 7);
                x13 ^= rotate(x9 + x5, 9);
                x1 ^= rotate(x13 + x9, 13);
                x5 ^= rotate(x1 + x13, 18);
                x14 ^= rotate(x10 + x6, 7);
                x2 ^= rotate(x14 + x10, 9);
                x6 ^= rotate(x2 + x14, 13);
                x10 ^= rotate(x6 + x2, 18);
                x3 ^= rotate(x15 + x11, 7);
                x7 ^= rotate(x3 + x15, 9);
                x11 ^= rotate(x7 + x3, 13);
                x15 ^= rotate(x11 + x7, 18);
                x1 ^= rotate(x0 + x3, 7);
                x2 ^= rotate(x1 + x0, 9);
                x3 ^= rotate(x2 + x1, 13);
                x0 ^= rotate(x3 + x2, 18);
                x6 ^= rotate(x5 + x4, 7);
                x7 ^= rotate(x6 + x5, 9);
                x4 ^= rotate(x7 + x6, 13);
                x5 ^= rotate(x4 + x7, 18);
                x11 ^= rotate(x10 + x9, 7);
                x8 ^= rotate(x11 + x10, 9);
                x9 ^= rotate(x8 + x11, 13);
                x10 ^= rotate(x9 + x8, 18);
                x12 ^= rotate(x15 + x14, 7);
                x13 ^= rotate(x12 + x15, 9);
                x14 ^= rotate(x13 + x12, 13);
                x15 ^= rotate(x14 + x13, 18);
            }

            x0 += j0;
            x1 += j1;
            x2 += j2;
            x3 += j3;
            x4 += j4;
            x5 += j5;
            x6 += j6;
            x7 += j7;
            x8 += j8;
            x9 += j9;
            x10 += j10;
            x11 += j11;
            x12 += j12;
            x13 += j13;
            x14 += j14;
            x15 += j15;

            mem = (unsigned int*)&A[blockno * XSALSA20_BLOCKSZ];
            *mem ^= x0;  mem++;
            *mem ^= x1;  mem++;
            *mem ^= x2;  mem++;
            *mem ^= x3;  mem++;
            *mem ^= x4;  mem++;
            *mem ^= x5;  mem++;
            *mem ^= x6;  mem++;
            *mem ^= x7;  mem++;
            *mem ^= x8;  mem++;
            *mem ^= x9;  mem++;
            *mem ^= x10;  mem++;
            *mem ^= x11;  mem++;
            *mem ^= x12;  mem++;
            *mem ^= x13;  mem++;
            *mem ^= x14;  mem++;
            *mem ^= x15;
            blockno++;
        }
    }
}

__global__ void convert_gaussian(unsigned* in, unsigned long long* out, unsigned long long q)
{
    int i = threadIdx.x + blockIdx.x * convertBlockSize;

    register float d = (float)in[i];

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

    register int dd = (int)d;

    if (dd < 0)
        out[i] = q + dd;
    else
        out[i] = dd;
}

__global__ void convert_range(unsigned long long* in, unsigned long long* out, unsigned long long q)
{
    int i = threadIdx.x + blockIdx.x * convertBlockSize;

    register double d = (double)in[i];

    d /= UINT64_MAX;

    d *= (double)(q - 1);

    out[i] = (unsigned long long)d;
}

__global__ void convert_ternary(unsigned char* in, unsigned long long* out, unsigned long long q)
{
    int i = threadIdx.x + blockIdx.x * convertBlockSize;

    register float d = (float)in[i];

    d /= (256.0f / 3);

    if (d >= 2)
        out[i] = 1;
    else if (d >= 1)
        out[i] = 0;
    else
        out[i] = q - 1;
}

void generate_random(unsigned char* a, unsigned n, cudaStream_t& stream)
{
    unsigned char* d_A = a;

    unsigned int NBLKS = n / 64, N;
    int threadsPerBlock, blocksPerGrid;
    size_t size, i;
    unsigned char k[32];
    uint64_t v_nonce;

    size = NBLKS * XSALSA20_BLOCKSZ;

    memset(k, 77, XSALSA20_CRYPTO_KEYBYTES);
    memset(h_nonce, 0, XSALSA20_CRYPTO_NONCEBYTES);

    cudaMemcpyToSymbolAsync(key, k, XSALSA20_CRYPTO_NONCEBYTES, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbolAsync(sigma, hsigma, 16, 0, cudaMemcpyHostToDevice);
    v_nonce = load_littleendian64(h_nonce);
    threadsPerBlock = THREADS_PER_BLOCK;

    cudaMemsetAsync(d_A, 0, size, stream);

    N = NBLKS;

    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VecCrypt << <blocksPerGrid, threadsPerBlock, 0, stream >> > (d_A, N, size, v_nonce, 1);
}

void generate_random_default(unsigned char* a, unsigned n)
{
    unsigned char* d_A = a;

    unsigned int NBLKS = n / 64, N;
    int threadsPerBlock, blocksPerGrid;
    size_t size, i;
    unsigned char k[32];
    uint64_t v_nonce;

    size = NBLKS * XSALSA20_BLOCKSZ;

    memset(k, 1, XSALSA20_CRYPTO_KEYBYTES);
    memset(h_nonce, 0, XSALSA20_CRYPTO_NONCEBYTES);

    cudaMemcpyToSymbolAsync(key, k, XSALSA20_CRYPTO_KEYBYTES, 0, cudaMemcpyHostToDevice); //re add async
    cudaMemcpyToSymbolAsync(sigma, hsigma, 16, 0, cudaMemcpyHostToDevice);
    v_nonce = load_littleendian64(h_nonce);
    threadsPerBlock = THREADS_PER_BLOCK;

    cudaMemsetAsync(d_A, 0, size); //re add async

    N = NBLKS;

    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VecCrypt << <blocksPerGrid, threadsPerBlock, 0, 0 >> > (d_A, N, size, v_nonce, 1);
}

void gaussian_dist(unsigned* in, unsigned long long* out, unsigned n, cudaStream_t& stream, unsigned long long q)
{
    int convert_block_amount = n / convertBlockSize;

    convert_gaussian << <convert_block_amount, convertBlockSize, 0, stream >> > (in, out, q);
}

void uniform_dist(unsigned long long* in, unsigned long long* out, unsigned n, cudaStream_t& stream, unsigned long long q)
{
    int convert_block_amount = n / convertBlockSize;

    convert_range << <convert_block_amount, convertBlockSize, 0, stream >> > (in, out, q);
}

void ternary_dist(unsigned char* in, unsigned long long* out, unsigned n, cudaStream_t& stream, unsigned long long q)
{
    int convert_block_amount = n / convertBlockSize;

    convert_ternary << <convert_block_amount, convertBlockSize, 0, stream >> > (in, out, q);
}


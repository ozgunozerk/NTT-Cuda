#pragma once

#include <vector>
#include <stdio.h>
using std::vector;

#include "ntt_60bit.cuh"
#include "uint128.h"
#include "poly_arithmetic.cuh"

#define small_block 128

__global__ void poly_add_xq_d(unsigned long long* c, unsigned n, int q_amount)
{
    int i = blockIdx.x * small_block + threadIdx.x;

    unsigned long long ra = c[i + (n * q_amount)] + c[i];

    if (ra > q_cons[i / n])
        ra -= q_cons[i / n];

    c[i + (n * q_amount)] = ra;
}

__global__ void poly_mul_int_xq_prodtgamma(unsigned long long* c, unsigned n)
{
    int i = blockIdx.x * small_block + threadIdx.x;
    unsigned long long q = q_cons[i / n];
    unsigned long long inv_punctured_q = prod_t_gamma_mod_q_cons[i / n];

    uint128_t ra;
    mul64(c[i], inv_punctured_q, ra);

    unsigned long long mu = mu_cons[i / n];
    int qbit = q_bit_cons[i / n];

    singleBarrett(ra, q, mu, qbit);

    c[i] = ra.low;
}

__global__ void poly_mul_int_xq_invpq(unsigned long long* c, unsigned n)
{
    int i = blockIdx.x * small_block + threadIdx.x;
    unsigned long long q = q_cons[i / n];
    unsigned long long inv_punctured_q = inv_punctured_q_cons[i / n];

    uint128_t ra;
    mul64(c[i], inv_punctured_q, ra);

    unsigned long long mu = mu_cons[i / n];
    int qbit = q_bit_cons[i / n];

    singleBarrett(ra, q, mu, qbit);

    c[i] = ra.low;
}



	/* DECRYPTION:
	c has (n * (q_amount-1) * 2) elements efficiently. But since in Encryption, we needed (n * q_amount * 2) elements for c's initial steps,
	it is been allocated (n * q_amount * 2) space for c, and we did not resize c for efficiency. After encryption, the last q's 
	(and their corresponding polynomials) of c0 and c1 are dropped from c. 
	Dropped parts corresponds to => c[8192:12288] and c[20480:24576]. These parts are not removed, but ignored.
	ALthough we only need (n * (q_amount-1) * 2) items for Decryption, we have to provide (n * q_amount * 2) items inside c array.
	
	Here is a how c_array should be crafted:

	c[0 : n * (q_amount-1)] = c0
	c[n * (q_amount-1) : n * q_amount] = PADDING (not important, can be 0's or whatever we like)
	c[n * q_amount : n * q_amount + n * (q_amount-1)] = c1
	c[n * q_amount + n * (q_amount-1) : n * q_amount * 2] = PADDING (not important, can be 0's or whatever we like)

	*/
void decryption_rns(unsigned long long* c, unsigned long long* secret_key,
    unsigned long long* q, vector<unsigned>& q_bit_lengths, vector<unsigned long long>& mu_array,
    unsigned long long* psi_table_device, unsigned long long* psiinv_table_device, int n, unsigned q_amount,
    vector<unsigned long long>& inv_punctured_q, unsigned long long* base_change_matrix_device,
    unsigned long long t, unsigned long long gamma, unsigned long long mu_gamma, vector<unsigned long long>& output_base,
    vector<unsigned>& output_base_bit_lengths, vector<unsigned long long>& neg_inv_q_mod_t_gamma,
    unsigned long long gamma_div_2, vector<unsigned long long> prod_t_gamma_mod_q) // hehehehe
{

    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * q_amount);
    for (int i = 0; i < q_amount; i++)
        cudaStreamCreate(&streams[i]);

    /*for (int i = 0; i < q_amount; i++)
    {
        half_poly_mul_device(c + i * n + (q_amount + 1) * n, secret_key + i * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n, psiinv_table_device + i * n);
        // c1 = c1 * sk

        poly_add_device(c + i * n + (q_amount + 1) * n, c + i * n, n, streams[i], q[i]);
        // c1 = c1 + c0
    }*/

    forwardNTT_batch(c + (q_amount + 1) * n, n, psi_table_device, q_amount, q_amount + 1);
    dim3 barrett_dim(n / 256, q_amount);
    barrett_batch << < barrett_dim, 256, 0, 0 >> > (c + (q_amount + 1) * n, secret_key, n, q_amount);
    inverseNTT_batch(c + (q_amount + 1) * n, n, psiinv_table_device, q_amount, q_amount + 1);

    poly_add_xq_d << < n * q_amount / small_block, small_block, 0, 0 >> > (c, n, q_amount + 1);

    /*for (int i = 0; i < q_amount; i++)
    {
        poly_mul_int(c + i * n + (q_amount + 1) * n, prod_t_gamma_mod_q[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i]);
        // c1 = c1 * prod_t_gamma_mod_q
        //cout << prod_t_gamma_mod_q[i] << endl;
    }*/

    poly_mul_int_xq_prodtgamma << <n * q_amount / small_block, small_block, 0, 0 >> > (c + (q_amount + 1) * n, n);

    // start of fast convert array

    // c1 / punc_q mod q
    /*for (int i = 0; i < q_amount; i++)
    {
        poly_mul_int(c + i * n + (q_amount + 1) * n, inv_punctured_q[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i]);
    }*/

    poly_mul_int_xq_invpq << <n * q_amount / small_block, small_block, 0, 0 >> > (c + (q_amount + 1) * n, n);

    //cudaStreamSynchronize(streams[q_amount - 1]);

    // multiply coeff[k] with base change matrix, add them together and split into 2 poly
    fast_convert_array_kernels(c + (q_amount + 1) * n, c, t, base_change_matrix_device, q_amount, gamma,
        output_base_bit_lengths[1], mu_gamma, streams[0], streams[1], n);

    // end of fast convert array

    // multiply polies by neg_inv_q_mod_t_gamma
    poly_mul_int_t(c, neg_inv_q_mod_t_gamma[0], n, streams[0], t);
    poly_mul_int(c + n, neg_inv_q_mod_t_gamma[1], n, streams[1], gamma, mu_gamma, output_base_bit_lengths[1]);

    //round
    dec_round(c, c + n * (q_amount - 1), t, gamma, gamma_div_2, n, streams[1]);
}

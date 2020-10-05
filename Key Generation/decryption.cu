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

void decryption_rns(unsigned long long** c0, unsigned long long** c1, unsigned long long** secret_key, 
    vector<unsigned long long>& q, vector<unsigned>& q_bit_lengths, vector<unsigned long long>& mu_array, 
    unsigned long long** psi_table_device, unsigned long long** psiinv_table_device, int n, unsigned q_amount,
    vector<unsigned long long>& inv_punctured_q, unsigned long long** base_change_matrix_device,
    unsigned long long t, unsigned long long gamma, unsigned long long mu_gamma, vector<unsigned long long>& output_base,
    vector<unsigned>& output_base_bit_lengths)
{
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * q_amount);
    for (int i = 0; i < q_amount; i++)
        cudaStreamCreate(&streams[i]);

    for (int i = 0; i < q_amount; i++)
    {
        half_poly_mul_device(c1[i], secret_key[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device[i], psiinv_table_device[i]);
        // c1 = c1 * sk

        poly_add_device(c1[i], c0[i], n, streams[i], q[i]);
        // c1 = c1 + c0
    }

    //start of fastbconvert array

    // c / punc_q mod q
    for (int i = 0; i < q_amount; i++)
    {
        poly_mul_int(c1[i], inv_punctured_q[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i]);
    }

    fast_convert_array_kernels(c1, c0, t, base_change_matrix_device, q_amount, gamma,
        output_base_bit_lengths[1], mu_gamma, streams[0], streams[1], n);
}

int main()
{
    int n = 1024 * 4;

    vector<unsigned long long> q = { 68719403009, 68719230977, 137438822401 };
    vector<unsigned long long> psi_roots = { 24250113, 29008497, 8625844 };
    vector<unsigned long long> psiinv_roots = { 60243494989, 37410665880, 5716440802 };
    vector<unsigned> q_bit_lengths = { 36, 36, 37 };
    vector<unsigned long long> mu_array = {};
    unsigned q_amount = q.size();

    vector<unsigned long long> punctured_q = {};
    vector<unsigned long long> inv_punctured_q = { 4548496129, 45637159345, 37067270992 };

    unsigned long long t = 1024;
    unsigned long long gamma = 2305843009213683713;

    vector<unsigned long long> output_base = { t, gamma };
    vector<unsigned> output_base_bit_lengths = { 10, 61 };
    unsigned long long mu_gamma;

    //generate mu parameter for barrett
    for (int i = 0; i < q_amount; i++)
    {
        unsigned int bit_length = q_bit_lengths[i];
        uint128_t mu1 = uint128_t::exp2(2 * bit_length);
        mu1 = mu1 / q[i];
        unsigned long long mu = mu1.low;

        mu_array.push_back(mu);
    }

    //generate mu parameter of gamma for barrett
    {
    unsigned int bit_length = output_base_bit_lengths[1];
    uint128_t mu1 = uint128_t::exp2(2 * bit_length);
    mu1 = mu1 / gamma;
    unsigned long long mu_gamma = mu1.low; 
    }

    //allocate memory for powers of psi root and psi inverse root
    //and fill those arrays
    unsigned long long** psi_table = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    unsigned long long** psiinv_table = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        psi_table[i] = (unsigned long long*)malloc(sizeof(unsigned long long) * n);
        psiinv_table[i] = (unsigned long long*)malloc(sizeof(unsigned long long) * n);

        fillTablePsi128(psi_roots[i], q[i], psiinv_roots[i], psi_table[i], psiinv_table[i], n);
    }

    //allocate memory for powers of psi root and psi inverse root on device
    //and copy their values from host to device
    unsigned long long** psi_table_device = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    unsigned long long** psiinv_table_device = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&psi_table_device[i], sizeof(unsigned long long) * n);
        cudaMalloc(&psiinv_table_device[i], sizeof(unsigned long long) * n);

        cudaMemcpy(psi_table_device[i], psi_table[i], sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(psiinv_table_device[i], psiinv_table[i], sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);
    }

    //calculate values of punctured q
    for (int i = 0; i < q_amount; i++)
    {
        uint128_t temp = 1;
        for (int j = 0; j < q_amount; j++)
        {
            if (i == j)
                continue;

            temp = host64x2(temp.low, q[j]) % q[i];
        }

        punctured_q.push_back(temp.low);
    }

    //pre-compute the values for base change matrix
    unsigned long long** base_change_matrix = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        base_change_matrix[i] = (unsigned long long*)malloc(sizeof(unsigned long long) * 2);
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < q_amount; j++)
        {
            base_change_matrix[i][j] = punctured_q[j] % output_base[i];
        }
    }

    //allocate memory for base change matrix on device and copy the values to it
    unsigned long long** base_change_matrix_device = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&base_change_matrix_device[i], sizeof(unsigned long long) * 2);
        cudaMemcpy(base_change_matrix_device[i], base_change_matrix[i], sizeof(unsigned long long) * 2, cudaMemcpyHostToDevice);
    }

    unsigned long long** secret_key = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    unsigned long long** c0 = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    unsigned long long** c1 = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);

    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&secret_key[i], sizeof(unsigned long long) * n);
        cudaMalloc(&c0[i], sizeof(unsigned long long) * n);
        cudaMalloc(&c1[i], sizeof(unsigned long long) * n);
    }

    decryption_rns(c0, c1, secret_key, q, q_bit_lengths, mu_array, psi_table_device, psiinv_table_device, 
        n, q_amount, inv_punctured_q, base_change_matrix_device, t, gamma, mu_gamma, output_base, output_base_bit_lengths);
}
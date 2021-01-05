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

void divide_and_round_q_last_inplace(unsigned long long* poly, unsigned N, cudaStream_t streams[], vector<unsigned long long> q, vector<unsigned> q_bit_lengths, 
    vector<unsigned long long> mu_array, vector<unsigned long long> inv_q_last_mod_q)
{
    unsigned q_amount = q.size();  // getting how many q's we have

    unsigned long long last_modulus = q[q_amount - 1];  // get the last q from the array
    unsigned long long half_last_modulus = last_modulus >> 1;  // divide it by half

    poly_add_integer_device_default(poly + N * (q_amount - 1), half_last_modulus, N, last_modulus);  
    // poly + N * (q_amount - 1) = getting the to the last q in the flattened array
    // adding half_last_modulus to it, in mod last_modulus
    // N is required for calling the kernel with optimal thread amount

    for (int i = 0; i < q_amount - 1; i++)  
    {
        unsigned long long half_mod = half_last_modulus % q[i];  // getting the half_last_modulus' mod in respect to every other q[x]
        divide_and_round_q_last_inplace_loop << <N / 256, 256, 0, streams[i] >> > (poly + N * i, poly + N * (q_amount - 1), q[i], half_mod, inv_q_last_mod_q[i], mu_array[i], q_bit_lengths[i]);  // too long to explain, inspect the comments in the function
    }
}

void rns_encryption(unsigned long long* c0, unsigned long long* c1, unsigned long long*** public_key, unsigned char* in, unsigned long long** u, unsigned long long*** e, unsigned N,
    cudaStream_t streams[], vector<unsigned long long> q, vector<unsigned> q_bit_lengths,
    vector<unsigned long long> mu_array, vector<unsigned long long> inv_q_last_mod_q, unsigned long long** psi_table_device, unsigned long long** psiinv_table_device,
    unsigned long long* m_poly_device, unsigned long long m_len, unsigned long long* qi_div_t_rns_array_device, unsigned long long* q_array_device, unsigned t)
{
    unsigned q_amount = q.size();  // getting how many q's do we have

    generate_random_default(in, sizeof(unsigned long long) * q_amount * N + sizeof(unsigned) * N * 2);  // default is for default stream: this is for synchronization issues
    // otherwise ternary distributions may run before this function, which is UNACCEPTABLE

    for (int i = 0; i < q_amount; i++)
    {
        ternary_dist(in + i * N, c0 + i * N, N, streams[i], q[i]);  // generate ternary dist poly directly into c0 and c1. c0 = c1, 
        ternary_dist(in + i * N, c1 + i * N, N, streams[i], q[i]);  // its represented by 'u' 
        // for ease of operations and memory allocation, we have generated 2 of them (identical), since we override some stuff in polynomial multiplication.
    }

    for (int i = 0; i < q_amount; i++)
    {
        gaussian_dist((unsigned*)(in + q_amount * N), e[0][i], N, streams[i], q[i]);  // this is again for generation ternary distribution, although it's name is gaussian
        // e0

        gaussian_dist((unsigned*)(in + q_amount * N), e[1][i], N, streams[i], q[i]);  // i was joking this is for gaussian
        // e1
    }

    // CAN should we delete the comment below?
    // olur
    /*c0 = self.pk[0] * u
        c1 = self.pk[1] * u
        c0 = c0 + e1
        c1 = c1 + e2*/

    for (int i = 0; i < q_amount; i++)
    {
        // multiply each public key with 'u'(c0 and c1). Remember that c0 and c1 are identical
        half_poly_mul_device(c0 + i * N, public_key[0][i], N, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device[i], psiinv_table_device[i]);  
        half_poly_mul_device(c1 + i * N, public_key[1][i], N, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device[i], psiinv_table_device[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        poly_add_device(c0 + i * N, e[0][i], N, streams[i], q[i]);  // add e0 to publickey[0]
        poly_add_device(c1 + i * N, e[1][i], N, streams[i], q[i]);  // add e1 to publickey[1]
    }

    divide_and_round_q_last_inplace(c0, N, streams, q, q_bit_lengths, mu_array, inv_q_last_mod_q);  // do that complicated stuff for each public key
    divide_and_round_q_last_inplace(c1, N, streams, q, q_bit_lengths, mu_array, inv_q_last_mod_q);

    weird_m_stuff << <N / 256, 256, 0, 0 >> > (m_len, m_poly_device, c0, t, qi_div_t_rns_array_device, q_array_device, q_amount, N);  // look at the comments in the function
}

int main()
{
    int n = 1024 * 4;

    vector<unsigned long long> q = { 68719403009, 68719230977, 137438822401 };  // the first value is 68719403009 because its bigger than 68719403008
    unsigned long long q_array[] = { 68719403009, 68719230977, 137438822401 };
    vector<unsigned long long> psi_roots = { 24250113, 29008497, 8625844 };
    vector<unsigned long long> psiinv_roots = { 60243494989, 37410665880, 5716440802 };
    vector<unsigned> q_bit_lengths = { 36, 36, 37 };
    vector<unsigned long long> mu_array = {};
    unsigned q_amount = q.size();
    vector<unsigned long long> inv_q_last_mod_q = { 20955999355, 17095778744 };
    unsigned long long q_mod_t = 1;
    vector<unsigned long long> qi_div_t_rns = { 67108792, 67108624, 134217600 }; // we don't need this vector
    unsigned long long qi_div_t_rns_array[] = { 67108792, 67108624, 134217600 };

    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * q_amount);
    for (int i = 0; i < q_amount; i++)
        cudaStreamCreate(&streams[i]);  // create streams for parallelism

    unsigned long long* q_array_device;
    cudaMalloc(&q_array_device, q_amount * sizeof(unsigned long long));
    cudaMemcpy(q_array_device, q_array, q_amount * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    unsigned long long* qi_div_t_rns_array_device;
    cudaMalloc(&qi_div_t_rns_array_device, (q_amount - 1)* sizeof(unsigned long long));
    cudaMemcpy(qi_div_t_rns_array_device, qi_div_t_rns_array, (q_amount - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    unsigned long long m = 100;  // our message to encrypt
    unsigned long long m_len = log2(m) + 1;  // length of m

    unsigned long long t = 1024;  // mathematical stuff that is beyond our comprehension

    //generate mu parameters for barrett
    for (int i = 0; i < q_amount; i++)
    {
        unsigned int bit_length = q_bit_lengths[i];
        uint128_t mu1 = uint128_t::exp2(2 * bit_length);
        mu1 = mu1 / q[i];
        unsigned long long mu = mu1.low;

        mu_array.push_back(mu);
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

    unsigned long long* c0;
    unsigned long long* c1;
    cudaMalloc(&c0, sizeof(unsigned long long) * n * q_amount);
    cudaMalloc(&c1, sizeof(unsigned long long) * n * q_amount);

    unsigned char* in;
    cudaMalloc(&in, sizeof(unsigned long long) * q_amount * n + sizeof(unsigned) * n * 2);

    unsigned long long** u = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&u[i], sizeof(unsigned long long) * n);
    }

    unsigned long long*** e = (unsigned long long***)malloc(sizeof(unsigned long long**) * 2);
    e[0] = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    e[1] = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < q_amount; j++)
        {
            cudaMalloc(&e[i][j], sizeof(unsigned long long) * n);
        }
    }

    unsigned long long*** public_key = (unsigned long long***)malloc(sizeof(unsigned long long**) * 2);
    public_key[0] = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    public_key[1] = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < q_amount; j++)
        {
            cudaMalloc(&public_key[i][j], sizeof(unsigned long long) * n);
        }
    }

    unsigned long long m_poly[4096];
    for (int i = 0; i < 4096; i++)
    {
        m_poly[i] = 0;
    }

    m_poly[2] = 1; m_poly[5] = 1; m_poly[6] = 1;

    unsigned long long* m_poly_device;
    cudaMalloc(&m_poly_device, 4096 * sizeof(unsigned long long));
    cudaMemcpy(m_poly_device, m_poly, 4096 * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    rns_encryption(c0, c1, public_key, in, u, e, n, streams, q, q_bit_lengths, mu_array, 
        inv_q_last_mod_q, psi_table_device, psiinv_table_device, m_poly_device, m_len, qi_div_t_rns_array_device, q_array_device, t);

}
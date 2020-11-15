#include <vector>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;

#include "helper.h"
#include "parameter.h"

#include "poly_arithmetic.cuh"
#include "distributions.cuh"

void print_array(unsigned long long a[])
{
    cout << "[";
    for (int i = 0; i < 4096; i++)
    {
        cout << a[i] << ", ";
    }
    cout << "]\n";
}

void divide_and_round_q_last_inplace(unsigned long long* poly, unsigned N, cudaStream_t streams[], vector<unsigned long long> q, vector<unsigned> q_bit_lengths,
    vector<unsigned long long> mu_array, vector<unsigned long long> inv_q_last_mod_q)
{
    unsigned q_amount = q.size();  // getting how many q's we have

    unsigned long long last_modulus = q[q_amount - 1];  // get the last q from the array
    unsigned long long half_last_modulus = last_modulus >> 1;  // divide it by 2

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

    generate_random_default(in, sizeof(char) * N + sizeof(unsigned) * N * 2);  // default is for default stream: this is for synchronization issues
    // otherwise ternary distributions may run before this function, which is UNACCEPTABLE

    for (int i = 0; i < q_amount; i++)
    {
        ternary_dist(in, c0 + i * N, N, streams[i], q[i]);  // generate ternary dist poly directly into c0 and c1. c0 = c1, 
        ternary_dist(in, c1 + i * N, N, streams[i], q[i]);  // its represented by 'u' 
        // for ease of operations and memory allocation, we have generated 2 of them (identical), since we override some stuff in polynomial multiplication.
    }

    for (int i = 0; i < q_amount; i++)
    {
        gaussian_dist((unsigned*)(in + N), e[0][i], N, streams[i], q[i]);  // this is again for generation ternary distribution, although it's name is gaussian
        // e0

        gaussian_dist((unsigned*)(in + N + N * 5), e[1][i], N, streams[i], q[i]);  // i was joking this is for gaussian
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

void decryption_rns(unsigned long long* c0, unsigned long long* c1, unsigned long long** secret_key,
    vector<unsigned long long>& q, vector<unsigned>& q_bit_lengths, vector<unsigned long long>& mu_array,
    unsigned long long** psi_table_device, unsigned long long** psiinv_table_device, int n, unsigned q_amount,
    vector<unsigned long long>& inv_punctured_q, unsigned long long* base_change_matrix_device,
    unsigned long long t, unsigned long long gamma, unsigned long long mu_gamma, vector<unsigned long long>& output_base,
    vector<unsigned>& output_base_bit_lengths, vector<unsigned long long>& neg_inv_q_mod_t_gamma,
    unsigned long long gamma_div_2) // hehehehe
{
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * q_amount);
    for (int i = 0; i < q_amount; i++)
        cudaStreamCreate(&streams[i]);

    for (int i = 0; i < q_amount; i++)
    {
        half_poly_mul_device(c1 + i * n, secret_key[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device[i], psiinv_table_device[i]);
        // c1 = c1 * sk

        poly_add_device(c1 + i * n, c0 + i * n, n, streams[i], q[i]);
        // c1 = c1 + c0
    }

    vector<unsigned long long> prod_t_gamma_mod_q = { 37067052033, 64547873793 };

    for (int i = 0; i < q_amount; i++)
    {
        poly_mul_int(c1 + i * n, prod_t_gamma_mod_q[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i]);
        // c1 = c1 * prod_t_gamma_mod_q
    }

    // start of fast convert array

    // c / punc_q mod q
    for (int i = 0; i < q_amount; i++)
    {
        poly_mul_int(c1 + i * n, inv_punctured_q[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i]);
    }

    cudaStreamSynchronize(streams[q_amount - 1]);

    // multiply coeff[k] with base change matrix, add them together and split into 2 poly
    fast_convert_array_kernels(c1, c0, t, base_change_matrix_device, q_amount, gamma,
        output_base_bit_lengths[1], mu_gamma, streams[0], streams[1], n);

    // end of fast convert array

    // multiply polies by neg_inv_q_mod_t_gamma
    poly_mul_int_t(c0, neg_inv_q_mod_t_gamma[0], n, streams[0], t);
    poly_mul_int(c0 + n, neg_inv_q_mod_t_gamma[1], n, streams[1], gamma, mu_gamma, output_base_bit_lengths[1]);

    //round
    dec_round(c0, c0 + n * (q_amount - 1), t, gamma, gamma_div_2, n, streams[1]);
}

int main()
{
    int n = 1024 * 4;

    vector<unsigned long long> q = { 68719403009, 68719230977, 137438822401 };
    unsigned long long q_array[] = { 68719403009, 68719230977, 137438822401 };
    vector<unsigned long long> psi_roots = { 24250113, 29008497, 8625844 };
    vector<unsigned long long> psiinv_roots = { 60243494989, 30331733829, 8970793855 };
    vector<unsigned long long> mu_array = {};
    vector<unsigned> q_bit_lengths = { 36, 36, 37 };
    unsigned q_amount = q.size();
    vector<unsigned long long> inv_q_last_mod_q = { 20955999355, 17095778744 };
    unsigned long long q_mod_t = 1;
    unsigned long long qi_div_t_rns_array[] = { 67108792, 67108624, 134217600 };

    // run operations on different q's with different streams
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * q_amount);
    for (int i = 0; i < q_amount; i++)
        cudaStreamCreate(&streams[i]);

    unsigned long long* q_array_device;
    cudaMalloc(&q_array_device, q_amount * sizeof(unsigned long long));
    cudaMemcpy(q_array_device, q_array, q_amount * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    unsigned long long* qi_div_t_rns_array_device;
    cudaMalloc(&qi_div_t_rns_array_device, q_amount * sizeof(unsigned long long));
    cudaMemcpy(qi_div_t_rns_array_device, qi_div_t_rns_array, q_amount * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    unsigned long long m = 100;  // our message to encrypt
    unsigned long long m_len = log2(m) + 1;  // length of m

    unsigned long long t = 1024;  // mathematical stuff that is beyond our comprehension

    unsigned char* in;
    cudaMalloc(&in, (sizeof(char) + sizeof(unsigned long long)) * q_amount * n + sizeof(unsigned) * n);

    unsigned long long** secret_key = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&secret_key[i], sizeof(unsigned long long) * n);
    }

    // we always have 2 public keys
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

    generate_random_default(in, (sizeof(char) + sizeof(unsigned long long)) * q_amount * n + sizeof(unsigned) * n);

    // convert random bytes to ternary distribution
    // use same byte sequence for each element of the secret key
    for (int i = 0; i < q_amount; i++)
    {
        ternary_dist(in, secret_key[i], n, streams[i], q[i]);
    }

    // convert random bytes to uniform distribution
    // use different byte sequences for each q
    for (int i = 0; i < q_amount; i++)
    {
        uniform_dist((unsigned long long*)(in + q_amount * n + i * n * sizeof(unsigned long long)), public_key[1][i], n, streams[i], q[i]);
    }

    // a temp array to store gaussian distribution values (e)
    unsigned long long** temp = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&temp[i], sizeof(unsigned long long) * n);
    }

    for (int i = 0; i < q_amount; i++)
    {
        gaussian_dist((unsigned*)(in + q_amount * n + q_amount * n * sizeof(unsigned long long)), temp[i], n, streams[i], q[i]);
    }

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

    forwardNTT(secret_key[0], n, streams[0], q[0], mu_array[0], q_bit_lengths[0], psi_table_device[0]);
    forwardNTT(secret_key[1], n, streams[1], q[1], mu_array[1], q_bit_lengths[1], psi_table_device[1]);
    forwardNTT(secret_key[2], n, streams[2], q[2], mu_array[2], q_bit_lengths[2], psi_table_device[2]);

    for (int i = 0; i < q_amount; i++)
    {
        cudaMemcpyAsync(public_key[0][i], public_key[1][i], sizeof(unsigned long long) * n, cudaMemcpyDeviceToDevice, streams[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        barrett << <n / 256, 256, 0, streams[i] >> > (public_key[0][i], secret_key[i], q[i], mu_array[i], q_bit_lengths[i]);
        inverseNTT(public_key[0][i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psiinv_table_device[i]);
        poly_add_device(public_key[0][i], temp[i], n, streams[i], q[i]);
        poly_negate_device(public_key[0][i], n, streams[i], q[i]);
    }

    forwardNTT(public_key[0][0], n, streams[0], q[0], mu_array[0], q_bit_lengths[0], psi_table_device[0]);
    forwardNTT(public_key[0][1], n, streams[1], q[1], mu_array[1], q_bit_lengths[1], psi_table_device[1]);
    forwardNTT(public_key[0][2], n, streams[2], q[2], mu_array[2], q_bit_lengths[2], psi_table_device[2]);

    unsigned long long* c0;
    unsigned long long* c1;
    cudaMalloc(&c0, sizeof(unsigned long long) * n * q_amount);
    cudaMalloc(&c1, sizeof(unsigned long long) * n * q_amount);

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

    vector<unsigned long long> punctured_q = {};
    vector<unsigned long long> inv_punctured_q = { 26179219651, 42540076863 };
    unsigned long long gamma = 2305843009213683713;
    unsigned long long gamma_div_2 = gamma >> 1;

    vector<unsigned long long> output_base = { t, gamma };
    vector<unsigned> output_base_bit_lengths = { 10, 61 };
    unsigned long long mu_gamma;

    vector<unsigned long long> neg_inv_q_mod_t_gamma = { 1023, 803320262470649134 };

    q.pop_back();
    q_amount = q.size();

    //generate mu parameter of gamma for barrett
    {
        unsigned int bit_length = output_base_bit_lengths[1];
        uint128_t mu1 = uint128_t::exp2(2 * bit_length);
        mu1 = mu1 / gamma;
        mu_gamma = mu1.low;
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
    unsigned long long* base_change_matrix = (unsigned long long*)malloc(sizeof(unsigned long long) * q_amount * 2);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < q_amount; j++)
        {
            //base_change_matrix = [t0, t1, t2, 
                                  //g0, g1, g2] but flattened xd
            base_change_matrix[i * q_amount + j] = punctured_q[j] % output_base[i];
        }
    }

    /*for (int i = 0; i < q_amount * 2; i++)
    {
        printf("%llu\n", base_change_matrix[i]);
    }*/

    //allocate memory for base change matrix on device and copy the values to it
    unsigned long long* base_change_matrix_device;
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&base_change_matrix_device, sizeof(unsigned long long) * 2 * q_amount);
        cudaMemcpy(base_change_matrix_device, base_change_matrix, sizeof(unsigned long long) * 2 * q_amount, cudaMemcpyHostToDevice);
    }

    decryption_rns(c0, c1, secret_key, q, q_bit_lengths, mu_array, psi_table_device, psiinv_table_device,
        n, q_amount, inv_punctured_q, base_change_matrix_device, t, gamma, mu_gamma, output_base, output_base_bit_lengths,
        neg_inv_q_mod_t_gamma, gamma_div_2);

    unsigned long long plain_poly[4096];

    cudaMemcpy(plain_poly, c0 + n * (q_amount - 1), n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cout << "[";
    for (int i = 0; i < n; i++)
    {
        cout << plain_poly[i] << ", ";
    }
    cout << "]uyhug";

    return 0;
}
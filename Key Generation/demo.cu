#include <vector>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;

#include "helper.h"
#include "parameter.h"

#include "poly_arithmetic.cuh"
#include "distributions.cuh"
#include "bfv_encryption.cuh"

void decryption_rns(unsigned long long* c, unsigned long long** secret_key,
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

    for (int i = 0; i < q_amount; i++)
    {
        half_poly_mul_device(c + i * n + (q_amount + 1) * n, secret_key[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n, psiinv_table_device + i * n);
        // c1 = c1 * sk

        poly_add_device(c + i * n + (q_amount + 1) * n, c + i * n, n, streams[i], q[i]);
        // c1 = c1 + c0
    }

    for (int i = 0; i < q_amount; i++)
    {
        poly_mul_int(c + i * n + (q_amount + 1) * n, prod_t_gamma_mod_q[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i]);
        // c1 = c1 * prod_t_gamma_mod_q
    }

    // start of fast convert array

    // c1 / punc_q mod q
    for (int i = 0; i < q_amount; i++)
    {
        poly_mul_int(c + i * n + (q_amount + 1) * n, inv_punctured_q[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i]);
    }

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

void keygen_rns(unsigned char in[], int q_amount, unsigned long long* q, unsigned n, unsigned long long** secret_key, unsigned long long* public_key,
    cudaStream_t* streams, unsigned long long** temp, vector<unsigned long long> mu_array, vector<unsigned> q_bit_lengths,
    unsigned long long* psi_table_device, unsigned long long* psiinv_table_device)
{
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
        uniform_dist((unsigned long long*)(in + n + i * n * sizeof(unsigned long long)), public_key + i * n + q_amount * n, n, streams[i], q[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        gaussian_dist((unsigned*)(in + n + q_amount * n * sizeof(unsigned long long)), temp[i], n, streams[i], q[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        forwardNTT(secret_key[i], n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n);
    }

    for (int i = 0; i < q_amount; i++)
    {
        cudaMemcpyAsync(public_key + i * n, public_key + i * n + q_amount * n, sizeof(unsigned long long) * n, cudaMemcpyDeviceToDevice, streams[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        barrett << <n / 256, 256, 0, streams[i] >> > (public_key + i * n, secret_key[i], q[i], mu_array[i], q_bit_lengths[i]);
        inverseNTT(public_key + i * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psiinv_table_device + i * n);
        poly_add_device(public_key + i * n, temp[i], n, streams[i], q[i]);
        poly_negate_device(public_key + i * n, n, streams[i], q[i]);
    }

    for (int i = 0; i < q_amount; i++)
    {
        forwardNTT(public_key + i * n, n, streams[i], q[i], mu_array[i], q_bit_lengths[i], psi_table_device + i * n);
    }
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    int n = 1024 * 8;

    unsigned long long t = 1024;  // mathematical stuff that is beyond our comprehension
    
    /*unsigned long long q_array[] = { 36028797012606977, 36028797010444289, 36028797009985537, 36028797005856769, 36028797005529089, 36028797005135873, 36028797003694081, 36028797003563009, 36028797001138177 };
    vector<unsigned long long> psi_roots = { 768741990072, 3911086673862, 5947090524825, 47595902954, 2691682578057, 3903338373, 235185854118, 1769787302793, 3151164484090 };*/

    /*unsigned long long q_array[] = { 274877562881, 274877202433, 274877153281 };
    vector<unsigned long long> psi_roots = { 71485851, 33872056, 22399294 };*/

    /*unsigned long long q_array[] = { 68719403009, 68719230977, 137438822401 };
    vector<unsigned long long> psi_roots = { 24250113, 29008497, 8625844 };*/

    unsigned long long q_array[] = { 8796092858369, 8796092792833, 17592186028033, 17592185438209 };
    vector<unsigned long long> psi_roots = { 1734247217, 304486499, 331339694, 9366611238 };

    /*unsigned long long q_array[] = { 1125899904679937, 1125899903991809, 1125899903827969, 1125899903795201, 1125899903500289 };
    vector<unsigned long long> psi_roots = { 184459094098, 125929543876, 13806300337, 10351677219, 68423600398 };*/

    /*unsigned long long q_array[] = { 281474976546817, 281474976317441, 281474975662081, 562949952798721, 562949952700417, 562949952274433, 562949951979521, 562949951881217, 1125899904679937 };
    vector<unsigned long long> psi_roots = { 23720796222, 21741529212, 13412349256, 1196930505, 31695302805, 6575376104, 394024808, 45092463253, 184459094098 };*/

    vector<unsigned long long> psiinv_roots;
    vector<unsigned long long> mu_array = {};
    unsigned q_amount = sizeof(q_array) / sizeof(unsigned long long);
    vector<unsigned> q_bit_lengths;
    unsigned* q_bit_lengths_p = (unsigned*)malloc(sizeof(unsigned) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        q_bit_lengths.push_back(log2(q_array[i]) + 1);
        q_bit_lengths_p[i] = q_bit_lengths[i];
    }
    cudaMemcpyToSymbolAsync(q_bit_cons, q_bit_lengths_p, sizeof(unsigned long long) * q_amount, 0, cudaMemcpyHostToDevice, 0);
    vector<unsigned long long> inv_q_last_mod_q;
    unsigned long long* inv_q_last_mod_q_p = (unsigned long long*)malloc(sizeof(unsigned long long) * (q_amount - 1));
    for (int i = 0; i < q_amount - 1; i++)
    {
        inv_q_last_mod_q.push_back(modinv128(q_array[q_amount - 1] % q_array[i], q_array[i]));
        inv_q_last_mod_q_p[i] = inv_q_last_mod_q[i];
    }
    cudaMemcpyToSymbolAsync(inv_q_last_mod_q_cons, inv_q_last_mod_q_p, sizeof(unsigned long long) * (q_amount - 1), 0, cudaMemcpyHostToDevice, 0);
    unsigned long long q_mod_t = 1;
    //unsigned long long qi_div_t_rns_array[] = { 67108792, 67108624, 134217600 };
    unsigned long long* qi_div_t_rns_array = (unsigned long long*)malloc(sizeof(unsigned long long) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        qi_div_t_rns_array[i] = q_array[i] / t;
    }
    vector<unsigned long long> punctured_q = {};
    //vector<unsigned long long> inv_punctured_q = { 26179219651, 42540076863 };
    vector<unsigned long long> inv_punctured_q;
    unsigned long long gamma = 2305843009213683713;
    unsigned long long gamma_div_2 = gamma >> 1;

    for (int i = 0; i < q_amount; i++)
        psiinv_roots.push_back(modinv128(psi_roots[i], q_array[i]));

    vector<unsigned long long> output_base = { t, gamma };
    vector<unsigned> output_base_bit_lengths = { 10, 61 };
    unsigned long long mu_gamma;

    unsigned long long mult_t = 1, mult_g = 1;
    for (int i = 0; i < q_amount - 1; i++)
    {
        mult_t = (host64x2(mult_t, q_array[i]) % t).low;
        mult_g = (host64x2(mult_g, q_array[i]) % gamma).low;
    }
    mult_t = t - modinv128(mult_t, t);
    mult_g = gamma - modinv128(mult_g, gamma);

    vector<unsigned long long> neg_inv_qs_mult_mod_t_gamma = { mult_t, mult_g }; // all qs multiplied mod t and gamma then taken inverse then negated

    //vector<unsigned long long> prod_t_gamma_mod_q = { 37067052033, 64547873793 };
    vector<unsigned long long> prod_t_gamma_mod_q;
    uint128_t prod_t_gamma = host64x2(t, gamma);
    for (int i = 0; i < q_amount; i++)
    {
        prod_t_gamma_mod_q.push_back((prod_t_gamma % q_array[i]).low);
    }

    cudaMemcpyToSymbolAsync(q_cons, q_array, sizeof(unsigned long long) * q_amount, 0, cudaMemcpyHostToDevice);

    // run operations on different q's with different streams
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * q_amount * 2);
    for (int i = 0; i < q_amount * 2; i++)
        cudaStreamCreate(&streams[i]);

    unsigned long long* q_array_device;
    cudaMalloc(&q_array_device, q_amount * sizeof(unsigned long long));
    cudaMemcpy(q_array_device, q_array, q_amount * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    unsigned long long* qi_div_t_rns_array_device;
    cudaMalloc(&qi_div_t_rns_array_device, q_amount * sizeof(unsigned long long));
    cudaMemcpy(qi_div_t_rns_array_device, qi_div_t_rns_array, q_amount * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    unsigned char* in;
    cudaMalloc(&in, (sizeof(char) + sizeof(unsigned long long)) * q_amount * n + sizeof(unsigned) * n);

    unsigned long long** secret_key = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&secret_key[i], sizeof(unsigned long long) * n);
    }

    // we always have 2 public keys
    unsigned long long* public_key;
    cudaMalloc(&public_key, sizeof(unsigned long long) * 2 * n * q_amount);

    // a temp array to store gaussian distribution values (e)
    unsigned long long** temp = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&temp[i], sizeof(unsigned long long) * n);
    }

    //generate mu parameters for barrett
    for (int i = 0; i < q_amount; i++)
    {
        unsigned int bit_length = q_bit_lengths[i];
        uint128_t mu1 = uint128_t::exp2(2 * bit_length);
        mu1 = mu1 / q_array[i];
        unsigned long long mu = mu1.low;

        mu_array.push_back(mu);
    }

    unsigned long long* mu_array_p = (unsigned long long*)malloc(sizeof(unsigned long long) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        mu_array_p[i] = mu_array[i];
    }
    cudaMemcpyToSymbolAsync(mu_cons, mu_array_p, sizeof(unsigned long long) * q_amount, 0, cudaMemcpyHostToDevice, 0);

    //allocate memory for powers of psi root and psi inverse root
    //and fill those arrays
    unsigned long long** psi_table = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    unsigned long long** psiinv_table = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        psi_table[i] = (unsigned long long*)malloc(sizeof(unsigned long long) * n);
        psiinv_table[i] = (unsigned long long*)malloc(sizeof(unsigned long long) * n);

        fillTablePsi128(psi_roots[i], q_array[i], psiinv_roots[i], psi_table[i], psiinv_table[i], n);
    }

    //allocate memory for powers of psi root and psi inverse root on device
    //and copy their values from host to device
    unsigned long long* psi_table_device = (unsigned long long*)malloc(sizeof(unsigned long long) * q_amount * n);
    unsigned long long* psiinv_table_device = (unsigned long long*)malloc(sizeof(unsigned long long) * q_amount * n);
    cudaMalloc(&psi_table_device, sizeof(unsigned long long) * n * q_amount);
    cudaMalloc(&psiinv_table_device, sizeof(unsigned long long) * n * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMemcpy(psi_table_device + i * n, psi_table[i], sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(psiinv_table_device + i * n, psiinv_table[i], sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);
    }

    unsigned long long* c;
    cudaMalloc(&c, sizeof(unsigned long long) * n * q_amount * 2);

    unsigned long long** u = (unsigned long long**)malloc(sizeof(unsigned long long*) * q_amount);
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&u[i], sizeof(unsigned long long) * n);
    }

    unsigned long long* e;
    cudaMalloc(&e, sizeof(unsigned long long) * n * q_amount * 2);

    unsigned long long* m_poly = (unsigned long long*)malloc(sizeof(unsigned long long) * n);

    randomArray128(m_poly, n, t);

    unsigned long long* m_poly_device;
    cudaMalloc(&m_poly_device, n * sizeof(unsigned long long));
    cudaMemcpy(m_poly_device, m_poly, n * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    q_amount--;

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

            temp = host64x2(temp.low, q_array[j]) % q_array[i];
        }

        punctured_q.push_back(temp.low);
        inv_punctured_q.push_back(modinv128(temp.low, q_array[i]));
    }

    //pre-compute the values for base change matrix
    unsigned long long* base_change_matrix = (unsigned long long*)malloc(sizeof(unsigned long long) * q_amount * 2);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < q_amount; j++)
        {
            //base_change_matrix = [t0, t1, t2, 
                                  //g0, g1, g2] but flattened xd
            uint128_t temp = 1;
            for (int k = 0; k < q_amount; k++)
            {
                if (j == k)
                    continue;
                temp = host64x2(temp.low, q_array[k]) % output_base[i];
            }
            base_change_matrix[i * q_amount + j] = temp.low;
        }
    }

    //allocate memory for base change matrix on device and copy the values to it
    unsigned long long* base_change_matrix_device;
    for (int i = 0; i < q_amount; i++)
    {
        cudaMalloc(&base_change_matrix_device, sizeof(unsigned long long) * 2 * q_amount);
        cudaMemcpy(base_change_matrix_device, base_change_matrix, sizeof(unsigned long long) * 2 * q_amount, cudaMemcpyHostToDevice);
    }

    keygen_rns(in, q_amount + 1, q_array, n, secret_key, public_key, streams, temp, mu_array, q_bit_lengths, psi_table_device, psiinv_table_device);

    cudaEventRecord(start);
    encryption_rns(c, public_key, in, u, e, n, streams, q_array, q_bit_lengths, mu_array,
        inv_q_last_mod_q, psi_table_device, psiinv_table_device, m_poly_device, qi_div_t_rns_array_device, q_array_device, t, q_amount + 1);
    cudaEventRecord(stop);

    decryption_rns(c, secret_key, q_array, q_bit_lengths, mu_array, psi_table_device, psiinv_table_device,
        n, q_amount, inv_punctured_q, base_change_matrix_device, t, gamma, mu_gamma, output_base, output_base_bit_lengths,
        neg_inv_qs_mult_mod_t_gamma, gamma_div_2, prod_t_gamma_mod_q);

    unsigned long long* plain_poly = (unsigned long long*)malloc(sizeof(unsigned long long) * n);

    cudaMemcpy(plain_poly, c + n * (q_amount - 1), n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++)
    {
        if (m_poly[i] != plain_poly[i])
            cout << "Error " << i << endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << milliseconds << " millisec." << endl;

    /*cout << "[";
    for (int i = 0; i < 5; i++)
    {
        cout << plain_poly[i] << ", ";
    }
    cout << "]";*/

    return 0;
}
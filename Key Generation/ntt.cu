#include "ntt.cuh"

__device__ __forceinline__ void singleBarrett(uint128_t& a, unsigned long long& q, unsigned long long& mu, int& qbit)
{
    uint128_t rx;

    rx = a >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(a, rx);

    if (a >= q)
        a -= q;
}

template<unsigned l, unsigned N>
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[])
{
    register int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

#pragma unroll
    for (int length = l; length < N; length *= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / 1024 / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q)
                target_result -= q;

            shared_array[target_index] = target_result;

            if (first_target_value < second_target_value)
                first_target_value += q;

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }

}

template<unsigned l, unsigned N>
__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[])
{
    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

    register unsigned long long q2 = (q + 1) >> 1;

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (N / l)];
    }

    __syncthreads();

#pragma unroll
    for (int length = (N / 2); length >= l; length /= 2)
    {
        register int step = (N / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (N / 1024 / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (N / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            if (target_result >= q)
                target_result -= q;

            if (target_result & 1)
                shared_array[target_index] = (target_result >> 1) + q2;
            else
                shared_array[target_index] = (target_result >> 1);

            if (first_target_value < second_target_value)
                first_target_value += q;

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);

            register unsigned long long temp_storage_low = temp_storage.low;
            if (temp_storage_low & 1)
                shared_array[target_index + step] = (temp_storage_low >> 1) + q2;
            else
                shared_array[target_index + step] = (temp_storage_low >> 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (N / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (N / l)] = shared_array[global_tid];
    }
}

template<unsigned l, unsigned N>
__global__ void CTBasedNTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[])
{
    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psi = psi_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_result = first_target_value + second_target_value;

    if (target_result >= q)
        target_result -= q;

    a[target_index] = target_result;

    if (first_target_value < second_target_value)
        first_target_value += q;

    a[target_index + step] = first_target_value - second_target_value;
}

template<unsigned l, unsigned N>
__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[])
{
    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (N / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psiinv = psiinv_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    if (target_result >= q)
        target_result -= q;

    register unsigned long long q2 = (q + 1) >> 1;

    if (target_result & 1)
        target_result = (target_result >> 1) + q2;
    else
        target_result = (target_result >> 1);

    a[target_index] = target_result;

    if (first_target_value < second_target_value)
        first_target_value += q;

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);

    register unsigned long long temp_storage_low = temp_storage.low;
    if (temp_storage_low & 1)
        temp_storage_low = (temp_storage_low >> 1) + q2;
    else
        temp_storage_low = (temp_storage_low >> 1);

    a[target_index + step] = temp_storage_low;
}

__host__ void forwardNTTdouble(unsigned long long* device_a, unsigned long long* device_b, unsigned N, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{
    if (N == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (N == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (N == 8192)
    {
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (N == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
}

__host__ void forwardNTT(unsigned long long* device_a, unsigned N, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{
    if (N == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (N == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (N == 8192)
    {
        CTBasedNTTInner<1, 8192> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else if (N == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
    else
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
    }
}

__host__ void inverseNTT(unsigned long long* device_a, unsigned N, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psiinv_powers)
{
    if (N == 32768)
    {
        GSBasedINTTInnerSingle<16, 32768> << <16, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<8, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (N == 16384)
    {
        GSBasedINTTInnerSingle<8, 16384> << <8, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<4, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (N == 8192)
    {
        GSBasedINTTInnerSingle<4, 8192> << <4, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<2, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
        GSBasedINTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else if (N == 4096)
    {
        GSBasedINTTInnerSingle<2, 4096> << <2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);

        GSBasedINTTInner<1, 4096> << <4096 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
    else
    {
        GSBasedINTTInnerSingle<1, 2048> << <1, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);
    }
}
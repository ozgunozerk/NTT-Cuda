#pragma once

#include <stdlib.h>

#include "uint128.h"
#include <random>

unsigned long long modpow128(unsigned long long a, unsigned long long b, unsigned long long mod)
{
    unsigned long long res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        uint128_t t128 = host64x2(a, a);
        a = (t128 % mod).low;
        if (b & 1)
        {
            uint128_t r128 = host64x2(res, a);
            res = (r128 % mod).low;
        }

    }
    return res;
}

unsigned modpow64(unsigned a, unsigned b, unsigned mod)
{
    unsigned res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        unsigned long long t64 = (unsigned long long)a * a;
        a = t64 % mod;
        if (b & 1)
        {
            unsigned long long r64 = (unsigned long long)a * res;
            res = r64 % mod;
        }

    }
    return res;
}

unsigned long long bitReverse(unsigned long long a, int bit_length)
{
    unsigned long long res = 0;

    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    return res;
}

std::random_device dev;
std::mt19937_64 rng(dev());

void randomArray128(unsigned long long a[], int n, unsigned long long q)
{
    std::uniform_int_distribution<unsigned long long> randnum(0, q);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void randomArray64(unsigned a[], int n, unsigned q)
{
    std::uniform_int_distribution<unsigned> randnum(0, q);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

unsigned long long* refPolyMul128(unsigned long long a[], unsigned long long b[], unsigned long long m, int n)
{
    unsigned long long* c = (unsigned long long*)malloc(sizeof(unsigned long long) * n * 2);
    unsigned long long* d = (unsigned long long*)malloc(sizeof(unsigned long long) * n);

    for (int i = 0; i < (n * 2); i++)
    {
        c[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i + j] = (host64x2(a[i], b[j]) % m).low + c[i + j] % m;
            c[i + j] %= m;
        }
    }

    for (int i = 0; i < n; i++)
    {

        if (c[i] < c[i + n])
            c[i] += m;

        d[i] = (c[i] - c[i + n]) % m;
    }

    free(c);

    return d;
}

unsigned* refPolyMul64(unsigned a[], unsigned b[], unsigned m, int n)
{
    unsigned* c = (unsigned*)malloc(sizeof(unsigned) * n * 2);
    unsigned* d = (unsigned*)malloc(sizeof(unsigned) * n);

    for (int i = 0; i < (n * 2); i++)
    {
        c[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i + j] = ((unsigned long long)a[i] * b[j]) % m + c[i + j] % m;
            c[i + j] %= m;
        }
    }

    for (int i = 0; i < n; i++)
    {

        if (c[i] < c[i + n])
            c[i] += m;

        d[i] = (c[i] - c[i + n]) % m;
    }

    free(c);

    return d;
}


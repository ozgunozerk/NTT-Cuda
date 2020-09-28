#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <string>
#include <math.h>

class uint128_t
{
public:
	
	uint64_t low;
	uint64_t high;

	__host__ __device__ __forceinline__ uint128_t()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t(const uint64_t& x)
	{
		low = x;
		high = 0;
	}

	__device__ __forceinline__ uint128_t(const uint64_t& x, const uint64_t& y)
	{
		low = x * y;
		high = __umul64hi(x, y);
	}

	__device__ __forceinline__ static void mult2x64(uint128_t& r, const uint64_t& x, const uint64_t& y)
	{
		r.high = __umul64hi(x, y);
		r.low = x * y;
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_t& r)
	{
		low = r.low;
		high = r.high;
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)
	{
		low = r;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t operator<<(const unsigned& shift)
	{
		uint128_t z;

		z.high = high << shift;
		z.high = (low >> (64 - shift)) | z.high;
		z.low = low << shift;

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator>>(const unsigned& shift)
	{
		uint128_t z;

		z.low = low >> shift;
		z.low = (high << (64 - shift)) | z.low;
		z.high = high >> shift;

		return z;
	}

	__host__ __device__ __forceinline__ static void shiftr(uint128_t& x, const unsigned& shift)
	{
		x.low = x.low >> shift;
		x.low = (x.high << (64 - shift)) | x.low;
		x.high = x.high >> shift;

	}

	__host__ static uint128_t exp2(const int& e)
	{
		uint128_t z;

		if (e < 64)
			z.low = 1ull << e;
		else
			z.high = 1ull << (e - 64);

		return z;
	}

	__host__ static int log_2(const uint128_t& x)
	{
		int z = 0;

		if (x.high != 0)
			z = log2((float)x.high) + 64;
		else
			z = log2((float)x.low);

		return z;
	}

	__host__ __device__ __forceinline__ static int clz(uint128_t x)
	{
		unsigned cnt = 0;

		if (x.high == 0)
		{
			while (x.low != 0)
			{
				cnt++;
				x.low = x.low >> 1;
			}

			return 128 - cnt;
		}		
		else
		{
			while (x.high != 0)
			{
				cnt++;
				x.high = x.high >> 1;
			}

			return 64 - cnt;
		}
	}

};

__host__ __device__ __forceinline__ static void operator<<=(uint128_t& x, const unsigned& shift)
{
	x.low = x.low >> shift;
	x.low = (x.high << (64 - shift)) | x.low;
	x.high = x.high >> shift;

}

__host__ __device__ __forceinline__ bool operator==(const uint128_t& l, const uint128_t& r)
{
	if ((l.low == r.low) && (l.high == r.high))
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_t& l, const uint128_t& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low < r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_t& l, const uint64_t& r)
{
	if (l.high != 0)
		return false;
	else if (l.low > r)
		return false;
	else
		return true;
}

__host__ __device__ __forceinline__ bool operator>(const uint128_t& l, const uint128_t& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low > r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<=(const uint128_t& l, const uint128_t& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low <= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator>=(const uint128_t& l, const uint128_t& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low >= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ uint128_t operator+(const uint128_t& x, const uint128_t& y)
{
	uint128_t z;

	z.low = x.low + y.low;
	z.high = x.high + y.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator+(const uint128_t& x, const uint64_t& y)
{
	uint128_t z;

	z.low = x.low + y;
	z.high = x.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator-(const uint128_t& x, const uint128_t& y)
{
	uint128_t z;

	z.low = x.low - y.low;
	z.high = x.high - y.high - (x.low < y.low);

	return z;
	
}

__host__ __device__ __forceinline__ void operator-=(uint128_t& x, const uint128_t& y)
{
	x.high = x.high - y.high - (x.low < y.low);
	x.low = x.low - y.low;
}

__host__ __device__ __forceinline__ uint128_t operator-(const uint128_t& x, const uint64_t& y)
{
	uint128_t z;

	z.low = x.low - y;
	z.high = x.high - (x.low < y);

	return z;

}

__host__ __device__ __forceinline__ uint128_t operator/(uint128_t x, const uint64_t& y)
{
	uint128_t z;
	uint128_t ycomp(y);
	uint128_t d(y);

	unsigned shift = uint128_t::clz(d) - uint128_t::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		z = z << 1;
		if (d <= x)
		{
			x = x - d;
			z = z + 1;
		}
		d = d >> 1;
	}

	z = z << 1;
	if (d <= x)
	{
		x = x - d;
		z = z + 1;
	}
	d = d >> 1;

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator%(uint128_t x, const uint64_t& y)
{
	if (x < y)
		return x;

	uint128_t z;
	uint128_t ycomp(y);
	uint128_t d(y);

	unsigned shift = uint128_t::clz(d) - uint128_t::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		z = z << 1;
		if (d <= x)
		{
			x = x - d;
			z = z + 1;
		}
		d = d >> 1;
	}

	z = z << 1;
	if (d <= x)
	{
		x = x - d;
		z = z + 1;
	}
	d = d >> 1;

	return x;
}

__host__ __device__ __forceinline__ static uint128_t host64x2(const uint64_t& x, const uint64_t& y)
{
	uint128_t z;

	uint128_t ux(x);
	uint128_t uy(y);

	int shift = 0;

	// hello elementary school
	while (uy.low != 0)
	{
		if (uy.low & 1)
		{
			if (shift == 0)
				z = z + ux;
			else
				z = z + (ux << shift);
		}

		shift++;

		uint128_t::shiftr(uy, 1);

	}

	return z;
}

//still slower even with all the inline directives
__device__ __forceinline__ void device64x2n(const uint64_t& x, const uint64_t& y, uint128_t& z)
{
	z.high = __umul64hi(x, y);
	z.low = x * y;
}

//still slower even with all the inline directives
__device__ __forceinline__ void device64x2s(const uint64_t& y, uint128_t& z)
{
	z.high = __umul64hi(z.low, y);
	z.low = z.low * y;
}


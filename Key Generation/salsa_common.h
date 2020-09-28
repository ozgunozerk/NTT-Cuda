#ifndef __SALSA_COMMON_H__
#define __SALSA_COMMON_H__

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#define ROUNDS 20
#ifndef UINT64_MAX
#define	UINT64_MAX (18446744073709551615ULL)
#endif

#define THREADS_PER_BLOCK (128)
#define XSALSA20_CRYPTO_KEYBYTES 32
#define XSALSA20_CRYPTO_NONCEBYTES 24
#define XSALSA20_BLOCKSZ 64
#define CTR_INBLOCK_SZ (16)
#define CTR_KS_SZ (XSALSA20_BLOCKSZ)
#define BLOCKS_PER_CHUNK_1X 4
#define BLOCKS_PER_CHUNK_2X 1
#define NUM_STREAMS 1

#define convertBlockSize 64

#define dstdev 3.2
#define dmean 0

#endif

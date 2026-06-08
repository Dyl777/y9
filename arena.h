#pragma once
#ifndef ARENA_H
#define ARENA_H
#include<stdlib.h>
/* Per-file SIMD is inlined in headers like tensor.h; avoid including

 * the deprecated central simd.h here to prevent conflicting typedefs

 * (y9_v8f) when headers inline different backends.

 */


typedef struct Arena {
	char *base;
	size_t size;
	size_t offset;
} Arena;

// prototype definitions
void arena_init(Arena *A, int size);
void *arena_alloc(Arena *A, int size);

void arena_init(Arena *A, int size) {
	A->base = (char *)malloc(size);
	A->size = size;
	A->offset = 0;
}

void *arena_alloc(Arena *A, int size) {
	/* SIMD Alignment: choose alignment according to native backend width
	 * (NEON/SSE=16, AVX2=32, AVX-512=64). This keeps arena consistent with
	 * the vector width in use.
	 */
	/* Choose alignment locally via native compiler feature macros.

	 * This avoids depending on any shared library-specific define and

	 * lets each file express its alignment policy with #if/#elif.

	 */

	#if defined(__AVX512F__)
		size_t align = 64;
	#elif defined(__AVX2__)
		size_t align = 32;
	#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
		size_t align = 16;
	#elif defined(__SSE4_2__) || defined(__SSE2__)
		size_t align = 16;
	#elif defined(__ALTIVEC__)
		size_t align = 16;
	#else
	size_t align = 8; 

	#endif

	/* Bitmask alignment: (addr + align-1) & ~(align-1) rounds up to boundary
	 * Example (align=32): addr=100, 100+31=131, 131 & ~31 = 128
	 */
	size_t current = (size_t)(A->base + A->offset);
	size_t aligned = (current + align - 1) & ~(align - 1);

	size_t padding = aligned - current;
	A->offset += padding;

	void *ptr = A->base + A->offset;
	A->offset += size;
	return ptr;
}	


/* SIMD Alignment Requirements:
     * - SSE (128-bit): 16-byte alignment
     * - AVX/AVX2 (256-bit): 32-byte alignment  <-- We use this
     * - AVX-512 (512-bit): 64-byte alignment
     * - ARM NEON (128-bit): 16-byte alignment
     *
     * Unaligned loads/stores work on x86 but cross cache-line boundaries
     * and cause performance penalties. Aligned is always faster.
     *
     * We use 32-byte alignment for AVX2 compatibility. This covers
     * SSE (16B divides into 32B) and works for AVX-512 if we upgrade later.
	 Bitmask alignment: (addr + 31) & ~31 rounds up to next 32B boundary
     * Example: addr=100 (0x64), 100+31=131, 131 & ~31 = 128 (0x80)
     */

#endif
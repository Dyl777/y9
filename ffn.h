#pragma once
#ifndef FFN_H
#define FFN_H
#include "tensor.h"
#include "config.h"

/* SIMD API is provided by tensor.h (inlined there). Do not redefine here. */

#if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
#include <omp.h>
#endif

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

typedef struct {
	// create the required tensors
	Tensor *w1;
	Tensor *w2;
	Tensor *h1;
	Tensor *a1; //2 neurons

	// save weights
	Tensor *dw1;
	Tensor *dw2;
	Tensor *da1;
	Tensor *dh1;

	// output tensor
	Tensor *out;
	Tensor *inputs;
	bool save_inputs;
	// dw2

    //use a linked list to construct more layers
    //without doing one by one

} FFN;

Tensor *ffn_forward(Arena *A, Tensor *x, FFN *y);
Tensor *__nag_ffn_forward(Arena *A, Tensor *x, FFN *f);
Tensor *ffn_backward(Arena *A, FFN *f, Tensor *x, Tensor *loss);
FFN *ffn_create(Arena *A, int input_dim, int hidden_dim);
Tensor *relu(Tensor *x);
Tensor *__ag_relu_forward(Tensor *x);
void sgd_optimizer(Tensor *a, Tensor *b, float lr);
void ffn_init_params(FFN *f);
//bool is_exploding(Tensor *x);
//void clip_gradient(Tensor *x);

void sgd_optimizer(Tensor *w, Tensor *dw, float lr) {
	assert(w->info.shape[0] == dw->info.shape[0] && w->info.shape[1] == dw->info.shape[1]);
	int size = tensor_size(w);
	/* OpenMP disabled: small vector (<=4096 elements), thread overhead > benefit */
	/* #if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
	#pragma omp parallel for schedule(static)
	#endif */
	for (int i = 0; i <= size - 8; i += 8) {
		#if defined(__AVX512F__) || defined(__AVX2__)
			__m256 vw = _mm256_loadu_ps(&w->_data[i]);
			__m256 vdw = _mm256_loadu_ps(&dw->_data[i]);
			__m256 vlr = _mm256_set1_ps(lr);
			#if defined(__FMA__)
				vw = _mm256_fnmadd_ps(vdw, vlr, vw);
			#else
				vw = _mm256_sub_ps(vw, _mm256_mul_ps(vdw, vlr));
			#endif
			_mm256_storeu_ps(&w->_data[i], vw);
		#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
			float32x4_t vw0 = vld1q_f32(&w->_data[i]);
			float32x4_t vw1 = vld1q_f32(&w->_data[i + 4]);
			float32x4_t vdw0 = vld1q_f32(&dw->_data[i]);
			float32x4_t vdw1 = vld1q_f32(&dw->_data[i + 4]);
			float32x4_t vlr = vdupq_n_f32(lr);
			vw0 = vmlsq_f32(vw0, vdw0, vlr);
			vw1 = vmlsq_f32(vw1, vdw1, vlr);
			vst1q_f32(&w->_data[i], vw0);
			vst1q_f32(&w->_data[i + 4], vw1);
		#elif defined(__SSE4_2__) || defined(__SSE2__)
			__m128 vw0 = _mm_loadu_ps(&w->_data[i]);
			__m128 vw1 = _mm_loadu_ps(&w->_data[i + 4]);
			__m128 vdw0 = _mm_loadu_ps(&dw->_data[i]);
			__m128 vdw1 = _mm_loadu_ps(&dw->_data[i + 4]);
			__m128 vlr = _mm_set1_ps(lr);
			vw0 = _mm_sub_ps(vw0, _mm_mul_ps(vdw0, vlr));
			vw1 = _mm_sub_ps(vw1, _mm_mul_ps(vdw1, vlr));
			_mm_storeu_ps(&w->_data[i], vw0);
			_mm_storeu_ps(&w->_data[i + 4], vw1);
		#else
			w->_data[i+0] = w->_data[i+0] - lr * dw->_data[i+0];
			w->_data[i+1] = w->_data[i+1] - lr * dw->_data[i+1];
			w->_data[i+2] = w->_data[i+2] - lr * dw->_data[i+2];
			w->_data[i+3] = w->_data[i+3] - lr * dw->_data[i+3];
			w->_data[i+4] = w->_data[i+4] - lr * dw->_data[i+4];
			w->_data[i+5] = w->_data[i+5] - lr * dw->_data[i+5];
			w->_data[i+6] = w->_data[i+6] - lr * dw->_data[i+6];
			w->_data[i+7] = w->_data[i+7] - lr * dw->_data[i+7];
		#endif
	}
	// Scalar tail (not worth parallelizing, just 0-7 elements)
	int tail_start = (size / 8) * 8;
	for (int i = tail_start; i < size; i++) w->_data[i] = w->_data[i] - lr * dw->_data[i];
}

void ffn_init_params(FFN *F) {
	tensor_randomize_weights(F->w1);
	tensor_randomize_weights(F->w1);
}

Tensor *ffn_backward(Arena *A, FFN *f, Tensor *x, Tensor *dout) {

	f->dw2 = __nag_tensor_matmul(A, tensor_transpose(f->a1), dout);
	f->da1 = __nag_tensor_matmul(A, dout, tensor_transpose(f->w2));
	f->dh1 = relu_backward(f->da1, f->h1); 
	f->dw1 = __nag_tensor_matmul(A, tensor_transpose(x), f->dh1);
	Tensor *dx = __nag_tensor_matmul(A, f->dh1, tensor_transpose(f->w1));

	return dx;
}

FFN *ffn_create(Arena *A, int input_dim, int hidden_dim) {
	int ndim = 2;
	int *shape1 = (int*) arena_alloc(A, ndim * sizeof(int));
	int *shape2 = (int*) arena_alloc(A, ndim * sizeof(int));

	shape1[0] = input_dim;
	shape1[1] = hidden_dim;

	shape2[0] = hidden_dim;
	shape2[1] = input_dim;


	FFN *f = (FFN*) arena_alloc(A, sizeof(FFN));
	f->w1 = tensor_create_weights_new(A, ndim, shape1);
	f->w2 = tensor_create_weights_new(A, ndim, shape2);
	//f->inputs = tensor_create_new(A, ndim, shape1);

	return f;
}



// Tensor *ffn_forward(Arena *A, Tensor *x, FFN *f) {
// 	assert(x->info.shape[1] == f->w1->info.shape[0]);
// 	if (f->save_inputs == true) {
// 		f->inputs = x;
// 	}
// 	Tensor *h1 = tensor_matmul(A, x, f->w1);
// 	f->h1 = h1;
// 	f->a1 = relu(f->h1);
// 	assert(f->a1->info.shape[1] == f->w2->info.shape[0]);
// 	f->out = tensor_matmul(A, f->a1, f->w2);
	

// 	return f->out;
// }	

Tensor *__nag_ffn_forward(Arena *A, Tensor *x, FFN *f) {
	assert(x->info.shape[1] == f->w1->info.shape[0]);
	if (f->save_inputs == true) {
		f->inputs = x;
	}
	Tensor *h1 = __nag_tensor_matmul(A, x, f->w1);
	f->h1 = h1;
	f->a1 = __nag_relu_forward(f->h1);
	assert(f->a1->info.shape[1] == f->w2->info.shape[0]);
	f->out = __nag_tensor_matmul(A, f->a1, f->w2);
	return f->out;
}

Tensor *__ag_ffn_forward(Arena *A, Tensor *x, FFN *f) {
	assert(x->info.shape[1] == f->w1->info.shape[0]);
	if (f->save_inputs == true) {
		f->inputs = x;
	}
	Tensor *h1 = __ag_tensor_matmul_forward(A, x, f->w1);
	f->h1 = h1;
	f->a1 = __ag_relu_forward(f->h1);
	assert(f->a1->info.shape[1] == f->w2->info.shape[0]);
	f->out = __ag_tensor_matmul_forward(A, f->a1, f->w2);
	

	return f->out;
}	


// Tensor *relu(Tensor *x) {
// 	int size = x->info.shape[0] * x->info.shape[1];
// 	y9_v8f vzero = y9_v8f_zero();
// 	/* OpenMP disabled: small vector (<=2048 elements), thread overhead > benefit */
// 	/* #if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
// 	#pragma omp parallel for schedule(static)
// 	#endif */
// 	for (int i = 0; i <= size - 8; i += 8) {
// 		y9_v8f v = y9_v8f_loadu(&x->_data[i]);
// 		y9_v8f_storeu(&x->_data[i], y9_v8f_max(v, vzero));
// 	}
// 	// Scalar tail
// 	int tail_start = (size / 8) * 8;
// 	for (int i = tail_start; i < size; i++) x->_data[i] = MAX(0.0f, x->_data[i]);
// 	return x;
// }

Tensor *__ag_relu_forward(Tensor *x) {
	//int size = x->info.shape[0] * x->info.shape[1];
	/* OpenMP disabled: small vector (<=2048 elements), thread overhead > benefit */
	/* #if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
	#pragma omp parallel for schedule(static)
	#endif */
	for (int i = 0; i <= x->info.sz - 8; i += 8) {
		#if defined(__AVX512F__) || defined(__AVX2__)
			__m256 v = _mm256_loadu_ps(&x->_data[i]);
			__m256 vzero = _mm256_setzero_ps();
			v = _mm256_max_ps(v, vzero);
			_mm256_storeu_ps(&x->_data[i], v);
		#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
			float32x4_t v0 = vld1q_f32(&x->_data[i]);
			float32x4_t v1 = vld1q_f32(&x->_data[i + 4]);
			float32x4_t vzero = vdupq_n_f32(0.0f);
			v0 = vmaxq_f32(v0, vzero);
			v1 = vmaxq_f32(v1, vzero);
			vst1q_f32(&x->_data[i], v0);
			vst1q_f32(&x->_data[i + 4], v1);
		#elif defined(__SSE4_2__) || defined(__SSE2__)
			__m128 v0 = _mm_loadu_ps(&x->_data[i]);
			__m128 v1 = _mm_loadu_ps(&x->_data[i + 4]);
			__m128 vzero = _mm_setzero_ps();
			v0 = _mm_max_ps(v0, vzero);
			v1 = _mm_max_ps(v1, vzero);
			_mm_storeu_ps(&x->_data[i], v0);
			_mm_storeu_ps(&x->_data[i + 4], v1);
		#else
			x->_data[i+0] = MAX(0.0f, x->_data[i+0]);
			x->_data[i+1] = MAX(0.0f, x->_data[i+1]);
			x->_data[i+2] = MAX(0.0f, x->_data[i+2]);
			x->_data[i+3] = MAX(0.0f, x->_data[i+3]);
			x->_data[i+4] = MAX(0.0f, x->_data[i+4]);
			x->_data[i+5] = MAX(0.0f, x->_data[i+5]);
			x->_data[i+6] = MAX(0.0f, x->_data[i+6]);
			x->_data[i+7] = MAX(0.0f, x->_data[i+7]);
		#endif
	}
	// Scalar tail
	int tail_start = (x->info.sz / 8) * 8;
	for (int i = tail_start; i < x->info.sz; i++) x->_data[i] = MAX(0.0f, x->_data[i]);
	return x;
}

// Tensor *forward(Arena *A, Tensor *x) {
// 	int shape1[2] = {32, 128};
// 	int shape2[2] = {128, 32};
// 	Tensor *w1 = tensor_create_weights(2, shape1);
// 	Tensor *w2 = tensor_create_weights(2, shape2);
// 	Tensor *h1 = tensor_matmul(A, x, w1);
// 	Tensor *a1 = relu(h1);
// 	Tensor *out = tensor_matmul(A, a1, w2);

// 	return out;
// }	

//int main() {
//	int BACH_SIZE = 10;
//	int EPOCHS = 2;
//	for (int i = 0; i < EPOCHS; i++) {
//		for (int j = 0; j < BACH_SIZE; j++) {
//			// .. Rest of logic comes here
//		}
//	}
//	int ndim = 2;
//	int *shape_tokens = malloc(ndim * sizeof(int));
//	int *shape_weights= malloc(ndim * sizeof(int));
//
//	shape_tokens[0] = SEQ_LEN;
//	shape_tokens[1] = EMB_DIM;
//
//	shape_weights[0] = EMB_DIM;
//	shape_weights[1] = EMB_DIM;
//
//	int num_heads = 8;
//
//	// define token tensors
//	Tensor *tokens = tensor_create(ndim, shape_tokens);
//
//
//	// define FFN weights
//	FFN *f = ffn_create(32, 128);
//	//ffn_backward(f);
//
//	int heads = 8;
//	MHA *mha = mha_create(heads, SEQ_LEN, EMB_DIM);
//	Tensor *score = mha_forward(tokens, mha);
//	Tensor *ln1 = layer_norm(score);
//	Tensor *res = ffn_forward(ln1, f);
//	Tensor *pred = layer_norm(res);
//	
//	// Backward pass functions start here
//	Tensor *target = tensor_create(ndim, shape_tokens);
//	Tensor *loss = tensor_mse_loss(pred, target);
//	Tensor *final = ffn_backward(f, tokens, loss);
//
//	tensor_shape(final);
//	tensor_shape(tokens);
//	tensor_shape(mha->out);
//
//	Tensor *mha_back = mha_backward(mha, final, tokens);
//	tensor_get(mha_back);
//
//	return 0;
//}


#endif
//layernorm
#pragma once
#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"
#include "config.h"

/* SIMD API is provided by tensor.h (inlined there). Do not redefine here. */

#if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
#include <omp.h>
#endif

typedef struct {
	int features; // in out case it's d_model OR embedding dimension,
							// as layer_norm is calculated along last dimention
	Tensor *beta; // Learnable shift. shape: features OR emb_dim
	Tensor *gamma; // Learnable sclae. shape: features OR emb_dim
	Tensor *d_beta;
	Tensor *d_gamma;
	Tensor *x_hat; // Normalized(x)
	float *var; // cached per row variance
} LayerNorm;

Tensor *layer_norm_forward(Arena *A, LayerNorm *ln, Tensor *t);
LayerNorm *layer_norm_create(int features);
LayerNorm *layer_norm_create_new(Arena *A, int features);
void layer_norm_backward(LayerNorm *ln, Tensor *x, Tensor *dy, Tensor *dx, float lr);
void layer_norm_init_params(LayerNorm *ln);



void layer_norm_init_params(LayerNorm *ln) {
	tensor_randomize_weights(ln->beta);
	tensor_randomize_weights(ln->gamma);
	tensor_randomize_weights(ln->d_beta);
	tensor_randomize_weights(ln->d_gamma);
}

Tensor *layer_norm_forward(Arena *A, LayerNorm *ln, Tensor *x) {

    int rows = x->info.shape[0];
    int cols = x->info.shape[1];

    //if (ln->var) free(ln->var);
		// TODO: Check this logic below:
    ln->var = (float*) arena_alloc(A, rows * sizeof(float));

    //if (ln->x_hat) tensor_free(ln->x_hat);
    ln->x_hat = tensor_create_new(A, 2, x->info.shape);

    // output tensor y
    Tensor *y = tensor_create_new(A, 2, x->info.shape);

	/* OpenMP disabled: rows too small (32 elements), thread overhead > benefit */
	/* #if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
	#pragma omp parallel for schedule(static) if(rows > 8)
	#endif */
	for (int i = 0; i < rows; i++) {

        float *x_row   = x->_data + i * cols;
        float *xh_row  = ln->x_hat->_data + i * cols;
        float *y_row   = y->_data + i * cols;

        float mu = mean(x_row, cols);

        // SIMD variance calculation
        float var_sum = 0.0f;
        int k = 0;
        #if defined(__AVX512F__) || defined(__AVX2__)
        	__m256 vmu = _mm256_set1_ps(mu);
        	__m256 vvar = _mm256_setzero_ps();
        	for (; k <= cols - 8; k += 8) {
        		__m256 v = _mm256_loadu_ps(&x_row[k]);
        		__m256 d = _mm256_sub_ps(v, vmu);
        		#if defined(__FMA__)
        			vvar = _mm256_fmadd_ps(d, d, vvar);
        		#else
        			vvar = _mm256_add_ps(vvar, _mm256_mul_ps(d, d));
        		#endif
        	}
        	float tv[8];
        	_mm256_storeu_ps(tv, vvar);
        	var_sum += tv[0] + tv[1] + tv[2] + tv[3] + tv[4] + tv[5] + tv[6] + tv[7];
        #elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
        	float32x4_t vmu4 = vdupq_n_f32(mu);
        	float32x4_t vvar0 = vdupq_n_f32(0.0f);
        	float32x4_t vvar1 = vdupq_n_f32(0.0f);
        	for (; k <= cols - 8; k += 8) {
        		float32x4_t v0 = vld1q_f32(&x_row[k]);
        		float32x4_t v1 = vld1q_f32(&x_row[k + 4]);
        		float32x4_t d0 = vsubq_f32(v0, vmu4);
        		float32x4_t d1 = vsubq_f32(v1, vmu4);
        		vvar0 = vmlaq_f32(vvar0, d0, d0);
        		vvar1 = vmlaq_f32(vvar1, d1, d1);
        	}
        	float tv[8];
        	vst1q_f32(&tv[0], vvar0);
        	vst1q_f32(&tv[4], vvar1);
        	var_sum += tv[0] + tv[1] + tv[2] + tv[3] + tv[4] + tv[5] + tv[6] + tv[7];
        #elif defined(__SSE4_2__) || defined(__SSE2__)
        	__m128 vmu4 = _mm_set1_ps(mu);
        	__m128 vvar0 = _mm_setzero_ps();
        	__m128 vvar1 = _mm_setzero_ps();
        	for (; k <= cols - 8; k += 8) {
        		__m128 v0 = _mm_loadu_ps(&x_row[k]);
        		__m128 v1 = _mm_loadu_ps(&x_row[k + 4]);
        		__m128 d0 = _mm_sub_ps(v0, vmu4);
        		__m128 d1 = _mm_sub_ps(v1, vmu4);
        		vvar0 = _mm_add_ps(vvar0, _mm_mul_ps(d0, d0));
        		vvar1 = _mm_add_ps(vvar1, _mm_mul_ps(d1, d1));
        	}
        	float tv0[4], tv1[4];
        	_mm_storeu_ps(tv0, vvar0);
        	_mm_storeu_ps(tv1, vvar1);
        	var_sum += tv0[0] + tv0[1] + tv0[2] + tv0[3] + tv1[0] + tv1[1] + tv1[2] + tv1[3];
        #endif
        for (; k < cols; k++) {
            float diff = x_row[k] - mu;
            var_sum += diff * diff;
        }

        float var = var_sum / cols;
        ln->var[i] = var;
        float inv_std = 1.0f / sqrtf(var + EPS);
        int c = 0;
        for (; c <= cols - 8; c += 8) {
        	#if defined(__AVX512F__) || defined(__AVX2__)
        		__m256 v = _mm256_loadu_ps(&x_row[c]);
        		__m256 vmu = _mm256_set1_ps(mu);
        		__m256 vinv = _mm256_set1_ps(inv_std);
        		__m256 xhat = _mm256_mul_ps(_mm256_sub_ps(v, vmu), vinv);
        		_mm256_storeu_ps(&xh_row[c], xhat);

        		__m256 vgamma = _mm256_loadu_ps(&ln->gamma->_data[c]);
        		__m256 vbeta  = _mm256_loadu_ps(&ln->beta->_data[c]);
        		#if defined(__FMA__)
        			__m256 vy = _mm256_fmadd_ps(vgamma, xhat, vbeta);
        		#else
        			__m256 vy = _mm256_add_ps(_mm256_mul_ps(vgamma, xhat), vbeta);
        		#endif
        		_mm256_storeu_ps(&y_row[c], vy);
        	#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
        		float32x4_t vmu4 = vdupq_n_f32(mu);
        		float32x4_t vinv4 = vdupq_n_f32(inv_std);
        		float32x4_t v0 = vld1q_f32(&x_row[c]);
        		float32x4_t v1 = vld1q_f32(&x_row[c + 4]);
        		float32x4_t xh0 = vmulq_f32(vsubq_f32(v0, vmu4), vinv4);
        		float32x4_t xh1 = vmulq_f32(vsubq_f32(v1, vmu4), vinv4);
        		vst1q_f32(&xh_row[c], xh0);
        		vst1q_f32(&xh_row[c + 4], xh1);

        		float32x4_t g0 = vld1q_f32(&ln->gamma->_data[c]);
        		float32x4_t g1 = vld1q_f32(&ln->gamma->_data[c + 4]);
        		float32x4_t b0 = vld1q_f32(&ln->beta->_data[c]);
        		float32x4_t b1 = vld1q_f32(&ln->beta->_data[c + 4]);
        		float32x4_t y0 = vmlaq_f32(b0, g0, xh0);
        		float32x4_t y1 = vmlaq_f32(b1, g1, xh1);
        		vst1q_f32(&y_row[c], y0);
        		vst1q_f32(&y_row[c + 4], y1);
        	#elif defined(__SSE4_2__) || defined(__SSE2__)
        		__m128 v0 = _mm_loadu_ps(&x_row[c]);
        		__m128 v1 = _mm_loadu_ps(&x_row[c + 4]);
        		__m128 vmu4 = _mm_set1_ps(mu);
        		__m128 vinv4 = _mm_set1_ps(inv_std);
        		__m128 xh0 = _mm_mul_ps(_mm_sub_ps(v0, vmu4), vinv4);
        		__m128 xh1 = _mm_mul_ps(_mm_sub_ps(v1, vmu4), vinv4);
        		_mm_storeu_ps(&xh_row[c], xh0);
        		_mm_storeu_ps(&xh_row[c + 4], xh1);

        		__m128 g0 = _mm_loadu_ps(&ln->gamma->_data[c]);
        		__m128 g1 = _mm_loadu_ps(&ln->gamma->_data[c + 4]);
        		__m128 b0 = _mm_loadu_ps(&ln->beta->_data[c]);
        		__m128 b1 = _mm_loadu_ps(&ln->beta->_data[c + 4]);
        		__m128 y0 = _mm_add_ps(_mm_mul_ps(g0, xh0), b0);
        		__m128 y1 = _mm_add_ps(_mm_mul_ps(g1, xh1), b1);
        		_mm_storeu_ps(&y_row[c], y0);
        		_mm_storeu_ps(&y_row[c + 4], y1);
        	#endif
        }
        for (; c < cols; c++) {
            float x_hat = (x_row[c] - mu) * inv_std;
            xh_row[c] = x_hat;
            y_row[c] = ln->gamma->_data[c] * x_hat + ln->beta->_data[c];
        }
    }

    return y;
}

// MHA -> LAYER NORM -> FFN -> LAYER NORM -> Loss
// Loss -> LAYER NORM -> FFN -> LAYER NORM -> MHA
// Loss(ln2, target) -> LAYER NORM(L2, ffn) -> FFN(ln1, F) -> LAYER NORM(L1, attn) -> input(X)

void layer_norm_backward(LayerNorm *ln, Tensor *x, Tensor *dy, Tensor *dx, float lr) {
    int rows = x->info.shape[0];
    int cols = x->info.shape[1];

    // 1. Calculate gradients for gamma and beta (accumulate over rows)
    for (int i = 0; i < rows; i++) {
        for (int c = 0; c < cols; c++) {
            float dy_val = dy->_data[i * cols + c];
            float xh_val = ln->x_hat->_data[i * cols + c];

            ln->d_gamma->_data[c] += dy_val * xh_val;
            ln->d_beta->_data[c]  += dy_val;
        }
    }

    // 2. Calculate gradient for input x
    for (int i = 0; i < rows; i++) {
        float *dy_row = dy->_data + i * cols;
        float *xh_row = ln->x_hat->_data + i * cols;
        float *dx_row = dx->_data + i * cols;
        
        float var = ln->var[i];
        float inv_std = 1.0f / sqrtf(var + EPS);

        // Intermediate terms for the simplified LayerNorm gradient formula
        float sum_dy_gamma = 0.0f;
        float sum_dy_gamma_xhat = 0.0f;

        for (int c = 0; c < cols; c++) {
            float dy_gamma = dy_row[c] * ln->gamma->_data[c];
            sum_dy_gamma += dy_gamma;
            sum_dy_gamma_xhat += dy_gamma * xh_row[c];
        }

        // The "one-liner" derivative formula for LayerNorm:
        // dx = (1/N) * inv_std * [N*dy_gamma - sum(dy_gamma) - x_hat*sum(dy_gamma*x_hat)]
        for (int c = 0; c < cols; c++) {
            float dy_gamma = dy_row[c] * ln->gamma->_data[c];
            dx_row[c] = (1.0f / cols) * inv_std * (
                (cols * dy_gamma) - sum_dy_gamma - (xh_row[c] * sum_dy_gamma_xhat)
            );
        }
    }
}


//Tensor *layer_norm_forward(LayerNorm *ln, Tensor *t) {
//	int rows = t->shape[0];
//	int cols = t->shape[1];
//	if (ln->var != NULL) free(ln->var);
//	ln->var = malloc(rows * sizeof(float));
//
//	for (int i = 0; i < rows; i++) {
//		float *row = t->data + (i * cols);
//		
//		// row mean
//		float row_mean = mean(row, cols);
//
//		// row variance
//		float var_sum = 0.0f;
//		for (int k = 0; k < cols; k++) {
//			float diff = row[k] - row_mean;
//			var_sum += (diff * diff);
//		}
//		float var_mean = var_sum / (float) cols;
//		
//		ln->var[i] = var_mean;
//
//		for (int c = 0; c < cols; c++) {
//			row[c] = (row[c] - row_mean) / sqrtf(var_mean + EPS);
//		}
//	}
//
//	ln->x_hat = t;
//	t = tensor_scaler_multiplication(t, GEMMA);
//	t = tensor_scaler_addition(t, BETA);
//	return t;
//}

LayerNorm *layer_norm_create(int features) {
	LayerNorm *ln = (LayerNorm*) malloc(sizeof(LayerNorm));
	if (!ln) {
		fprintf(stderr, "Allocation failed\n");
		exit(1);
	}
	ln->features = features;
	int ndim = 2;
	int shape[2] = {1, features};
	ln->beta = tensor_create_weights(ndim, shape);
	ln->gamma = tensor_create_weights(ndim, shape);
	ln->d_gamma = tensor_create_weights(ndim, shape);
	ln->d_beta = tensor_create_weights(ndim, shape);
	ln->x_hat = NULL;
	ln->var = NULL; // forward activations cache initially NULL
	
	return ln;
}

LayerNorm *layer_norm_create_new(Arena *A, int features) {
	LayerNorm *ln = (LayerNorm*) arena_alloc(A, sizeof(LayerNorm));
	//if (!ln) {
	//	fprintf(stderr, "Allocation failed\n");
	//	exit(1);
	//}
	ln->features = features;
	int ndim = 2;
	int *shape = (int*) arena_alloc(A, ndim * sizeof(int));
	shape[0] = 1;
	shape[1] = features;

	ln->beta = tensor_create_weights_new(A, ndim, shape);
	ln->gamma = tensor_create_weights_new(A, ndim, shape);
	ln->d_gamma = tensor_create_weights_new(A, ndim, shape);
	ln->d_beta = tensor_create_weights_new(A, ndim, shape);

	ln->x_hat = NULL;
	ln->var = NULL; // forward activations cache initially NULL
	
	return ln;
}

//int main() {
//	int ndim = 2;
//	int *shape_tokens = malloc(ndim * sizeof(int));
//	
//	shape_tokens[0] = SEQ_LEN;
//	shape_tokens[1] = EMB_DIM;
//
//	LayerNorm *ln = layer_norm_create(EMB_DIM);
//	printf("before:\n");
//	printf("%p\n", ln->x_hat);
//	
//	printf("created layernorm\n");
//	Tensor *tokens = tensor_create(ndim, shape_tokens);
//
//	int *shape_weights = malloc(ndim * sizeof(int));
//
//	shape_weights[0] = EMB_DIM;
//	shape_weights[1] = EMB_DIM;
//
//	int heads = 8;
//
//	MHA *mha = mha_create(heads, SEQ_LEN, EMB_DIM);
//	Tensor *score = mha_forward(tokens, mha);
//	Tensor *t = layer_norm_forward(ln, score);
//	printf("after:\n");
//	tensor_shape(ln->x_hat);
//	printf("success!\n");
//	tensor_shape(t);
//	tensor_get(t);
//	return 0;
//}

Tensor* __ag_layer_norm_forward(Arena *A, LayerNorm *ln, Tensor *x){
    /*instead of float maths, we need Tensor graph maths here
    each operation should have their own relative tensor
    tensor mean, tensor_variance, tensor_add, tensor_sub etc ROW WIS,
    then create computational graph using tensors*/
    
    // Enable gradients on learnable parameters FIRST
    ln->gamma->requires_grad = true;
    ln->beta->requires_grad = true;
    // Also enable on input if not already set
    if(x->requires_grad == false && x->parents == NULL) {
        x->requires_grad = true;
    }
    
    int rows = x->info.shape[0];
    int cols = x->info.shape[1];
    int ndim = x->ndim;

    int* y_shape = arena_alloc(A, ndim * sizeof(int));
    y_shape[0] = rows;
    y_shape[1] = cols;

    Tensor* y = tensor_create_new(A, ndim, y_shape);
    Tensor* mean = __ag_tensor_mean(A, x);
    //Docs reference: pytorch for expand_cols
    Tensor* mean_exp = __ag_tensor_expand_cols_forward(A, mean, x->info.shape[1]);
    Tensor* diff = __ag_tensor_sub_forward(A, x, mean_exp);
    Tensor* sq = __ag_tensor_sq_forward(A, diff);
    Tensor* var = __ag_tensor_mean(A, sq);
    Tensor* var_exp = __ag_tensor_expand_cols_forward(A, var, x->info.shape[1]);
    //Tensor* eps = tensor_fill_like(A, var_exp, 1e-5);
    //Tensor* var_eps = __ag_tensor_add_forward(A, var_exp, eps);
    Tensor* std = __ag_tensor_sqrt_forward(A, var_exp);
    Tensor* out = __ag_tensor_div_forward(A, diff, std);
    
    ln->var = var->_data; //not break backward compatibility and rather use the variance as a tensor only
    //when required and just store it as a float*
    ln->x_hat = out;

    Tensor* gamma_exp = __ag_tensor_expand_rows(A, ln->gamma, rows);
    // Tensor* gamma_transpose = tensor_transpose(ln->gamma);
    // Tensor* gamma_exp = __ag_tensor_expand_cols_forward(A, gamma_transpose, out->info.shape[0]);
    //Tensor* yhat = __ag_tensor_hadamard_mul_forward(A, tensor_transpose(gamma_exp), out);
    Tensor* yhat = __ag_tensor_hadamard_mul_forward(A, gamma_exp, out);


    // Tensor* beta_transpose = tensor_transpose(ln->beta);
    // Tensor* beta_exp = tensor_transpose(__ag_tensor_expand_cols_forward(A, beta_transpose, out->info.shape[0]));


    //Tensor* yhat = __ag_tensor_scaling(A, tensor_transpose(gamma_exp), out);

    Tensor* beta_exp = __ag_tensor_expand_rows(A, ln->beta, rows);

    //tensor_randomize_weights(gamma_exp);
    //tensor_randomize_weights(beta_exp);

    //Tensor* yhat = __ag_tensor_scaling(A, gamma_exp, out);

    y = __ag_tensor_add_forward(A, yhat, beta_exp);
    //y = __ag_tensor_add_forward(A, __ag_tensor_hadamard_mul_forward(A, ln->gamma, out), ln->beta);

    return y;
}


#endif
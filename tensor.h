#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<stdalign.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include "arena.h"

#if defined(__AVX512F__) || defined(__AVX2__)
 #include <immintrin.h>
#elif defined(__SSE4_2__) || defined(__SSE2__)
 #include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
 #include <arm_neon.h>
#endif

#if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
#include <omp.h>
#endif

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX

// CPU feature detection
// static int cpu_features_checked = 0;
// static int has_avx = 0;
// static int has_avx2 = 0;

typedef struct Arena Arena; //the tensor gets all allocated or deallocated together
typedef struct Tensor Tensor;

typedef struct Op {
	void(*backward)(struct Tensor* self);
} Op;

struct Tensor {

    struct { //force compiler to start at cache line
	int *shape;
	int *stride;
    int sz;
    } info;
 
	int ndim;
	float *_data;

	//modifying for autograd using a DAG
	Op* ops; //operations that created this tensor
	Tensor** parents; //array of tensor pointers [*A, *B], i will make it just be 2
	float* grad; 
	int num_parents;
	bool requires_grad; //add this as the last bit to an int, won't matter because cpu instruction set will still take more than 1 cycle
	//why a DAG instead of tree structure??

};

//implement 2 tensor matmul ops, one for forward pass, the other for backward
//forward for building the comp.graph, backwards for computing those gradients 
//and updating params in the optimiser step

// prototypes definition
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_create_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights(int ndim, int *shape);
Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_softmax(Arena* A, Tensor *a);
Tensor *tensor_transpose(Tensor *t);
Tensor *relu_backward(Tensor *x, Tensor *y);
Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target);
Tensor *tensor_mse_loss_manual(Arena *A, Tensor *pred, Tensor *target);
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_scalar_multiplication(Tensor *x, float a);
Tensor *tensor_scalar_addition(Tensor *x, float a);
void tensor_fill_zeros(Tensor *a);
void tensor_add_inplace(Tensor **a, Tensor **b);
//void tensor_free(Tensor *t);
void tensor_get(Tensor *t);
void tensor_check(char *name, Tensor *x);
int tensor_size(Tensor *t);
float loss_value(Tensor *a, Tensor *b);
//void tensor_shape(Tensor *t);
//bool is_exploding(Tensor *x);
void clip_gradient(Tensor *x);

// Non-autograd (__nag_) function declarations
Tensor* __nag_tensor_matmul(Arena* A, Tensor* a, Tensor* b);
Tensor* __nag_tensor_softmax(Arena* A, Tensor* a);
Tensor* __nag_relu_forward(Tensor* x);

// Autograd backward prototypes (must be declared before forwards that reference them)
void __ag_tensor_mean_backward(Tensor* o);
Tensor* __ag_tensor_expand_cols_forward(Arena* A, Tensor* mean, int cols_r);
void __ag_tensor_expand_cols_backward(Tensor* o);
void __ag_tensor_sq_backward(Tensor* o);
void __ag_tensor_sqrt_backward(Tensor* o);
void __ag_tensor_transpose_backward(Tensor* o);
Tensor* __ag_tensor_transpose_forward(Arena* A, Tensor* in);
void __ag_tensor_matmul_backward(Tensor* loss);
void __ag_tensor_add_backward(Tensor* o);
void __ag_tensor_sub_backward(Tensor* o);
void __ag_tensor_div_backward(Tensor* o);
void __ag_tensor_hadamard_mul_backward(Tensor* o);
// Slice/scatter for MHA autograd support
void __ag_tensor_slice_backward(Tensor* o);
Tensor* __ag_tensor_slice_forward(Arena* A, Tensor* in, int start_row, int start_col, int num_rows, int num_cols);
void __ag_tensor_scatter_backward(Tensor* o);
Tensor* __ag_tensor_scatter_forward(Arena* A, Tensor* base, Tensor* src, int start_row, int start_col);

// DAG visualization for autograd
void tensor_print_dag(Tensor* root, const char* name);
void tensor_print_dag_recursive(Tensor* node, int depth, int* visited, int max_visited, const char* prefix);
const char* tensor_get_op_name(Tensor* node);
void tensor_print_node_box(Tensor* node, int depth, const char* prefix, int is_last);
void print_box_line(int width, const char* left, const char* right);

// Autograd backward pass - triggers automatic backward through computation graph
void tensor_backward(Tensor* root);

// Debug helper to check for NaN in gradients
static inline void check_nan_grad(Tensor* t, const char* name) {
	if(!t || !t->grad) return;
	for(int i = 0; i < t->info.sz; i++) {
		if(isnan(t->grad[i]) || isinf(t->grad[i])) {
			printf("[NaN DETECTED] %s->grad[%d] = %f\n", name, i, t->grad[i]);
			return;
		}
	}
}

// new methods added for model struct
void tensor_randomize_weights(Tensor *x);
void tensor_randomize(Tensor *x);

void __ag_tensor_mean_backward(Tensor* o){
	if(!o || !o->parents) return;
   Tensor* a = o->parents[0];
	if(!a) return;
   // ensure grad buffer exists on parent
   int rows = a->info.shape[0];
   int cols = a->info.shape[1];
   if(!a->grad){
	   a->grad = (float*)malloc(rows * cols * sizeof(float));
	   if(a->grad) memset(a->grad, 0, rows * cols * sizeof(float));
   }
   // o is (rows x 1), propagate mean gradient evenly across columns
   for(int r=0;r<rows;r++){
	   float g = o->grad ? o->grad[r] : 0.0f;
	   float per = g / (float)cols;
	   for(int c=0;c<cols;c++) a->grad[r*cols + c] += per;
   }
	return;
}

static inline void ensure_grad_buffer(Tensor* t){
	if(!t) return;
	if(t->grad) return;
	int sz = 1;
	for(int i=0;i<t->ndim;i++) sz *= t->info.shape[i];
	t->info.sz = sz;
	t->grad = (float*)malloc(sz * sizeof(float));
	if(t->grad) memset(t->grad, 0, sz * sizeof(float));
}

float mean(float *arr, int size) {
	float sum = 0.0f;
	int i = 0;
	for (; i <= size - 8; i += 8) {
		#if defined(__AVX512F__) || defined(__AVX2__)
			__m256 v = _mm256_loadu_ps(&arr[i]);
			__m256 tmp = _mm256_hadd_ps(v, v);
			tmp = _mm256_hadd_ps(tmp, tmp);
			float t[8];
			_mm256_storeu_ps(t, tmp);
			sum += t[0] + t[4];
		#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
			float32x4_t v0 = vld1q_f32(&arr[i]);
			float32x4_t v1 = vld1q_f32(&arr[i + 4]);
			float32x2_t s0 = vadd_f32(vget_low_f32(v0), vget_high_f32(v0));
			float32x2_t s1 = vadd_f32(vget_low_f32(v1), vget_high_f32(v1));
			s0 = vpadd_f32(s0, s0);
			s1 = vpadd_f32(s1, s1);
			sum += vget_lane_f32(s0, 0) + vget_lane_f32(s1, 0);
		#elif defined(__SSE4_2__) || defined(__SSE2__)
			__m128 v0 = _mm_loadu_ps(&arr[i]);
			__m128 v1 = _mm_loadu_ps(&arr[i + 4]);
			__m128 sh0 = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 3, 0, 1));
			__m128 sh1 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(2, 3, 0, 1));
			__m128 s0 = _mm_add_ps(v0, sh0);
			__m128 s1 = _mm_add_ps(v1, sh1);
			sh0 = _mm_shuffle_ps(s0, s0, _MM_SHUFFLE(1, 0, 3, 2));
			sh1 = _mm_shuffle_ps(s1, s1, _MM_SHUFFLE(1, 0, 3, 2));
			s0 = _mm_add_ps(s0, sh0);
			s1 = _mm_add_ps(s1, sh1);
			sum += ((float*)&s0)[0] + ((float*)&s1)[0];
		#else
			sum += arr[i+0] + arr[i+1] + arr[i+2] + arr[i+3] + arr[i+4] + arr[i+5] + arr[i+6] + arr[i+7];
		#endif
	}
	for (; i < size; i++) sum += arr[i];
	return sum / (float) size;
}

Tensor* __ag_tensor_mean(Arena* A, Tensor* a){
	//computes row wise mean
	//out_shape = (rows, 1)
	int* shape_r = arena_alloc(A, a->ndim * sizeof(int)); //ndim = 2
    shape_r[0] = a->info.shape[0];
	shape_r[1] = 1;
	Tensor* r = tensor_create_new(A, a->ndim, shape_r);
	if(a->requires_grad){
		//build computation graph
		r->requires_grad = true;
		r->num_parents = 1;
		r->parents = arena_alloc(A, sizeof(Tensor*));
		r->parents[0] = a;
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_mean_backward;
		r->ops = op;  
		r->grad = arena_alloc(A, shape_r[0] * 1 * sizeof(float));  
	}
	
	    for (int i = 0; i < shape_r[0]; i++) {
           float *x_row = a->_data + i * a->info.shape[1];
           r->_data[i] = mean(x_row, a->info.shape[1]);
		}
  return r;
}

void __ag_tensor_transpose_backward(Tensor* out) {
	if(!out || !out->parents) return;
	Tensor* in = out->parents[0];
	if(!in) return;
	ensure_grad_buffer(in);
	
	int rows_out = out->info.shape[0];
	int cols_out = out->info.shape[1];
	// Transpose gradient: out[i,j] = in[j,i], so dout[i,j] -> din[j,i]
	for(int i = 0; i < rows_out; i++) {
		for(int j = 0; j < cols_out; j++) {
			float g = out->grad ? out->grad[i * cols_out + j] : 0.0f;
			in->grad[j * rows_out + i] += g;
		}
	}
}

Tensor* __ag_tensor_transpose_forward(Arena* A, Tensor* in) {
	assert(in->ndim == 2);
	int rows = in->info.shape[0];
	int cols = in->info.shape[1];
	
	int ndim = 2;
	int* shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = cols;
	shape[1] = rows;
	
	Tensor* r = tensor_create_new(A, ndim, shape);
	
	// Perform transpose
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			r->_data[j * rows + i] = in->_data[i * cols + j];
		}
	}
	
	if(in->requires_grad) {
		r->requires_grad = true;
		r->num_parents = 1;
		r->parents = arena_alloc(A, sizeof(Tensor*));
		r->parents[0] = in;
		r->grad = arena_alloc(A, shape[0] * shape[1] * sizeof(float));
		memset(r->grad, 0, shape[0] * shape[1] * sizeof(float));
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_transpose_backward;
		r->ops = op;
	}
	
	return r;
}

Tensor* __ag_tensor_expand_rows(Arena* A, Tensor* in, int rows_r){
   return __ag_tensor_transpose_forward(A, __ag_tensor_expand_cols_forward(A, __ag_tensor_transpose_forward(A, in), rows_r));
}

//void __ag_tensor_expand_cols_backward(Tensor* o);

Tensor* __ag_tensor_expand_cols_forward(Arena* A, Tensor* mean, int cols_r){
//Takes the mean value for each row
//and expands its column number of times to match the other tensor
//reduce_mean = [m1],
//              [m2],
//              [m3]
//expand_mean = [m1, m1, m1]
//              [m2, m2, m2]
//              [m3, m3, m3]
assert(mean->ndim == 2);
assert(mean->info.shape[1] == 1);
int rows = mean->info.shape[0];
int ndim = 2;

int* shape_r = arena_alloc(A, ndim * sizeof(int));
shape_r[0] = rows;
shape_r[1] = cols_r;

Tensor* r = tensor_create_new(A, ndim, shape_r);

if(mean->requires_grad){
    r->requires_grad = true;
	r->num_parents = 1;
	r->parents = arena_alloc(A, sizeof(Tensor*));
	r->parents[0] = mean;
	Op* op = arena_alloc(A, sizeof(Op));
	op->backward = __ag_tensor_expand_cols_backward;
	r->ops = op;
	r->grad = arena_alloc(A, shape_r[0] * cols_r * sizeof(float));
}

   //IMPORTANT!!!
   //row_offset = r_i * row_stride;
   //col_offset = c * col_strid;
   //index = row_offset + col_offset;
   for(int r_i = 0; r_i < rows; r_i++){
	float v = mean->_data[r_i];
	for(int c = 0; c < cols_r; c++){
      r->_data[r_i * cols_r + c] = v;
	}
   }

 return r;
}

//void __ag_tensor_sq_backward(Tensor* o);

Tensor* __ag_tensor_sq_forward(Arena* A, Tensor* a){
  //assert(a->info.shape[0] == b->info.shape[0] && a->info.shape[1] == b->info.shape[1]);
//   int rows = a->info.shape[0];
//   int cols = a->info.shape[1];
//   int ndim = a->ndim;

  int* shape_r = arena_alloc(A, a->ndim * sizeof(int));
  shape_r[0] = a->info.shape[0];
  shape_r[1] = a->info.shape[1];
  
  Tensor* r = tensor_create_new(A, a->ndim, shape_r);

if(a->requires_grad){
    r->requires_grad = true;
	r->num_parents = 1;
	r->parents = arena_alloc(A, sizeof(Tensor*));
	r->parents[0] = a;
	Op* op = arena_alloc(A, sizeof(Op));
	op->backward = __ag_tensor_sq_backward;
	r->ops = op;
	r->grad = arena_alloc(A, a->info.sz * sizeof(float));
}
  
  for(int i = 0; i < r->info.sz; i++){
	r->_data[i] = a->_data[i] * a->_data[i];
  }
  
  return r;
}

//void __ag_tensor_sqrt_backward(Tensor* o);

Tensor* __ag_tensor_sqrt_forward(Arena* A, Tensor* a){
  //assert(a->info.shape[0] == b->info.shape[0] && a->info.shape[1] == b->info.shape[1]);
//   int rows = a->info.shape[0];
//   int cols = a->info.shape[1];
//   int ndim = a->ndim;

  int* shape_r = arena_alloc(A, a->ndim * sizeof(int));
  shape_r[0] = a->info.shape[0];
  shape_r[1] = a->info.shape[1];
  
  Tensor* r = tensor_create_new(A, a->ndim, shape_r);

if(a->requires_grad){
    r->requires_grad = true;
	r->num_parents = 1;
	r->parents = arena_alloc(A, sizeof(Tensor*));
	r->parents[0] = a;
	Op* op = arena_alloc(A, sizeof(Op));
	op->backward = __ag_tensor_sqrt_backward;
	r->ops = op;
	r->grad = arena_alloc(A, a->info.sz * sizeof(float));
}
  
  for(int i = 0; i < r->info.sz; i++){
	r->_data[i] = sqrtf(a->_data[i] + 1e-8f);
  }
  
  return r;
}
//Tensor __ag_tensor_variance(int* row1, int* row2);

// Slice/scatter for MHA autograd support
// NOTE: Slice offset stored in output tensor's stride field (hijacking for backward)
void __ag_tensor_slice_backward(Tensor* o){
	if(!o || !o->parents) return;
	Tensor* in = o->parents[0];
	if(!in) return;
	ensure_grad_buffer(in);
	
	int rows = o->info.shape[0];
	int cols = o->info.shape[1];
	int in_cols = in->info.shape[1];
	
	// Get slice start offset from stride (where we stored it in forward)
	int start_row = o->info.stride[0];
	int start_col = o->info.stride[1];
	
	// Scatter gradients back to their original positions in input
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			float g = o->grad ? o->grad[i * cols + j] : 0.0f;
			int in_idx = (start_row + i) * in_cols + (start_col + j);
			in->grad[in_idx] += g;
		}
	}
}

Tensor* __ag_tensor_slice_forward(Arena* A, Tensor* in, int start_row, int start_col, int num_rows, int num_cols){
	int ndim = 2;
	int* shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = num_rows;
	shape[1] = num_cols;
	
	Tensor* r = tensor_create_new(A, ndim, shape);
	
	// Copy data from input slice
	for(int i = 0; i < num_rows; i++){
		for(int j = 0; j < num_cols; j++){
			r->_data[i * num_cols + j] = in->_data[(start_row + i) * in->info.shape[1] + (start_col + j)];
		}
	}
	
	if(in->requires_grad){
		r->requires_grad = true;
		r->num_parents = 1;
		r->parents = arena_alloc(A, sizeof(Tensor*));
		r->parents[0] = in;
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_slice_backward;
		r->ops = op;
		r->grad = arena_alloc(A, num_rows * num_cols * sizeof(float));
		memset(r->grad, 0, num_rows * num_cols * sizeof(float));
		// Store slice offset in stride for backward pass
		r->info.stride[0] = start_row;
		r->info.stride[1] = start_col;
	}
	
	return r;
}

void __ag_tensor_scatter_backward(Tensor* o){
	if(!o || !o->parents) return;
	Tensor* base = o->parents[0];
	Tensor* src = o->parents[1];
	if(base) ensure_grad_buffer(base);
	if(src) ensure_grad_buffer(src);
	
	int out_rows = o->info.shape[0];
	int out_cols = o->info.shape[1];
	
	// Get scatter offset from stride
	int start_row = o->info.stride[0];
	int start_col = o->info.stride[1];
	
	// Gradients to base: copy from output except scattered region
	if(base && base->grad) {
		for(int i = 0; i < out_rows; i++){
			for(int j = 0; j < out_cols; j++){
				float g = o->grad ? o->grad[i * out_cols + j] : 0.0f;
				base->grad[i * out_cols + j] += g;
			}
		}
	}
	
	// Gradients to src: extract from scattered region
	if(src && src->grad) {
		int src_rows = src->info.shape[0];
		int src_cols = src->info.shape[1];
		for(int i = 0; i < src_rows; i++){
			for(int j = 0; j < src_cols; j++){
				float g = o->grad ? o->grad[(start_row + i) * out_cols + (start_col + j)] : 0.0f;
				src->grad[i * src_cols + j] += g;
			}
		}
	}
}

Tensor* __ag_tensor_scatter_forward(Arena* A, Tensor* base, Tensor* src, int start_row, int start_col){
	int ndim = 2;
	int* shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = base->info.shape[0];
	shape[1] = base->info.shape[1];
	
	Tensor* r = tensor_create_new(A, ndim, shape);
	memcpy(r->_data, base->_data, base->info.sz * sizeof(float));
	
	int src_rows = src->info.shape[0];
	int src_cols = src->info.shape[1];
	for(int i = 0; i < src_rows; i++){
		for(int j = 0; j < src_cols; j++){
			r->_data[(start_row + i) * shape[1] + (start_col + j)] = src->_data[i * src_cols + j];
		}
	}
	
	if(base->requires_grad || src->requires_grad){
		r->requires_grad = true;
		r->num_parents = 2;
		r->parents = arena_alloc(A, 2 * sizeof(Tensor*));
		r->parents[0] = base;
		r->parents[1] = src;
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_scatter_backward;
		r->ops = op;
		r->grad = arena_alloc(A, shape[0] * shape[1] * sizeof(float));
		memset(r->grad, 0, shape[0] * shape[1] * sizeof(float));
		// Store scatter offset in stride for backward
		r->info.stride[0] = start_row;
		r->info.stride[1] = start_col;
	}
	
	return r;
}

//Autograd supported ops

//multiplies A and B, populates the ops => pointer to the backward function
//keeps track of parents
//__ag stands for being autograd supported
Tensor* __ag_tensor_matmul_forward(Arena* A, Tensor* a, Tensor* b){
	//check the dimensionality
	// if (a->info.shape[1] != b->info.shape[0]) {
	// 	fprintf(stderr, "DEBUG matmul shape mismatch: a->shape={%d, %d}, b->shape={%d, %d}\n",
	// 		a->info.shape[0], a->info.shape[1], b->info.shape[0], b->info.shape[1]);
	// }
	assert(a->info.shape[1] == b->info.shape[0]);

	int ndim_r = 2;
	int rows_a = a->info.shape[0];
	int cols_a = a->info.shape[1];
	int cols_b = b->info.shape[1];

	int* shape_r = arena_alloc(A, ndim_r * sizeof(int));
	shape_r[0] = a->info.shape[0];
	shape_r[1] = b->info.shape[1];

	Tensor* r = tensor_create_new(A, ndim_r, shape_r);

	if(a->requires_grad == true || b->requires_grad == true){
        r->requires_grad = true;
		r->num_parents = 2;
		r->parents = arena_alloc(A, r->num_parents * sizeof(Tensor*));
		r->parents[0] = a;
		r->parents[1] = b;

		int out_size = a->info.shape[0] * b->info.shape[1];
		r->grad = arena_alloc(A, out_size * sizeof(float));
		//treebased memAllocator

		//lets define the ops
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_matmul_backward;
		r->ops = op;
	}

	// Zero initialize result
	memset(r->_data, 0, rows_a * cols_b * sizeof(float));

	// Cache-friendly blocking with SIMD
	const int BLOCK_M = 64;  // Rows block
	const int BLOCK_N = 64;  // Cols block
	const int BLOCK_K = 64;  // Inner dim block

#if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
	#pragma omp parallel for
#endif
	for (int ii = 0; ii < rows_a; ii += BLOCK_M) {
		for (int jj = 0; jj < cols_b; jj += BLOCK_N) {
			for (int kk = 0; kk < cols_a; kk += BLOCK_K) {
				int i_end = (ii + BLOCK_M < rows_a) ? ii + BLOCK_M : rows_a;
				int j_end = (jj + BLOCK_N < cols_b) ? jj + BLOCK_N : cols_b;
				int k_end = (kk + BLOCK_K < cols_a) ? kk + BLOCK_K : cols_a;

				for (int i = ii; i < i_end; i++) {
					for (int j = jj; j < j_end; j++) {
						float sum = r->_data[i * cols_b + j];
						int k = kk;

						float vsum_scalar = 0.0f;
						for (; k <= k_end - 8; k += 8) {
							#if defined(__AVX512F__) || defined(__AVX2__)
								__m256 va = _mm256_loadu_ps(&a->_data[i * a->info.stride[0] + k * a->info.stride[1]]);
							#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
								float32x4_t va0 = vld1q_f32(&a->_data[i * a->info.stride[0] + (k+0) * a->info.stride[1]]);
								float32x4_t va1 = vld1q_f32(&a->_data[i * a->info.stride[0] + (k+4) * a->info.stride[1]]);
							#elif defined(__SSE4_2__) || defined(__SSE2__)
								__m128 va0 = _mm_loadu_ps(&a->_data[i * a->info.stride[0] + (k+0) * a->info.stride[1]]);
								__m128 va1 = _mm_loadu_ps(&a->_data[i * a->info.stride[0] + (k+4) * a->info.stride[1]]);
							#endif
							// gather B column into a vector (still scalar loads, but vector math)
							float bg[8] = {
								b->_data[k * b->info.stride[0] + j * b->info.stride[1]],

								b->_data[(k+1) * b->info.stride[0] + j * b->info.stride[1]],
								b->_data[(k+2) * b->info.stride[0] + j * b->info.stride[1]],
								b->_data[(k+3) * b->info.stride[0] + j * b->info.stride[1]],
								b->_data[(k+4) * b->info.stride[0] + j * b->info.stride[1]],
								b->_data[(k+5) * b->info.stride[0] + j * b->info.stride[1]],
								b->_data[(k+6) * b->info.stride[0] + j * b->info.stride[1]],
								b->_data[(k+7) * b->info.stride[0] + j * b->info.stride[1]],
							};
							#if defined(__AVX512F__) || defined(__AVX2__)
								__m256 vb = _mm256_loadu_ps(bg);
								__m256 vm = _mm256_mul_ps(va, vb);
								__m256 tmp = _mm256_hadd_ps(vm, vm);
								tmp = _mm256_hadd_ps(tmp, tmp);
								float t[8];
								_mm256_storeu_ps(t, tmp);
								vsum_scalar += t[0] + t[4];
							#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
								float32x4_t vb0 = vld1q_f32(&bg[0]);
								float32x4_t vb1 = vld1q_f32(&bg[4]);
								float32x4_t vm0 = vmulq_f32(va0, vb0);
								float32x4_t vm1 = vmulq_f32(va1, vb1);
								float32x2_t s0 = vadd_f32(vget_low_f32(vm0), vget_high_f32(vm0));
								float32x2_t s1 = vadd_f32(vget_low_f32(vm1), vget_high_f32(vm1));
								s0 = vpadd_f32(s0, s0);
								s1 = vpadd_f32(s1, s1);
								vsum_scalar += vget_lane_f32(s0, 0) + vget_lane_f32(s1, 0);
							#elif defined(__SSE4_2__) || defined(__SSE2__)
								__m128 vb0 = _mm_loadu_ps(&bg[0]);
								__m128 vb1 = _mm_loadu_ps(&bg[4]);
								__m128 vm0 = _mm_mul_ps(va0, vb0);
								__m128 vm1 = _mm_mul_ps(va1, vb1);
								float t0[4], t1[4];
								_mm_storeu_ps(t0, vm0);
								_mm_storeu_ps(t1, vm1);
								vsum_scalar += t0[0]+t0[1]+t0[2]+t0[3] + t1[0]+t1[1]+t1[2]+t1[3];
							#else
								vsum_scalar += bg[0]*a->_data[i * a->info.stride[0] + (k+0) * a->info.stride[1]];
								vsum_scalar += bg[1]*a->_data[i * a->info.stride[0] + (k+1) * a->info.stride[1]];
								vsum_scalar += bg[2]*a->_data[i * a->info.stride[0] + (k+2) * a->info.stride[1]];
								vsum_scalar += bg[3]*a->_data[i * a->info.stride[0] + (k+3) * a->info.stride[1]];
								vsum_scalar += bg[4]*a->_data[i * a->info.stride[0] + (k+4) * a->info.stride[1]];
								vsum_scalar += bg[5]*a->_data[i * a->info.stride[0] + (k+5) * a->info.stride[1]];
								vsum_scalar += bg[6]*a->_data[i * a->info.stride[0] + (k+6) * a->info.stride[1]];
								vsum_scalar += bg[7]*a->_data[i * a->info.stride[0] + (k+7) * a->info.stride[1]];
							#endif
						}
						sum += vsum_scalar;
						for (; k < k_end; k++) {
							sum += a->_data[i * a->info.stride[0] + k * a->info.stride[1]] * b->_data[k * b->info.stride[0] + j * b->info.stride[1]];
						}

						r->_data[i * cols_b + j] = sum;
					}
				}
			}
		}
	}

	return r;
}

void __ag_tensor_matmul_backward(Tensor* loss){
	if(!loss || !loss->parents) return;
	Tensor* a = loss->parents[0];
	Tensor* b = loss->parents[1];
	if(!a || !b) return;
	ensure_grad_buffer(a);
	ensure_grad_buffer(b);
	int M = a->info.shape[0];
	int K = a->info.shape[1];
	int N = b->info.shape[1];
	// loss->_data is upstream gradients of shape M x N
	// dA[i,k] = sum_j upstream[i,j] * b[k,j]
	for(int i=0;i<M;i++){
		for(int k=0;k<K;k++){
			float sum = 0.0f;
			for(int j=0;j<N;j++){
				float g = loss->grad ? loss->grad[i*N + j] : 0.0f;
				sum += g * b->_data[k * N + j];
			}
			a->grad[i*K + k] += sum;
		}
	}
	// dB[k,j] = sum_i a[i,k] * upstream[i,j]
	for(int k=0;k<K;k++){
		for(int j=0;j<N;j++){
			float sum = 0.0f;
			for(int i=0;i<M;i++){
				sum += a->_data[i*K + k] * (loss->grad ? loss->grad[i*N + j] : 0.0f);
			}
			b->grad[k*N + j] += sum;
		}
	}
	return;

}



Tensor* tensor_create(int ndim, int* shape){
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if(!t){
        fprintf(stderr, "failed to create Tensor due to malloc failure\n-> aborting..");
		return NULL;
    }

    t->info.shape = (int*) malloc(ndim * sizeof(int)); //no need to place them in the same block, avoid algorithmic overhead
    t->info.stride = (int*) malloc(ndim * sizeof(int));

    t->ndim = ndim; //analogous to rank atleast in this case

    //t->info.shape = shape;
	    // Copy shape values instead of assigning pointer
    for (int i = 0; i < ndim; i++) {
        t->info.shape[i] = shape[i];
    }

    int _size = 1; //pointer walk
	for (const int *p = shape, *end = shape + ndim; p < end; ++p) {
		_size *= (size_t)*p;
	}

	//ndim - 1 > is always 1, fastest changing dimension
	// for next ones wer reverse loop and assign
	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
	t->info.stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; i--) {
		t->info.stride[i] = t->info.stride[i + 1] * t->info.shape[i + 1];
	}
	// define the data now
	t->_data = (float*)malloc(_size * sizeof(float));
	for (int i = 0; i < _size; i++) {
		t->_data[i] = (rand() % 10) + 1.0f;
	}
	t->info.sz = _size;  // Set the size field

	//-----------------AUTOGRAD SUPPORT---------------
	t->parents = NULL;
	t->ops = NULL;
	t->grad = NULL;
	t->num_parents = 0;
	//-------------------------------------------------

    return t;
}

Tensor* tensor_create_weights(int ndim, int* shape){
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if(!t){
        fprintf(stderr, "failed to create Tensor due to malloc failure\n-> aborting..");
		return NULL;
    }

    t->info.shape = (int*) malloc(ndim * sizeof(int)); //no need to place them in the same block, avoid algorithmic overhead
    t->info.stride = (int*) malloc(ndim * sizeof(int));

    t->ndim = ndim; //analogous to rank atleast in this case

    //t->info.shape = shape;
	// Copy shape values instead of assigning pointer
    for (int i = 0; i < ndim; i++) {
        t->info.shape[i] = shape[i];
    }

    int _size = 1; //pointer walk
	for (const int *p = shape, *end = shape + ndim; p < end; ++p) {
		_size *= (size_t)*p;
	}

	//ndim - 1 > is always 1, fastest changing dimension
	// for next ones wer reverse loop and assign
	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
	t->info.stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; i--) {
		t->info.stride[i] = t->info.stride[i + 1] * t->info.shape[i + 1];
	}
	// define the data now
	t->_data = (float*)malloc(_size * sizeof(float));
	for (int i = 0; i < _size; i++) {
		t->_data[i] = RAND_FLOAT;
	}	

    return t;
}

Tensor *tensor_create_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	t->ndim = ndim;
	t->info.shape = arena_alloc(A, ndim * sizeof(int));
	t->info.stride = arena_alloc(A, ndim * sizeof(int));

	// define the shape of Tensor
	int total = 1;
	for (int i = ndim - 1; i >= 0; i--) {
		t->info.shape[i] = shape[i];
		t->info.stride[i] = total;
		total *= shape[i];
	}
	t->_data = arena_alloc(A, total * sizeof(float));
	t->info.sz = total;
	//-----------------AUTOGRAD SUPPORT---------------
	t->parents = NULL;
	t->ops = NULL;
	t->grad = NULL;
	t->num_parents = 0;
	//-------------------------------------------------

	return t;
}

void tensor_randomize_weights(Tensor *x) {
	for (int i = 0; i < x->info.sz; i++) {
		x->_data[i] = RAND_FLOAT;
	}
}

void tensor_randomize(Tensor *x) {
	for (int i = 0; i < x->info.sz; i++) {
		x->_data[i] = (rand() % 10) + 1.0f;
	}
}

void tensor_add_inplace(Tensor **a, Tensor **b) {
	assert((*a)->info.shape != (*b)->info.shape);
	int rows = (*a)->info.shape[0];
	int cols = (*a)->info.shape[1];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * cols + j;
			(*a)->_data[idx] = (*b)->_data[i];
		}
	}
}

void clip_gradient(Tensor *x) {
    float threshold = 1.0f;
    float MX = 0.0f;
    bool has_bad = false;
    int sz = x->info.sz;

    // SIMD pass: find max abs + NaN/Inf check
    int i = 0;
    for (; i <= sz - 8; i += 8) {
		float tmp[8];
		#if defined(__AVX512F__) || defined(__AVX2__)
			__m256 v = _mm256_loadu_ps(&x->_data[i]);
			_mm256_storeu_ps(tmp, v);
		#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
			float32x4_t v0 = vld1q_f32(&x->_data[i]);
			float32x4_t v1 = vld1q_f32(&x->_data[i + 4]);
			vst1q_f32(&tmp[0], v0);
			vst1q_f32(&tmp[4], v1);
		#elif defined(__SSE4_2__) || defined(__SSE2__)
			__m128 v0 = _mm_loadu_ps(&x->_data[i]);
			__m128 v1 = _mm_loadu_ps(&x->_data[i + 4]);
			_mm_storeu_ps(&tmp[0], v0);
			_mm_storeu_ps(&tmp[4], v1);
		#else
			memcpy(tmp, &x->_data[i], sizeof(tmp));
		#endif
		for (int k = 0; k < 8; k++) {
			float g = tmp[k];
			if (!isfinite(g)) { has_bad = true; break; }
			float av = fabsf(g);
			if (av > MX) MX = av;
		}
		if (has_bad) break;
	}

	for (; i < sz && !has_bad; i++) {
		if (isinf(x->_data[i]) || isnan(x->_data[i])) { has_bad = true; break; }
		MX = (fabsf(x->_data[i]) > MX) ? fabsf(x->_data[i]) : MX;
	}

	if (has_bad) {
		memset(x->_data, 0, sz * sizeof(float));
		return;
	}

	if (MX > threshold) {
		float scale = threshold / MX;
		i = 0;
		for (; i <= sz - 8; i += 8) {
			#if defined(__AVX512F__) || defined(__AVX2__)
				__m256 v = _mm256_loadu_ps(&x->_data[i]);
				__m256 vs = _mm256_set1_ps(scale);
				_mm256_storeu_ps(&x->_data[i], _mm256_mul_ps(v, vs));
			#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
				float32x4_t v0 = vld1q_f32(&x->_data[i]);
				float32x4_t v1 = vld1q_f32(&x->_data[i + 4]);
				float32x4_t vs = vdupq_n_f32(scale);
				vst1q_f32(&x->_data[i], vmulq_f32(v0, vs));
				vst1q_f32(&x->_data[i + 4], vmulq_f32(v1, vs));
			#elif defined(__SSE4_2__) || defined(__SSE2__)
				__m128 v0 = _mm_loadu_ps(&x->_data[i]);
				__m128 v1 = _mm_loadu_ps(&x->_data[i + 4]);
				__m128 vs = _mm_set1_ps(scale);
				_mm_storeu_ps(&x->_data[i], _mm_mul_ps(v0, vs));
				_mm_storeu_ps(&x->_data[i + 4], _mm_mul_ps(v1, vs));
			#else
				x->_data[i+0] *= scale;
				x->_data[i+1] *= scale;
				x->_data[i+2] *= scale;
				x->_data[i+3] *= scale;
				x->_data[i+4] *= scale;
				x->_data[i+5] *= scale;
				x->_data[i+6] *= scale;
				x->_data[i+7] *= scale;
			#endif
		}
		for (; i < sz; i++) x->_data[i] *= scale;
	}
}

bool is_exploding(Tensor *x) {
	//int size = tensor_size(x);
	for (int i = 0; i < x->info.sz; i++) {
		float v = x->_data[i];
		if (isnan(v) || isinf(v)) {
			return true;
		}
	}
	return false;
}

Tensor *tensor_scalar_multiplication(Tensor *x, float val) {
	int rows = x->info.shape[0];
	int cols = x->info.shape[1];
	//int size = tensor_size(x);
	for (int i = 0; i < x->info.sz; i++) {
		x->_data[i * cols + rows] = val * x->_data[i * cols + rows];
	}
	return x;
}

Tensor *tensor_scalar_addition(Tensor *x, float val) {
	int rows = x->info.shape[0];
	int cols = x->info.shape[1];
	for (int i = 0; i < x->info.sz; i++) {
		x->_data[i * cols + rows] = val + x->_data[i * cols + rows];
	}
	return x;
}
	

Tensor *tensor_add(Tensor *a, Tensor *b) {
	assert(a->info.shape != b->info.shape);
	int ndim = 2;
	int rows = a->info.shape[0];
	int cols = a->info.shape[1];
	int shape[2] = {rows, cols};
	Tensor *res = tensor_create(ndim, shape);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * res->info.shape[1] + j;
			res->_data[idx] = a->_data[idx] + b->_data[idx];
		}
	}
	return res;
}

// void __ag_tensor_add_backward(Tensor* upstream){
// 	if(!upstream || !upstream->parents) return;
// 	Tensor* a = upstream->parents[0];
// 	Tensor* b = upstream->parents[1];
// 	if(a) ensure_grad_buffer(a);
// 	if(b) ensure_grad_buffer(b);

// 	int sz = tensor_size(upstream);
// 	for(int i=0;i<sz;i++){
// 		float g = upstream->_data[i];
// 		if(a) a->grad[i] += g;
// 		if(b) b->grad[i] += g;
// 	}
// 	return;
// }

Tensor */*tensor_softmax*/__ag_tensor_softmax_forward(Arena* A, Tensor *t) {
	//Tensor *r = malloc(sizeof(Tensor));
	Tensor *r = arena_alloc(A, sizeof(Tensor));
	if (!r) {return NULL;}
	r->info.shape = t->info.shape;
	r->info.stride = t->info.stride;
	r->ndim = t->ndim;
	//r->_data = malloc(r->info.shape[0] * r->info.shape[1] * sizeof(float));
	r->_data = arena_alloc(A, r->info.shape[0] * r->info.shape[1] * sizeof(float));

	int rows = t->info.shape[0];
	int cols = t->info.shape[1];

	/* OpenMP disabled: row size typically small (16-32), thread overhead > benefit */
	/* #if defined(_OPENMP) && !defined(Y9_DISABLE_OPENMP)
	#pragma omp parallel for
	#endif */
	for (int i = 0; i < rows; i++) {
		float max = -INFINITY;
		int j = 0;

		#if defined(__AVX512F__) || defined(__AVX2__)
			__m256 vmax = _mm256_set1_ps(-INFINITY);
		#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
			float32x4_t vmax0 = vdupq_n_f32(-INFINITY);
			float32x4_t vmax1 = vdupq_n_f32(-INFINITY);
		#elif defined(__SSE4_2__) || defined(__SSE2__)
			__m128 vmax0 = _mm_set1_ps(-INFINITY);
			__m128 vmax1 = _mm_set1_ps(-INFINITY);
		#endif
		for (; j <= cols - 8; j += 8) {
			#if defined(__AVX512F__) || defined(__AVX2__)
				__m256 v = _mm256_loadu_ps(&t->_data[i * cols + j]);
				vmax = _mm256_max_ps(vmax, v);
			#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
				float32x4_t v0 = vld1q_f32(&t->_data[i * cols + j]);
				float32x4_t v1 = vld1q_f32(&t->_data[i * cols + j + 4]);
				vmax0 = vmaxq_f32(vmax0, v0);
				vmax1 = vmaxq_f32(vmax1, v1);
			#elif defined(__SSE4_2__) || defined(__SSE2__)
				__m128 v0 = _mm_loadu_ps(&t->_data[i * cols + j]);
				__m128 v1 = _mm_loadu_ps(&t->_data[i * cols + j + 4]);
				vmax0 = _mm_max_ps(vmax0, v0);
				vmax1 = _mm_max_ps(vmax1, v1);
			#endif
		}
		float temp[8];
		#if defined(__AVX512F__) || defined(__AVX2__)
			_mm256_storeu_ps(temp, vmax);
		#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
			vst1q_f32(&temp[0], vmax0);
			vst1q_f32(&temp[4], vmax1);
		#elif defined(__SSE4_2__) || defined(__SSE2__)
			_mm_storeu_ps(&temp[0], vmax0);
			_mm_storeu_ps(&temp[4], vmax1);
		#endif
		for (int k = 0; k < 8; k++) if (temp[k] > max) max = temp[k];
		for (; j < cols; j++) {
			if (t->_data[i * cols + j] > max) max = t->_data[i * cols + j];
		}

		float sum = 0.0f;
		// Process exp with scalar expf (no standard SIMD exp in AVX)
		j = 0;
		for (; j < cols; j++) {
			float exp_val = expf(t->_data[i * cols + j] - max);
			r->_data[i * cols + j] = exp_val;
			sum += exp_val;
		}

		j = 0;
		float inv_sum = 1.0f / sum;
		for (; j <= cols - 8; j += 8) {
			#if defined(__AVX512F__) || defined(__AVX2__)
				__m256 v = _mm256_loadu_ps(&r->_data[i * cols + j]);
				__m256 vinv = _mm256_set1_ps(inv_sum);
				_mm256_storeu_ps(&r->_data[i * cols + j], _mm256_mul_ps(v, vinv));
			#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
				float32x4_t v0 = vld1q_f32(&r->_data[i * cols + j]);
				float32x4_t v1 = vld1q_f32(&r->_data[i * cols + j + 4]);
				float32x4_t vinv = vdupq_n_f32(inv_sum);
				vst1q_f32(&r->_data[i * cols + j], vmulq_f32(v0, vinv));
				vst1q_f32(&r->_data[i * cols + j + 4], vmulq_f32(v1, vinv));
			#elif defined(__SSE4_2__) || defined(__SSE2__)
				__m128 v0 = _mm_loadu_ps(&r->_data[i * cols + j]);
				__m128 v1 = _mm_loadu_ps(&r->_data[i * cols + j + 4]);
				__m128 vinv = _mm_set1_ps(inv_sum);
				_mm_storeu_ps(&r->_data[i * cols + j], _mm_mul_ps(v0, vinv));
				_mm_storeu_ps(&r->_data[i * cols + j + 4], _mm_mul_ps(v1, vinv));
			#endif
		}
		for (; j < cols; j++) r->_data[i * cols + j] /= sum;
	}
	return r;
}

void tensor_get(Tensor *t) {
	if (!t) return;
	/*int size = 1;
	for (int i = 0; i < t->ndim; i++) {
		size *= t->info.shape[i];
	}*/
	//printf("size: %d\n", size);
	for (int i = 0; i < t->info.sz; i++) {
		printf("%0.2f ", t->_data[i]);
	}		
}

Tensor *tensor_transpose(Tensor *a) {
	int ndim = 2;
	int *shape = malloc(ndim * sizeof(int));
	shape[0] = a->info.shape[1];
	shape[1] = a->info.shape[0];
	Tensor *t = tensor_create(ndim, shape);
	
	int rows_a = a->info.shape[0];
	int cols_a = a->info.shape[1];
	for (int i = 0; i < rows_a; i++) {
		for (int j = 0; j < cols_a; j++) {
			t->_data[j * t->info.stride[0] + i * t->info.stride[1]] = a->_data[i * a->info.stride[0] + j * a->info.stride[1]];
		}
	}

	return t;
}

Tensor *relu_backward(Tensor *x, Tensor *y) {
	// x is the activation, y is the pre-activation (h1)
	// ReLU'(h1) = 1 if h1 > 0, else 0
	// Then multiply element-wise by upstream gradient (x)
	int rows = x->info.shape[0];
	int cols = x->info.shape[1];
	Tensor *res = tensor_create(2, x->info.shape);
	int size = rows * cols;
	int i = 0;
	#if defined(__AVX512F__) || defined(__AVX2__)
		__m256 vzero = _mm256_setzero_ps();
	#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
		float32x4_t vzero0 = vdupq_n_f32(0.0f);
		float32x4_t vzero1 = vdupq_n_f32(0.0f);
	#elif defined(__SSE4_2__) || defined(__SSE2__)
		__m128 vzero0 = _mm_setzero_ps();
		__m128 vzero1 = _mm_setzero_ps();
	#endif
	for (; i <= size - 8; i += 8) {
		#if defined(__AVX512F__) || defined(__AVX2__)
			__m256 vgrad = _mm256_loadu_ps(&x->_data[i]);
			__m256 vpre = _mm256_loadu_ps(&y->_data[i]);
			__m256 mask = _mm256_cmp_ps(vpre, vzero, _CMP_GT_OQ);
			_mm256_storeu_ps(&res->_data[i], _mm256_blendv_ps(vzero, vgrad, mask));
		#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
			float32x4_t vgrad0 = vld1q_f32(&x->_data[i]);
			float32x4_t vgrad1 = vld1q_f32(&x->_data[i + 4]);
			float32x4_t vpre0 = vld1q_f32(&y->_data[i]);
			float32x4_t vpre1 = vld1q_f32(&y->_data[i + 4]);
			uint32x4_t m0 = vcgtq_f32(vpre0, vzero0);
			uint32x4_t m1 = vcgtq_f32(vpre1, vzero1);
			vst1q_f32(&res->_data[i], vbslq_f32(m0, vgrad0, vzero0));
			vst1q_f32(&res->_data[i + 4], vbslq_f32(m1, vgrad1, vzero1));
		#elif defined(__SSE4_2__) || defined(__SSE2__)
			__m128 vgrad0 = _mm_loadu_ps(&x->_data[i]);
			__m128 vgrad1 = _mm_loadu_ps(&x->_data[i + 4]);
			__m128 vpre0 = _mm_loadu_ps(&y->_data[i]);
			__m128 vpre1 = _mm_loadu_ps(&y->_data[i + 4]);
			__m128 m0 = _mm_cmpgt_ps(vpre0, vzero0);
			__m128 m1 = _mm_cmpgt_ps(vpre1, vzero1);
			__m128 r0 = _mm_or_ps(_mm_and_ps(m0, vgrad0), _mm_andnot_ps(m0, vzero0));
			__m128 r1 = _mm_or_ps(_mm_and_ps(m1, vgrad1), _mm_andnot_ps(m1, vzero1));
			_mm_storeu_ps(&res->_data[i], r0);
			_mm_storeu_ps(&res->_data[i + 4], r1);
		#endif
	}
	for (; i < size; i++) res->_data[i] = (y->_data[i] > 0.0f) ? x->_data[i] : 0.0f;
	return res;
}

//exploiting OS-level memory access bugs or running over a hypervisor to implement in RDMA
//can specific bus channels between ram and storage and ram and cpu be blocked
//and just allow only for communication between these channels
//deterministic threads for proper allocation tracking and knowing when to expect data
//exploiting free dangling pointers from other programs, binding with 'restrict' pointers
//to prevent other programs from taking advantage of this vulnerability
//wifi (bluetooth mesh later) for crosscommunication distributed inference
//escalation privileges to allow for extended lowlevel functionalities (emulating hypervisor functionalities)
//apart from the hypervisor or machines not supporting it
//movement optimisation through compression

void tensor_check(char *name, Tensor *x) {
	if (!x) {
		fprintf(stderr, "tensor_check: %s is NULL\n", name);
		return;
	}
	for (int i = 0; i < x->info.sz; i++) {
		if (isnan(x->_data[i]) || isinf(x->_data[i])) {
			fprintf(stderr, "tensor_check: %s has NaN or Inf at index %d\n", name, i);
			return;
		}
	}
}

int tensor_size(Tensor *t) {
	if (!t) return 0;
	int size = 1;
	for (int i = 0; i < t->ndim; i++) {
		size *= t->info.shape[i];
	}
	t->info.sz = size;
	return size;
}

Tensor *tensor_create_weights_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	t->ndim = ndim;
	t->info.shape = arena_alloc(A, ndim * sizeof(int));
	t->info.stride = arena_alloc(A, ndim * sizeof(int));

	int total = 1;
	for (int i = ndim - 1; i >= 0; i--) {
		t->info.shape[i] = shape[i];
		t->info.stride[i] = total;
		total *= shape[i];
	}
	t->_data = arena_alloc(A, total * sizeof(float));
	for (int i = 0; i < total; i++) {
		t->_data[i] = RAND_FLOAT;
	}
	t->info.sz = total;

	//-----------------AUTOGRAD SUPPORT---------------
	t->parents = NULL;
	t->ops = NULL;
	t->grad = NULL;
	t->num_parents = 0;
	t->requires_grad = true;  // Weight tensors need gradients for training
	//-------------------------------------------------

	return t;
}

void __ag_tensor_add_backward(Tensor* upstream){
	if(!upstream || !upstream->parents) return;
	Tensor* a = upstream->parents[0];
	Tensor* b = upstream->parents[1];
	if(a) ensure_grad_buffer(a);
	if(b) ensure_grad_buffer(b);
	int sz = tensor_size(upstream);
	for(int i=0;i<sz;i++){
		float g = upstream->grad ? upstream->grad[i] : 0.0f;
		if(a) a->grad[i] += g;
		if(b) b->grad[i] += g;
	}
	return;
}

void __ag_tensor_sub_backward(Tensor* upstream){
	if(!upstream || !upstream->parents) return;
	Tensor* a = upstream->parents[0];
	Tensor* b = upstream->parents[1];
	if(a) ensure_grad_buffer(a);
	if(b) ensure_grad_buffer(b);
	int sz = tensor_size(upstream);
	for(int i=0;i<sz;i++){
		float g = upstream->grad ? upstream->grad[i] : 0.0f;
		if(a) a->grad[i] += g;
		if(b) b->grad[i] -= g;
	}
	return;
}

void __ag_tensor_div_backward(Tensor* upstream){
	if(!upstream || !upstream->parents) return;
	Tensor* a = upstream->parents[0];
	Tensor* b = upstream->parents[1];
	if(a) ensure_grad_buffer(a);
	if(b) ensure_grad_buffer(b);
	int sz = tensor_size(upstream);
	for(int i=0;i<sz;i++){
		float g = upstream->grad ? upstream->grad[i] : 0.0f;
		float bv = b->_data[i];
		float av = a->_data[i];
		if(a) a->grad[i] += g / (bv + 1e-8f);
		if(b) b->grad[i] -= g * av / (bv * bv + 1e-8f);
	}
	return;
}

void __ag_tensor_hadamard_mul_backward(Tensor* upstream){
	if(!upstream || !upstream->parents) return;
	Tensor* a = upstream->parents[0];
	Tensor* b = upstream->parents[1];
	if(a) ensure_grad_buffer(a);
	if(b) ensure_grad_buffer(b);
	int sz = tensor_size(upstream);
	for(int i=0;i<sz;i++){
		float g = upstream->grad ? upstream->grad[i] : 0.0f;
		if(a) a->grad[i] += g * b->_data[i];
		if(b) b->grad[i] += g * a->_data[i];
	}
	return;
}

void __ag_tensor_sq_backward(Tensor* upstream){
	if(!upstream || !upstream->parents) return;
	Tensor* a = upstream->parents[0];
	if(!a) return;
	ensure_grad_buffer(a);
	int sz = tensor_size(upstream);
	for(int i=0;i<sz;i++){
		float g = upstream->grad ? upstream->grad[i] : 0.0f;
		a->grad[i] += 2.0f * a->_data[i] * g;
	}
	return;
}

void __ag_tensor_sqrt_backward(Tensor* upstream){
	if(!upstream || !upstream->parents) return;
	Tensor* a = upstream->parents[0];
	if(!a) return;
	ensure_grad_buffer(a);
	int sz = tensor_size(upstream);
	for(int i=0;i<sz;i++){
		float g = upstream->grad ? upstream->grad[i] : 0.0f;
		// sqrt(x) grad = 0.5 / sqrt(x), but we need input to sqrt which is variance + EPS
		// Actually a here is var_exp (variance expanded), so a->_data[i] is already variance
		float denom = 2.0f * sqrtf(a->_data[i] + 1e-8f);
		if(denom > 1e-8f) a->grad[i] += g / denom;
	}
	return;
}

void __ag_tensor_expand_cols_backward(Tensor* upstream){
	if(!upstream || !upstream->parents) return;
	Tensor* mean = upstream->parents[0];
	if(!mean) return;
	ensure_grad_buffer(mean);
	int rows = mean->info.shape[0];
	int cols = upstream->info.shape[1];
	for(int r=0;r<rows;r++){
		float s = 0.0f;
		for(int c=0;c<cols;c++) {
			float g = upstream->grad ? upstream->grad[r*cols + c] : 0.0f;
			s += g;
		}
		mean->grad[r] += s;
	}
	return;
}

//elementwise multiplication
Tensor* __ag_tensor_hadamard_mul_forward(Arena* A, Tensor* a, Tensor* b){
assert(a->info.shape != b->info.shape);
	int ndim = 2;
	int rows = a->info.shape[0];
	int cols = a->info.shape[1];
	int shape[2] = {rows, cols};
	Tensor *res = tensor_create(ndim, shape);


	if(a->requires_grad == true || b->requires_grad == true){
        res->requires_grad = true;
		res->num_parents = 2;
		res->parents = arena_alloc(A, res->num_parents * sizeof(Tensor*));
		res->parents[0] = a;
		res->parents[1] = b;

		int out_size = a->info.shape[0] * b->info.shape[1];
		res->grad = arena_alloc(A, out_size * sizeof(float));
		//treebased memAllocator

		//lets define the ops
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_hadamard_mul_backward;
		res->ops = op;
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * res->info.shape[1] + j;
			res->_data[idx] = a->_data[idx] * b->_data[idx];
		}
	}

	return res;
}

Tensor* __ag_tensor_add_forward(Arena* A, Tensor* a, Tensor* b){
assert(a->info.shape != b->info.shape);
	int ndim = 2;
	int rows = a->info.shape[0];
	int cols = a->info.shape[1];
	int shape[2] = {rows, cols};
	Tensor *res = tensor_create(ndim, shape);


	if(a->requires_grad == true || b->requires_grad == true){
        res->requires_grad = true;
		res->num_parents = 2;
		res->parents = arena_alloc(A, res->num_parents * sizeof(Tensor*));
		res->parents[0] = a;
		res->parents[1] = b;

		int out_size = a->info.shape[0] * b->info.shape[1];
		res->grad = arena_alloc(A, out_size * sizeof(float));
		//treebased memAllocator

		//lets define the ops
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_add_backward;
		res->ops = op;
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * res->info.shape[1] + j;
			res->_data[idx] = a->_data[idx] + b->_data[idx];
		}
	}

	return res;
}

Tensor* __ag_tensor_sub_forward(Arena* A, Tensor* a, Tensor* b){
assert(a->info.shape != b->info.shape);
	int ndim = 2;
	int rows = a->info.shape[0];
	int cols = a->info.shape[1];
	int shape[2] = {rows, cols};
	Tensor *res = tensor_create(ndim, shape);


	if(a->requires_grad == true || b->requires_grad == true){
        res->requires_grad = true;
		res->num_parents = 2;
		res->parents = arena_alloc(A, res->num_parents * sizeof(Tensor*));
		res->parents[0] = a;
		res->parents[1] = b;

		int out_size = a->info.shape[0] * b->info.shape[1];
		res->grad = arena_alloc(A, out_size * sizeof(float));
		//treebased memAllocator

		//lets define the ops
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_sub_backward;
		res->ops = op;
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * res->info.shape[1] + j;
			res->_data[idx] = a->_data[idx] - b->_data[idx];
		}
	}

	return res;
}

Tensor* __ag_tensor_div_forward(Arena* A, Tensor* a, Tensor* b){
assert(a->info.shape != b->info.shape);
	int ndim = 2;
	int rows = a->info.shape[0];
	int cols = a->info.shape[1];

	int shape[2] = {rows, cols};
	Tensor *res = tensor_create(ndim, shape);


	if(a->requires_grad == true || b->requires_grad == true){
        res->requires_grad = true;
		res->num_parents = 2;
		res->parents = arena_alloc(A, res->num_parents * sizeof(Tensor*));
		res->parents[0] = a;
		res->parents[1] = b;

		int out_size = a->info.shape[0] * b->info.shape[1];
		res->grad = arena_alloc(A, out_size * sizeof(float));
		//treebased memAllocator

		//lets define the ops
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_div_backward;
		res->ops = op;
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * res->info.shape[1] + j;
			res->_data[idx] = a->_data[idx] / b->_data[idx];
		}
	}

	return res;
}


void tensor_fill_zeros(Tensor *a) {
	if (!a) return;
	for (int i = 0; i < a->info.sz; i++) {
		a->_data[i] = 0.0f;
	}
	if (a->grad) {
		for (int i = 0; i < a->info.sz; i++) {
			a->grad[i] = 0.0f;
		}
	}
}

float loss_value(Tensor *a, Tensor *b) {
	if (!a || !b) return 0.0f;
	int sz = a->info.sz;
	if (sz <= 0) return 0.0f;
	float sum = 0.0f;
	for (int i = 0; i < sz; i++) {
		float d = a->_data[i] - b->_data[i];
		sum += d * d;
	}
	return sum / (float)sz;
}

void __ag_tensor_mse_loss_backward(Tensor* out) {
	if(!out || !out->parents) return;
	Tensor* pred = out->parents[0];
	Tensor* target = out->parents[1];  // not used for grad but needed for shape
	if(!pred) return;
	ensure_grad_buffer(pred);
	
	int sz = pred->info.sz;
	float inv = 2.0f / (float)sz;
	
	// Gradient of MSE w.r.t. pred: 2*(pred - target)/sz
	for(int i = 0; i < sz; i++) {
		float target_val = target ? target->_data[i] : 0.0f;
		float upstream = out->grad ? out->grad[i] : 1.0f;
		pred->grad[i] += (pred->_data[i] - target_val) * inv * upstream;
	}
}

// Manual/non-autograd MSE loss (kept for backward compatibility)
Tensor *tensor_mse_loss_manual(Arena *A, Tensor *pred, Tensor *target) {
	if (!pred || !target) return NULL;
	int ndim = pred->ndim;
	int* shape = arena_alloc(A, ndim * sizeof(int));
	for (int i = 0; i < ndim; i++) shape[i] = pred->info.shape[i];
	Tensor *dx = tensor_create_new(A, ndim, shape);
	int sz = pred->info.sz;
	if (sz <= 0) return dx;
	float inv = 2.0f / (float)sz;
	for (int i = 0; i < sz; i++) {
		dx->_data[i] = (pred->_data[i] - target->_data[i]) * inv;
	}
	return dx;
}

// Autograd-enabled MSE loss
Tensor *__ag_tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target) {
	if (!pred || !target) return NULL;
	int ndim = pred->ndim;
	int* shape = arena_alloc(A, ndim * sizeof(int));
	for (int i = 0; i < ndim; i++) shape[i] = pred->info.shape[i];
	Tensor *loss = tensor_create_new(A, ndim, shape);
	
	int sz = pred->info.sz;
	if (sz <= 0) return loss;
	
	// Compute MSE: (pred - target)^2 / sz
	float inv = 1.0f / (float)sz;
	for (int i = 0; i < sz; i++) {
		float diff = pred->_data[i] - target->_data[i];
		loss->_data[i] = diff * diff * inv;
	}
	
	// Build autograd graph if pred requires grad
	if(pred->requires_grad || target->requires_grad) {
		loss->requires_grad = true;
		loss->num_parents = 2;
		loss->parents = arena_alloc(A, 2 * sizeof(Tensor*));
		loss->parents[0] = pred;
		loss->parents[1] = target;
		loss->grad = arena_alloc(A, sz * sizeof(float));
		memset(loss->grad, 0, sz * sizeof(float));
		
		Op* op = arena_alloc(A, sizeof(Op));
		op->backward = __ag_tensor_mse_loss_backward;
		loss->ops = op;
	}
	
	return loss;
}

// Default uses autograd version
Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target) {
	return __ag_tensor_mse_loss(A, pred, target);
}

Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b) {
	return __ag_tensor_matmul_forward(A, a, b);
}

Tensor *tensor_softmax(Arena* A, Tensor *t) {
	return __ag_tensor_softmax_forward(A, t);
}

// DAG visualization implementation
const char* tensor_get_op_name(Tensor* node) {
	if(!node || !node->ops) return "input/leaf";
	void(*bw)(Tensor*) = node->ops->backward;
	if(bw == __ag_tensor_matmul_backward) return "matmul";
	if(bw == __ag_tensor_add_backward) return "add";
	if(bw == __ag_tensor_sub_backward) return "sub";
	if(bw == __ag_tensor_div_backward) return "div";
	if(bw == __ag_tensor_hadamard_mul_backward) return "hadamard_mul";
	if(bw == __ag_tensor_mean_backward) return "mean";
	if(bw == __ag_tensor_expand_cols_backward) return "expand_cols";
	if(bw == __ag_tensor_sq_backward) return "square";
	if(bw == __ag_tensor_sqrt_backward) return "sqrt";
	if(bw == __ag_tensor_mse_loss_backward) return "mse_loss";
	if(bw == __ag_tensor_transpose_backward) return "transpose";
	if(bw == __ag_tensor_slice_backward) return "slice";
	if(bw == __ag_tensor_scatter_backward) return "scatter";
	return "unknown_op";
}

// Print horizontal line for box
void print_box_line(int width, const char* left, const char* right) {
	printf("%s", left);
	for(int i = 0; i < width; i++) printf("-");
	printf("%s", right);
}

// Print a node as an ASCII box with connecting lines
void tensor_print_node_box(Tensor* node, int depth, const char* prefix, int is_last) {
	if(!node) return;
	
	const char* op_name = tensor_get_op_name(node);
	int shape[2] = {0, 0};
	if(node->info.shape) {
		shape[0] = node->info.shape[0];
		shape[1] = node->info.shape[1];
	}
	
	// Print prefix and connector
	printf("%s", prefix);
	if(depth > 0) {
		printf("%s", is_last ? "`-- " : "|-- ");
	}
	
	// Build full metadata rows (keep all information and make it easier to read)
	char row1[192];
	char row2[192];
	char row3[192];
	char row4[192];
	char row5[192];
	snprintf(row1, sizeof(row1), " op=%s ", op_name);
	snprintf(row2, sizeof(row2), " shape=[%d,%d] size=%d ndim=%d ", shape[0], shape[1], node->info.sz, node->ndim);
	snprintf(row3, sizeof(row3), " requires_grad=%d parents=%d ptr=%p ", node->requires_grad ? 1 : 0, node->num_parents, (void*)node);
	if(node->_data && node->info.sz > 0) {
		snprintf(row4, sizeof(row4), " data=[%.6f, %.6f, %.6f ...] ", node->_data[0],
			node->info.sz > 1 ? node->_data[1] : 0.0f,
			node->info.sz > 2 ? node->_data[2] : 0.0f);
	} else {
		snprintf(row4, sizeof(row4), " data=[NULL] ");
	}
	if(node->grad && node->info.sz > 0) {
		snprintf(row5, sizeof(row5), " grad=[%.6f, %.6f, %.6f ...] ", node->grad[0],
			node->info.sz > 1 ? node->grad[1] : 0.0f,
			node->info.sz > 2 ? node->grad[2] : 0.0f);
	} else {
		snprintf(row5, sizeof(row5), " grad=[NULL] ");
	}
	int box_width = (int)strlen(row1);
	if((int)strlen(row2) > box_width) box_width = (int)strlen(row2);
	if((int)strlen(row3) > box_width) box_width = (int)strlen(row3);
	if((int)strlen(row4) > box_width) box_width = (int)strlen(row4);
	if((int)strlen(row5) > box_width) box_width = (int)strlen(row5);
	
	// Print top border of box
	printf("+");
	for(int i = 0; i < box_width; i++) printf("-");
	printf("+\n");
	
	// Print prefix continuation and all rows
	printf("%s", prefix);
	if(depth > 0) {
		printf("%s   ", is_last ? "    " : "|   ");
	}
	printf("|%s", row1); for(int i = 0; i < box_width - (int)strlen(row1); i++) printf(" "); printf("|\n");
	printf("%s", prefix); if(depth > 0) printf("%s   ", is_last ? "    " : "|   ");
	printf("|%s", row2); for(int i = 0; i < box_width - (int)strlen(row2); i++) printf(" "); printf("|\n");
	printf("%s", prefix); if(depth > 0) printf("%s   ", is_last ? "    " : "|   ");
	printf("|%s", row3); for(int i = 0; i < box_width - (int)strlen(row3); i++) printf(" "); printf("|\n");
	printf("%s", prefix); if(depth > 0) printf("%s   ", is_last ? "    " : "|   ");
	printf("|%s", row4); for(int i = 0; i < box_width - (int)strlen(row4); i++) printf(" "); printf("|\n");
	printf("%s", prefix); if(depth > 0) printf("%s   ", is_last ? "    " : "|   ");
	printf("|%s", row5); for(int i = 0; i < box_width - (int)strlen(row5); i++) printf(" "); printf("|\n");
	
	// Print bottom border
	printf("%s", prefix);
	if(depth > 0) {
		printf("%s   ", is_last ? "    " : "|   ");
	}
	printf("+");
	for(int i = 0; i < box_width; i++) printf("-");
	printf("+");
}

void tensor_print_dag_recursive(Tensor* node, int depth, int* visited, int max_visited, const char* prefix) {
	if(!node) return;
	
	// Check if already visited (prevent cycles in printing)
	for(int i = 0; i < max_visited; i++) {
		if(visited[i] == (int)(size_t)node) {
			// Already visited - print reference only
			printf("%s", prefix);
			if(depth > 0) printf("    ");
			printf("[see above: %s@%p]\n", tensor_get_op_name(node), (void*)node);
			return;
		}
	}
	
	// Mark as visited
	for(int i = 0; i < max_visited; i++) {
		if(visited[i] == 0) {
			visited[i] = (int)(size_t)node;
			break;
		}
	}
	
	// Count parents for determining "is_last"
	int num_parents = node->parents ? node->num_parents : 0;
	
	// Print this node as a box
	int is_last = 1; // For root, always use last style
	tensor_print_node_box(node, depth, prefix, is_last);
	printf("\n");
	
	// Recursively print parents with updated prefix
	if(node->parents && node->num_parents > 0) {
		for(int i = 0; i < node->num_parents; i++) {
			char new_prefix[256];
			snprintf(new_prefix, sizeof(new_prefix), "%s%s", 
				prefix, 
				(depth > 0 || i < node->num_parents - 1) ? "|   " : "    ");
			
			// If this is the last parent, don't continue the vertical line
			if(i == node->num_parents - 1) {
				snprintf(new_prefix, sizeof(new_prefix), "%s    ", prefix);
			}
			
			tensor_print_dag_recursive(node->parents[i], depth + 1, visited, max_visited, new_prefix);
		}
	}
}

void tensor_print_dag(Tensor* root, const char* name) {
	if(!root) {
		printf("\n+============================================================================+\n");
		printf("|  COMPUTATION GRAPH (DAG): %-48s|\n", name ? name : "unnamed (NULL)");
		printf("+============================================================================+\n\n");
		return;
	}
	
	printf("\n+============================================================================+\n");
	printf("|  COMPUTATION GRAPH (DAG): %-48s|\n", name ? name : "unnamed");
	printf("+============================================================================+\n");
	printf("|  Root: shape=[%d,%d] size=%d requires_grad=%s ptr=%p |\n",
		root->info.shape[0], root->info.shape[1],
		root->info.sz, 
		root->requires_grad ? "true " : "false",
		(void*)root);
	printf("+============================================================================+\n");
	printf("\n  Flow: INPUTS (below) --> OUTPUT (root above)\n\n");
	
	int visited[1000] = {0};
	tensor_print_dag_recursive(root, 0, visited, 1000, "");
	
	printf("\n+============================================================================+\n\n");
}

Tensor* __nag_tensor_matmul(Arena* A, Tensor* a, Tensor* b) {
	// Non-autograd matmul - no graph building, pure computation
	assert(a->info.shape[1] == b->info.shape[0]);
	
	int ndim_r = 2;
	int rows_a = a->info.shape[0];
	int cols_a = a->info.shape[1];
	int cols_b = b->info.shape[1];
	
	int* shape_r = arena_alloc(A, ndim_r * sizeof(int));
	shape_r[0] = rows_a;
	shape_r[1] = cols_b;
	
	Tensor* r = tensor_create_new(A, ndim_r, shape_r);
	
	// Zero initialize result
	memset(r->_data, 0, rows_a * cols_b * sizeof(float));
	
	// Simple triple-loop matmul (no autograd tracking) //use SIMD here
	for(int i = 0; i < rows_a; i++) {
		for(int k = 0; k < cols_a; k++) {
			float a_val = a->_data[i * cols_a + k];
			for(int j = 0; j < cols_b; j++) {
				r->_data[i * cols_b + j] += a_val * b->_data[k * cols_b + j];
			}
		}
	}

	
	/* IF YOU WANT TO USE BLAS
	// Use BLAS for GEMM
	#define _USE_CBLAS_
	#include <cblas.h>
    #endif
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows_a, cols_b, cols_a,
                1.0,
                a->data, cols_a,
                b->data, cols_b,
                0.0,
                r->data, cols_b);*/
	
	return r;
}

Tensor* __nag_tensor_softmax(Arena* A, Tensor* a) {
	// Non-autograd softmax - no graph building
	int ndim = 2;
	int rows = a->info.shape[0];
	int cols = a->info.shape[1];
	
	int* shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = rows;
	shape[1] = cols;
	
	Tensor* r = tensor_create_new(A, ndim, shape);
	
	// Row-wise softmax
	for(int i = 0; i < rows; i++) {
		// Find max for numerical stability
		float max_val = a->_data[i * cols];
		for(int j = 1; j < cols; j++) {
			if(a->_data[i * cols + j] > max_val) max_val = a->_data[i * cols + j];
		}
		
		// Compute exp and sum
		float sum = 0.0f;
		for(int j = 0; j < cols; j++) {
			r->_data[i * cols + j] = expf(a->_data[i * cols + j] - max_val);
			sum += r->_data[i * cols + j];
		}
		
		// Normalize
		float inv_sum = 1.0f / sum;
		for(int j = 0; j < cols; j++) {
			r->_data[i * cols + j] *= inv_sum;
		}
	}
	
	return r;
}

Tensor* __nag_relu_forward(Tensor* x) {
	// Non-autograd ReLU - modifies in-place, no graph tracking
	int size = x->info.sz;
	
	// SIMD with AVX2
	#if defined(__AVX2__)
		__m256 vzero = _mm256_setzero_ps();
		int i = 0;
		for(; i <= size - 8; i += 8) {
			__m256 v = _mm256_loadu_ps(&x->_data[i]);
			v = _mm256_max_ps(v, vzero);
			_mm256_storeu_ps(&x->_data[i], v);
		}
		// Scalar tail
		for(; i < size; i++) {
			if(x->_data[i] < 0.0f) x->_data[i] = 0.0f;
		}
	#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
		float32x4_t vzero = vdupq_n_f32(0.0f);
		int i = 0;
		for(; i <= size - 4; i += 4) {
			float32x4_t v = vld1q_f32(&x->_data[i]);
			v = vmaxq_f32(v, vzero);
			vst1q_f32(&x->_data[i], v);
		}
		for(; i < size; i++) {
			if(x->_data[i] < 0.0f) x->_data[i] = 0.0f;
		}
	#else
		// Scalar fallback
		for(int i = 0; i < size; i++) {
			if(x->_data[i] < 0.0f) x->_data[i] = 0.0f;
		}
	#endif
	
	return x;
}

// Autograd backward pass - traverses computation graph in reverse topological order
// and calls backward functions to compute gradients
void tensor_backward_recursive(Tensor* node, int* visited, int max_visited) {
	if(!node) return;
	
	// Check if already visited (prevent cycles)
	for(int i = 0; i < max_visited; i++) {
		if(visited[i] == (int)(size_t)node) return;
	}
	
	// Mark as visited
	for(int i = 0; i < max_visited; i++) {
		if(visited[i] == 0) {
			visited[i] = (int)(size_t)node;
			break;
		}
	}
	
	// First, call this node's backward function to propagate gradients to parents
	if(node->ops && node->ops->backward) {
		node->ops->backward(node);
	}
	
	// Then recursively backward parents (now they have upstream gradients)
	if(node->parents) {
		for(int i = 0; i < node->num_parents; i++) {
			tensor_backward_recursive(node->parents[i], visited, max_visited);
		}
	}
}

void tensor_backward(Tensor* root) {
	if(!root) {
		fprintf(stderr, "tensor_backward: root is NULL\n");
		return;
	}
	
	if(!root->requires_grad) {
		fprintf(stderr, "tensor_backward: root does not require grad\n");
		return;
	}
	
	// Initialize grad buffer for root with seed gradient of 1.0
	if(!root->grad) {
		root->grad = (float*)malloc(root->info.sz * sizeof(float));
	}
	if(root->grad) {
		for(int i = 0; i < root->info.sz; i++) {
			root->grad[i] = 1.0f; // d(root)/d(root) = 1
		}
	}
	
	int visited[1000] = {0};
	tensor_backward_recursive(root, visited, 1000);
}


// Fast index calculation methods for performance
static inline int fast_index(const int* indices, const Tensor* t) {
	if (!indices || !t || !t->info.shape || t->ndim <= 0) return 0;
	int index = 0;
	int stride = 1;
	for (int i = t->ndim - 1; i >= 0; --i) {
		index += indices[i] * stride;
		stride *= t->info.shape[i];
	}
	return index;
}
// Fast index calculation without bounds checking (for performance)
static inline int fast_index_3d(int c, int h, int w, const Tensor* t) {
	// For shape [channels, height, width]
	return c * t->info.shape[1] * t->info.shape[2] + h * t->info.shape[2] + w;
} // Optimized for 3D access


static inline int fast_index_4d(int f, int c, int h, int w, const Tensor* t) {
	// For shape [filters, channels, height, width]
	return f * t->info.shape[1] * t->info.shape[2] * t->info.shape[3] +
	       c * t->info.shape[2] * t->info.shape[3] +
	       h * t->info.shape[3] + w;
} // Optimized for 4D access


static float _rand_uniform01(void) {
	return (float)rand() / (float)RAND_MAX;
}

static float _rand_normal01(void) {
	/* Box-Muller transform */
	float u1 = _rand_uniform01();
	float u2 = _rand_uniform01();
	if (u1 < 1e-12f) u1 = 1e-12f;
	float r = sqrtf(-2.0f * logf(u1));
	float theta = 6.28318530718f * u2;
	return r * cosf(theta);
}

static void _tensor_init_xavier_uniform(Tensor* w, int fan_in, int fan_out) {
	if (!w || !w->_data || w->info.sz <= 0) return;
	if (fan_in <= 0) fan_in = 1;
	if (fan_out <= 0) fan_out = 1;
	float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
	for (int i = 0; i < w->info.sz; i++) {
		float u = _rand_uniform01();
		w->_data[i] = (2.0f * u - 1.0f) * limit;
	}
}

static void _tensor_init_he_normal(Tensor* w, int fan_in) {
	if (!w || !w->_data || w->info.sz <= 0) return;
	if (fan_in <= 0) fan_in = 1;
	float std = sqrtf(2.0f / (float)fan_in);
	for (int i = 0; i < w->info.sz; i++) {
		w->_data[i] = _rand_normal01() * std;
	}
}

typedef struct _adam_state {
	Tensor* m;
	Tensor* v;
	int t;
} _adam_state;

static _adam_state _adam_state_create(Arena* A, const Tensor* w) {
	_adam_state st;
	memset(&st, 0, sizeof(st));
	if (!A || !w) return st;
	int ndim = w->ndim;
	Tensor* m = tensor_create_new(A, ndim, w->info.shape);
	Tensor* v = tensor_create_new(A, ndim, w->info.shape);
	tensor_fill_zeros(m);
	tensor_fill_zeros(v);
	st.m = m;
	st.v = v;
	st.t = 0;
	return st;
}

static void _adam_step(Tensor* w, const Tensor* dw, _adam_state* st, float lr, float beta1, float beta2, float eps) {
	if (!w || !dw || !st || !st->m || !st->v) return;
	if (!w->_data || !dw->_data || !st->m->_data || !st->v->_data) return;
	if (w->info.sz != dw->info.sz) return;
	if (eps <= 0.0f) eps = 1e-8f;
	st->t += 1;

	float b1t = powf(beta1, (float)st->t);
	float b2t = powf(beta2, (float)st->t);
	float inv1 = 1.0f / (1.0f - b1t);
	float inv2 = 1.0f / (1.0f - b2t);

	for (int i = 0; i < w->info.sz; i++) {
		float g = dw->_data[i];
		float m = st->m->_data[i] = beta1 * st->m->_data[i] + (1.0f - beta1) * g;
		float v = st->v->_data[i] = beta2 * st->v->_data[i] + (1.0f - beta2) * (g * g);
		float mhat = m * inv1;
		float vhat = v * inv2;
		w->_data[i] -= lr * mhat / (sqrtf(vhat) + eps);
	}
}

typedef struct _muon_state {
	Tensor* v;
} _muon_state;

static _muon_state _muon_state_create(Arena* A, const Tensor* w) {
	_muon_state st;
	memset(&st, 0, sizeof(st));
	if (!A || !w) return st;
	Tensor* v = tensor_create_new(A, w->ndim, w->info.shape);
	tensor_fill_zeros(v);
	st.v = v;
	return st;
}

static void _muon_step(Tensor* w, const Tensor* dw, _muon_state* st, float lr, float momentum) {
	/* MUON: implemented here as SGD + momentum (velocity). */
	if (!w || !dw || !st || !st->v) return;
	if (!w->_data || !dw->_data || !st->v->_data) return;
	if (w->info.sz != dw->info.sz) return;
	if (momentum < 0.0f) momentum = 0.0f;
	if (momentum > 1.0f) momentum = 1.0f;

	for (int i = 0; i < w->info.sz; i++) {
		float v = st->v->_data[i] = momentum * st->v->_data[i] - lr * dw->_data[i];
		w->_data[i] += v;
	}
}

typedef float (*_newton_fn)(float x, void* ctx);

static float _newton_raphson(_newton_fn f, _newton_fn df, void* ctx, float x0, int max_iter, float tol) {
	if (!f || !df) return x0;
	if (max_iter <= 0) max_iter = 20;
	if (tol <= 0.0f) tol = 1e-6f;
	float x = x0;
	for (int i = 0; i < max_iter; i++) {
		float fx = f(x, ctx);
		float dfx = df(x, ctx);
		if (fabsf(dfx) < 1e-12f) break;
		float step = fx / dfx;
		x = x - step;
		if (fabsf(step) < tol) break;
	}
	return x;
}

#endif

/*"C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -Wall -Wextra test.c*/

/*"C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/g++.exe" -std=c++17 -Wall -Wextra -o test.exe test.c*/
/* Full non-autograd CNN training: Conv2D -> Pool2D -> LN -> FFN -> LN */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "./conv2d.h"
#include "./pool.h"
#include "./ln.h"
#include "./ffn.h"
#include "./threadpool.c"
#include "./conv2d.c"

#define CONV_EPOCHS      3
#define CONV_NUM_BATCHES 32
#define CONV_IN_C        3
#define CONV_H           16
#define CONV_W           16
#define CONV_FILTERS     8
#define CONV_K           3
#define CONV_STRIDE      1
#define CONV_PAD         1
#define POOL_SIZE        2
#define POOL_STRIDE      2
#define CONV_HIDDEN      64
#define CONV_LR          1e-3f
#define CONV_ARENA_BYTES (256 * 1024 * 1024)

int main(void) {
	srand((unsigned)time(NULL));

	Arena* A = (Arena*)malloc(sizeof(Arena));
	arena_init(A, CONV_ARENA_BYTES);

	int in_bytes = CONV_IN_C * CONV_H * CONV_W * (int)sizeof(float);
	int conv_h = conv2d_out_dim(CONV_H, CONV_PAD, CONV_K, CONV_STRIDE);
	int conv_w = conv2d_out_dim(CONV_W, CONV_PAD, CONV_K, CONV_STRIDE);
	int pool_h = pool2d_out_dim(conv_h, POOL_SIZE, POOL_STRIDE);
	int pool_w = pool2d_out_dim(conv_w, POOL_SIZE, POOL_STRIDE);
	int rows = pool_h * pool_w;
	int token_bytes = rows * CONV_FILTERS * (int)sizeof(float);

	float* data_store = (float*)arena_alloc(A, (size_t)CONV_NUM_BATCHES * (size_t)in_bytes);
	float* target_store = (float*)arena_alloc(A, (size_t)CONV_NUM_BATCHES * (size_t)token_bytes);
	for (int i = 0; i < CONV_NUM_BATCHES * CONV_IN_C * CONV_H * CONV_W; i++) {
		data_store[i] = (float)rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < CONV_NUM_BATCHES * rows * CONV_FILTERS; i++) {
		target_store[i] = (float)rand() / (float)RAND_MAX;
	}

	Conv2D* conv = conv2d_create(A, CONV_IN_C, CONV_FILTERS, CONV_K, CONV_STRIDE, CONV_PAD);
	Pool2D* pool = pool2d_create(A, POOL_SIZE, POOL_STRIDE, MaxPool2D);
	LayerNorm* L1 = layer_norm_create_new(A, CONV_FILTERS);
	layer_norm_init_params(L1);
	LayerNorm* L2 = layer_norm_create_new(A, CONV_FILTERS);
	layer_norm_init_params(L2);
	FFN* f = ffn_create(A, CONV_FILTERS, CONV_HIDDEN);
	ffn_init_params(f);

	int ishape[3] = { CONV_IN_C, CONV_H, CONV_W };
	int pool_shape[3] = { CONV_FILTERS, pool_h, pool_w };
	int token_shape[2] = { rows, CONV_FILTERS };

	FILE* logf = fopen("loss_conv2d_nag.csv", "w");
	if (!logf) return 1;
	fprintf(logf, "step,loss\n");

	printf("Starting nag training: Conv->Pool->LN->FFN->LN\n");
	clock_t start_time = clock();
	int global_step = 0;

	for (int e = 1; e <= CONV_EPOCHS; e++) {
		for (int b = 0; b < CONV_NUM_BATCHES; b++) {
			float* batch_ptr = data_store + b * (in_bytes / (int)sizeof(float));
			float* target_ptr = target_store + b * (token_bytes / (int)sizeof(float));

			Tensor* input = tensor_create_new(A, 3, ishape);
			Tensor* target = tensor_create_new(A, 2, token_shape);
			memcpy(input->_data, batch_ptr, (size_t)in_bytes);
			memcpy(target->_data, target_ptr, (size_t)token_bytes);

			Tensor* conv_out = __nag_conv2d_forward(A, conv, input);
			Tensor* pool_out = __nag_pool2d_forward(A, pool, conv_out);
			Tensor* tokens = tensor_create_new(A, 2, token_shape);
			conv3d_to_tokens(pool_out, tokens);

			Tensor* ln1 = layer_norm_forward(A, L1, tokens);
			Tensor* ffn_out = __nag_ffn_forward(A, ln1, f);
			Tensor* ln2 = layer_norm_forward(A, L2, ffn_out);

			float loss_to_show = loss_value(ln2, target);
			Tensor* loss = tensor_mse_loss_manual(A, ln2, target);

			Tensor* dx_ffn = tensor_create_new(A, 2, token_shape);
			layer_norm_backward(L2, ffn_out, loss, dx_ffn, CONV_LR);
			Tensor* dx_ln1 = ffn_backward(A, f, ln1, dx_ffn);
			Tensor* dx_tokens = tensor_create_new(A, 2, token_shape);
			layer_norm_backward(L1, tokens, dx_ln1, dx_tokens, CONV_LR);

			Tensor* pool_dout = tensor_create_new(A, 3, pool_shape);
			tensor_fill_zeros(pool_dout);
			conv_tokens_to_3d_grad(dx_tokens, pool_dout);
			Tensor* conv_dout = __nag_pool2d_backward(A, pool, pool_dout);
			(void)__nag_conv2d_backward(A, conv, conv_dout);

			clip_gradient(conv->dweights);
			clip_gradient(conv->dbiases);
			clip_gradient(f->dw1);
			clip_gradient(f->dw2);
			clip_gradient(f->da1);
			clip_gradient(f->dh1);
			clip_gradient(L1->d_gamma);
			clip_gradient(L1->d_beta);
			clip_gradient(L2->d_gamma);
			clip_gradient(L2->d_beta);

			sgd_optimizer(conv->weights, conv->dweights, CONV_LR);
			sgd_optimizer(conv->biases, conv->dbiases, CONV_LR);
			sgd_optimizer(f->w1, f->dw1, CONV_LR);
			sgd_optimizer(f->w2, f->dw2, CONV_LR);
			sgd_optimizer(f->a1, f->da1, CONV_LR);
			sgd_optimizer(f->h1, f->dh1, CONV_LR);
			sgd_optimizer(L1->gamma, L1->d_gamma, CONV_LR);
			sgd_optimizer(L1->beta, L1->d_beta, CONV_LR);
			sgd_optimizer(L2->gamma, L2->d_gamma, CONV_LR);
			sgd_optimizer(L2->beta, L2->d_beta, CONV_LR);

			tensor_fill_zeros(L1->d_gamma);
			tensor_fill_zeros(L1->d_beta);
			tensor_fill_zeros(L2->d_gamma);
			tensor_fill_zeros(L2->d_beta);

			if (b % 4 == 0) {
				printf("Loss: %f, Epoch: %d, Batch: %d\n", loss_to_show, e, b);
				fprintf(logf, "%d,%f\n", global_step, loss_to_show);
				global_step++;
			}
		}
	}

	printf("conv2d nag training finished in %.3f seconds\n",
	       (double)(clock() - start_time) / (double)CLOCKS_PER_SEC);
	fclose(logf);
	return 0;
}

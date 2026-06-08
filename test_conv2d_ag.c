/* Full autograd CNN training: Conv2D -> Pool2D -> LN -> FFN -> LN */
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

static void optimize_tensor_sgd(Tensor* w, float lr) {
	if (!w || !w->grad) return;
	Tensor grad = {0};
	grad._data = w->grad;
	grad.info = w->info;
	grad.ndim = w->ndim;
	clip_gradient(&grad);
	sgd_optimizer(w, &grad, lr);
	memset(w->grad, 0, w->info.sz * sizeof(float));
}

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
	conv->weights->requires_grad = true;
	conv->biases->requires_grad = true;
	Pool2D* pool = pool2d_create(A, POOL_SIZE, POOL_STRIDE, MaxPool2D);

	LayerNorm* L1 = layer_norm_create_new(A, CONV_FILTERS);
	layer_norm_init_params(L1);
	L1->gamma->requires_grad = true;
	L1->beta->requires_grad = true;

	FFN* f = ffn_create(A, CONV_FILTERS, CONV_HIDDEN);
	ffn_init_params(f);
	f->w1->requires_grad = true;
	f->w2->requires_grad = true;

	LayerNorm* L2 = layer_norm_create_new(A, CONV_FILTERS);
	layer_norm_init_params(L2);
	L2->gamma->requires_grad = true;
	L2->beta->requires_grad = true;

	int ishape[3] = { CONV_IN_C, CONV_H, CONV_W };
	int token_shape[2] = { rows, CONV_FILTERS };

	FILE* logf = fopen("loss_conv2d_ag.csv", "w");
	if (!logf) return 1;
	fprintf(logf, "step,loss\n");

	printf("Starting autograd training: Conv->Pool->LN->FFN->LN\n");
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
			input->requires_grad = true;

			Tensor* conv_out = __ag_conv2d_forward(A, conv, input);
			Tensor* pool_out = __ag_pool2d_forward(A, pool, conv_out);
			Tensor* tokens = __ag_conv3d_to_tokens_forward(A, pool_out);
			Tensor* ln1 = __ag_layer_norm_forward(A, L1, tokens);
			Tensor* ffn_out = __ag_ffn_forward(A, ln1, f);
			Tensor* ln2 = __ag_layer_norm_forward(A, L2, ffn_out);

			Tensor* loss = tensor_mse_loss(A, ln2, target);
			float loss_to_show = loss_value(ln2, target);

			tensor_backward(loss);

			optimize_tensor_sgd(conv->weights, CONV_LR);
			optimize_tensor_sgd(conv->biases, CONV_LR);
			optimize_tensor_sgd(f->w1, CONV_LR);
			optimize_tensor_sgd(f->w2, CONV_LR);
			optimize_tensor_sgd(L1->gamma, CONV_LR);
			optimize_tensor_sgd(L1->beta, CONV_LR);
			optimize_tensor_sgd(L2->gamma, CONV_LR);
			optimize_tensor_sgd(L2->beta, CONV_LR);

			if (b % 4 == 0) {
				printf("Loss: %f, Epoch: %d, Batch: %d\n", loss_to_show, e, b);
				fprintf(logf, "%d,%f\n", global_step, loss_to_show);
				global_step++;
			}
		}
	}

	printf("conv2d autograd training finished in %.3f seconds\n",
	       (double)(clock() - start_time) / (double)CLOCKS_PER_SEC);
	fclose(logf);
	return 0;
}

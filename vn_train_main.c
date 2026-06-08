#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "tensor.h"
#include "attn.h"
#include "ln.h"
#include "ffn.h"
#include "config.h"
#include "vn_transf_mdl.h"

int main() {
	fprintf(stderr, "DEBUG: Entering main\n");
	fflush(stderr);
	
	srand(time(NULL));

	size_t SIZE = 1024 * 1024 * 1024;
	Arena *A = malloc(sizeof(Arena));
	arena_init(A, SIZE);
	printf("Arena initilized\n");

	int ndim = 2;
	int shape[2] = {SEQ_LEN, EMB_DIM};
	
	// Creating actual data tensor
	Tensor *T = tensor_create_new(A, 2, shape);
	printf("Tensor T created, shape: %d x %d\n", T->info.shape[0], T->info.shape[1]);
	tensor_randomize(T); 
	printf("Tensor T randomized\n");
	int size = tensor_size(T);
	printf("Tensor size: %d\n", size);

	// Create global MHA
	// we are here now....
	//MHA *M = mha_create_new(A, HEADS, SEQ_LEN, EMB_DIM);

	// Target tensor to compare the output against
	int shape_target[2] = {SEQ_LEN, EMB_DIM};
	printf("Creating target tensor...\n");
	Tensor *target = tensor_create_new(A, ndim, shape_target);
	printf("Target tensor created\n");
	tensor_randomize(target);
	printf("Target randomized\n");

	// define batches for Actual tensor
	int num_chunks = SEQ_LEN / BATCH_SIZE;
	printf("num_chunks = %d (SEQ_LEN=%d / BATCH_SIZE=%d)\n", num_chunks, SEQ_LEN, BATCH_SIZE);

	// CREATED THESET COMPONENETS OUTSIDE FOR TESTING!!!
	printf("Creating MHA...\n");
	fflush(stdout);
	MHA *m_batch = mha_create_new(A, HEADS, BATCH_SIZE, EMB_DIM);
	printf("MHA created at %p\n", (void*)m_batch);
	fflush(stdout);
	
	// Initialize parameters for MHA
	printf("Initializing MHA params...\n");
	fflush(stdout);
	mha_init_params(m_batch);
	printf("MHA params initialized\n");
	fflush(stdout);

	printf("Creating L1...\n");
	LayerNorm *L1 = layer_norm_create_new(A, EMB_DIM);
	layer_norm_init_params(L1);
	printf("L1 created\n");

	printf("Creating FFN...\n");
	FFN *f = ffn_create(A, EMB_DIM, 128);
	ffn_init_params(f);
	printf("FFN created\n");

	printf("Creating L2...\n");
	LayerNorm *L2 = layer_norm_create_new(A, EMB_DIM);
	layer_norm_init_params(L2);
	printf("L2 created\n");

	// Log file
	printf("Opening loss.csv...\n");
	FILE *logf = fopen("loss.csv", "w");
	if (!logf) {
		printf("Failed to open loss.csv!\n");
		return 1;
	}
	fprintf(logf, "step,loss\n");
	printf("Starting training loop...\n");
	clock_t start_time = clock();
	int global_step = 0;
	for (int e = 1; e <= EPOCHS; e++) {
		printf("Epoch %d started\n", e);
		for (int b = 0; b < num_chunks; b++) {
			if (b == 0) printf("Processing batch %d/%d\n", b, num_chunks);

			float *batch_ptr = T->_data + b * BATCH_SIZE * EMB_DIM;
			float *target_ptr = target->_data + b * BATCH_SIZE * EMB_DIM;

			int shape_local[2] = {BATCH_SIZE, EMB_DIM};
			Tensor *batch_tensor = tensor_create_new(A, 2, shape_local);
			Tensor *target_batch = tensor_create_new(A, 2, shape_local);
			
			// TODO: Check this code down, may have bugs!!
			memcpy(batch_tensor->_data, batch_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));
			memcpy(target_batch->_data, target_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));

			Tensor *attn_score = __nag_mha_forward(A, batch_tensor, m_batch);
			clip_gradient(attn_score);

			//tensor_get(attn_score);
			tensor_check("attn_score_forward", attn_score);
			//tensor_get(attn_score);

			// Apply layer_norm
			Tensor *ln1 = layer_norm_forward(A, L1, attn_score);
			tensor_check("ln1_forward", ln1);
			//printf("LayerNorm #1 ran successfully!\n");

			// Create FFN feed-forward NN and run ffn_forward pass
			//Tensor *ffn_ln = ffn_forward(A, ln1, f);
			Tensor *ffn_ln = __nag_ffn_forward(A, ln1, f);
			tensor_check("ffn_ln_forward", ffn_ln);

			// Apply layer_norm
			Tensor *ln2 = layer_norm_forward(A, L2, ffn_ln);
			tensor_check("ln2_forward", ln2);
			//tensor_shape(ln2);
			//printf("LayerNorm #2 ran successfully!\n");
		
			Tensor *loss = tensor_mse_loss_manual(A, ln2, target_batch);
			float loss_to_show = loss_value(ln2, target_batch);

			Tensor *dx_for_ffn = tensor_create_new(A, 2, shape_local);
			Tensor *dx_for_mha = tensor_create_new(A, 2, shape_local);
			//printf("All good\n");
			layer_norm_backward(L2, ffn_ln, loss, dx_for_ffn, LR);
			Tensor *ffn_backpass = ffn_backward(A, f, ln1, dx_for_ffn);
			layer_norm_backward(L1, attn_score, ffn_backpass, dx_for_mha, LR);
			
			Tensor *mha_backpass = mha_backward(A, m_batch, dx_for_mha, batch_tensor);

			//printf("w1 shape: \n");
			//tensor_shape(f->w1);

			//printf("dw1 shape: \n");
			//tensor_shape(f->dw1);

			clip_gradient(f->dw1);
			clip_gradient(f->dw2);
			clip_gradient(f->da1);
			clip_gradient(f->dh1);
			sgd_optimizer(f->w1, f->dw1, LR);
			sgd_optimizer(f->w2, f->dw2, LR);
			sgd_optimizer(f->a1, f->da1, LR);
			sgd_optimizer(f->h1, f->dh1, LR);

			tensor_fill_zeros(f->dw1);
			tensor_fill_zeros(f->dw2);
			tensor_fill_zeros(f->da1);
			tensor_fill_zeros(f->dh1);

			clip_gradient(m_batch->dwq);
			clip_gradient(m_batch->dwk);
			clip_gradient(m_batch->dwv);
			sgd_optimizer(m_batch->wq, m_batch->dwq, LR);
			sgd_optimizer(m_batch->wk, m_batch->dwk, LR);
			sgd_optimizer(m_batch->wv, m_batch->dwv, LR);

			tensor_fill_zeros(m_batch->dwq);
			tensor_fill_zeros(m_batch->dwk);
			tensor_fill_zeros(m_batch->dwv);

			clip_gradient(L1->d_gamma);
			clip_gradient(L1->d_beta);
			clip_gradient(L2->d_gamma);
			clip_gradient(L2->d_beta);

			sgd_optimizer(L1->gamma, L1->d_gamma, LR);
			sgd_optimizer(L1->beta,  L1->d_beta,  LR);
			sgd_optimizer(L2->gamma, L2->d_gamma, LR);
			sgd_optimizer(L2->beta,  L2->d_beta,  LR);

			// Zero out gradients
			tensor_fill_zeros(L1->d_gamma);
			tensor_fill_zeros(L1->d_beta);
			tensor_fill_zeros(L2->d_gamma);
			tensor_fill_zeros(L2->d_beta);

			// Copy the chunks back to main tensor
			memcpy(batch_ptr, mha_backpass->_data, BATCH_SIZE * EMB_DIM * sizeof(float));
			//printf("batch tensor copied to the main\n");

			// free memory
			//tensor_free(batch_tensor);
			//tensor_free(attn_score);
			//tensor_free(ln1);
			//tensor_free(ffn_ln);
			//tensor_free(ln2);
			//printf("Running Epoch: %d, Batch: %d\n", e, b);
			if (b % 10 == 0) {
				printf("Loss: %f, after Epochs: %d\n", loss_to_show, e);
				fprintf(logf, "%d,%f\n", global_step, loss_to_show);
				global_step++;
			}
		}
	}
	clock_t end_time = clock();
	double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
	printf("Training finished in %.3f seconds!\n", elapsed);
	//tensor_shape(T);

	return 0;
}

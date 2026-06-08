/* Full autograd transformer training pipeline using MUON (stateful optimizer in tensor.h) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "./tensor.h"
#include "./attn.h"
#include "./ln.h"
#include "./ffn.h"
#include "./config.h"

static void optimize_muon(Tensor* w, _muon_state* st, float lr, float momentum) {
	if (!w || !w->grad || !st) return;
	Tensor grad = {0};
	grad._data = w->grad;
	grad.info = w->info;
	grad.ndim = w->ndim;
	clip_gradient(&grad);
	_muon_step(w, &grad, st, lr, momentum);
	memset(w->grad, 0, w->info.sz * sizeof(float));
}

int main(void) {
	srand((unsigned)time(NULL));

	size_t arena_size = 512 * 1024 * 1024;
	Arena* A = (Arena*)malloc(sizeof(Arena));
	arena_init(A, arena_size);
	printf("Arena initialized\n");

	int shape[2] = { SEQ_LEN, EMB_DIM };
	Tensor* T = tensor_create_new(A, 2, shape);
	tensor_randomize(T);

	Tensor* target = tensor_create_new(A, 2, shape);
	tensor_randomize(target);

	int num_chunks = SEQ_LEN / BATCH_SIZE;
	printf("num_chunks = %d\n", num_chunks);

	MHA* m_batch = mha_create_new(A, HEADS, BATCH_SIZE, EMB_DIM);
	mha_init_params(m_batch);
	m_batch->wq->requires_grad = true;
	m_batch->wk->requires_grad = true;
	m_batch->wv->requires_grad = true;

	LayerNorm* L1 = layer_norm_create_new(A, EMB_DIM);
	layer_norm_init_params(L1);
	L1->gamma->requires_grad = true;
	L1->beta->requires_grad = true;

	FFN* f = ffn_create(A, EMB_DIM, HIDDEN_DIM);
	ffn_init_params(f);
	f->w1->requires_grad = true;
	f->w2->requires_grad = true;

	LayerNorm* L2 = layer_norm_create_new(A, EMB_DIM);
	layer_norm_init_params(L2);
	L2->gamma->requires_grad = true;
	L2->beta->requires_grad = true;

	_muon_state muon_wq = _muon_state_create(A, m_batch->wq);
	_muon_state muon_wk = _muon_state_create(A, m_batch->wk);
	_muon_state muon_wv = _muon_state_create(A, m_batch->wv);
	_muon_state muon_w1 = _muon_state_create(A, f->w1);
	_muon_state muon_w2 = _muon_state_create(A, f->w2);
	_muon_state muon_g1 = _muon_state_create(A, L1->gamma);
	_muon_state muon_b1 = _muon_state_create(A, L1->beta);
	_muon_state muon_g2 = _muon_state_create(A, L2->gamma);
	_muon_state muon_b2 = _muon_state_create(A, L2->beta);

	const float momentum = 0.9f;

	FILE* logf = fopen("loss_muon.csv", "w");
	if (!logf) {
		printf("Failed to open loss_muon.csv\n");
		return 1;
	}
	fprintf(logf, "step,loss\n");

	printf("Starting MUON autograd training loop...\n");
	clock_t start_time = clock();
	int global_step = 0;

	for (int e = 1; e <= EPOCHS; e++) {
		printf("Epoch %d started\n", e);
		for (int b = 0; b < num_chunks; b++) {
			if (b == 0) printf("Processing batch %d/%d\n", b, num_chunks);

			float* batch_ptr = T->_data + b * BATCH_SIZE * EMB_DIM;
			float* target_ptr = target->_data + b * BATCH_SIZE * EMB_DIM;

			int shape_local[2] = { BATCH_SIZE, EMB_DIM };
			Tensor* batch_tensor = tensor_create_new(A, 2, shape_local);
			Tensor* target_batch = tensor_create_new(A, 2, shape_local);
			memcpy(batch_tensor->_data, batch_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));
			memcpy(target_batch->_data, target_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));

			Tensor* attn_score = mha_forward(A, batch_tensor, m_batch);
			tensor_check("attn_score_forward", attn_score);

			Tensor* ln1 = __ag_layer_norm_forward(A, L1, attn_score);
			tensor_check("ln1_forward", ln1);

			Tensor* ffn_ln = __ag_ffn_forward(A, ln1, f);
			tensor_check("ffn_ln_forward", ffn_ln);

			Tensor* ln2 = __ag_layer_norm_forward(A, L2, ffn_ln);
			tensor_check("ln2_forward", ln2);

			Tensor* loss = tensor_mse_loss(A, ln2, target_batch);
			float loss_to_show = loss_value(ln2, target_batch);

			tensor_backward(loss);
			printf("Backward complete, loss=%f\n", loss_to_show);

			optimize_muon(f->w1, &muon_w1, LR, momentum);
			optimize_muon(f->w2, &muon_w2, LR, momentum);

			optimize_muon(m_batch->wq, &muon_wq, LR, momentum);
			optimize_muon(m_batch->wk, &muon_wk, LR, momentum);
			optimize_muon(m_batch->wv, &muon_wv, LR, momentum);

			optimize_muon(L1->gamma, &muon_g1, LR, momentum);
			optimize_muon(L1->beta, &muon_b1, LR, momentum);
			optimize_muon(L2->gamma, &muon_g2, LR, momentum);
			optimize_muon(L2->beta, &muon_b2, LR, momentum);

			if (b % 10 == 0) {
				printf("Loss: %f, Epoch: %d\n", loss_to_show, e);
				fprintf(logf, "%d,%f\n", global_step, loss_to_show);
				global_step++;
			}
		}
	}

	clock_t end_time = clock();
	double elapsed = (double)(end_time - start_time) / (double)CLOCKS_PER_SEC;
	printf("MUON autograd training finished in %.3f seconds\n", elapsed);
	fclose(logf);
	return 0;
}

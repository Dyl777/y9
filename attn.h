#pragma once
#ifndef ATTENTION_H
#define ATTENTION_H
#include "tensor.h"
#include "arena.h"

typedef struct {
	int num_heads;
	int dk;

	// weights needed for backward pass
	Tensor *wq;
	Tensor *wk;
	Tensor *wv;
	Tensor *wo;

	// cached values, intrinsics to handle this for optimal performance
	Tensor *Q;
	Tensor *K;
	Tensor *V;
	Tensor *out;

	// gradients for optimizers
	Tensor *dwq;
	Tensor *dwk;
	Tensor *dwv;
	Tensor *dQ;
	Tensor *dK;
	Tensor *dV;
	
} MHA;

MHA *mha_create(int heads, int seq_len, int emb_dim);
MHA *mha_create_new(Arena *A, int heads, int seq_len, int emb_dim);
Tensor *scaled_dot_product_attention(Arena *A, Tensor *Q, Tensor *K, Tensor *V, int heads);
Tensor *mha_forward(Arena *A, Tensor *t, MHA *m);
Tensor *__nag_scaled_dot_product_attention(Arena *A, Tensor *Q, Tensor *K, Tensor *V, int dk);
Tensor *__nag_mha_forward(Arena *A, Tensor *t, MHA *mha);
Tensor *mha_backward(Arena *A, MHA *m, Tensor *t, Tensor *tokens);
void mha_init_params(MHA *m);

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX
#define EMB_DIM 32 // out model dimension, i.e Embedding size for each token
#define EPS 1e-5

// Since we already slice the heads from our LOSS matrix
// now we have dO_i.... dO..heads-1 for i = 0 ... num_heads
// Now let's use the chain rule identity
// we know that Y = A @ B the chain rule becomes 
// dA = dY @ B_transpose
// dB = A_transpose @ dY
//
// for our sliced heads from h0...h(num_heads)
// we have dO_i = heads defined above
// so we can do something like below
// output = attention_score @ V
// d_attention_score = output @ V_transpose
// d_V = attention_score_transpose @ output

// Do calculations for Attention_score
// TODO: Need to fix the architecture issue for thie Recalculation!

Tensor *softmax_gradient(Tensor *A, Tensor *dA) {

	int rows = A->info.shape[0];
	int cols = A->info.shape[1];
	int dS_shape[2] = {rows, cols};
	Tensor *dS = tensor_create(2, dS_shape);

	for (int i = 0; i < rows; i++) {
		float sum = 0.0f;
		for (int j = 0; j < cols; j++) {
			sum += dA->_data[i * cols +j] * A->_data[i * cols + j];
		}

		for (int j = 0; j < cols; j++) {
			dA->_data[i * cols +j] = dA->_data[i * cols + j] - sum;
		}
	}
	return dS;
}

void mha_init_params(MHA *m) {
	tensor_randomize_weights(m->wq);
	tensor_randomize_weights(m->wk);
	tensor_randomize_weights(m->wv);
	tensor_randomize_weights(m->wo);
}

void mha_backward_temp_weights(Arena *AA, Tensor *dO, Tensor *A, Tensor *B, Tensor **dA, Tensor **dV) {
	*dA = __nag_tensor_matmul(AA, dO, tensor_transpose(B));
	*dV = __nag_tensor_matmul(AA, tensor_transpose(A), dO);
	//tensor_shape(*(dA));
	//tensor_shape(*(dV));
}

Tensor *mha_backward(Arena *A, MHA *m, Tensor *dx, Tensor *tokens) {
	int ndim = 2;
	int heads = m->num_heads;
	int dk = m->dk;
	int shape[2] = {dx->info.shape[0], dk};
	int rows = dx->info.shape[0];
	int cols = dx->info.shape[1];

	// Resultant matrix needs to be returend summing all the heads
	Tensor *dX_total = tensor_create_new(A, 2, tokens->info.shape);
	int dw_shape[2] = {tokens->info.shape[1], tokens->info.shape[1]};

	m->dwq = tensor_create_new(A, 2, dw_shape);
	m->dwk = tensor_create_new(A, 2, dw_shape);
	m->dwv = tensor_create_new(A, 2, dw_shape);

	int dQ_shape[2] = {16, 32};
	m->dQ = tensor_create_new(A, 2, dQ_shape);
	m->dV = tensor_create_new(A, 2, dQ_shape);
	m->dK = tensor_create_new(A, 2, dQ_shape);

	for (int k = 0; k < heads; k++) {
		Tensor *head = tensor_create_weights_new(A, 2, shape);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				int src = i * cols + j + k * dk;
				int dest = i * dk + j;
				head->_data[dest] = dx->_data[src];
			}
		}

		// slicing the Q, K and V tensors for head 'k'
		Tensor *Qk = tensor_create_weights_new(A, ndim, shape);
		Tensor *Kk = tensor_create_weights_new(A, ndim, shape);
		Tensor *Vk = tensor_create_weights_new(A, ndim, shape);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				int src = i * cols + j + k * dk;
				int dest = i * dk + j;
				Qk->_data[dest] = m->Q->_data[src];
				Kk->_data[dest] = m->K->_data[src];
				Vk->_data[dest] = m->V->_data[src];
			}
		}
		float scale = 1.0f / sqrtf(dk);
		Tensor *Kt = tensor_transpose(Kk);
		Tensor *QKt = __nag_tensor_matmul(A, Qk, Kt);
		for (int i = 0; i < QKt->info.shape[0]; i++) {
			for (int j = 0; j < QKt->info.shape[1]; j++) {
				QKt->_data[i * QKt->info.shape[1] + j] *= scale;
			}
		}
		//Tensor *Ak = tensor_softmax(A, QKt);
		Tensor *Ak = __nag_tensor_softmax(A, QKt);

		
		int ashape[2] = {rows, rows};
		Tensor *dAk = tensor_create_weights_new(A, ndim, ashape);
		Tensor *dVk = tensor_create_weights_new(A, ndim, shape);

		mha_backward_temp_weights(A, head, Ak, Vk, &dAk, &dVk);

		// so we got the derivatives of Value matrix and Attetion score matrix
		// Now we need to produce grandients of weight matrices that produced 'V' i.e wv
		// Since V = X @ wv => dwv = X^T @ dV
		// PROBLEM LIES HERE!!! BUGGGG
		//m->dwv = tensor_matmul(A, tensor_transpose(tokens), dVk);
		Tensor *dSk = softmax_gradient(dAk, Ak);

		Tensor *dKk = __nag_tensor_matmul(A, tensor_transpose(dSk), Qk);
		Tensor *dQk = __nag_tensor_matmul(A, dSk, Kk);
		
		// Scaling dQk and dKk
		for (int i = 0; i < dQk->info.shape[0]; i++) {
			for (int j = 0; j < dQk->info.shape[1]; j++) {
				dQk->_data[i * dQk->info.shape[1] +j] *= scale;
			}
		}

		for (int i = 0; i < dKk->info.shape[0]; i++) {
			for (int j = 0; j < dKk->info.shape[1]; j++) {
				dKk->_data[i * dKk->info.shape[1] +j] *= scale;
			}
		}
		//
		// finding gradients for input 'X' weights
	  Tensor *dwq = __nag_tensor_matmul(A, tensor_transpose(tokens), dQk);
		Tensor *dwk = __nag_tensor_matmul(A, tensor_transpose(tokens), dKk);
		Tensor *dwv = __nag_tensor_matmul(A, tensor_transpose(tokens), dVk);


		// TODO: Accumulation step tensor_inplace_gradients
		int w_rows = tokens->info.shape[1];
		int local_cols = dwq->info.shape[1];
		//printf("local_cols: %d\n", local_cols);
		
		for (int i = 0; i < w_rows; i++) {
			for (int j = 0; j < local_cols; j++) {
				int dest = i * w_rows + j + k * local_cols; // for dw matrices rows and cols are same [x, x]
				int src = i * local_cols + j;
				m->dwq->_data[dest] += dwq->_data[src];
				m->dwk->_data[dest] += dwk->_data[src];
				m->dwv->_data[dest] += dwv->_data[src];
			}
		}


		int Q_rows = tokens->info.shape[0];
		int Q_local_cols = dQk->info.shape[1];

		for (int i = 0; i < Q_rows; i++) {
			for (int j = 0; j < Q_local_cols; j++) {
				int dest = i * 32 + j + k * Q_local_cols;
				int src = i * Q_local_cols + j;

				m->dQ->_data[dest] += dQk->_data[src];
				m->dK->_data[dest] += dKk->_data[src];
				m->dV->_data[dest] += dVk->_data[src];

			}
		}
	}

	//tensor_shape(m->dwq);
	//tensor_shape(m->dwk);
	//tensor_shape(m->dwv);

	//printf("dQ shape: \n");
	//tensor_shape(m->dQ);
	//printf("dK shape: \n");
	//tensor_shape(m->dK);
	//printf("dV shape: \n");
	//tensor_shape(m->dV);
	
	Tensor *dx1 = __nag_tensor_matmul(A, m->dQ, tensor_transpose(m->wq));
	Tensor *dx2 = __nag_tensor_matmul(A, m->dK, tensor_transpose(m->wk));
	Tensor *dx3 = __nag_tensor_matmul(A, m->dV, tensor_transpose(m->wv));

	//printf("dx1 shape: \n");
	//tensor_shape(dx1);
	//printf("dx2 shape: \n");
	//tensor_shape(dx2);
	//printf("dx3 shape: \n");
	//tensor_shape(dx3);

	Tensor *temp = tensor_add(dx1, dx2);
	temp = tensor_add(temp, dx3);

	tensor_add_inplace(&dX_total, &temp);
	return dX_total;
}


Tensor *mha_forward(Arena *A, Tensor *t, MHA *mha) {
	// Debug: Check shapes before matmul
	// fprintf(stderr, "DEBUG mha_forward: t->shape={%d, %d}, wq->shape={%d, %d}, wk->shape={%d, %d}, wv->shape={%d, %d}\n",
	// 	t->info.shape[0], t->info.shape[1],
	// 	mha->wq->info.shape[0], mha->wq->info.shape[1],
	// 	mha->wk->info.shape[0], mha->wk->info.shape[1],
	// 	mha->wv->info.shape[0], mha->wv->info.shape[1]);

	// mha->Q = tensor_matmul(A, t, mha->wq);
	// mha->K = tensor_matmul(A, t, mha->wk);
	// mha->V = tensor_matmul(A, t, mha->wv);

	//------------------AUTOGRAD SUPPORTED---------------
	// fprintf(stderr, "DEBUG before Q matmul: t=%p shape={%d,%d}, wq=%p shape={%d,%d}\n",
	// 	(void*)t, t->info.shape[0], t->info.shape[1],
	// 	(void*)mha->wq, mha->wq->info.shape[0], mha->wq->info.shape[1]);
	mha->Q = __ag_tensor_matmul_forward(A, t, mha->wq);
	// fprintf(stderr, "DEBUG before K matmul: t=%p shape={%d,%d}, wk=%p shape={%d,%d}\n",
	// 	(void*)t, t->info.shape[0], t->info.shape[1],
	// 	(void*)mha->wk, mha->wk->info.shape[0], mha->wk->info.shape[1]);
	mha->K = __ag_tensor_matmul_forward(A, t, mha->wk);
	// fprintf(stderr, "DEBUG before V matmul: t=%p shape={%d,%d}, wv=%p shape={%d,%d}\n",
	// 	(void*)t, t->info.shape[0], t->info.shape[1],
	// 	(void*)mha->wv, mha->wv->info.shape[0], mha->wv->info.shape[1]);
	mha->V = __ag_tensor_matmul_forward(A, t, mha->wv);
	//---------------------------------------

	// extract the required parameters
	int rows = t->info.shape[0];
	int cols = t->info.shape[1];
	int heads = mha->num_heads;
	int dk = mha->dk;


	mha->out = tensor_create_new(A, 2, t->info.shape);
	int common_shape[2] = {rows, dk};

	for (int k = 0; k < heads; k++) {
		// Autograd-enabled slicing for Q, K, V heads
		Tensor *Q_h = __ag_tensor_slice_forward(A, mha->Q, 0, k * dk, rows, dk);
		Tensor *K_h = __ag_tensor_slice_forward(A, mha->K, 0, k * dk, rows, dk);
		Tensor *V_h = __ag_tensor_slice_forward(A, mha->V, 0, k * dk, rows, dk);

		Tensor *head_out = scaled_dot_product_attention(A, Q_h, K_h, V_h, dk);
		
		// Autograd-enabled scatter to output
		mha->out = __ag_tensor_scatter_forward(A, mha->out, head_out, 0, k * dk);
	}

	return mha->out;
}	

Tensor *__nag_mha_forward(Arena *A, Tensor *t, MHA *mha) {
	mha->Q = __nag_tensor_matmul(A, t, mha->wq);
	mha->K = __nag_tensor_matmul(A, t, mha->wk);
	mha->V = __nag_tensor_matmul(A, t, mha->wv);

	int rows = t->info.shape[0];
	int cols = t->info.shape[1];
	int heads = mha->num_heads;
	int dk = mha->dk;

	mha->out = tensor_create_new(A, 2, t->info.shape);
	for (int k = 0; k < heads; k++) {
		int shape_local[2] = {rows, dk};
		Tensor *Q_h = tensor_create_new(A, 2, shape_local);
		Tensor *K_h = tensor_create_new(A, 2, shape_local);
		Tensor *V_h = tensor_create_new(A, 2, shape_local);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				int src = i * cols + j + k * dk;
				int dest = i * dk + j;
				Q_h->_data[dest] = mha->Q->_data[src];
				K_h->_data[dest] = mha->K->_data[src];
				V_h->_data[dest] = mha->V->_data[src];
			}
		}

		Tensor *head_out = __nag_scaled_dot_product_attention(A, Q_h, K_h, V_h, dk);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				int src = i * dk + j;
				int dest = i * cols + j + k * dk;
				mha->out->_data[dest] = head_out->_data[src];
			}
		}
	}

	return mha->out;
}

Tensor *__nag_scaled_dot_product_attention(Arena *A, Tensor *Q, Tensor *K, Tensor *V, int dk) {
	Tensor *kt = tensor_transpose(K);
	Tensor *qkt = __nag_tensor_matmul(A, Q, kt);
	for (int i = 0; i < qkt->info.shape[0]; i++) {
		for (int j = 0; j < qkt->info.shape[1]; j++) {
			qkt->_data[i * qkt->info.shape[1] + j] = qkt->_data[i * qkt->info.shape[1] + j] / sqrtf(dk);
		}
	}
	Tensor *qkt_soft = __nag_tensor_softmax(A, qkt);
	Tensor *ret = __nag_tensor_matmul(A, qkt_soft, V);
	return ret;
}


Tensor *scaled_dot_product_attention(Arena *A, Tensor *Q, Tensor *K, Tensor *V, int dk) {
	// fprintf(stderr, "DEBUG sdpa: Q=%p shape={%d,%d}, K=%p shape={%d,%d}, V=%p shape={%d,%d}, dk=%d\n",
	// 	(void*)Q, Q->info.shape[0], Q->info.shape[1],
	// 	(void*)K, K->info.shape[0], K->info.shape[1],
	// 	(void*)V, V->info.shape[0], V->info.shape[1], dk);
	Tensor *kt = __ag_tensor_transpose_forward(A, K);
	// fprintf(stderr, "DEBUG sdpa: kt shape={%d,%d}\n", kt->info.shape[0], kt->info.shape[1]);
	//Tensor *qkt = tensor_matmul(A, Q, kt);
	Tensor *qkt = __ag_tensor_matmul_forward(A, Q, kt);
	// fprintf(stderr, "DEBUG sdpa: qkt shape={%d,%d}\n", qkt->info.shape[0], qkt->info.shape[1]);
	for (int i = 0; i < qkt->info.shape[0]; i++) {
		for (int j = 0; j < qkt->info.shape[1]; j++) {
			qkt->_data[i * qkt->info.shape[1] + j] = qkt->_data[i * qkt->info.shape[1] +j] / sqrtf(dk);
		}
	}
	//Tensor *qkt_soft = tensor_softmax(A, qkt); // RAND_FLOAT is random we'll calculate this later
	Tensor *qkt_soft = __ag_tensor_softmax_forward(A, qkt);
	//Tensor *ret = tensor_matmul(A, qkt_soft, V);
	Tensor *ret = __ag_tensor_matmul_forward(A, qkt_soft, V);
	return ret;
}

MHA *mha_create_new(Arena *A, int num_heads, int seq_len, int emb_dim) {
	MHA *mha = (MHA*) arena_alloc(A, sizeof(MHA));
	int ndim = 2;
	int *shape_weights = (int*) arena_alloc(A, ndim * sizeof(int));
	int *shape_tokens = (int*) arena_alloc(A, ndim * sizeof(int));

	shape_weights[0] = emb_dim;
	shape_weights[1] = emb_dim;

	shape_tokens[0] = seq_len;
	shape_tokens[1] = emb_dim;

	mha->wq = tensor_create_weights_new(A, ndim, shape_weights);
	mha->wk = tensor_create_weights_new(A, ndim, shape_weights);
	mha->wv = tensor_create_weights_new(A, ndim, shape_weights);
	mha->wo = tensor_create_weights_new(A, ndim, shape_weights); // output weights
	
	// define the tensor
	mha->Q = tensor_create_weights_new(A, ndim, shape_tokens);
	mha->K = tensor_create_weights_new(A, ndim, shape_tokens);
	mha->V = tensor_create_weights_new(A, ndim, shape_tokens);
	mha->out = tensor_create_weights_new(A, ndim, shape_tokens);

	mha->num_heads= num_heads;
	mha->dk = emb_dim / mha->num_heads;

	// Initialize gradient fields to NULL
	mha->dwq = NULL;
	mha->dwk = NULL;
	mha->dwv = NULL;
	mha->dQ = NULL;
	mha->dK = NULL;
	mha->dV = NULL;

	return mha;
}

MHA *mha_create(int num_heads, int seq_len, int emb_dim) {
	MHA *mha = (MHA*) malloc(sizeof(MHA));
	int ndim = 2;
	int shape_weights[2] = {emb_dim, emb_dim};
	int shape_tokens[2] = {seq_len, emb_dim};
	mha->wq = tensor_create_weights(ndim, shape_weights);
	mha->wk = tensor_create_weights(ndim, shape_weights);
	mha->wv = tensor_create_weights(ndim, shape_weights);
	mha->wo = tensor_create_weights(ndim, shape_weights); // output weights
	
	// define the tensor
	mha->Q = tensor_create(ndim, shape_tokens);
	mha->K = tensor_create(ndim, shape_tokens);
	mha->V = tensor_create(ndim, shape_tokens);
	mha->out = tensor_create(ndim, shape_tokens);
	mha->num_heads= num_heads;
	mha->dk = emb_dim / mha->num_heads;
	return mha;
}

	

//int main() {
//	//int seed = 32;
//	//srand(seed);
//	int ndim = 2;
//
//	int shape_tokens[2] = {SEQ_LEN, EMB_DIM};
//	int shape_weights[2] = {EMB_DIM, EMB_DIM};
//
//	Tensor *tokens = tensor_create(ndim, shape_tokens);
//	
//	int heads = 8;
//	int HEAD_DIM = EMB_DIM / heads;
//	MHA *mha = mha_create(heads, SEQ_LEN, EMB_DIM);
//	printf("MHA created\n");
//	Tensor *multi_head = mha_forward(tokens, mha);
//	tensor_shape(multi_head);
//	tensor_get(multi_head);
//
//	return 0;
//}

#endif
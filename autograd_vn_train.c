/* Full autograd training loop example (copied from vn_train_main.c)
 * This standalone test reproduces the full training example using
 * autograd-capable ops
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "tensor.h"
#include "attn.h"
#include "ln.h"
#include "ffn.h"
#include "config.h"
#include "vn_transf_mdl.h"

typedef struct progress_config {
	int show_progress_bar;
	int progress_bar_width;
	int show_accuracy;
} progress_config;

static void print_epoch_progress(const progress_config* cfg, int epoch0, int epochs, float loss, float accuracy, long epoch_duration_ms) {
	if (!cfg) return;
	if (!cfg->show_progress_bar) {
		printf("Epoch %d/%d - Loss: %.6f", epoch0 + 1, epochs, (double)loss);
		if (cfg->show_accuracy) printf(" - Accuracy: %.2f%%", (double)(accuracy * 100.0f));
		printf("\n");
		return;
	}

	int width = (cfg->progress_bar_width > 0) ? cfg->progress_bar_width : 30;
	int progress = (int)(((double)(epoch0 + 1) / (double)epochs) * (double)width);
	if (progress < 0) progress = 0;
	if (progress > width) progress = width;

	printf("\rEpoch %4d/%d [", epoch0 + 1, epochs);
	for (int i = 0; i < width; ++i) {
		if (i < progress) {
			putchar('█');
		} else if (i == progress) {
			putchar('▌');
		} else {
			putchar(' ');
		}
	}
	printf("] ");

	double percent = ((double)(epoch0 + 1) / (double)epochs) * 100.0;
	printf("%5.1f%% ", percent);
	printf("Loss: %.6f", (double)loss);
	if (cfg->show_accuracy) printf(" Acc: %.2f%%", (double)(accuracy * 100.0f));
	printf(" (%ldms/epoch)", epoch_duration_ms);

	if (epoch0 < epochs - 1) {
		int remaining_epochs = epochs - epoch0 - 1;
		long eta_ms = epoch_duration_ms * (long)remaining_epochs;
		if (eta_ms > 1000) {
			printf(" ETA: %lds", eta_ms / 1000);
		} else {
			printf(" ETA: %ldms", eta_ms);
		}
	}

	fflush(stdout);
}

int main() {
    fprintf(stderr, "DEBUG: Entering autograd_vn_train test\n");
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

    int shape_target[2] = {SEQ_LEN, EMB_DIM};
    printf("Creating target tensor...\n");
    Tensor *target = tensor_create_new(A, ndim, shape_target);
    printf("Target tensor created\n");
    tensor_randomize(target);
    printf("Target randomized\n");

    int num_chunks = SEQ_LEN / BATCH_SIZE;
    printf("num_chunks = %d (SEQ_LEN=%d / BATCH_SIZE=%d)\n", num_chunks, SEQ_LEN, BATCH_SIZE);

    printf("Creating MHA...\n");
    fflush(stdout);
    MHA *m_batch = mha_create_new(A, HEADS, BATCH_SIZE, EMB_DIM);
    printf("MHA created at %p\n", (void*)m_batch);
    fflush(stdout);

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

    FILE *logf = fopen("loss.csv", "w");
    if (!logf) {
        printf("Failed to open loss.csv!\n");
        return 1;
    }
    fprintf(logf, "step,loss\n");
    printf("Starting training loop...\n");
    clock_t start_time = clock();
    int global_step = 0;

	progress_config progress_cfg;
	progress_cfg.show_progress_bar = 1;
	progress_cfg.progress_bar_width = 30;
	progress_cfg.show_accuracy = 0;

    for (int e = 1; e <= EPOCHS; e++) {
        printf("Epoch %d started\n", e);
		clock_t epoch_start = clock();
        for (int b = 0; b < num_chunks; b++) {
            if (b == 0) printf("Processing batch %d/%d\n", b, num_chunks);

            float *batch_ptr = T->_data + b * BATCH_SIZE * EMB_DIM;
            float *target_ptr = target->_data + b * BATCH_SIZE * EMB_DIM;

            int shape_local[2] = {BATCH_SIZE, EMB_DIM};
            Tensor *batch_tensor = tensor_create_new(A, 2, shape_local);
            Tensor *target_batch = tensor_create_new(A, 2, shape_local);
            memcpy(batch_tensor->_data, batch_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));
            memcpy(target_batch->_data, target_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));

            // Forward (autograd-enabled ops)
            Tensor *attn_score = mha_forward(A, batch_tensor, m_batch);
            clip_gradient(attn_score);
            tensor_check("attn_score_forward", attn_score);

            Tensor *ln1 = __ag_layer_norm_forward(A, L1, attn_score);
            tensor_check("ln1_forward", ln1);

            Tensor *ffn_ln = __ag_ffn_forward(A, ln1, f);
            tensor_check("ffn_ln_forward", ffn_ln);

            Tensor *ln2 = __ag_layer_norm_forward(A, L2, ffn_ln);
            tensor_check("ln2_forward", ln2);

            Tensor *loss = tensor_mse_loss(A, ln2, target_batch);
            float loss_to_show = loss_value(ln2, target_batch);

            // AUTOMATIC BACKWARD - traverses computation graph and computes all gradients
            tensor_backward(loss);
            printf("Backward complete, loss=%f\n", loss_to_show);
            if (b == 0 || b % 100 == 0) {
                char dag_name[64];
                snprintf(dag_name, sizeof(dag_name), "loss (epoch=%d,batch=%d)", e, b);
                tensor_print_dag(loss, dag_name);
            }

            // Clip gradients to prevent exploding
            if(f->w1->grad) clip_gradient(f->w1);
            if(f->w2->grad) clip_gradient(f->w2);
            if(f->a1->grad) clip_gradient(f->a1);
            if(f->h1->grad) clip_gradient(f->h1);
            if(m_batch->wq->grad) clip_gradient(m_batch->wq);
            if(m_batch->wk->grad) clip_gradient(m_batch->wk);
            if(m_batch->wv->grad) clip_gradient(m_batch->wv);
            if(L1->gamma->grad) clip_gradient(L1->gamma);
            if(L1->beta->grad) clip_gradient(L1->beta);
            if(L2->gamma->grad) clip_gradient(L2->gamma);
            if(L2->beta->grad) clip_gradient(L2->beta);

            // MANUAL BACKWARD (alternative - kept for reference/debugging):
            // Tensor *dx_for_ffn = tensor_create_new(A, 2, shape_local);

            // Optimizer steps using AUTOGRAD gradients (stored in weight->grad)
            // Create temp tensors wrapping the grad pointers for sgd_optimizer
            Tensor grad_w1 = {0}, grad_w2 = {0}, grad_a1 = {0}, grad_h1 = {0};
            grad_w1._data = f->w1->grad; grad_w1.info = f->w1->info; grad_w1.ndim = f->w1->ndim;
            grad_w2._data = f->w2->grad; grad_w2.info = f->w2->info; grad_w2.ndim = f->w2->ndim;
            grad_a1._data = f->a1->grad; grad_a1.info = f->a1->info; grad_a1.ndim = f->a1->ndim;
            grad_h1._data = f->h1->grad; grad_h1.info = f->h1->info; grad_h1.ndim = f->h1->ndim;
            
            if (grad_w1._data) { clip_gradient(&grad_w1); sgd_optimizer(f->w1, &grad_w1, LR); }
            if (grad_w2._data) { clip_gradient(&grad_w2); sgd_optimizer(f->w2, &grad_w2, LR); }
            if (grad_a1._data) { clip_gradient(&grad_a1); sgd_optimizer(f->a1, &grad_a1, LR); }
            if (grad_h1._data) { clip_gradient(&grad_h1); sgd_optimizer(f->h1, &grad_h1, LR); }

            // Zero autograd gradients after optimizer step
            if(f->w1->grad) memset(f->w1->grad, 0, f->w1->info.sz * sizeof(float));
            if(f->w2->grad) memset(f->w2->grad, 0, f->w2->info.sz * sizeof(float));
            if(f->a1->grad) memset(f->a1->grad, 0, f->a1->info.sz * sizeof(float));
            if(f->h1->grad) memset(f->h1->grad, 0, f->h1->info.sz * sizeof(float));

            // MHA weights autograd optimizer
            Tensor grad_wq = {0}, grad_wk = {0}, grad_wv = {0};
            grad_wq._data = m_batch->wq->grad; grad_wq.info = m_batch->wq->info; grad_wq.ndim = m_batch->wq->ndim;
            grad_wk._data = m_batch->wk->grad; grad_wk.info = m_batch->wk->info; grad_wk.ndim = m_batch->wk->ndim;
            grad_wv._data = m_batch->wv->grad; grad_wv.info = m_batch->wv->info; grad_wv.ndim = m_batch->wv->ndim;
            
            if (grad_wq._data) { clip_gradient(&grad_wq); sgd_optimizer(m_batch->wq, &grad_wq, LR); }
            if (grad_wk._data) { clip_gradient(&grad_wk); sgd_optimizer(m_batch->wk, &grad_wk, LR); }
            if (grad_wv._data) { clip_gradient(&grad_wv); sgd_optimizer(m_batch->wv, &grad_wv, LR); }

            if(m_batch->wq->grad) memset(m_batch->wq->grad, 0, m_batch->wq->info.sz * sizeof(float));
            if(m_batch->wk->grad) memset(m_batch->wk->grad, 0, m_batch->wk->info.sz * sizeof(float));
            if(m_batch->wv->grad) memset(m_batch->wv->grad, 0, m_batch->wv->info.sz * sizeof(float));

            // LayerNorm autograd optimizer
            Tensor grad_gamma1 = {0}, grad_beta1 = {0}, grad_gamma2 = {0}, grad_beta2 = {0};
            grad_gamma1._data = L1->gamma->grad; grad_gamma1.info = L1->gamma->info; grad_gamma1.ndim = L1->gamma->ndim;
            grad_beta1._data = L1->beta->grad; grad_beta1.info = L1->beta->info; grad_beta1.ndim = L1->beta->ndim;
            grad_gamma2._data = L2->gamma->grad; grad_gamma2.info = L2->gamma->info; grad_gamma2.ndim = L2->gamma->ndim;
            grad_beta2._data = L2->beta->grad; grad_beta2.info = L2->beta->info; grad_beta2.ndim = L2->beta->ndim;
            
            if (grad_gamma1._data) { clip_gradient(&grad_gamma1); sgd_optimizer(L1->gamma, &grad_gamma1, LR); }
            if (grad_beta1._data)  { clip_gradient(&grad_beta1);  sgd_optimizer(L1->beta,  &grad_beta1,  LR); }
            if (grad_gamma2._data) { clip_gradient(&grad_gamma2); sgd_optimizer(L2->gamma, &grad_gamma2, LR); }
            if (grad_beta2._data)  { clip_gradient(&grad_beta2);  sgd_optimizer(L2->beta,  &grad_beta2,  LR); }

            if(L1->gamma->grad) memset(L1->gamma->grad, 0, L1->gamma->info.sz * sizeof(float));
            if(L1->beta->grad) memset(L1->beta->grad, 0, L1->beta->info.sz * sizeof(float));
            if(L2->gamma->grad) memset(L2->gamma->grad, 0, L2->gamma->info.sz * sizeof(float));
            if(L2->beta->grad) memset(L2->beta->grad, 0, L2->beta->info.sz * sizeof(float));

            if (b % 10 == 0) {
                printf("Loss: %f, after Epochs: %d\n", loss_to_show, e);
                fprintf(logf, "%d,%f\n", global_step, loss_to_show);
                global_step++;
            }

			if (b == num_chunks - 1) {
				clock_t epoch_end = clock();
				long epoch_duration_ms = (long)(((double)(epoch_end - epoch_start) * 1000.0) / (double)CLOCKS_PER_SEC);
				print_epoch_progress(&progress_cfg, e - 1, EPOCHS, loss_to_show, 0.0f, epoch_duration_ms);
				if (e == EPOCHS) printf("\n");
			}

            //IMPLEMENT THIS PROGRESSBAR THING ONLY INTO OUR CODE STYLE IN C, DON'T DELETE ANYTHING
            //IN THE CODE
            /*if (config.show_progress_bar) {
                int progress = static_cast<int>((static_cast<double>(epoch + 1) / epochs) * config.progress_bar_width);
                std::cout << "\rEpoch " << std::setw(4) << (epoch + 1) << "/" << epochs << " [";
                for (int i = 0; i < config.progress_bar_width; ++i) {
                    if (i < progress) {
                        std::cout << "█";
                    } else if (i == progress) {
                        std::cout << "▌";
                    } else {
                        std::cout << " ";
                    }
                }
                std::cout << "] ";
                
                // Progress percentage
                double percent = (static_cast<double>(epoch + 1) / epochs) * 100.0;
                std::cout << std::fixed << std::setprecision(1) << percent << "% ";
                
                // Loss
                std::cout << "Loss: " << std::fixed << std::setprecision(6) << loss;
                
                // Accuracy if requested
                if (config.show_accuracy) {
                    std::cout << " Acc: " << std::fixed << std::setprecision(2) << (accuracy * 100) << "%";
                }
                
                // Timing
                std::cout << " (" << epoch_duration.count() << "ms/epoch)";
                
                // ETA
                if (epoch < epochs - 1) {
                    int remaining_epochs = epochs - epoch - 1;
                    auto eta_ms = epoch_duration.count() * remaining_epochs;
                    if (eta_ms > 1000) {
                        std::cout << " ETA: " << (eta_ms / 1000) << "s";
                    } else {
                        std::cout << " ETA: " << eta_ms << "ms";
                    }
                }
                
                std::cout << std::flush;
            } else {
                // Simple line output without progress bar
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                         << " - Loss: " << std::fixed << std::setprecision(6) << loss;
                if (config.show_accuracy) {
                    std::cout << " - Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100) << "%";
                }
                std::cout << std::endl;
            }
        }
    }
    */
        }
    }
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training finished in %.3f seconds!\n", elapsed);

    /*IMPLEMENT ADAM, MUON, NEWTON-RAPHSON, HE/XAVIER INITIALISATION*/

    return 0;
}

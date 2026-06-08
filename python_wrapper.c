/* C wrapper for Python ctypes interface */
#include <stdlib.h>
#include <string.h>

/* Include all headers in a single translation unit to avoid multiple definition issues */
#include "config.h"
#include "arena.h"
#include "tensor.h"
#include "conv2d.h"
#include "pool.h"
#include "ln.h"
#include "ffn.h"
#include "attn.h"

/* Include the .c files that have non-inline implementations */
#include "conv2d.c"
#include "threadpool.c"

static Arena* g_arena = NULL;
static const size_t WRAPPER_ARENA_SIZE = 1024 * 1024 * 1024; // 1GB

/* Arena management */
void py_arena_init() {
    if (!g_arena) {
        g_arena = (Arena*)malloc(sizeof(Arena));
        arena_init(g_arena, WRAPPER_ARENA_SIZE);
    }
}

void py_arena_cleanup() {
    if (g_arena) {
        if (g_arena->base) {
            free(g_arena->base);
        }
        free(g_arena);
        g_arena = NULL;
    }
}

Arena* py_get_arena() {
    if (!g_arena) {
        py_arena_init();
    }
    return g_arena;
}

/* Tensor operations */
Tensor* py_tensor_create(int* shape, int ndim, int requires_grad) {
    Arena* A = py_get_arena();
    Tensor* t = tensor_create_new(A, ndim, shape);
    t->requires_grad = requires_grad;
    return t;
}

Tensor* py_tensor_zeros(int* shape, int ndim) {
    Arena* A = py_get_arena();
    Tensor* t = tensor_create_new(A, ndim, shape);
    tensor_fill_zeros(t);
    return t;
}

Tensor* py_tensor_randn(int* shape, int ndim) {
    Arena* A = py_get_arena();
    Tensor* t = tensor_create_weights_new(A, ndim, shape);
    return t;
}

void py_tensor_set_data(Tensor* t, float* data, int size) {
    if (t && t->_data) {
        memcpy(t->_data, data, size * sizeof(float));
    }
}

int py_tensor_ndim(Tensor* t) {
    return t ? t->ndim : 0;
}

int py_tensor_numel(Tensor* t) {
    return t ? t->info.sz : 0;
}

int* py_tensor_shape(Tensor* t) {
    return t ? t->info.shape : NULL;
}

float* py_tensor_data(Tensor* t) {
    return t ? t->_data : NULL;
}

int py_tensor_requires_grad(Tensor* t) {
    return t ? t->requires_grad : 0;
}

void py_tensor_set_requires_grad(Tensor* t, int req) {
    if (t) {
        t->requires_grad = req;
    }
}

float* py_tensor_grad(Tensor* t) {
    return t ? t->grad : NULL;
}

void py_tensor_backward(Tensor* t) {
    if (t) {
        tensor_backward(t);
    }
}

void py_tensor_zero_grad(Tensor* t) {
    if (t && t->grad) {
        memset(t->grad, 0, t->info.sz * sizeof(float));
    }
}

/* Tensor operations */
Tensor* py_tensor_matmul(Tensor* a, Tensor* b) {
    Arena* A = py_get_arena();
    return __ag_tensor_matmul_forward(A, a, b);
}

Tensor* py_tensor_add(Tensor* a, Tensor* b) {
    Arena* A = py_get_arena();
    int ndim = a->ndim;
    int* shape = (int*)arena_alloc(A, ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = a->info.shape[i];
    }
    Tensor* result = tensor_create_new(A, ndim, shape);
    for (int i = 0; i < result->info.sz; i++) {
        result->_data[i] = a->_data[i] + b->_data[i];
    }
    return result;
}

Tensor* py_tensor_sub(Tensor* a, Tensor* b) {
    Arena* A = py_get_arena();
    int ndim = a->ndim;
    int* shape = (int*)arena_alloc(A, ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = a->info.shape[i];
    }
    Tensor* result = tensor_create_new(A, ndim, shape);
    for (int i = 0; i < result->info.sz; i++) {
        result->_data[i] = a->_data[i] - b->_data[i];
    }
    return result;
}

Tensor* py_tensor_mul(Tensor* a, Tensor* b) {
    Arena* A = py_get_arena();
    int ndim = a->ndim;
    int* shape = (int*)arena_alloc(A, ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = a->info.shape[i];
    }
    Tensor* result = tensor_create_new(A, ndim, shape);
    for (int i = 0; i < result->info.sz; i++) {
        result->_data[i] = a->_data[i] * b->_data[i];
    }
    return result;
}

Tensor* py_tensor_relu(Tensor* x) {
    return __ag_relu_forward(x);
}

Tensor* py_tensor_mse_loss(Tensor* pred, Tensor* target) {
    Arena* A = py_get_arena();
    return tensor_mse_loss(A, pred, target);
}

/* Conv2D */
Conv2D* py_conv2d_create(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    Arena* A = py_get_arena();
    return conv2d_create(A, in_channels, out_channels, kernel_size, stride, padding);
}

Tensor* py_conv2d_forward(Conv2D* conv, Tensor* input) {
    Arena* A = py_get_arena();
    return __ag_conv2d_forward(A, conv, input);
}

/* Pool2D */
Pool2D* py_maxpool2d_create(int kernel_size, int stride) {
    Arena* A = py_get_arena();
    return pool2d_create(A, kernel_size, stride, MaxPool2D);
}

Pool2D* py_avgpool2d_create(int kernel_size, int stride) {
    Arena* A = py_get_arena();
    return pool2d_create(A, kernel_size, stride, AvgPool2D);
}

Tensor* py_pool2d_forward(Pool2D* pool, Tensor* input) {
    Arena* A = py_get_arena();
    return __ag_pool2d_forward(A, pool, input);
}

/* LayerNorm */
LayerNorm* py_layernorm_create(int normalized_shape) {
    Arena* A = py_get_arena();
    LayerNorm* ln = layer_norm_create_new(A, normalized_shape);
    layer_norm_init_params(ln);
    return ln;
}

Tensor* py_layernorm_forward(LayerNorm* ln, Tensor* input) {
    Arena* A = py_get_arena();
    return __ag_layer_norm_forward(A, ln, input);
}

/* FFN */
FFN* py_ffn_create(int input_dim, int hidden_dim) {
    Arena* A = py_get_arena();
    FFN* f = ffn_create(A, input_dim, hidden_dim);
    ffn_init_params(f);
    return f;
}

Tensor* py_ffn_forward(FFN* ffn, Tensor* input) {
    Arena* A = py_get_arena();
    return __ag_ffn_forward(A, input, ffn);
}

/* MHA */
MHA* py_mha_create(int num_heads, int seq_len, int emb_dim) {
    Arena* A = py_get_arena();
    MHA* mha = mha_create_new(A, num_heads, seq_len, emb_dim);
    mha_init_params(mha);
    return mha;
}

Tensor* py_mha_forward(MHA* mha, Tensor* input) {
    Arena* A = py_get_arena();
    return mha_forward(A, input, mha);
}

/* Optimizer */
void py_sgd_step(Tensor* param, float lr) {
    if (param && param->grad) {
        Tensor grad = {0};
        grad._data = param->grad;
        grad.info = param->info;
        grad.ndim = param->ndim;
        clip_gradient(&grad);
        sgd_optimizer(param, &grad, lr);
        memset(param->grad, 0, param->info.sz * sizeof(float));
    }
}

void py_clip_gradient(Tensor* t) {
    if (t) {
        clip_gradient(t);
    }
}

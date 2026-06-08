#include "conv2d.h"
#include "threadpool.h"
#include <math.h>
#include <string.h>

typedef struct {
	const float* data_im;
	float* data_col;
	int channels;
	int padded_h;
	int padded_w;
	int ksize;
	int stride;
	int output_h;
	int output_w;
	int col_cols;
} conv2d_im2col_ctx;

static void conv2d_im2col_channel_task(int c, void* raw) {
	conv2d_im2col_ctx* ctx = (conv2d_im2col_ctx*)raw;
	for (int kh = 0; kh < ctx->ksize; kh++) {
		for (int kw = 0; kw < ctx->ksize; kw++) {
			int row = c * ctx->ksize * ctx->ksize + kh * ctx->ksize + kw;
			for (int oh = 0; oh < ctx->output_h; oh++) {
				for (int ow = 0; ow < ctx->output_w; ow++) {
					int ih = oh * ctx->stride + kh;
					int iw = ow * ctx->stride + kw;
					int im_index = c * (ctx->padded_h * ctx->padded_w) + ih * ctx->padded_w + iw;
					int col_index = row * ctx->col_cols + oh * ctx->output_w + ow;
					ctx->data_col[col_index] = ctx->data_im[im_index];
				}
			}
		}
	}
}

void conv2d_im2col(const float* data_im, int channels, int padded_h, int padded_w,
                   int ksize, int stride, float* data_col) {
	int output_h = (padded_h - ksize) / stride + 1;
	int output_w = (padded_w - ksize) / stride + 1;
	int col_cols = output_h * output_w;

	conv2d_im2col_ctx ctx;
	ctx.data_im = data_im;
	ctx.data_col = data_col;
	ctx.channels = channels;
	ctx.padded_h = padded_h;
	ctx.padded_w = padded_w;
	ctx.ksize = ksize;
	ctx.stride = stride;
	ctx.output_h = output_h;
	ctx.output_w = output_w;
	ctx.col_cols = col_cols;

	int num_threads = threadpool_default_threads();
	if (num_threads > channels) num_threads = channels;
	if (num_threads < 1) num_threads = 1;
	parallel_for(0, channels, num_threads, conv2d_im2col_channel_task, &ctx);
}

typedef struct {
	const float* data_col;
	float* data_im;
	int channels;
	int padded_h;
	int padded_w;
	int ksize;
	int stride;
	int output_h;
	int output_w;
	int col_cols;
} conv2d_col2im_ctx;

static void conv2d_col2im_channel_task(int c, void* raw) {
	conv2d_col2im_ctx* ctx = (conv2d_col2im_ctx*)raw;
	for (int kh = 0; kh < ctx->ksize; kh++) {
		for (int kw = 0; kw < ctx->ksize; kw++) {
			int row = c * ctx->ksize * ctx->ksize + kh * ctx->ksize + kw;
			for (int oh = 0; oh < ctx->output_h; oh++) {
				for (int ow = 0; ow < ctx->output_w; ow++) {
					int ih = oh * ctx->stride + kh;
					int iw = ow * ctx->stride + kw;
					int col_index = row * ctx->col_cols + oh * ctx->output_w + ow;
					int im_index = c * (ctx->padded_h * ctx->padded_w) + ih * ctx->padded_w + iw;
					ctx->data_im[im_index] += ctx->data_col[col_index];
				}
			}
		}
	}
}

void conv2d_col2im(const float* data_col, int channels, int padded_h, int padded_w,
                   int ksize, int stride, float* data_im) {
	int output_h = (padded_h - ksize) / stride + 1;
	int output_w = (padded_w - ksize) / stride + 1;
	int col_cols = output_h * output_w;

	conv2d_col2im_ctx ctx;
	ctx.data_col = data_col;
	ctx.data_im = data_im;
	ctx.channels = channels;
	ctx.padded_h = padded_h;
	ctx.padded_w = padded_w;
	ctx.ksize = ksize;
	ctx.stride = stride;
	ctx.output_h = output_h;
	ctx.output_w = output_w;
	ctx.col_cols = col_cols;

	int num_threads = threadpool_default_threads();
	if (num_threads > channels) num_threads = channels;
	if (num_threads < 1) num_threads = 1;
	parallel_for(0, channels, num_threads, conv2d_col2im_channel_task, &ctx);
}

Tensor* conv2d_apply_padding(Arena* A, const Conv2D* c, Tensor* input) {
	assert(A && c && input);
	if (c->padding == 0) return input;

	int channels = input->info.shape[0];
	int height = input->info.shape[1];
	int width = input->info.shape[2];
	int pshape[3] = { channels, height + 2 * c->padding, width + 2 * c->padding };
	Tensor* padded = tensor_create_new(A, 3, pshape);
	tensor_fill_zeros(padded);

	for (int ch = 0; ch < channels; ch++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				int in_idx = conv2d_idx3(input, ch, h, w);
				int pad_idx = conv2d_idx3(padded, ch, h + c->padding, w + c->padding);
				padded->_data[pad_idx] = input->_data[in_idx];
			}
		}
	}
	return padded;
}

Tensor* conv2d_remove_padding(Arena* A, const Conv2D* c, Tensor* padded) {
	assert(A && c && padded);
	if (c->padding == 0) return padded;

	int channels = padded->info.shape[0];
	int height = padded->info.shape[1] - 2 * c->padding;
	int width = padded->info.shape[2] - 2 * c->padding;
	int oshape[3] = { channels, height, width };
	Tensor* out = tensor_create_new(A, 3, oshape);

	for (int ch = 0; ch < channels; ch++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				int pad_idx = conv2d_idx3(padded, ch, h + c->padding, w + c->padding);
				int out_idx = conv2d_idx3(out, ch, h, w);
				out->_data[out_idx] = padded->_data[pad_idx];
			}
		}
	}
	return out;
}

void conv2d_initialize_weights(Conv2D* c) {
	if (!c || !c->weights) return;
	int fan_in = c->input_channels * c->kernel_size * c->kernel_size;
	int fan_out = c->num_filters * c->kernel_size * c->kernel_size;
	(void)fan_in;
	(void)fan_out;
	tensor_randomize_weights(c->weights);
	tensor_fill_zeros(c->biases);
}

Tensor* conv2d_apply_convolution(Arena* A, Conv2D* c, Tensor* input) {
	return __nag_conv2d_forward(A, c, input);
}

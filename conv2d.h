#pragma once
#ifndef CONV2D_H
#define CONV2D_H

#include <assert.h>

#include "arena.h"
#include "tensor.h"
#include "threadpool.h"

typedef struct Conv2D {
	int input_channels;
	int num_filters;
	int kernel_size;
	int stride;
	int padding;

	Tensor* weights; /* [F, C, K, K] */
	Tensor* biases;  /* [F] */

	Tensor* last_input;
	Tensor* dweights;
	Tensor* dbiases;
	
	/*// Optimized workspace buffers (to avoid realloc each pass)
    Tensor col_buffer;
    Tensor padded_input_buffer;
    Tensor padded_input_gradients_buffer;*/
} Conv2D;

void conv2d_im2col(const float* data_im, int channels, int padded_h, int padded_w,
                   int ksize, int stride, float* data_col);
void conv2d_col2im(const float* data_col, int channels, int padded_h, int padded_w,
                   int ksize, int stride, float* data_im);
Tensor* conv2d_apply_padding(Arena* A, const Conv2D* c, Tensor* input);
Tensor* conv2d_remove_padding(Arena* A, const Conv2D* c, Tensor* padded);
void conv2d_initialize_weights(Conv2D* c);
Tensor* conv2d_apply_convolution(Arena* A, Conv2D* c, Tensor* input);

static inline int conv2d_idx3(const Tensor* t, int c, int h, int w);
static inline int conv2d_idx4(const Tensor* t, int a, int b, int c, int d);
/*YOU HAVEN'T IMPLEMENTED THIS Tensor im2col(const Tensor& input) const;
    Tensor col2im(const Tensor& col, const std::vector<int>& input_shape) const;
    Tensor apply_convolution(const Tensor& input) const;
    Tensor apply_padding(const Tensor& input) const;
    Tensor remove_padding(const Tensor& input) const;
	static void conv2d_im2col(const double* data_im, int channels, int padded_h, int padded_w,
                   int ksize, int stride,
                   double* data_col) {
    int output_h = (padded_h - ksize) / stride + 1;
    int output_w = (padded_w - ksize) / stride + 1;
    int col_cols = output_h * output_w;
    size_t num_threads = std::min(static_cast<size_t>(channels), static_cast<size_t>(std::thread::hardware_concurrency()));
    parallel_for(0, channels, num_threads, [&](size_t c) {
        for (int kh = 0; kh < ksize; ++kh) {
            for (int kw = 0; kw < ksize; ++kw) {
                int row = c * ksize * ksize + kh * ksize + kw;
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        int im_index = c * (padded_h * padded_w) + ih * padded_w + iw;
                        int col_index = row * col_cols + oh * output_w + ow;
                        data_col[col_index] = data_im[im_index];
                    }
                }
            }
        }
    });
}

// Reverse of im2col: scatter columns back into image gradients
static void conv2d_col2im(const double* data_col, int channels, int padded_h, int padded_w,
                   int ksize, int stride,
                   double* data_im) {
    int output_h = (padded_h - ksize) / stride + 1;
    int output_w = (padded_w - ksize) / stride + 1;
    int col_cols = output_h * output_w;
    size_t num_threads = std::min(static_cast<size_t>(channels), static_cast<size_t>(std::thread::hardware_concurrency()));
    parallel_for(0, channels, num_threads, [&](size_t c) {
        for (int kh = 0; kh < ksize; ++kh) {
            for (int kw = 0; kw < ksize; ++kw) {
                int row = c * ksize * ksize + kh * ksize + kw;
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        int col_index = row * col_cols + oh * output_w + ow;
                        int im_index = c * (padded_h * padded_w) + ih * padded_w + iw;
                        data_im[im_index] += data_col[col_index];
                    }
                }
            }
        }
    });
}

Conv2D::Conv2D(int num_filters, int kernel_size, int stride, int padding, Activation activation)
    : input_channels(-1), num_filters(num_filters), kernel_size(kernel_size), 
      stride(stride), padding(padding), activation_func(activation), weights_initialized(false) {
}

Conv2D::Conv2D(int input_channels, int num_filters, int kernel_size, int stride, int padding, Activation activation)
    : input_channels(input_channels), num_filters(num_filters), kernel_size(kernel_size),
      stride(stride), padding(padding), activation_func(activation), weights_initialized(false) {
    initialize_weights(input_channels, 0, 0);
}

Conv2D::~Conv2D() {
    // Smart pointers will automatically clean up
}

void Conv2D::initialize_weights(int input_channels, int height, int width) {
    this->input_channels = input_channels;
    
    // Initialize weights using Xavier/Glorot initialization
    // Shape: [num_filters, input_channels, kernel_size, kernel_size]
    std::vector<int> weight_shape = {num_filters, input_channels, kernel_size, kernel_size};
    weights = Tensor(weight_shape);
    
    // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
    int fan_in = input_channels * kernel_size * kernel_size;
    int fan_out = num_filters * kernel_size * kernel_size;
    double scale = std::sqrt(2.0 / (fan_in + fan_out));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, scale);
    
    for (int i = 0; i < weights.size(); ++i) {
        weights.at(i) = dis(gen);
    }
    
    // Initialize biases to zero
    std::vector<int> bias_shape = {num_filters};
    biases = Tensor::zeros(bias_shape);
    
    weights_initialized = true;
}

std::vector<int> Conv2D::calculate_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Input must be 3D: [channels, height, width]");
    }
    
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    return {num_filters, output_height, output_width};
}

Tensor Conv2D::apply_padding(const Tensor& input) const {
    if (padding == 0) {
        return input;
    }
    
    std::vector<int> input_shape = input.shape();
    int channels = input_shape[0];
    int height = input_shape[1];
    int width = input_shape[2];
    
    std::vector<int> padded_shape = {channels, height + 2 * padding, width + 2 * padding};
    Tensor padded = Tensor::zeros(padded_shape);
    
    // Use fast indexing for efficient copying
    const double* input_data = input.data_ptr();
    double* padded_data = padded.data_ptr();
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int input_idx = input.fast_index_3d(c, h, w);
                int padded_idx = padded.fast_index_3d(c, h + padding, w + padding);
                padded_data[padded_idx] = input_data[input_idx];
            }
        }
    }
    
    return padded;
}
*/

static inline void __ag_conv2d_backward_op(Tensor* out) {
	if (!out || !out->parents) return;
	Tensor* in = out->parents[0];
	Tensor* w = out->parents[1];
	Tensor* b = out->parents[2];
	Tensor* meta = out->parents[3];
	if (!in || !w || !b || !meta) return;
	ensure_grad_buffer(in);
	ensure_grad_buffer(w);
	ensure_grad_buffer(b);

	int input_channels = (int)meta->_data[0];
	int num_filters = (int)meta->_data[1];
	int kernel_size = (int)meta->_data[2];
	int stride = (int)meta->_data[3];
	int padding = (int)meta->_data[4];

	int in_h = in->info.shape[1];
	int in_w = in->info.shape[2];
	int out_h = out->info.shape[1];
	int out_w = out->info.shape[2];

	for (int f = 0; f < num_filters; f++) {
		for (int oh = 0; oh < out_h; oh++) {
			for (int ow = 0; ow < out_w; ow++) {
				int oi = conv2d_idx3(out, f, oh, ow);
				float g = out->grad ? out->grad[oi] : 0.0f;
				b->grad[f] += g;

				int in_y0 = oh * stride - padding;
				int in_x0 = ow * stride - padding;
				for (int ch = 0; ch < input_channels; ch++) {
					for (int ky = 0; ky < kernel_size; ky++) {
						int iy = in_y0 + ky;
						if ((unsigned)iy >= (unsigned)in_h) continue;
						for (int kx = 0; kx < kernel_size; kx++) {
							int ix = in_x0 + kx;
							if ((unsigned)ix >= (unsigned)in_w) continue;
							int ii = conv2d_idx3(in, ch, iy, ix);
							int wi = conv2d_idx4(w, f, ch, ky, kx);
							w->grad[wi] += in->_data[ii] * g;
							in->grad[ii] += w->_data[wi] * g;
						}
					}
				}
			}
		}
	}
}

static inline int conv2d_idx3(const Tensor* t, int c, int h, int w) {
	return c * (t->info.shape[1] * t->info.shape[2]) + h * t->info.shape[2] + w;
}

static inline int conv2d_idx4(const Tensor* t, int a, int b, int c, int d) {
	int s1 = t->info.shape[1];
	int s2 = t->info.shape[2];
	int s3 = t->info.shape[3];
	return ((a * s1 + b) * s2 + c) * s3 + d;
}

static inline int conv2d_out_dim(int in, int pad, int k, int stride) {
	return (in + 2 * pad - k) / stride + 1;
}

/* [C,H,W] feature map <-> [H*W, C] token matrix for FFN/LN */
static inline void conv3d_to_tokens(const Tensor* feat, Tensor* tokens) {
	int f = feat->info.shape[0];
	int h = feat->info.shape[1];
	int w = feat->info.shape[2];
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int row = y * w + x;
			for (int c = 0; c < f; c++) {
				tokens->_data[row * f + c] = feat->_data[conv2d_idx3(feat, c, y, x)];
			}
		}
	}
}

static inline void conv_tokens_to_3d_grad(const Tensor* tokens_dx, Tensor* feat_dx) {
	int f = feat_dx->info.shape[0];
	int h = feat_dx->info.shape[1];
	int w = feat_dx->info.shape[2];
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int row = y * w + x;
			for (int c = 0; c < f; c++) {
				feat_dx->_data[conv2d_idx3(feat_dx, c, y, x)] += tokens_dx->_data[row * f + c];
			}
		}
	}
}

static inline void __ag_conv3d_to_tokens_backward(Tensor* tokens) {
	if (!tokens || !tokens->parents) return;
	Tensor* feat = tokens->parents[0];
	if (!feat) return;
	ensure_grad_buffer(feat);
	conv_tokens_to_3d_grad(tokens, feat);
}

static inline Tensor* __ag_conv3d_to_tokens_forward(Arena* A, Tensor* feat) {
	int shape[2] = { feat->info.shape[1] * feat->info.shape[2], feat->info.shape[0] };
	Tensor* tokens = tensor_create_new(A, 2, shape);
	conv3d_to_tokens(feat, tokens);
	if (!feat->requires_grad) return tokens;

	tokens->requires_grad = true;
	tokens->num_parents = 1;
	tokens->parents = (Tensor**)arena_alloc(A, sizeof(Tensor*));
	tokens->parents[0] = feat;
	tokens->grad = (float*)arena_alloc(A, tokens->info.sz * sizeof(float));
	memset(tokens->grad, 0, tokens->info.sz * sizeof(float));
	Op* op = (Op*)arena_alloc(A, sizeof(Op));
	op->backward = __ag_conv3d_to_tokens_backward;
	tokens->ops = op;
	return tokens;
}

static inline Conv2D* conv2d_create(Arena* A, int input_channels, int num_filters, int kernel_size, int stride, int padding) {
	assert(A);
	Conv2D* c = (Conv2D*)arena_alloc(A, sizeof(Conv2D));
	c->input_channels = input_channels;
	c->num_filters = num_filters;
	c->kernel_size = kernel_size;
	c->stride = stride;
	c->padding = padding;
	c->last_input = NULL;
	c->dweights = NULL;
	c->dbiases = NULL;

	int wshape[4] = { num_filters, input_channels, kernel_size, kernel_size };
	c->weights = tensor_create_weights_new(A, 4, wshape);
	int bshape[1] = { num_filters };
	c->biases = tensor_create_new(A, 1, bshape);
	tensor_fill_zeros(c->biases);
	conv2d_initialize_weights(c);
	return c;
}

static inline Tensor* __nag_conv2d_forward(Arena* A, Conv2D* c, Tensor* input) {
	assert(A);
	assert(c);
	assert(input);
	assert(input->ndim == 3);
	assert(input->info.shape[0] == c->input_channels);
	assert(c->weights && c->biases);

	c->last_input = input;

	int in_h = input->info.shape[1];
	int in_w = input->info.shape[2];
	int out_h = conv2d_out_dim(in_h, c->padding, c->kernel_size, c->stride);
	int out_w = conv2d_out_dim(in_w, c->padding, c->kernel_size, c->stride);
	assert(out_h > 0 && out_w > 0);

	int oshape[3] = { c->num_filters, out_h, out_w };
	Tensor* out = tensor_create_new(A, 3, oshape);
	tensor_fill_zeros(out);

	for (int f = 0; f < c->num_filters; f++) {
		float b = c->biases->_data[f];
		for (int oh = 0; oh < out_h; oh++) {
			for (int ow = 0; ow < out_w; ow++) {
				float sum = b;
				int in_y0 = oh * c->stride - c->padding;
				int in_x0 = ow * c->stride - c->padding;
				for (int ch = 0; ch < c->input_channels; ch++) {
					for (int ky = 0; ky < c->kernel_size; ky++) {
						int iy = in_y0 + ky;
						if ((unsigned)iy >= (unsigned)in_h) continue;
						int ix0 = in_x0;
						int ix1 = in_x0 + c->kernel_size;
						if (ix0 >= 0 && ix1 <= in_w) {
							int ii0 = conv2d_idx3(input, ch, iy, ix0);
							int wi0 = conv2d_idx4(c->weights, f, ch, ky, 0);
							int kx = 0;
							#if defined(__AVX512F__) || defined(__AVX2__)
								__m256 vsum = _mm256_setzero_ps();
								for (; kx <= c->kernel_size - 8; kx += 8) {
									__m256 vin = _mm256_loadu_ps(&input->_data[ii0 + kx]);
									__m256 vw = _mm256_loadu_ps(&c->weights->_data[wi0 + kx]);
									#if defined(__FMA__)
										vsum = _mm256_fmadd_ps(vin, vw, vsum);
									#else
										vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vin, vw));
									#endif
								}
								float tmp[8];
								_mm256_storeu_ps(tmp, vsum);
								sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
							#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
								float32x4_t vsum0 = vdupq_n_f32(0.0f);
								float32x4_t vsum1 = vdupq_n_f32(0.0f);
								for (; kx <= c->kernel_size - 8; kx += 8) {
									float32x4_t vin0 = vld1q_f32(&input->_data[ii0 + kx]);
									float32x4_t vin1 = vld1q_f32(&input->_data[ii0 + kx + 4]);
									float32x4_t vw0 = vld1q_f32(&c->weights->_data[wi0 + kx]);
									float32x4_t vw1 = vld1q_f32(&c->weights->_data[wi0 + kx + 4]);
									vsum0 = vmlaq_f32(vsum0, vin0, vw0);
									vsum1 = vmlaq_f32(vsum1, vin1, vw1);
								}
								float tmp0[4], tmp1[4];
								vst1q_f32(tmp0, vsum0);
								vst1q_f32(tmp1, vsum1);
								sum += tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];
							#elif defined(__SSE4_2__) || defined(__SSE2__)
								__m128 vsum0 = _mm_setzero_ps();
								__m128 vsum1 = _mm_setzero_ps();
								for (; kx <= c->kernel_size - 8; kx += 8) {
									__m128 vin0 = _mm_loadu_ps(&input->_data[ii0 + kx]);
									__m128 vin1 = _mm_loadu_ps(&input->_data[ii0 + kx + 4]);
									__m128 vw0 = _mm_loadu_ps(&c->weights->_data[wi0 + kx]);
									__m128 vw1 = _mm_loadu_ps(&c->weights->_data[wi0 + kx + 4]);
									vsum0 = _mm_add_ps(vsum0, _mm_mul_ps(vin0, vw0));
									vsum1 = _mm_add_ps(vsum1, _mm_mul_ps(vin1, vw1));
								}
								float tmp0[4], tmp1[4];
								_mm_storeu_ps(tmp0, vsum0);
								_mm_storeu_ps(tmp1, vsum1);
								sum += tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];
							#endif
							for (; kx < c->kernel_size; kx++) {
								sum += input->_data[ii0 + kx] * c->weights->_data[wi0 + kx];
							}
						} else {
							for (int kx = 0; kx < c->kernel_size; kx++) {
								int ix = in_x0 + kx;
								if ((unsigned)ix >= (unsigned)in_w) continue;
								int ii = conv2d_idx3(input, ch, iy, ix);
								int wi = conv2d_idx4(c->weights, f, ch, ky, kx);
								sum += input->_data[ii] * c->weights->_data[wi];
							}
						}
					}
				}
				int oi = conv2d_idx3(out, f, oh, ow);
				out->_data[oi] = sum;
			}
		}
	}

	return out;
}

static inline Tensor* __nag_conv2d_backward(Arena* A, Conv2D* c, Tensor* dout) {
	assert(A);
	assert(c);
	assert(dout);
	assert(c->last_input);
	assert(c->last_input->ndim == 3);

	Tensor* input = c->last_input;
	int in_h = input->info.shape[1];
	int in_w = input->info.shape[2];
	int out_h = dout->info.shape[1];
	int out_w = dout->info.shape[2];

	int dx_shape[3] = { c->input_channels, in_h, in_w };
	Tensor* dx = tensor_create_new(A, 3, dx_shape);
	tensor_fill_zeros(dx);

	int dw_shape[4] = { c->num_filters, c->input_channels, c->kernel_size, c->kernel_size };
	c->dweights = tensor_create_new(A, 4, dw_shape);
	tensor_fill_zeros(c->dweights);
	int db_shape[1] = { c->num_filters };
	c->dbiases = tensor_create_new(A, 1, db_shape);
	tensor_fill_zeros(c->dbiases);

	for (int f = 0; f < c->num_filters; f++) {
		for (int oh = 0; oh < out_h; oh++) {
			for (int ow = 0; ow < out_w; ow++) {
				int oi = conv2d_idx3(dout, f, oh, ow);
				float g = dout->_data[oi];
				c->dbiases->_data[f] += g;

				int in_y0 = oh * c->stride - c->padding;
				int in_x0 = ow * c->stride - c->padding;
				for (int ch = 0; ch < c->input_channels; ch++) {
					for (int ky = 0; ky < c->kernel_size; ky++) {
						int iy = in_y0 + ky;
						if ((unsigned)iy >= (unsigned)in_h) continue;
						int ix0 = in_x0;
						int ix1 = in_x0 + c->kernel_size;
						if (ix0 >= 0 && ix1 <= in_w) {
							int ii0 = conv2d_idx3(input, ch, iy, ix0);
							int wi0 = conv2d_idx4(c->weights, f, ch, ky, 0);
							int dwi0 = conv2d_idx4(c->dweights, f, ch, ky, 0);
							int kx = 0;
							#if defined(__AVX512F__) || defined(__AVX2__)
								__m256 vg = _mm256_set1_ps(g);
								for (; kx <= c->kernel_size - 8; kx += 8) {
									__m256 vin = _mm256_loadu_ps(&input->_data[ii0 + kx]);
									__m256 vw = _mm256_loadu_ps(&c->weights->_data[wi0 + kx]);
									__m256 vdw = _mm256_loadu_ps(&c->dweights->_data[dwi0 + kx]);
									#if defined(__FMA__)
										vdw = _mm256_fmadd_ps(vin, vg, vdw);
									#else
										vdw = _mm256_add_ps(vdw, _mm256_mul_ps(vin, vg));
									#endif
									_mm256_storeu_ps(&c->dweights->_data[dwi0 + kx], vdw);
									__m256 vdx = _mm256_loadu_ps(&dx->_data[ii0 + kx]);
									#if defined(__FMA__)
										vdx = _mm256_fmadd_ps(vw, vg, vdx);
									#else
										vdx = _mm256_add_ps(vdx, _mm256_mul_ps(vw, vg));
									#endif
									_mm256_storeu_ps(&dx->_data[ii0 + kx], vdx);
								}
							#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
								float32x4_t vg0 = vdupq_n_f32(g);
								for (; kx <= c->kernel_size - 8; kx += 8) {
									float32x4_t vin0 = vld1q_f32(&input->_data[ii0 + kx]);
									float32x4_t vin1 = vld1q_f32(&input->_data[ii0 + kx + 4]);
									float32x4_t vw0 = vld1q_f32(&c->weights->_data[wi0 + kx]);
									float32x4_t vw1 = vld1q_f32(&c->weights->_data[wi0 + kx + 4]);
									float32x4_t vdw0 = vld1q_f32(&c->dweights->_data[dwi0 + kx]);
									float32x4_t vdw1 = vld1q_f32(&c->dweights->_data[dwi0 + kx + 4]);
									vdw0 = vmlaq_f32(vdw0, vin0, vg0);
									vdw1 = vmlaq_f32(vdw1, vin1, vg0);
									vst1q_f32(&c->dweights->_data[dwi0 + kx], vdw0);
									vst1q_f32(&c->dweights->_data[dwi0 + kx + 4], vdw1);
									float32x4_t vdx0 = vld1q_f32(&dx->_data[ii0 + kx]);
									float32x4_t vdx1 = vld1q_f32(&dx->_data[ii0 + kx + 4]);
									vdx0 = vmlaq_f32(vdx0, vw0, vg0);
									vdx1 = vmlaq_f32(vdx1, vw1, vg0);
									vst1q_f32(&dx->_data[ii0 + kx], vdx0);
									vst1q_f32(&dx->_data[ii0 + kx + 4], vdx1);
								}
							#elif defined(__SSE4_2__) || defined(__SSE2__)
								__m128 vg0 = _mm_set1_ps(g);
								for (; kx <= c->kernel_size - 8; kx += 8) {
									__m128 vin0 = _mm_loadu_ps(&input->_data[ii0 + kx]);
									__m128 vin1 = _mm_loadu_ps(&input->_data[ii0 + kx + 4]);
									__m128 vw0 = _mm_loadu_ps(&c->weights->_data[wi0 + kx]);
									__m128 vw1 = _mm_loadu_ps(&c->weights->_data[wi0 + kx + 4]);
									__m128 vdw0 = _mm_loadu_ps(&c->dweights->_data[dwi0 + kx]);
									__m128 vdw1 = _mm_loadu_ps(&c->dweights->_data[dwi0 + kx + 4]);
									vdw0 = _mm_add_ps(vdw0, _mm_mul_ps(vin0, vg0));
									vdw1 = _mm_add_ps(vdw1, _mm_mul_ps(vin1, vg0));
									_mm_storeu_ps(&c->dweights->_data[dwi0 + kx], vdw0);
									_mm_storeu_ps(&c->dweights->_data[dwi0 + kx + 4], vdw1);
									__m128 vdx0 = _mm_loadu_ps(&dx->_data[ii0 + kx]);
									__m128 vdx1 = _mm_loadu_ps(&dx->_data[ii0 + kx + 4]);
									vdx0 = _mm_add_ps(vdx0, _mm_mul_ps(vw0, vg0));
									vdx1 = _mm_add_ps(vdx1, _mm_mul_ps(vw1, vg0));
									_mm_storeu_ps(&dx->_data[ii0 + kx], vdx0);
									_mm_storeu_ps(&dx->_data[ii0 + kx + 4], vdx1);
								}
							#endif
							for (; kx < c->kernel_size; kx++) {
								c->dweights->_data[dwi0 + kx] += input->_data[ii0 + kx] * g;
								dx->_data[ii0 + kx] += c->weights->_data[wi0 + kx] * g;
							}
						} else {
							for (int kx = 0; kx < c->kernel_size; kx++) {
								int ix = in_x0 + kx;
								if ((unsigned)ix >= (unsigned)in_w) continue;
								int ii = conv2d_idx3(input, ch, iy, ix);
								int wi = conv2d_idx4(c->weights, f, ch, ky, kx);
								int dwi = conv2d_idx4(c->dweights, f, ch, ky, kx);
								c->dweights->_data[dwi] += input->_data[ii] * g;
								dx->_data[ii] += c->weights->_data[wi] * g;
							}
						}
					}
				}
			}
		}
	}

	return dx;
}

static inline Tensor* __ag_conv2d_forward(Arena* A, Conv2D* c, Tensor* input) {
	Tensor* out = __nag_conv2d_forward(A, c, input);
	if (!out) return NULL;
	if (!(input->requires_grad || (c->weights && c->weights->requires_grad) || (c->biases && c->biases->requires_grad))) return out;

	out->requires_grad = true;
	out->num_parents = 4;
	out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor*));
	out->parents[0] = input;
	out->parents[1] = c->weights;
	out->parents[2] = c->biases;
	int meta_shape[1] = { 5 };
	Tensor* meta = tensor_create_new(A, 1, meta_shape);
	meta->requires_grad = false;
	meta->_data[0] = (float)c->input_channels;
	meta->_data[1] = (float)c->num_filters;
	meta->_data[2] = (float)c->kernel_size;
	meta->_data[3] = (float)c->stride;
	meta->_data[4] = (float)c->padding;
	out->parents[3] = meta;
	out->grad = arena_alloc(A, out->info.sz * sizeof(float));
	memset(out->grad, 0, out->info.sz * sizeof(float));
	Op* op = arena_alloc(A, sizeof(Op));
	op->backward = __ag_conv2d_backward_op;
	out->ops = op;

	return out;
}

static inline Tensor* __ag_conv2d_backward(Arena* A, Conv2D* c, Tensor* dout) {
	(void)A;
	(void)c;
	return dout;
}

#endif
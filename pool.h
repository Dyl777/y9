#pragma once
#ifndef POOL_H
#define POOL_H

#include <assert.h>
#include <math.h>
#include <string.h>

#include "arena.h"
#include "tensor.h"
#include "threadpool.h"

typedef enum PoolType { MaxPool2D, AvgPool2D, MinPool2D, GlobalAvgPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D, L2Pool2D } PoolType;

typedef struct Pool2D {
	int pool_size;
	int stride;
	int dilation;
	int ceil_mode;
	int adaptive_out_h;
	int adaptive_out_w;
	Tensor* last_input;
	Tensor* mask;
	PoolType type;
} Pool2D;

static inline int pool2d_idx3(const Tensor* t, int c, int h, int w);

typedef struct __pool2d_ag_backward_ch_ctx {
	Tensor* out;
	Tensor* in;
	Tensor* mask;
	int pool_size;
	int stride;
	int dilation;
	int type;
} __pool2d_ag_backward_ch_ctx;

static inline void __pool2d_ag_backward_ch(int ch, void* vctx) {
	__pool2d_ag_backward_ch_ctx* ctx = (__pool2d_ag_backward_ch_ctx*)vctx;
	Tensor* out = ctx->out;
	Tensor* in = ctx->in;
	Tensor* mask = ctx->mask;
	int pool_size = ctx->pool_size;
	int stride = ctx->stride;
	int dilation = ctx->dilation;
	int type = ctx->type;

	int in_h = in->info.shape[1];
	int in_w = in->info.shape[2];
	int out_h = out->info.shape[1];
	int out_w = out->info.shape[2];

	for (int oh = 0; oh < out_h; oh++) {
		for (int ow = 0; ow < out_w; ow++) {
			int oi = pool2d_idx3(out, ch, oh, ow);
			float g = out->grad ? out->grad[oi] : 0.0f;

			if (type == (int)GlobalAvgPool2D) {
				float gg = g / (float)(in_h * in_w);
				for (int ih = 0; ih < in_h; ih++) {
					for (int iw = 0; iw < in_w; iw++) {
						int ii = pool2d_idx3(in, ch, ih, iw);
						in->grad[ii] += gg;
					}
				}
			} else if (type == (int)AdaptiveAvgPool2D) {
				int hstart = (oh * in_h) / out_h;
				int hend = ((oh + 1) * in_h + out_h - 1) / out_h;
				int wstart = (ow * in_w) / out_w;
				int wend = ((ow + 1) * in_w + out_w - 1) / out_w;
				int count = (hend - hstart) * (wend - wstart);
				float gg = (count > 0) ? (g / (float)count) : 0.0f;
				for (int ih = hstart; ih < hend; ih++) {
					for (int iw = wstart; iw < wend; iw++) {
						int ii = pool2d_idx3(in, ch, ih, iw);
						in->grad[ii] += gg;
					}
				}
			} else if (type == (int)L2Pool2D) {
				int count = 0;
				float sumsq = 0.0f;
				for (int kh = 0; kh < pool_size; kh++) {
					for (int kw = 0; kw < pool_size; kw++) {
						int ih = oh * stride + kh;
						int iw = ow * stride + kw;
						if (ih >= in_h || iw >= in_w) continue;
						int ii = pool2d_idx3(in, ch, ih, iw);
						float v = in->_data[ii];
						sumsq += v * v;
						count++;
					}
				}
				float l2 = (count > 0) ? sqrtf(sumsq / (float)count + 1e-8f) : 0.0f;
				float denom = (count > 0) ? (l2 * (float)count) : 1.0f;
				for (int kh = 0; kh < pool_size; kh++) {
					for (int kw = 0; kw < pool_size; kw++) {
						int ih = oh * stride + kh;
						int iw = ow * stride + kw;
						if (ih >= in_h || iw >= in_w) continue;
						int ii = pool2d_idx3(in, ch, ih, iw);
						in->grad[ii] += g * (in->_data[ii] / (denom + 1e-8f));
					}
				}
			} else if (type == (int)AvgPool2D) {
				int count = 0;
				for (int kh = 0; kh < pool_size; kh++) {
					for (int kw = 0; kw < pool_size; kw++) {
						int ih = oh * stride + kh;
						int iw = ow * stride + kw;
						if (ih < in_h && iw < in_w) count++;
					}
				}
				float gg = (count > 0) ? (g / (float)count) : 0.0f;
				for (int kh = 0; kh < pool_size; kh++) {
					for (int kw = 0; kw < pool_size; kw++) {
						int ih = oh * stride + kh;
						int iw = ow * stride + kw;
						if (ih >= in_h || iw >= in_w) continue;
						int ii = pool2d_idx3(in, ch, ih, iw);
						in->grad[ii] += gg;
					}
				}
			} else {
				if (!mask) continue;
				if (type == (int)AdaptiveMaxPool2D) {
					int hstart = (oh * in_h) / out_h;
					int hend = ((oh + 1) * in_h + out_h - 1) / out_h;
					int wstart = (ow * in_w) / out_w;
					int wend = ((ow + 1) * in_w + out_w - 1) / out_w;
					for (int ih = hstart; ih < hend; ih++) {
						for (int iw = wstart; iw < wend; iw++) {
							int mi = pool2d_idx3(mask, ch, ih, iw);
							if (mask->_data[mi] > 0.5f) {
								int ii = pool2d_idx3(in, ch, ih, iw);
								in->grad[ii] += g;
							}
						}
					}
				} else {
					for (int kh = 0; kh < pool_size; kh++) {
						for (int kw = 0; kw < pool_size; kw++) {
							int ih = oh * stride + kh * dilation;
							int iw = ow * stride + kw * dilation;
							if (ih >= in_h || iw >= in_w) continue;
							int mi = pool2d_idx3(mask, ch, ih, iw);
							if (mask->_data[mi] > 0.5f) {
								int ii = pool2d_idx3(in, ch, ih, iw);
								in->grad[ii] += g;
							}
						}
					}
				}
			}
		}
	}
}

static inline void __ag_pool2d_backward_op(Tensor* out) {
	if (!out || !out->parents) return;
	Tensor* in = out->parents[0];
	Tensor* mask = (out->num_parents > 1) ? out->parents[1] : NULL;
	Tensor* meta = (out->num_parents > 2) ? out->parents[2] : NULL;
	if (!in) return;
	ensure_grad_buffer(in);

	int pool_size = 0;
	int stride = 0;
	int type = 0;
	int dilation = 1;
	int ceil_mode = 0;
	int adaptive_out_h = 0;
	int adaptive_out_w = 0;
	if (meta && meta->_data && meta->info.sz >= 7) {
		pool_size = (int)meta->_data[0];
		stride = (int)meta->_data[1];
		type = (int)meta->_data[2];
		dilation = (int)meta->_data[3];
		ceil_mode = (int)meta->_data[4];
		adaptive_out_h = (int)meta->_data[5];
		adaptive_out_w = (int)meta->_data[6];
	} else if (meta && meta->_data && meta->info.sz >= 3) {
		pool_size = (int)meta->_data[0];
		stride = (int)meta->_data[1];
		type = (int)meta->_data[2];
	}
	if (dilation <= 0) dilation = 1;
	if (stride <= 0) stride = pool_size;
	(void)ceil_mode;
	(void)adaptive_out_h;
	(void)adaptive_out_w;

	int c = in->info.shape[0];
	__pool2d_ag_backward_ch_ctx ctx;
	ctx.out = out;
	ctx.in = in;
	ctx.mask = mask;
	ctx.pool_size = pool_size;
	ctx.stride = stride;
	ctx.dilation = dilation;
	ctx.type = type;
	parallel_for(0, c, 0, __pool2d_ag_backward_ch, &ctx);
}

static inline int pool2d_out_dim(int in, int pool_size, int stride) {
	return (in - pool_size) / stride + 1;
}

static inline int pool2d_out_dim_ceil(int in, int eff_k, int stride) {
	return (in - eff_k + stride - 1) / stride + 1;
}

static inline int pool2d_idx3(const Tensor* t, int c, int h, int w) {
	(void)t;
	return c * (t->info.shape[1] * t->info.shape[2]) + h * t->info.shape[2] + w;
}

static inline Pool2D* pool2d_create(Arena* A, int pool_size, int stride, PoolType type) {
	Pool2D* p = (Pool2D*)arena_alloc(A, sizeof(Pool2D));
	p->pool_size = pool_size;
	p->stride = (stride <= 0) ? pool_size : stride;
	p->dilation = 1;
	p->ceil_mode = 0;
	p->adaptive_out_h = 0;
	p->adaptive_out_w = 0;
	p->last_input = NULL;
	p->mask = NULL;
	p->type = type;
	return p;
}

static inline void pool2d_set_adaptive(Pool2D* p, int out_h, int out_w) {
	if (!p) return;
	p->adaptive_out_h = out_h;
	p->adaptive_out_w = out_w;
}

static inline void pool2d_set_dilation_ceil(Pool2D* p, int dilation, int ceil_mode) {
	if (!p) return;
	p->dilation = (dilation <= 0) ? 1 : dilation;
	p->ceil_mode = (ceil_mode != 0);
}

typedef struct __pool2d_forward_ch_ctx {
	Pool2D* p;
	Tensor* input;
	Tensor* out;
	int in_h;
	int in_w;
	int out_h;
	int out_w;
	int eff_k;
} __pool2d_forward_ch_ctx;

static inline void __pool2d_forward_ch(int ch, void* vctx);

static inline Tensor* __nag_pool2d_forward(Arena* A, Pool2D* p, Tensor* input) {
	assert(p);
	assert(input);
	assert(input->ndim == 3);

	p->last_input = input;

	int c = input->info.shape[0];
	int in_h = input->info.shape[1];
	int in_w = input->info.shape[2];
	int out_h = 0;
	int out_w = 0;
	int eff_k = (p->pool_size - 1) * ((p->dilation <= 0) ? 1 : p->dilation) + 1;
	if (p->type == GlobalAvgPool2D) {
		out_h = 1;
		out_w = 1;
	} else if (p->type == AdaptiveAvgPool2D || p->type == AdaptiveMaxPool2D) {
		out_h = (p->adaptive_out_h > 0) ? p->adaptive_out_h : 1;
		out_w = (p->adaptive_out_w > 0) ? p->adaptive_out_w : 1;
	} else if ((p->type == MaxPool2D || p->type == MinPool2D) && p->ceil_mode) {
		out_h = pool2d_out_dim_ceil(in_h, eff_k, p->stride);
		out_w = pool2d_out_dim_ceil(in_w, eff_k, p->stride);
	} else {
		out_h = pool2d_out_dim(in_h, p->pool_size, p->stride);
		out_w = pool2d_out_dim(in_w, p->pool_size, p->stride);
	}
	assert(out_h > 0 && out_w > 0);

	int out_shape[3] = { c, out_h, out_w };
	Tensor* out = tensor_create_new(A, 3, out_shape);
	tensor_fill_zeros(out);

	if (p->type == MaxPool2D || p->type == MinPool2D || p->type == AdaptiveMaxPool2D) {
		int in_shape[3] = { c, in_h, in_w };
		p->mask = tensor_create_new(A, 3, in_shape);
		tensor_fill_zeros(p->mask);
	}

	__pool2d_forward_ch_ctx fwd_ctx;
	fwd_ctx.p = p;
	fwd_ctx.input = input;
	fwd_ctx.out = out;
	fwd_ctx.in_h = in_h;
	fwd_ctx.in_w = in_w;
	fwd_ctx.out_h = out_h;
	fwd_ctx.out_w = out_w;
	fwd_ctx.eff_k = eff_k;
	{
		int num_threads = threadpool_default_threads();
		if (num_threads > c) num_threads = c;
		if (num_threads < 1) num_threads = 1;
		parallel_for(0, c, num_threads, __pool2d_forward_ch, &fwd_ctx);
	}

	return out;
}

static inline void __pool2d_forward_ch(int ch, void* vctx) {
		__pool2d_forward_ch_ctx* ctx = (__pool2d_forward_ch_ctx*)vctx;
		Pool2D* p = ctx->p;
		Tensor* input = ctx->input;
		Tensor* out = ctx->out;
		int in_h = ctx->in_h;
		int in_w = ctx->in_w;
		int out_h = ctx->out_h;
		int out_w = ctx->out_w;
		int eff_k = ctx->eff_k;

		for (int oh = 0; oh < out_h; oh++) {
			for (int ow = 0; ow < out_w; ow++) {
				float best = 0.0f;
				int best_h = -1;
				int best_w = -1;
				float sum = 0.0f;
				int count = 0;

				if (p->type == GlobalAvgPool2D) {
					for (int ih = 0; ih < in_h; ih++) {
						int base = pool2d_idx3(input, ch, ih, 0);
						for (int iw = 0; iw < in_w; iw++) {
							sum += input->_data[base + iw];
							count++;
						}
					}
				} else if (p->type == AdaptiveAvgPool2D || p->type == AdaptiveMaxPool2D) {
					int outH = out_h;
					int outW = out_w;
					int hstart = (oh * in_h) / outH;
					int hend = ((oh + 1) * in_h + outH - 1) / outH;
					int wstart = (ow * in_w) / outW;
					int wend = ((ow + 1) * in_w + outW - 1) / outW;
					for (int ih = hstart; ih < hend; ih++) {
						int base = pool2d_idx3(input, ch, ih, wstart);
						for (int iw = wstart; iw < wend; iw++) {
							float v = input->_data[base + (iw - wstart)];
							if (p->type == AdaptiveAvgPool2D) {
								sum += v;
								count++;
							} else {
								if (best_h < 0 || v > best) {
									best = v;
									best_h = ih;
									best_w = iw;
								}
							}
						}
					}
				} else {
					for (int kh = 0; kh < p->pool_size; kh++) {
						int ih = oh * p->stride + kh * ((p->dilation <= 0) ? 1 : p->dilation);
						if (ih >= in_h) continue;
						int iw0 = ow * p->stride;
						if (iw0 >= in_w) continue;
						int dilation = (p->dilation <= 0) ? 1 : p->dilation;
						int base = pool2d_idx3(input, ch, ih, iw0);

						if (p->type == AvgPool2D && dilation == 1) {
							int max_w = iw0 + p->pool_size;
							if (max_w > in_w) max_w = in_w;
							int width = max_w - iw0;
							int kw = 0;
						#if defined(__AVX512F__) || defined(__AVX2__)
							__m256 vsum = _mm256_setzero_ps();
							for (; kw <= width - 8; kw += 8) {
								__m256 v = _mm256_loadu_ps(&input->_data[base + kw]);
								vsum = _mm256_add_ps(vsum, v);
							}
							float tmp[8];
							_mm256_storeu_ps(tmp, vsum);
							sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
						#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
							float32x4_t vsum0 = vdupq_n_f32(0.0f);
							float32x4_t vsum1 = vdupq_n_f32(0.0f);
							for (; kw <= width - 8; kw += 8) {
								float32x4_t v0 = vld1q_f32(&input->_data[base + kw]);
								float32x4_t v1 = vld1q_f32(&input->_data[base + kw + 4]);
								vsum0 = vaddq_f32(vsum0, v0);
								vsum1 = vaddq_f32(vsum1, v1);
							}
							float tmp0[4], tmp1[4];
							vst1q_f32(tmp0, vsum0);
							vst1q_f32(tmp1, vsum1);
							sum += tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];
						#elif defined(__SSE4_2__) || defined(__SSE2__)
							__m128 vsum0 = _mm_setzero_ps();
							__m128 vsum1 = _mm_setzero_ps();
							for (; kw <= width - 8; kw += 8) {
								__m128 v0 = _mm_loadu_ps(&input->_data[base + kw]);
								__m128 v1 = _mm_loadu_ps(&input->_data[base + kw + 4]);
								vsum0 = _mm_add_ps(vsum0, v0);
								vsum1 = _mm_add_ps(vsum1, v1);
							}
							float tmp0[4], tmp1[4];
							_mm_storeu_ps(tmp0, vsum0);
							_mm_storeu_ps(tmp1, vsum1);
							sum += tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];
						#endif
						for (; kw < width; kw++) sum += input->_data[base + kw];
						count += width;
					} else if (p->type == AvgPool2D) {
						for (int kw = 0; kw < p->pool_size; kw++) {
							int iw = iw0 + kw * dilation;
							if (iw >= in_w) continue;
							float v = input->_data[base + (iw - iw0)];
							sum += v;
							count++;
						}
					} else if (p->type == L2Pool2D) {
						for (int kw = 0; kw < p->pool_size; kw++) {
							int iw = iw0 + kw * dilation;
							if (iw >= in_w) continue;
							float v = input->_data[base + (iw - iw0)];
							sum += v * v;
							count++;
						}
					} else {
						for (int kw = 0; kw < p->pool_size; kw++) {
							int iw = iw0 + kw * dilation;
							if (iw >= in_w) continue;
							float v = input->_data[base + (iw - iw0)];
							if (best_h < 0) {
								best = v;
								best_h = ih;
								best_w = iw;
							} else if (p->type == MinPool2D) {
								if (v < best) {
									best = v;
									best_h = ih;
									best_w = iw;
								}
							} else {
								if (v > best) {
									best = v;
									best_h = ih;
									best_w = iw;
								}
							}
						}
					}
					}
				}

				int oi = pool2d_idx3(out, ch, oh, ow);
				if (p->type == AvgPool2D || p->type == GlobalAvgPool2D || p->type == AdaptiveAvgPool2D) {
					out->_data[oi] = (count > 0) ? (sum / (float)count) : 0.0f;
				} else if (p->type == L2Pool2D) {
					out->_data[oi] = (count > 0) ? sqrtf(sum / (float)count + 1e-8f) : 0.0f;
				} else {
					out->_data[oi] = best;
					if ((p->type == MaxPool2D || p->type == MinPool2D || p->type == AdaptiveMaxPool2D) && p->mask && best_h >= 0 && best_w >= 0) {
						int mi = pool2d_idx3(p->mask, ch, best_h, best_w);
						p->mask->_data[mi] = 1.0f;
					}
				}
			}
		}
}

typedef struct __pool2d_backward_ch_ctx {
	Pool2D* p;
	Tensor* dout;
	Tensor* dx;
	int in_h;
	int in_w;
	int out_h;
	int out_w;
} __pool2d_backward_ch_ctx;

static inline void __pool2d_backward_ch(int ch, void* vctx);

static inline Tensor* __nag_pool2d_backward(Arena* A, Pool2D* p, Tensor* dout) {
	assert(p);
	assert(dout);
	assert(p->last_input);
	assert(p->last_input->ndim == 3);

	int c = p->last_input->info.shape[0];
	int in_h = p->last_input->info.shape[1];
	int in_w = p->last_input->info.shape[2];
	int dx_shape[3] = { c, in_h, in_w };
	Tensor* dx = tensor_create_new(A, 3, dx_shape);
	tensor_fill_zeros(dx);

	int out_h = dout->info.shape[1];
	int out_w = dout->info.shape[2];

	__pool2d_backward_ch_ctx bwd_ctx;
	bwd_ctx.p = p;
	bwd_ctx.dout = dout;
	bwd_ctx.dx = dx;
	bwd_ctx.in_h = in_h;
	bwd_ctx.in_w = in_w;
	bwd_ctx.out_h = out_h;
	bwd_ctx.out_w = out_w;
	{
		int num_threads = threadpool_default_threads();
		if (num_threads > c) num_threads = c;
		if (num_threads < 1) num_threads = 1;
		parallel_for(0, c, num_threads, __pool2d_backward_ch, &bwd_ctx);
	}

	return dx;
}

static inline void __pool2d_backward_ch(int ch, void* vctx) {
		__pool2d_backward_ch_ctx* ctx = (__pool2d_backward_ch_ctx*)vctx;
		Pool2D* p = ctx->p;
		Tensor* dout = ctx->dout;
		Tensor* dx = ctx->dx;
		int in_h = ctx->in_h;
		int in_w = ctx->in_w;
		int out_h = ctx->out_h;
		int out_w = ctx->out_w;

		for (int oh = 0; oh < out_h; oh++) {
			for (int ow = 0; ow < out_w; ow++) {
				int oi = pool2d_idx3(dout, ch, oh, ow);
				float g = dout->_data[oi];

				if (p->type == AvgPool2D) {
					int count = 0;
					for (int kh = 0; kh < p->pool_size; kh++) {
						for (int kw = 0; kw < p->pool_size; kw++) {
							int ih = oh * p->stride + kh;
							int iw = ow * p->stride + kw;
							if (ih < in_h && iw < in_w) count++;
						}
					}
					float gg = (count > 0) ? (g / (float)count) : 0.0f;
					for (int kh = 0; kh < p->pool_size; kh++) {
						for (int kw = 0; kw < p->pool_size; kw++) {
							int ih = oh * p->stride + kh;
							int iw = ow * p->stride + kw;
							if (ih >= in_h || iw >= in_w) continue;
							int di = pool2d_idx3(dx, ch, ih, iw);
							dx->_data[di] += gg;
						}
					}
				} else if (p->type == GlobalAvgPool2D) {
					float gg = g / (float)(in_h * in_w);
					for (int ih = 0; ih < in_h; ih++) {
						for (int iw = 0; iw < in_w; iw++) {
							int di = pool2d_idx3(dx, ch, ih, iw);
							dx->_data[di] += gg;
						}
					}
				} else if (p->type == AdaptiveAvgPool2D) {
					int outH = out_h;
					int outW = out_w;
					int hstart = (oh * in_h) / outH;
					int hend = ((oh + 1) * in_h + outH - 1) / outH;
					int wstart = (ow * in_w) / outW;
					int wend = ((ow + 1) * in_w + outW - 1) / outW;
					int count = (hend - hstart) * (wend - wstart);
					float gg = (count > 0) ? (g / (float)count) : 0.0f;
					for (int ih = hstart; ih < hend; ih++) {
						for (int iw = wstart; iw < wend; iw++) {
							int di = pool2d_idx3(dx, ch, ih, iw);
							dx->_data[di] += gg;
						}
					}
				} else if (p->type == L2Pool2D) {
					int count = 0;
					float sumsq = 0.0f;
					for (int kh = 0; kh < p->pool_size; kh++) {
						for (int kw = 0; kw < p->pool_size; kw++) {
							int ih = oh * p->stride + kh;
							int iw = ow * p->stride + kw;
							if (ih >= in_h || iw >= in_w) continue;
							int ii = pool2d_idx3(p->last_input, ch, ih, iw);
							float v = p->last_input->_data[ii];
							sumsq += v * v;
							count++;
						}
					}
					float l2 = (count > 0) ? sqrtf(sumsq / (float)count + 1e-8f) : 0.0f;
					float denom = (count > 0) ? (l2 * (float)count) : 1.0f;
					for (int kh = 0; kh < p->pool_size; kh++) {
						for (int kw = 0; kw < p->pool_size; kw++) {
							int ih = oh * p->stride + kh;
							int iw = ow * p->stride + kw;
							if (ih >= in_h || iw >= in_w) continue;
							int ii = pool2d_idx3(p->last_input, ch, ih, iw);
							int di = pool2d_idx3(dx, ch, ih, iw);
							dx->_data[di] += g * (p->last_input->_data[ii] / (denom + 1e-8f));
						}
					}
				} else {
					assert(p->mask);
					if (p->type == AdaptiveMaxPool2D) {
						int outH = out_h;
						int outW = out_w;
						int hstart = (oh * in_h) / outH;
						int hend = ((oh + 1) * in_h + outH - 1) / outH;
						int wstart = (ow * in_w) / outW;
						int wend = ((ow + 1) * in_w + outW - 1) / outW;
						for (int ih = hstart; ih < hend; ih++) {
							for (int iw = wstart; iw < wend; iw++) {
								int mi = pool2d_idx3(p->mask, ch, ih, iw);
								if (p->mask->_data[mi] > 0.5f) {
									int di = pool2d_idx3(dx, ch, ih, iw);
									dx->_data[di] += g;
								}
							}
						}
					} else {
						for (int kh = 0; kh < p->pool_size; kh++) {
							for (int kw = 0; kw < p->pool_size; kw++) {
								int ih = oh * p->stride + kh * ((p->dilation <= 0) ? 1 : p->dilation);
								int iw = ow * p->stride + kw * ((p->dilation <= 0) ? 1 : p->dilation);
								if (ih >= in_h || iw >= in_w) continue;
								int mi = pool2d_idx3(p->mask, ch, ih, iw);
								if (p->mask->_data[mi] > 0.5f) {
									int di = pool2d_idx3(dx, ch, ih, iw);
									dx->_data[di] += g;
								}
							}
						}
					}
				}
			}
		}
}

static inline Tensor* __ag_pool2d_forward(Arena* A, Pool2D* p, Tensor* input) {
	Tensor* out = __nag_pool2d_forward(A, p, input);
	if (!out) return NULL;
	if (!(input->requires_grad)) return out;

	out->requires_grad = true;
	out->num_parents = 3;
	out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor*));
	out->parents[0] = input;
	out->parents[1] = p->mask;
	int meta_shape[1] = { 7 };
	Tensor* meta = tensor_create_new(A, 1, meta_shape);
	meta->requires_grad = false;
	meta->_data[0] = (float)p->pool_size;
	meta->_data[1] = (float)p->stride;
	meta->_data[2] = (float)p->type;
	meta->_data[3] = (float)p->dilation;
	meta->_data[4] = (float)p->ceil_mode;
	meta->_data[5] = (float)p->adaptive_out_h;
	meta->_data[6] = (float)p->adaptive_out_w;
	out->parents[2] = meta;

	out->grad = arena_alloc(A, out->info.sz * sizeof(float));
	memset(out->grad, 0, out->info.sz * sizeof(float));
	Op* op = arena_alloc(A, sizeof(Op));
	op->backward = __ag_pool2d_backward_op;
	out->ops = op;

	return out;
}

static inline Tensor* __ag_pool2d_backward(Arena* A, Pool2D* p, Tensor* dout) {
	(void)p;
	/* In this codebase, backward is driven via tensor_backward(). */
	return dout;
}

#endif
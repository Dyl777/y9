







## y9 Code Overview

**y9** is a minimal C implementation of a **Transformer neural network** with training capability. It's a from-scratch deep learning framework focused on educational clarity.

---

## Architecture

### Core Components

| File | Purpose |
|------|---------|
| [arena.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/arena.h:0:0-0:0) | **Linear memory allocator** - bump allocator for efficient memory management |
| [tensor.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/tensor.h:0:0-0:0) | **Tensor operations** - matrix math, matmul, softmax, MSE loss, gradients |
| [attn.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/attn.h:0:0-0:0) | **Multi-Head Attention (MHA)** - the core Transformer attention mechanism |
| [ffn.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/ffn.h:0:0-0:0) | **Feed-Forward Network** - 2-layer MLP with ReLU activation |
| [ln.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/ln.h:0:0-0:0) | **Layer Normalization** - with learnable gamma/beta parameters |
| [config.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/config.h:0:0-0:0) | **Hyperparameters** - dimensions, learning rate, training config |
| [vn_transf_mdl.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/vn_transf_mdl.h:0:0-0:0) | **Model header** - component integration |
| [vn_train_main.c](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/vn_train_main.c:0:0-0:0) | **Training loop** - full forward/backward pass with SGD optimizer |

---

## Key Design Decisions

**Memory Management** (`@arena.h:1-28`)
- Uses arena/bump allocator - fast sequential allocation, no individual frees
- 1GB arena pre-allocated, simple offset-based allocation

**Tensor Structure** (`@tensor.h:22-32`)
```c
typedef struct {
    alignas(64) struct {
        int *shape;
        int *stride;
        int sz;
    } info;
    int ndim;
    float *_data;
} Tensor;
```
- Cache-line aligned for performance
- Stride-based indexing for flexible tensor views

**Model Dimensions** (`@config.h:5-16`)
- `EMB_DIM = 32` - embedding dimension
- `HEADS = 8` - attention heads (head_dim = 4)
- `BATCH_SIZE = 16`, `SEQ_LEN = 16000` - training batch/sequence
- `HIDDEN_DIM = 128` - FFN hidden layer

---

## Training Pipeline (`@vn_train_main.c:61-173`)

```
Input (16, 32) → MHA → LayerNorm → FFN → LayerNorm → MSE Loss
                     ↑___________↓_________________↓
                          Backward Pass (SGD)
```

**Forward Pass:**
1. **MHA** - multi-head self-attention across 8 heads
2. **LayerNorm #1** - normalize + scale/shift
3. **FFN** - expand to 128 dims, ReLU, project back to 32
4. **LayerNorm #2** - final normalization
5. **MSE Loss** - compare with target

**Backward Pass:**
- Gradient clipping for stability ([clip_gradient](cci:1://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/tensor.h:179:0-211:1))
- SGD optimizer updates weights
- Zero gradients for next batch

---

## Notable Features

- **Manual backpropagation** - custom gradient computation for each layer
- **Gradient clipping** - prevents exploding gradients
- **Batch training** - processes 16,000 sequence length in chunks of 16
- **Loss logging** - CSV export for monitoring


# Attention mechanism and layer connections

## Self-Attention in [attn.h](cci:7://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/attn.h:0:0-0:0)

**Forward Pass** (`@attn.h:251-297`):

```c
// 1. Project input into Q, K, V matrices
mha->Q = tensor_matmul(A, t, mha->wq);  // X @ Wq
mha->K = tensor_matmul(A, t, mha->wk);  // X @ Wk  
mha->V = tensor_matmul(A, t, mha->wv);  // X @ Wv

// 2. Split into 8 heads, each gets 4 dimensions (32/8)
for each head k:
    slice Q, K, V into Q_h, K_h, V_h (shape: [16, 4])
    
// 3. Scaled dot-product attention per head
head_out = softmax(Q @ K^T / sqrt(4)) @ V

// 4. Concatenate all heads back together
write head_out into mha->out at correct offset
```

**Backward Pass** (`@attn.h:97-248`) uses the chain rule:
- `dAk` = gradient through softmax
- `dQk, dKk, dVk` = gradients for Q, K, V per head
- `dwq, dwk, dwv` = gradients for weight matrices (computed as `X^T @ dQ/K/V`)
- Accumulates across all 8 heads

---

## Layer Connections in [main()](cci:1://file:///c:/Users/AMBE/Downloads/ts-cpl/y5/y9/vn_train_main.c:12:0-177:1) (`@vn_train_main.c:75-107`)

The data flows like this:

```
batch_tensor (16, 32)
    ↓
mha_forward() → attn_score (16, 32)      [MHA: self-attention]
    ↓
layer_norm_forward(L1) → ln1 (16, 32)    [Add & Norm pattern]
    ↓
ffn_forward() → ffn_ln (16, 32)           [FFN: 32→128→32]
    ↓
layer_norm_forward(L2) → ln2 (16, 32)    [Add & Norm]
    ↓
tensor_mse_loss(ln2, target) → loss
```

**Backward** goes in reverse:
```c
layer_norm_backward(L2, ffn_ln, loss, dx_for_ffn, LR);  // dL/d(ffn_ln)
ffn_backward(A, f, ln1, dx_for_ffn);                  // dL/d(ln1)
layer_norm_backward(L1, attn_score, ffn_backpass, dx_for_mha, LR);  // dL/d(attn)
mha_backward(A, m_batch, dx_for_mha, batch_tensor);   // dL/d(X)
```

---

## LayerNorm Usage

**Forward** (`@ln.h:49-97`):
```c
// For each row (token):
mu = mean(row)
var = variance(row)
x_hat = (row - mu) / sqrt(var + EPS)   // normalize
y = gamma * x_hat + beta               // scale & shift (learnable)
```

**Caches** `x_hat` and `var` for backward pass.

**Backward** (`@ln.h:103-146`):
- Computes `d_gamma`, `d_beta` (parameter gradients)
- Computes `dx` (input gradient) using the "one-liner" formula

---

This implements a **single Transformer encoder block**:

```
Input → MHA → Add&Norm → FFN → Add&Norm → Output
```

The "Add" (residual connection) is **implicit** here - the code doesn't show `x + sublayer(x)` but the structure follows the pattern. LayerNorm is applied **after** each sublayer (post-LN variant).

The main function loops through batches and epochs, doing:
1. **Forward**: Compute prediction
2. **Loss**: MSE against target
3. **Backward**: Compute all gradients
4. **Optimize**: SGD updates weights
5. **Zero**: Clear gradients for next batch

The code uses **2D tensors** (shape: `[rows, cols]`) by treating dimensions differently than standard deep learning frameworks:

## How It Handles Batching

Instead of 3D `[batch, seq, embed]`, it uses **2D slices per batch** (`@vn_train_main.c:62-73`):

```c
// Large tensor: [16000, 32] - all data
Tensor *T = tensor_create_new(A, 2, shape);  // shape = {SEQ_LEN, EMB_DIM}

// Process in chunks of 16 (BATCH_SIZE)
float *batch_ptr = T->_data + b * BATCH_SIZE * EMB_DIM;  // pointer arithmetic
int shape_local[2] = {BATCH_SIZE, EMB_DIM};
Tensor *batch_tensor = tensor_create_new(A, 2, shape_local);
memcpy(batch_tensor->_data, batch_ptr, ...);  // copy chunk
```

The batch dimension is handled **outside** the tensor shape via pointer slicing.

---

## How It Handles Multi-Head Attention

Instead of 4D `[batch, heads, seq, head_dim]`, it **slices the embedding dimension** (`@attn.h:266-294`):

```c
// Full Q, K, V: [16, 32] - 2D
// Split into 8 heads, each with 4 dims (32/8)

for (int k = 0; k < 8; k++) {           // 8 heads
    Tensor *Q_h = create_new(A, 2, {16, 4});  // 2D per head
    
    // Slice: extract columns k*4 to (k+1)*4 from the 32-dim embedding
    for (i, j) {
        int src = i * 32 + j + k * 4;   // source in full Q [16,32]
        int dest = i * 4 + j;           // dest in Q_h [16,4]
        Q_h->_data[dest] = mha->Q->_data[src];
    }
}
```

Each head is a **2D tensor** `[seq_len, head_dim]` (e.g., `[16, 4]`). The heads dimension is handled by a `for` loop, not a tensor dimension.

---

## Why 2D?

**Pros:**
- Simpler code - no complex N-dimensional indexing
- Easier to debug and understand
- Pointer arithmetic replaces fancy striding

**Cons:**
- No broadcasting across batch/heads
- Manual slicing with loops and memcpy
- Harder to extend to variable batch sizes

This is a **minimal educational implementation** - production frameworks (PyTorch, JAX) use 3D/4D tensors with automatic broadcasting.

---

## Commands Used During This Session

Below are the terminal commands used while validating/fixing autograd vs non-autograd paths and DAG output.

```powershell
ls
git status --short
git ls-files
ls | sort-object Length -Descending | select-object -first 15 Name,Length

& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -g "vn_train_main.c" -o "vn_train_main.exe"
& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -g "autograd_vn_train.c" -o "autograd_vn_train.exe"

./vn_train_main.exe
./autograd_vn_train.exe

ls "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin"
& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gdb64.exe" -q -batch -ex run -ex bt --args "C:/Users/AMBE/Downloads/ts-cpl/y5/y9/autograd_vn_train.exe"

& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -g "autograd_vn_train.c" -o "autograd_vn_train.exe"; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; ./autograd_vn_train.exe

& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -g "vn_train_main.c" -o "vn_train_main.exe"
& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -g "autograd_vn_train.c" -o "autograd_vn_train.exe"
```

### CNN + optimizer tests (`y9/test/`)

Single translation unit (includes `threadpool.c` / `conv2d.c` where needed — same pattern as `vn_train_main.c`):

```powershell
cd y9
& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -Wall -Wextra -I. test/test_conv2d_nag.c -o test/test_conv2d_nag.exe
& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -Wall -Wextra -I. test/test_conv2d_ag.c -o test/test_conv2d_ag.exe
& "C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/gcc.exe" -std=c99 -O0 -Wall -Wextra -I. test/test_muon_autograd.c -o test/test_muon_autograd.exe
./test/test_conv2d_nag.exe
./test/test_conv2d_ag.exe
./test/test_muon_autograd.exe
```

Pipelines: **Conv → Pool → flatten → LN → FFN → LN → loss** (nag manual backward; ag `tensor_backward`).

### Notes

- Some command attempts were interrupted while troubleshooting environment issues (disk space / shell interruptions), then re-run successfully.
- The final successful compile/run commands are included above.

---

## Python Interface

A PyTorch-like Python interface is available using nanobind bindings.

### Installation

```bash
pip install -e .
```

### Usage Example

```python
import y9
from y9 import Tensor, Linear, Conv2d, MaxPool2d, LayerNorm, Sequential, relu, mse_loss, SGD

# Create a simple model
model = Sequential(
    Linear(128, 64),
    relu,
    Linear(64, 32),
)

# Create optimizer
optimizer = SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    x = Tensor.randn([32, 128])
    x.requires_grad = True
    target = Tensor.randn([32, 32])
    
    # Forward pass
    output = model(x)
    loss = mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

### Available Modules

- `Tensor` - PyTorch-like tensor with autograd
- `Linear` - Fully connected layer
- `Conv2d` - 2D convolution layer
- `MaxPool2d` - 2D max pooling
- `AvgPool2d` - 2D average pooling
- `LayerNorm` - Layer normalization
- `MultiheadAttention` - Multi-head attention
- `FeedForward` - Feed-forward network
- `Sequential` - Sequential container
- `SGD` - SGD optimizer

### Running Examples

```bash
python example_usage.py
```
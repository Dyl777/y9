"""
PyTorch-like interface for the y9 C backend
"""
import numpy as np
import ctypes
import os
import atexit
from typing import List, Optional, Union

# Load the shared library
_lib_path = os.path.join(os.path.dirname(__file__), 'y9_wrapper.dll')
if not os.path.exists(_lib_path):
    _lib_path = os.path.join(os.path.dirname(__file__), 'y9_wrapper.so')
if not os.path.exists(_lib_path):
    _lib_path = os.path.join(os.path.dirname(__file__), 'y9_wrapper.dylib')

_lib = None
try:
    _lib = ctypes.CDLL(_lib_path)
    
    # Define function signatures
    _lib.py_arena_init.argtypes = []
    _lib.py_arena_init.restype = None
    
    _lib.py_arena_cleanup.argtypes = []
    _lib.py_arena_cleanup.restype = None
    
    _lib.py_tensor_create.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    _lib.py_tensor_create.restype = ctypes.c_void_p
    
    _lib.py_tensor_zeros.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _lib.py_tensor_zeros.restype = ctypes.c_void_p
    
    _lib.py_tensor_randn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _lib.py_tensor_randn.restype = ctypes.c_void_p
    
    _lib.py_tensor_set_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    _lib.py_tensor_set_data.restype = None
    
    _lib.py_tensor_ndim.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_ndim.restype = ctypes.c_int
    
    _lib.py_tensor_numel.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_numel.restype = ctypes.c_int
    
    _lib.py_tensor_shape.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_shape.restype = ctypes.POINTER(ctypes.c_int)
    
    _lib.py_tensor_data.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_data.restype = ctypes.POINTER(ctypes.c_float)
    
    _lib.py_tensor_requires_grad.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_requires_grad.restype = ctypes.c_int
    
    _lib.py_tensor_set_requires_grad.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.py_tensor_set_requires_grad.restype = None
    
    _lib.py_tensor_grad.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_grad.restype = ctypes.POINTER(ctypes.c_float)
    
    _lib.py_tensor_backward.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_backward.restype = None
    
    _lib.py_tensor_zero_grad.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_zero_grad.restype = None
    
    _lib.py_tensor_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_tensor_matmul.restype = ctypes.c_void_p
    
    _lib.py_tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_tensor_add.restype = ctypes.c_void_p
    
    _lib.py_tensor_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_tensor_sub.restype = ctypes.c_void_p
    
    _lib.py_tensor_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_tensor_mul.restype = ctypes.c_void_p
    
    _lib.py_tensor_relu.argtypes = [ctypes.c_void_p]
    _lib.py_tensor_relu.restype = ctypes.c_void_p
    
    _lib.py_tensor_mse_loss.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_tensor_mse_loss.restype = ctypes.c_void_p
    
    _lib.py_conv2d_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib.py_conv2d_create.restype = ctypes.c_void_p
    
    _lib.py_conv2d_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_conv2d_forward.restype = ctypes.c_void_p
    
    _lib.py_maxpool2d_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _lib.py_maxpool2d_create.restype = ctypes.c_void_p
    
    _lib.py_avgpool2d_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _lib.py_avgpool2d_create.restype = ctypes.c_void_p
    
    _lib.py_pool2d_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_pool2d_forward.restype = ctypes.c_void_p
    
    _lib.py_layernorm_create.argtypes = [ctypes.c_int]
    _lib.py_layernorm_create.restype = ctypes.c_void_p
    
    _lib.py_layernorm_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_layernorm_forward.restype = ctypes.c_void_p
    
    _lib.py_ffn_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _lib.py_ffn_create.restype = ctypes.c_void_p
    
    _lib.py_ffn_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_ffn_forward.restype = ctypes.c_void_p
    
    _lib.py_mha_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib.py_mha_create.restype = ctypes.c_void_p
    
    _lib.py_mha_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.py_mha_forward.restype = ctypes.c_void_p
    
    _lib.py_sgd_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
    _lib.py_sgd_step.restype = None
    
    _lib.py_clip_gradient.argtypes = [ctypes.c_void_p]
    _lib.py_clip_gradient.restype = None
    
    # Initialize arena
    _lib.py_arena_init()
    
    # Register cleanup
    atexit.register(_lib.py_arena_cleanup)
    
except OSError:
    print("Warning: y9 C backend not available. Build the wrapper first.")


class Tensor:
    """PyTorch-like Tensor class wrapping the C backend"""
    
    def __init__(self, data=None, shape=None, requires_grad=False, _tensor=None):
        if _tensor is not None:
            # Wrap existing C tensor
            self._tensor = _tensor
        elif data is not None:
            # Create from numpy array or list
            if isinstance(data, np.ndarray):
                data_flat = data.astype(np.float32).flatten()
                shape = list(data.shape) if hasattr(data, 'shape') else [len(data)]
            elif isinstance(data, list):
                # Flatten nested lists
                def flatten(lst):
                    for item in lst:
                        if isinstance(item, list):
                            yield from flatten(item)
                        else:
                            yield item
                data_flat = np.array(list(flatten(data)), dtype=np.float32)
                if shape is None:
                    # Infer shape from nested list structure
                    def get_shape(lst):
                        if isinstance(lst, list):
                            return [len(lst)] + get_shape(lst[0]) if lst else []
                        return []
                    shape = get_shape(data)
            else:
                data_flat = np.array([float(data)], dtype=np.float32)
                shape = [1]
            
            if _lib is None:
                raise RuntimeError("C backend not available")
            
            # Create tensor
            shape_arr = (ctypes.c_int * len(shape))(*shape)
            self._tensor = _lib.py_tensor_create(shape_arr, len(shape), int(requires_grad))
            
            # Set data
            data_arr = (ctypes.c_float * len(data_flat))(*data_flat)
            _lib.py_tensor_set_data(self._tensor, data_arr, len(data_flat))
        elif shape is not None:
            # Create tensor with given shape
            if _lib is None:
                raise RuntimeError("C backend not available")
            shape_arr = (ctypes.c_int * len(shape))(*shape)
            self._tensor = _lib.py_tensor_zeros(shape_arr, len(shape))
        else:
            raise ValueError("Either data, shape, or _tensor must be provided")
    
    @property
    def shape(self):
        if _lib is None:
            raise RuntimeError("C backend not available")
        shape_ptr = _lib.py_tensor_shape(self._tensor)
        ndim = _lib.py_tensor_ndim(self._tensor)
        return tuple([shape_ptr[i] for i in range(ndim)])
    
    @property
    def ndim(self):
        if _lib is None:
            raise RuntimeError("C backend not available")
        return _lib.py_tensor_ndim(self._tensor)
    
    @property
    def numel(self):
        if _lib is None:
            raise RuntimeError("C backend not available")
        return _lib.py_tensor_numel(self._tensor)
    
    @property
    def requires_grad(self):
        if _lib is None:
            raise RuntimeError("C backend not available")
        return bool(_lib.py_tensor_requires_grad(self._tensor))
    
    @requires_grad.setter
    def requires_grad(self, value):
        if _lib is None:
            raise RuntimeError("C backend not available")
        _lib.py_tensor_set_requires_grad(self._tensor, int(value))
    
    @property
    def grad(self):
        if _lib is None:
            raise RuntimeError("C backend not available")
        grad_ptr = _lib.py_tensor_grad(self._tensor)
        if grad_ptr:
            numel = self.numel
            grad_data = np.ctypeslib.as_array(ctypes.cast(grad_ptr, ctypes.POINTER(ctypes.c_float)), shape=(numel,))
            return Tensor(data=np.array(grad_data, dtype=np.float32), shape=self.shape, _tensor=None)
        return None
    
    def backward(self):
        if _lib is None:
            raise RuntimeError("C backend not available")
        _lib.py_tensor_backward(self._tensor)
    
    def zero_grad(self):
        if _lib is None:
            raise RuntimeError("C backend not available")
        _lib.py_tensor_zero_grad(self._tensor)
    
    def numpy(self):
        """Convert to numpy array"""
        if _lib is None:
            raise RuntimeError("C backend not available")
        data_ptr = _lib.py_tensor_data(self._tensor)
        if data_ptr:
            numel = self.numel
            data = np.ctypeslib.as_array(ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_float)), shape=(numel,))
            return data.reshape(self.shape).copy()
        return np.array([], dtype=np.float32)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    # Operator overloading
    def __add__(self, other):
        if _lib is None:
            raise RuntimeError("C backend not available")
        if isinstance(other, Tensor):
            result_ptr = _lib.py_tensor_add(self._tensor, other._tensor)
            return Tensor(_tensor=result_ptr)
        elif isinstance(other, (int, float)):
            # Scalar addition
            result = Tensor(shape=self.shape, requires_grad=self.requires_grad)
            data = self.numpy()
            if data is not None and len(data) > 0:
                result._set_data_from_numpy(data + other)
            return result
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if _lib is None:
            raise RuntimeError("C backend not available")
        if isinstance(other, Tensor):
            result_ptr = _lib.py_tensor_sub(self._tensor, other._tensor)
            return Tensor(_tensor=result_ptr)
        elif isinstance(other, (int, float)):
            result = Tensor(shape=self.shape, requires_grad=self.requires_grad)
            data = self.numpy()
            if data is not None and len(data) > 0:
                result._set_data_from_numpy(data - other)
            return result
        return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(shape=self.shape, requires_grad=self.requires_grad)
            data = self.numpy()
            if data is not None and len(data) > 0:
                result._set_data_from_numpy(other - data)
            return result
        return NotImplemented
    
    def __mul__(self, other):
        if _lib is None:
            raise RuntimeError("C backend not available")
        if isinstance(other, Tensor):
            result_ptr = _lib.py_tensor_mul(self._tensor, other._tensor)
            return Tensor(_tensor=result_ptr)
        elif isinstance(other, (int, float)):
            result = Tensor(shape=self.shape, requires_grad=self.requires_grad)
            data = self.numpy()
            if data is not None and len(data) > 0:
                result._set_data_from_numpy(data * other)
            return result
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __matmul__(self, other):
        if _lib is None:
            raise RuntimeError("C backend not available")
        if isinstance(other, Tensor):
            result_ptr = _lib.py_tensor_matmul(self._tensor, other._tensor)
            return Tensor(_tensor=result_ptr)
        return NotImplemented
    
    def _set_data_from_numpy(self, data):
        """Helper to set tensor data from numpy array"""
        if _lib is None:
            raise RuntimeError("C backend not available")
        data_flat = data.astype(np.float32).flatten()
        data_arr = (ctypes.c_float * len(data_flat))(*data_flat)
        _lib.py_tensor_set_data(self._tensor, data_arr, len(data_flat))
    
    @staticmethod
    def zeros(shape):
        """Create tensor filled with zeros"""
        if _lib is None:
            raise RuntimeError("C backend not available")
        shape_arr = (ctypes.c_int * len(shape))(*shape)
        tensor_ptr = _lib.py_tensor_zeros(shape_arr, len(shape))
        return Tensor(_tensor=tensor_ptr)
    
    @staticmethod
    def ones(shape):
        """Create tensor filled with ones"""
        t = Tensor.zeros(shape)
        data = t.numpy()
        if data is not None and len(data) > 0:
            t._set_data_from_numpy(np.ones_like(data))
        return t
    
    @staticmethod
    def randn(shape):
        """Create tensor with random normal values"""
        if _lib is None:
            raise RuntimeError("C backend not available")
        shape_arr = (ctypes.c_int * len(shape))(*shape)
        tensor_ptr = _lib.py_tensor_randn(shape_arr, len(shape))
        return Tensor(_tensor=tensor_ptr)
    
    @staticmethod
    def from_numpy(array):
        """Create tensor from numpy array"""
        return Tensor(data=array)


class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def parameters(self):
        """Return an iterator over module parameters"""
        for param in self._parameters.values():
            yield param
    
    def named_parameters(self):
        """Return an iterator over module parameters, yielding both name and parameter"""
        for name, param in self._parameters.items():
            yield name, param
    
    def train(self):
        """Set the module in training mode"""
        pass
    
    def eval(self):
        """Set the module in evaluation mode"""
        pass


class Linear(Module):
    """Linear layer (fully connected)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight tensor
        weight_shape = [in_features, out_features]
        self.weight = Tensor.randn(weight_shape)
        self.weight.requires_grad = True
        self._parameters['weight'] = self.weight
        
        if bias:
            self.bias = Tensor.zeros([out_features])
            self.bias.requires_grad = True
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, in_features]
        # weight: [in_features, out_features]
        # result: [batch_size, out_features]
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2d(Module):
    """2D Convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        self._conv = _lib.py_conv2d_create(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x: Tensor) -> Tensor:
        if _lib is None:
            raise RuntimeError("C backend not available")
        result_ptr = _lib.py_conv2d_forward(self._conv, x._tensor)
        return Tensor(_tensor=result_ptr)


class MaxPool2d(Module):
    """2D Max pooling layer"""
    
    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        self._pool = _lib.py_maxpool2d_create(kernel_size, self.stride)
    
    def forward(self, x: Tensor) -> Tensor:
        if _lib is None:
            raise RuntimeError("C backend not available")
        result_ptr = _lib.py_pool2d_forward(self._pool, x._tensor)
        return Tensor(_tensor=result_ptr)


class AvgPool2d(Module):
    """2D Average pooling layer"""
    
    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        self._pool = _lib.py_avgpool2d_create(kernel_size, self.stride)
    
    def forward(self, x: Tensor) -> Tensor:
        if _lib is None:
            raise RuntimeError("C backend not available")
        result_ptr = _lib.py_pool2d_forward(self._pool, x._tensor)
        return Tensor(_tensor=result_ptr)


class LayerNorm(Module):
    """Layer Normalization"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        self._ln = _lib.py_layernorm_create(normalized_shape)
    
    def forward(self, x: Tensor) -> Tensor:
        if _lib is None:
            raise RuntimeError("C backend not available")
        result_ptr = _lib.py_layernorm_forward(self._ln, x._tensor)
        return Tensor(_tensor=result_ptr)


class MultiheadAttention(Module):
    """Multi-Head Attention layer"""
    
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        self._mha = None
        self._seq_len = None
    
    def _ensure_mha(self, seq_len: int):
        if self._mha is None or self._seq_len != seq_len:
            self._seq_len = seq_len
            self._mha = _lib.py_mha_create(self.num_heads, seq_len, self.embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        seq_len = x.shape[0] if self.batch_first else x.shape[1]
        self._ensure_mha(seq_len)
        
        result_ptr = _lib.py_mha_forward(self._mha, x._tensor)
        return Tensor(_tensor=result_ptr)


class FeedForward(Module):
    """Feed-forward network (2-layer MLP)"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        self._ffn = _lib.py_ffn_create(input_dim, hidden_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        if _lib is None:
            raise RuntimeError("C backend not available")
        result_ptr = _lib.py_ffn_forward(self._ffn, x._tensor)
        return Tensor(_tensor=result_ptr)


class Sequential(Module):
    """Sequential container for modules"""
    
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


# Functional API
def relu(x: Tensor) -> Tensor:
    """ReLU activation function"""
    if _lib is None:
        raise RuntimeError("C backend not available")
    result_ptr = _lib.py_tensor_relu(x._tensor)
    return Tensor(_tensor=result_ptr)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss"""
    if _lib is None:
        raise RuntimeError("C backend not available")
    result_ptr = _lib.py_tensor_mse_loss(pred._tensor, target._tensor)
    return Tensor(_tensor=result_ptr)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication"""
    if _lib is None:
        raise RuntimeError("C backend not available")
    result_ptr = _lib.py_tensor_matmul(a._tensor, b._tensor)
    return Tensor(_tensor=result_ptr)


# Optimizer
class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        """Perform a single optimization step"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out the gradients of all parameters"""
        for param in self.params:
            if param.requires_grad:
                param.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, params, lr: float, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
    
    def step(self):
        """Perform a single optimization step"""
        if _lib is None:
            raise RuntimeError("C backend not available")
        
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                _lib.py_clip_gradient(param._tensor)
                _lib.py_sgd_step(param._tensor, ctypes.c_float(self.lr))

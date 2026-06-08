"""
Example usage of y9 PyTorch-like interface
"""
import numpy as np
from y9 import Tensor, Linear, Conv2d, MaxPool2d, LayerNorm, FeedForward, Sequential, relu, mse_loss, SGD


def example_linear():
    """Example with Linear layer"""
    print("=== Linear Layer Example ===")
    
    # Create input tensor
    x = Tensor.randn([32, 128])  # batch_size=32, input_dim=128
    x.requires_grad = True
    
    # Create linear layer
    linear = Linear(128, 64)
    
    # Forward pass
    output = linear(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Create target and compute loss
    target = Tensor.randn([32, 64])
    loss = mse_loss(output, target)
    print(f"Loss: {loss.numpy().mean()}")
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer = SGD(linear.parameters(), lr=0.001)
    optimizer.step()
    optimizer.zero_grad()
    
    print()


def example_conv2d():
    """Example with Conv2D layer"""
    print("=== Conv2D Layer Example ===")
    
    # Create input tensor (batch_size=4, channels=3, height=32, width=32)
    x = Tensor.randn([4, 3, 32, 32])
    x.requires_grad = True
    
    # Create Conv2D layer
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    
    # Forward pass
    output = conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Create target and compute loss
    target_shape = output.shape
    target = Tensor.randn(target_shape)
    loss = mse_loss(output, target)
    print(f"Loss: {loss.numpy().mean()}")
    
    # Backward pass
    loss.backward()
    
    print()


def example_cnn():
    """Example with a simple CNN"""
    print("=== CNN Example ===")
    
    # Create input tensor
    x = Tensor.randn([4, 3, 32, 32])
    x.requires_grad = True
    
    # Build a simple CNN
    model = Sequential(
        Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        relu,
        MaxPool2d(kernel_size=2, stride=2),
        Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        relu,
        MaxPool2d(kernel_size=2, stride=2),
    )
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Create target and compute loss
    target = Tensor.randn(output.shape)
    loss = mse_loss(output, target)
    print(f"Loss: {loss.numpy().mean()}")
    
    # Backward pass
    loss.backward()
    
    print()


def example_transformer_block():
    """Example with a transformer-like block"""
    print("=== Transformer Block Example ===")
    
    # Create input tensor (seq_len=128, embed_dim=32)
    x = Tensor.randn([128, 32])
    x.requires_grad = True
    
    # Build a simple transformer block
    model = Sequential(
        LayerNorm(32),
        FeedForward(32, 64),
        LayerNorm(32),
    )
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Create target and compute loss
    target = Tensor.randn(output.shape)
    loss = mse_loss(output, target)
    print(f"Loss: {loss.numpy().mean()}")
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer = SGD(model.parameters(), lr=0.001)
    optimizer.step()
    optimizer.zero_grad()
    
    print()


def example_training_loop():
    """Example training loop"""
    print("=== Training Loop Example ===")
    
    # Create a simple model
    model = Sequential(
        Linear(128, 64),
        relu,
        Linear(64, 32),
    )
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 3
    batch_size = 32
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch in range(10):  # 10 batches per epoch
            # Create random batch
            x = Tensor.randn([batch_size, 128])
            x.requires_grad = True
            target = Tensor.randn([batch_size, 32])
            
            # Forward pass
            output = model(x)
            
            # Compute loss
            loss = mse_loss(output, target)
            total_loss += loss.numpy().mean()
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / 10
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print()


def example_tensor_operations():
    """Example with tensor operations"""
    print("=== Tensor Operations Example ===")
    
    # Create tensors
    a = Tensor.randn([4, 4])
    b = Tensor.randn([4, 4])
    a.requires_grad = True
    b.requires_grad = True
    
    # Operations
    c = a + b
    d = a - b
    e = a * b
    f = a @ b  # Matrix multiplication
    
    print(f"a shape: {a.shape}")
    print(f"b shape: {b.shape}")
    print(f"a + b shape: {c.shape}")
    print(f"a - b shape: {d.shape}")
    print(f"a * b shape: {e.shape}")
    print(f"a @ b shape: {f.shape}")
    
    # ReLU
    g = relu(a)
    print(f"relu(a) shape: {g.shape}")
    
    print()


if __name__ == "__main__":
    print("y9 PyTorch-like Interface Examples\n")
    
    try:
        example_tensor_operations()
        example_linear()
        example_conv2d()
        example_cnn()
        example_transformer_block()
        example_training_loop()
        
        print("All examples completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install the package with: pip install -e .")

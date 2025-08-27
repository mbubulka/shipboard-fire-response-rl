"""Basic PyTorch test"""
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Simple tensor operation
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Sum:", x.sum().item())
print("✅ PyTorch working correctly")

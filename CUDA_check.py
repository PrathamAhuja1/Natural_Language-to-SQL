import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current CUDA Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())


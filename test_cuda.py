import torch

# 打印可用的GPU设备数量
print(torch.cuda.device_count())

# 打印是否可以使用CUDA，即是否可以在GPU上运行计算
print(torch.cuda.is_available())

# 打印是否可以使用cuDNN，这是一个用于深度神经网络的库，它提供了优化的计算和内存访问模式
print(torch.backends.cudnn.is_available)

# 打印CUDA的版本号
print(torch.cuda_version)

# 打印cuDNN的版本号
print(torch.backends.cudnn.version())
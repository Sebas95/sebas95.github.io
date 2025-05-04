import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

tensor_transform = transforms.ToTensor()
normalization_transform = transforms.Normalize((0.5,), (0.5,))
transform = transforms.Compose([tensor_transform, normalization_transform])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Check if CUDA is available
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
    
print(f"Using device: {device}")

for x_train, y_train in train_loader:
    
	x_train = x_train.to(device)
	y_train = y_train.to(device)
    
	print(x_train.shape)
	print(y_train.shape)
	break

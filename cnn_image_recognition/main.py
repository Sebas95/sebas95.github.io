from torchvision import datasets, transforms


tensor_transform = transforms.ToTensor()
normalization_transform = transforms.Normalize((0.5,), (0.5,))
transform = transforms.Compose([tensor_transform, normalization_transform])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



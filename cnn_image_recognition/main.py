import torch
import os 

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import datetime

# Local imports
from conv_net import ConvNet

model_path = './checkpoints/persistedmodel.pth'

# Retrieve the data set 
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


def train_loop(model, criterion):
	num_epochs = 100
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
		
	losses = []
	epoch_number = []
	for epoch in range(num_epochs):
		epoch_losses = []
		for x_train, y_train in train_loader:
		
			x_train = x_train.to(device)
			y_train = y_train.to(device)
			
			# Clean up old gradientes
			optimizer.zero_grad()
			
			# Forward pass
			outputs = model(x_train)
		    
			# Compute loss
			loss = criterion(outputs, y_train)
		    
			# Backward pass
			loss.backward()
		    
			# Weigth update
			optimizer.step()
		    
			# Optics collection
			epoch_losses.append(loss)
		
		# Get the lost of last epoch 
		epoch_loss = torch.stack(epoch_losses).mean().item()  
		losses.append(epoch_loss)
		epoch_number.append(epoch)
		
		print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")
		
	return model
	

def test_loop(model, criterion):
	
	model.eval()  # Set model to evaluation mode
	test_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for inputs, labels in test_loader:
			inputs, labels = inputs.to(device), labels.to(device)
		        
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			test_loss += loss.item() * inputs.size(0)
		        
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		avg_loss = test_loss / total
		accuracy = correct / total

		print(f'Test Loss: {avg_loss:.4f} | Accuracy: {accuracy * 100:.2f}%')


# Start the training loop
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
	
if os.path.exists(model_path):
	model.load_state_dict(torch.load(model_path))
else:
	model = train_loop(model, criterion)
	torch.save(model.state_dict(), f'./checkpoints/persistedmodel.pth')
	
# Test loop
test_loop(model, criterion)


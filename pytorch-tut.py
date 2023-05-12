import torch
import numpy as np

# Create a tensor from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# Create a tensor from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Create a tensor from another tensor
x_ones = torch.ones_like(x_data) # Copies properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # Overrides the data type
print(f"Random Tensor: \n {x_rand} \n")

# Set tensor dimensions using a tuple of dimensions
shape = (2,3) # A 2 X 3 shape, a list containing 2 lists, each containing 3 items.
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# By default, tensors are created on the CPU.
# We need to explicitly move tensors to the GPU using .to() method (after checking for GPU availability).
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = ones_tensor.to("cuda")
    
# Datasets and Dataloaders
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Load training data examples from TorchVision
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Load test data examples from Torchvision
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Use Matplotlib to iterate and visualize datasets
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the Dataloader

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

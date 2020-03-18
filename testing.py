import AutoEncoder
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# Getting random sample from testing set
to_tensor = torchvision.transforms.ToTensor()
test_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=to_tensor)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
sample = next(iter(test_dataloader))[0]

# Utility for reshaping the images in a format
# That the matplotlib library can plot.
def reshape_rgb(to_reshape):
    """
    Converts an RGB image with shape (3, M, N)
    to shape (M, N, 3)
    """
    M = to_reshape.shape[1]
    N = to_reshape.shape[2]
    reshaped = np.zeros((M, N, 3))
    for i in range(M):
        for j in range(N):
            for k in range(3):
                reshaped[i][j][k] = to_reshape[k][i][j]
    return reshaped

# Displaying original sample image
img1 = reshape_rgb(sample.numpy()[0])
fig, axes = plt.subplots(3, 1)
axes[0].imshow(img1)

# Loading AutoEncoder
net = AutoEncoder.AutoEncoder()
loaded = torch.load('neuralnet', map_location=torch.device('cpu'))
net.load_state_dict(loaded)
net.eval()

# Encoding image and displaying it
encoded = net.encode(sample)
img2 = reshape_rgb(encoded.detach().numpy()[0])
axes[1].imshow(img2)

# Decoding image and displaying it
decoded = net.decode(encoded)
img3 = reshape_rgb(decoded.detach().numpy()[0])
axes[2].imshow(img3)

# Calculating and printing loss
criterion = nn.MSELoss()
print("Calculated loss: {:3.6f}".format(float(criterion(decoded, sample))))

axes[0].title.set_text('3 Channel Original image (32x32)')
axes[1].title.set_text('3 Channel Encoded image (15x15)')
axes[2].title.set_text('3 Channel Recovered image (32x32)')

axes[0].set_yticks([])
axes[0].set_xticks([])
axes[1].set_yticks([])
axes[1].set_xticks([])
axes[2].set_yticks([])
axes[2].set_xticks([])

plt.show()

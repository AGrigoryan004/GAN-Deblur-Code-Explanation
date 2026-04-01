import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import ImageFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



class BlurImage:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


transform_clean = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

transform_blur = transforms.Compose([
    transforms.Resize((96, 96)),
    BlurImage(radius=2),
    transforms.ToTensor()
])


class DeblurSTL10(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.clean_dataset = torchvision.datasets.STL10(
            root="./data",
            split=split,
            download=True,
            transform=transform_clean
        )
        self.blur_dataset = torchvision.datasets.STL10(
            root="./data",
            split=split,
            download=True,
            transform=transform_blur
        )

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        clean_img, label = self.clean_dataset[idx]
        blur_img, _ = self.blur_dataset[idx]
        return blur_img, clean_img


train_dataset = DeblurSTL10(split="train")
test_dataset = DeblurSTL10(split="test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetDeblur(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        out = self.out_act(out)
        return out

model = UNetDeblur().to(device)

l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def combined_loss(output, target):
    return 0.8 * l1_loss(output, target) + 0.2 * mse_loss(output, target)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for blurred, clean in train_loader:
        blurred = blurred.to(device)
        clean = clean.to(device)

        output = model(blurred)
        loss = combined_loss(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


model.eval()

blurred, clean = next(iter(test_loader))
blurred = blurred.to(device)
clean = clean.to(device)

with torch.no_grad():
    restored = model(blurred)

blurred = blurred.cpu()
clean = clean.cpu()
restored = restored.cpu()

fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i in range(5):
    axes[0, i].imshow(clean[i].permute(1, 2, 0))
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(blurred[i].permute(1, 2, 0))
    axes[1, i].set_title("Blurred")
    axes[1, i].axis("off")

    axes[2, i].imshow(restored[i].permute(1, 2, 0))
    axes[2, i].set_title("Restored")
    axes[2, i].axis("off")

plt.tight_layout()
plt.show()


torch.save(model.state_dict(), "deblur_unet_stl10.pth")
print("Model saved as deblur_unet_stl10.pth")

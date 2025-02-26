import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# 配置参数
hr_dir = "Flickr2K_HR"
lr_dir = "Flickr2K_LR_bicubic"
scale = "x2"  # 可选 "x2", "x3", "x4"
image_size = 128
batch_size = 32
epochs = 5
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 自定义数据集
class Flickr2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = os.path.join(lr_dir, scale)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.hr_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.image_files[idx])
        # 拼接低分辨率图像路径
        lr_image_name = self.image_files[idx].replace('.png', f'{scale}.png')
        lr_image_path = os.path.join(self.lr_dir, lr_image_name)
        
        # 检查文件是否存在，不存在则跳过
        if not os.path.exists(lr_image_path):
            print(f"文件未找到：{lr_image_path}")
            return self.__getitem__((idx + 1) % len(self.image_files))
        
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        return lr_image, hr_image

# 加载数据集
dataset = Flickr2KDataset(hr_dir, lr_dir, scale, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 自定义 CNN 模型
class CustomSRCNN(nn.Module):
    def __init__(self):
        super(CustomSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# 初始化模型
model = CustomSRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
# 训练模型
def train():
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0  # 每个 epoch 开始前清零
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # 记录损失
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    # 绘制损失曲线
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(list(range(1, epochs+1)), loss_history, label='训练损失')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

# 模型评估
# 修改 evaluate 函数
def evaluate():
    model.eval()
    psnr_values, ssim_values = [], []
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs).cpu().numpy()
            hr_imgs = hr_imgs.cpu().numpy()
            
            for i in range(len(outputs)):
                # 还原像素值到 [0, 1]
                output = (outputs[i].transpose(1,2,0) + 1) / 2.0
                hr_image = (hr_imgs[i].transpose(1,2,0) + 1) / 2.0
                
                # 获取图像尺寸
                h, w = output.shape[:2]
                win_size = min(h, w, 7)  # 确保 win_size 不超过图像最小尺寸
                
                # 计算 PSNR 和 SSIM
                psnr_values.append(psnr(hr_image, output, data_range=1))
                ssim_values.append(ssim(hr_image, 
                                        output, 
                                        win_size=win_size,
                                        channel_axis=-1,
                                        data_range=1))
    
    print(f"平均 PSNR: {np.mean(psnr_values):.2f}, 平均 SSIM: {np.mean(ssim_values):.4f}")

if __name__ == "__main__":
    train()
    evaluate()

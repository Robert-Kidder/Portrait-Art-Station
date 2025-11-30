
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

# ==========================================
# Part 1: PyTorch Fast Neural Style 模型架构定义
# (必须包含这个类，否则无法加载官方的 .pth 权重)
# ==========================================
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


# ==========================================
# Part 2: 核心处理函数 (Style Transfer + Segmentation)
# ==========================================

def portrait_style_transfer(
    content_img: Image.Image, 
    model_path: str, 
    use_gpu: bool = False
) -> Image.Image:
    """
    执行语义感知的风格迁移。
    
    Args:
        content_img: PIL Image, 用户上传的原图
        model_path: str, .pth 模型文件的路径
        use_gpu: bool, 是否使用 CUDA
        
    Returns:
        PIL Image: 处理后的最终融合图像
    """
    
    # ---------------------------
    # 1. 准备工作：尺寸与设备
    # ---------------------------
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # 将 PIL 转换为 Numpy (用于 MediaPipe)
    img_np = np.array(content_img)
    original_h, original_w, _ = img_np.shape
    
    # ---------------------------
    # 2. 步骤 A (分割): 使用 MediaPipe 提取人像 Mask
    # ---------------------------
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # MediaPipe 需要 RGB 输入 (PIL 默认是 RGB，np.array 也是 RGB)
        results = selfie_segmentation.process(img_np)
        
        # 获取分割掩码 (Mask)，值为 0.0 ~ 1.0 (float32)
        # 1.0 代表肯定是人，0.0 代表肯定是背景
        mask = results.segmentation_mask
        
    if mask is None:
        print("未检测到人像，返回全图风格化结果")
        # 如果没检测到人，就把 Mask 全设为 0 (全背景)
        mask = np.zeros((original_h, original_w), dtype=np.float32)

    # ---------------------------
    # 3. 步骤 B (风格化): PyTorch 推理
    # ---------------------------
    # 加载风格迁移模型
    style_model = TransformerNet()
    
    state_dict = torch.load(model_path, map_location=device)

    for key in list(state_dict.keys()):
        if 'running_mean' in key or 'running_var' in key:
            del state_dict[key]
            
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    style_model.eval()

    # 预处理图像：PIL -> Tensor
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)) # 许多预训练模型需要 0-255 的输入范围
    ])
    content_tensor = content_transform(content_img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output_tensor = style_model(content_tensor)

    # 后处理：Tensor -> Numpy
    output_tensor = output_tensor.cpu().squeeze(0)
    output_tensor = output_tensor.clamp(0, 255).numpy()
    output_tensor = output_tensor.transpose(1, 2, 0).astype("uint8") # (C,H,W) -> (H,W,C)
    
    # ---------------------------
    # 4. 关键步骤：尺寸对齐 (Resize)
    # ---------------------------
    # 风格迁移网络可能会因为 Padding 或 stride 导致输出尺寸与原图有细微差异
    # 或者用户为了加速，之前缩小了原图。
    # 强制将“风格图”和“Mask”都调整回“原图”的大小。
    stylized_img_resized = cv2.resize(output_tensor, (original_w, original_h))
    
    # ---------------------------
    # 5. 步骤 C (融合): Alpha Blending
    # ---------------------------
    
    # 优化点：对 Mask 进行高斯模糊 (羽化)，避免边缘锯齿
    # Stack Mask to 3 channels to match image shape (H, W, 3)
    mask_3d = np.stack((mask,) * 3, axis=-1)
    
    # 模糊 Mask 边缘
    mask_3d = cv2.GaussianBlur(mask_3d, (21, 21), 0) 
    
    # 融合公式：
    # Result = (人像原图 * Mask) + (风格化背景 * (1 - Mask))
    # img_np 是 uint8，计算前转为 float 以防溢出
    img_original_float = img_np.astype(np.float32)
    img_stylized_float = stylized_img_resized.astype(np.float32)
    
    final_image = (img_original_float * mask_3d) + (img_stylized_float * (1.0 - mask_3d))
    
    # 转回 uint8 并生成 PIL Image
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    return Image.fromarray(final_image)
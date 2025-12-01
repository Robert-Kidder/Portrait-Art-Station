import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import gc # 引入垃圾回收

# ==========================================
# Part 1: 模型定义 (保持不变)
# ==========================================
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
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
# Part 2: 核心处理函数 (增强抗压能力)
# ==========================================

def load_optimized_model(model_path, device):
    """
    辅助函数：加载并量化模型，大幅降低内存占用
    """
    model = TransformerNet()
    state_dict = torch.load(model_path, map_location=device)
    for key in list(state_dict.keys()):
        if 'running_mean' in key or 'running_var' in key:
            del state_dict[key]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # ⚡ 动态量化：将 FP32 转为 INT8，内存占用减半，CPU 推理提速
    if device.type == 'cpu':
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Conv2d, torch.nn.InstanceNorm2d}, dtype=torch.qint8
            )
        except Exception:
            pass # 如果量化失败，回退到普通模式
            
    return model

def portrait_style_transfer(content_img: Image.Image, model_path: str, use_gpu: bool = False) -> Image.Image:
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # 再次强制限制尺寸，防止前端漏网之鱼
    if max(content_img.size) > 650:
        content_img.thumbnail((650, 650), Image.Resampling.LANCZOS)

    img_np = np.array(content_img)
    original_h, original_w, _ = img_np.shape
    
    # ---------------------------
    # 1. 分割 (MediaPipe) - 使用上下文管理
    # ---------------------------
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        results = selfie_segmentation.process(img_np)
        mask = results.segmentation_mask
        
    if mask is None:
        mask = np.zeros((original_h, original_w), dtype=np.float32)

    # ---------------------------
    # 2. 风格化 (PyTorch) - 使用优化后的模型加载
    # ---------------------------
    style_model = load_optimized_model(model_path, device)

    from torchvision import transforms
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_tensor = content_transform(content_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = style_model(content_tensor)

    # ⚡ 显式内存释放
    del content_tensor
    del style_model # 立即销毁模型对象
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_tensor = output_tensor.cpu().squeeze(0)
    output_tensor = output_tensor.clamp(0, 255).numpy()
    output_tensor = output_tensor.transpose(1, 2, 0).astype("uint8")
    
    # ---------------------------
    # 3. 融合
    # ---------------------------
    stylized_img_resized = cv2.resize(output_tensor, (original_w, original_h))
    
    mask_3d = np.stack((mask,) * 3, axis=-1)
    mask_3d = cv2.GaussianBlur(mask_3d, (15, 15), 0) # 稍微减小卷积核以提速
    
    img_original_float = img_np.astype(np.float32)
    img_stylized_float = stylized_img_resized.astype(np.float32)
    
    final_image = (img_original_float * mask_3d) + (img_stylized_float * (1.0 - mask_3d))
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    
    gc.collect() # 强制运行垃圾回收
    
    return Image.fromarray(final_image)
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError, ImageFile
import torch
import torchvision.transforms as transforms
import os
import io
import gc  # 引入垃圾回收模块

# 导入工具库
# 确保 style_transfer_utils.py 在同一目录下
from style_transfer_utils import TransformerNet, portrait_style_transfer

# ==========================================
# 0. 全局设置：允许加载截断/不完整的图片
# ==========================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. 页面配置与 CSS
# ==========================================
st.set_page_config(
    page_title="艺术风格迁移实验室",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    [data-testid="stDecoration"] { visibility: hidden; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    h1 { font-weight: 700; color: #333; text-align: center; padding-bottom: 20px; }
    [data-testid="stSidebar"] h1 { text-align: left; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    .stAlert { border-radius: 12px; border: none; background-color: #f8f9fa; border-left: 5px solid #11998e; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心辅助函数 (究极增强版)
# ==========================================

MAX_IMAGE_SIZE = 1000

def load_and_resize_image(image_file, max_size=MAX_IMAGE_SIZE):
    """
    安全加载并缩放图片。
    针对无扩展名的长文件名、WebP、HEIC、截断图片进行了全面防御。
    """
    try:
        # 1. 基础检查
        if image_file is None: return None
        image_file.seek(0)
        file_bytes = image_file.read()
        if len(file_bytes) == 0:
            st.error("⚠️ 错误：上传的文件大小为 0，请重新上传。")
            return None
            
        # 2. 准备 BytesIO 流
        image_stream = io.BytesIO(file_bytes)
        
        # 3. 尝试打开图片 (多策略尝试)
        image = None
        error_msg = ""
        
        # 策略 A: 让 PIL 自动嗅探 (不设置 name，纯靠字节头)
        try:
            image_stream.seek(0)
            image_stream.name = "" # 清空名字，防止干扰
            image = Image.open(image_stream)
            image.load() # 强制读取数据
        except Exception:
            # 策略 B: 强制伪装成 JPG (应对无后缀的 JPG)
            try:
                image_stream.seek(0)
                image_stream.name = "force_detect.jpg"
                image = Image.open(image_stream)
                image.load()
            except Exception:
                # 策略 C: 强制伪装成 PNG
                try:
                    image_stream.seek(0)
                    image_stream.name = "force_detect.png"
                    image = Image.open(image_stream)
                    image.load()
                except Exception as e:
                    error_msg = str(e)
                    image = None

        # 4. 如果所有策略都失败
        if image is None:
            st.error(f"⚠️ 无法识别图片格式。请注意：\n1. 本系统暂不支持 HEIC (iPhone) 格式，请在手机相册设置中改为“兼容性最佳”或截图上传。\n2. 原始报错: {error_msg}")
            return None
        
        # 5. 修复手机拍摄图片的旋转问题 (EXIF Orientation)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass 
        
        # 6. 强制转换为 RGB (去除 Alpha 通道，防止 RGBA 报错)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 7. 计算缩放比例 (防止内存溢出)
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image

    except Exception as e:
        st.error(f"处理图片时发生未知错误: {e}")
        return None

# ==========================================
# 3. 模型加载逻辑
# ==========================================

STYLE_MODELS = {
    "✨ 马赛克 (Mosaic)": "saved_models/mosaic.pth",
    "🍬 糖果世界 (Candy)": "saved_models/candy.pth",
    "☔ 雨之公主 (Rain Princess)": "saved_models/rain_princess.pth",
    "🎨 乌德尼 (Udnie)": "saved_models/udnie.pth"
}

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNet()
    try:
        state_dict = torch.load(model_path, map_location=device)
        # 移除多余的 keys
        for key in list(state_dict.keys()):
            if 'running_mean' in key or 'running_var' in key:
                del state_dict[key]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        return None

def global_style_transfer(content_img, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_model = load_model(model_path)
    if style_model is None: return None

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_tensor = content_transform(content_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = style_model(content_tensor)

    output_tensor = output_tensor.cpu().squeeze(0).clamp(0, 255).numpy()
    output_tensor = output_tensor.transpose(1, 2, 0).astype("uint8")
    
    del content_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return Image.fromarray(output_tensor)

# ==========================================
# 4. 侧边栏
# ==========================================
st.sidebar.title("⚙️ 设置面板")
st.sidebar.markdown("上传图片并选择你喜欢的艺术风格。")

uploaded_file = st.sidebar.file_uploader(
    "1️⃣ 上传一张照片...", 
    type=["jpg", "jpeg", "png", "webp"], # 显式允许 webp
    help="支持 JPG, PNG, WEBP。大图将自动优化。"
)

selected_style_name = st.sidebar.selectbox("2️⃣ 选择艺术风格", list(STYLE_MODELS.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 创新功能")
use_portrait_mode = st.sidebar.checkbox(
    "🛡️ 人像保护模式", value=True,
    help="勾选后，系统将自动识别人物，仅对背景进行风格化。"
)

generate_btn = st.sidebar.button("开始创作 ✨")

# ==========================================
# 5. 主界面
# ==========================================
st.title("艺术风格迁移实验室")
st.markdown("<p style='text-align: center; color: #666; margin-bottom: 30px;'>基于深度语义感知的智能风格迁移系统</p>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👋 欢迎体验！请点击左侧侧边栏 (电脑) 或左上角箭头 (手机) 上传图片。")
    col_spacer1, col_img, col_spacer2 = st.columns([3, 4, 3])
    with col_img:
        local_image_path = "mosaic.jpg"
        if os.path.exists(local_image_path):
            st.image(Image.open(local_image_path), caption="效果预览：马赛克风格", use_container_width=True)
        else:
            st.warning(f"⚠️ 提示：未在当前目录下找到 '{local_image_path}'。")
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🛡️ 人像保护")
        st.caption("智能分割前景人物，拒绝五官乱码与变形。")
    with c2:
        st.markdown("#### ⚡ 极速推理")
        st.caption("毫秒级生成速度，大图自动优化。")
    with c3:
        st.markdown("#### 📱 全端适配")
        st.caption("无论手机还是电脑，随时随地开启创作。")

else:
    # 加载图片
    content_image = load_and_resize_image(uploaded_file)
    
    # 只有当图片成功加载时才继续
    if content_image is not None:
        col_input, col_output = st.columns(2)
        with col_input:
            st.markdown("##### 📸 原始图像")
            st.image(content_image, use_container_width=True)

        if generate_btn:
            model_path = STYLE_MODELS[selected_style_name]
            if not os.path.exists(model_path):
                st.error(f"❌ 模型文件未找到：{model_path}。")
            else:
                with col_output:
                    st.markdown(f"##### 🎨 艺术化结果")
                    status_box = st.empty()
                    progress_bar = st.progress(0)
                    
                    try:
                        if use_portrait_mode:
                            status_box.info("🔍 正在识别人物并融合背景...")
                            progress_bar.progress(30)
                            output_image = portrait_style_transfer(
                                content_image, model_path, use_gpu=torch.cuda.is_available()
                            )
                        else:
                            status_box.info("⚡ 正在进行全局风格渲染...")
                            progress_bar.progress(50)
                            output_image = global_style_transfer(content_image, model_path)
                        
                        progress_bar.progress(100)
                        progress_bar.empty()
                        status_box.success("✨ 生成成功！")
                        st.image(output_image, use_container_width=True)
                        
                        buf = io.BytesIO()
                        output_image.save(buf, format="JPEG", quality=95)
                        byte_im = buf.getvalue()
                        st.download_button(
                            label="📥 保存高清图片", data=byte_im,
                            file_name="art_style_result.jpg", mime="image/jpeg",
                            use_container_width=True
                        )
                        
                        gc.collect()
                        
                    except Exception as e:
                        status_box.error("处理出错，可能是图片过于复杂或内存不足。")
                        st.error(f"Error: {e}")
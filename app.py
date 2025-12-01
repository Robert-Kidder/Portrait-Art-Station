import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError, ImageFile
import torch
import torchvision.transforms as transforms
import os
import io
import gc
import time
import filelock # 需要 pip install filelock，虽然标准库没有，但Streamlit环境通常有，如果没有则使用简易实现

# 导入工具库
from style_transfer_utils import TransformerNet, portrait_style_transfer, load_style_model

# ==========================================
# 0. 全局设置 & 并发控制
# ==========================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ⚡ 极限压缩：为了50人并发，必须牺牲分辨率
# 600px 在手机上看已经足够清晰，且内存占用极低
MAX_IMAGE_SIZE = 600 

# 定义文件锁路径 (实现简单的排队机制)
LOCK_FILE = "gpu_resource.lock"

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
    .stAlert { border-radius: 12px; border: none; background-color: #f8f9fa; border-left: 5px solid #11998e; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 辅助函数：安全图片加载 (极限压缩版)
# ==========================================
def load_and_resize_image(image_file, max_size=MAX_IMAGE_SIZE):
    try:
        if image_file is None: return None
        image_file.seek(0)
        file_bytes = image_file.read()
        if len(file_bytes) == 0: return None
            
        clean_stream = io.BytesIO(file_bytes)
        clean_stream.name = "temp.jpg" # 强制改名，避开长文件名 bug
        
        try:
            image = Image.open(clean_stream)
            image.load()
        except Exception:
            # 备用方案：盲开
            clean_stream.seek(0)
            clean_stream.name = None
            try:
                image = Image.open(clean_stream)
                image.load()
            except:
                st.error("无法解析图片。")
                return None

        try:
            image = ImageOps.exif_transpose(image)
        except: pass
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # ⚡ 强制 Resize：这一步是防崩溃的核心
        # 不管原图多大，进内存前先砍一刀
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        return None

# ==========================================
# 3. 模型加载逻辑 (带缓存与量化)
# ==========================================

STYLE_MODELS = {
    "✨ 马赛克 (Mosaic)": "saved_models/mosaic.pth",
    "🍬 糖果世界 (Candy)": "saved_models/candy.pth",
    "☔ 雨之公主 (Rain Princess)": "saved_models/rain_princess.pth",
    "🎨 乌德尼 (Udnie)": "saved_models/udnie.pth"
}

@st.cache_resource(max_entries=2) # 限制缓存数量，节省内存
def load_cached_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 调用 utils 里的量化加载函数
    return load_style_model(model_path, device)

def global_style_transfer(content_img, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用缓存模型
    style_model = load_cached_model(model_path)
    
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
    gc.collect()
        
    return Image.fromarray(output_tensor)

# ==========================================
# 4. 界面与主逻辑
# ==========================================
st.sidebar.title("⚙️ 设置面板")
uploaded_file = st.sidebar.file_uploader("1️⃣ 上传照片", type=["jpg", "png", "webp"])
selected_style_name = st.sidebar.selectbox("2️⃣ 选择风格", list(STYLE_MODELS.keys()))
st.sidebar.markdown("---")
use_portrait_mode = st.sidebar.checkbox("🛡️ 人像保护模式", value=True)
generate_btn = st.sidebar.button("开始创作 ✨")

# 服务器状态指示
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

st.title("艺术风格迁移实验室")

if uploaded_file:
    content_image = load_and_resize_image(uploaded_file)
    
    if content_image:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📸 原始图像")
            st.image(content_image, use_container_width=True)

        if generate_btn:
            model_path = STYLE_MODELS[selected_style_name]
            if not os.path.exists(model_path):
                st.error("模型丢失")
            else:
                with col2:
                    st.markdown("##### 🎨 艺术化结果")
                    status_place = st.empty()
                    
                    # ------------------------------------------------
                    # 🔒 核心并发锁机制：排队系统
                    # ------------------------------------------------
                    from filelock import FileLock, Timeout
                    lock = FileLock(LOCK_FILE + ".lock")
                    
                    try:
                        status_place.info("⌛ 正在排队等待服务器资源...")
                        
                        # 尝试获取锁，等待最多 15 秒
                        with lock.acquire(timeout=15):
                            status_place.info("🚀 正在处理中... (请勿刷新)")
                            progress_bar = st.progress(0)
                            
                            # 模拟处理延迟，避免瞬间抢占
                            progress_bar.progress(20)
                            
                            if use_portrait_mode:
                                output_image = portrait_style_transfer(
                                    content_image, model_path, use_gpu=False
                                )
                            else:
                                output_image = global_style_transfer(content_image, model_path)
                            
                            progress_bar.progress(100)
                            progress_bar.empty()
                            status_place.success("✨ 完成！")
                            st.image(output_image, use_container_width=True)
                            
                            # 下载按钮
                            buf = io.BytesIO()
                            output_image.save(buf, format="JPEG", quality=85) # 稍微降低质量以加速下载
                            st.download_button("📥 下载图片", buf.getvalue(), "art.jpg", "image/jpeg", use_container_width=True)
                            
                            # 强制回收
                            del output_image
                            gc.collect()

                    except Timeout:
                        status_place.warning("⚠️ 服务器繁忙 (排队人数 > 50)，请等待 10 秒后重试！")
                    except Exception as e:
                        status_place.error(f"处理中断: {str(e)}")
                        gc.collect()
    else:
        st.error("图片无法加载，请重试。")

else:
    # 欢迎页代码保持精简
    st.info("👋 欢迎！由于服务器资源有限，请大家排队上传。")
    if os.path.exists("mosaic.jpg"):
        st.image(Image.open("mosaic.jpg"), width=300)
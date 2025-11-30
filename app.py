import streamlit as st
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import os
import io
import gc  # 引入垃圾回收模块

# 导入工具库
from style_transfer_utils import TransformerNet, portrait_style_transfer

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
# 2. 核心辅助函数 (新增：缩放与安全加载)
# ==========================================

# 🔴 关键修改：限制图片最大尺寸，防止内存溢出
MAX_IMAGE_SIZE = 1000  # 设置最长边为 1000 像素，平衡速度与画质

def load_and_resize_image(image_file, max_size=MAX_IMAGE_SIZE):
    """
    安全加载并缩放图片。
    1. 解决 Image.open 的并发报错 (通过 .copy())
    2. 解决大文件导致的内存崩溃 (通过 resize)
    """
    try:
        image = Image.open(image_file)
        
        # 修复手机上传图片可能出现的旋转问题 (EXIF Orientation)
        image = ImageOps.exif_transpose(image)
        
        # 强制转换为 RGB，防止 RGBA 或 CMYK 导致后续报错
        image = image.convert('RGB')
        
        # 计算缩放比例
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        st.error(f"图片加载失败: {e}")
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
    
    # 🔴 显式清理显存/内存
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
    type=["jpg", "jpeg", "png"],
    help="建议上传包含人物的自拍或生活照，以体验人像保护功能。大图将自动压缩至 1000px。"
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
    # 🔴 使用新的安全加载函数
    content_image = load_and_resize_image(uploaded_file)
    
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
                            label="📥 保存高清大图", data=byte_im,
                            file_name="art_style_result.jpg", mime="image/jpeg",
                            use_container_width=True
                        )
                        
                        # 🔴 运行结束后清理内存
                        gc.collect()
                        
                    except Exception as e:
                        status_box.error("处理出错，可能是图片过于复杂或内存不足。")
                        st.error(f"Error: {e}")
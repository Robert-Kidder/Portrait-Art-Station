import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError, ImageFile
import torch
import torchvision.transforms as transforms
import os
import io
import gc  # 引入垃圾回收模块
from filelock import FileLock, Timeout # 引入文件锁用于排队

# 导入工具库
# 确保 style_transfer_utils.py 在同一目录下
from style_transfer_utils import TransformerNet, portrait_style_transfer, load_optimized_model

# ==========================================
# 0. 全局设置：允许加载截断/不完整的图片
# ==========================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ⚡ 抗压修改：将最大尺寸限制为 600px
# 50人并发下，1000px 会导致内存溢出，600px 是流畅演示的最佳平衡点
MAX_IMAGE_SIZE = 600 

# 定义并发锁文件路径
LOCK_FILE = "processing.lock"

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
# 2. 核心辅助函数 (匿名化处理长文件名 + 尺寸压缩)
# ==========================================

def load_and_resize_image(image_file, max_size=MAX_IMAGE_SIZE):
    """
    安全加载并缩放图片。
    通过创建全新、短命名的 BytesIO 流，彻底解决长文件名导致的报错。
    """
    try:
        if image_file is None: return None
        
        # 1. 读取原始数据的二进制流
        image_file.seek(0)
        file_bytes = image_file.read()
        
        if len(file_bytes) == 0:
            st.error("⚠️ 错误：上传的文件内容为空。")
            return None
            
        # 2. 创建一个新的、干净的内存流
        # 这一步切断了与原始 UploadedFile (及其长文件名) 的联系
        clean_stream = io.BytesIO(file_bytes)
        
        # 3. 【关键步骤】强制赋予一个短的、安全的假名字
        # 无论原图叫什么，PIL 现在只认为它叫 "temp.jpg"
        clean_stream.name = "temp.jpg"
        
        # 4. 尝试打开
        image = None
        try:
            image = Image.open(clean_stream)
            image.load() # 立即解码，测试文件完整性
        except Exception:
            # 如果当做 JPG 失败，尝试当做 PNG
            clean_stream.seek(0)
            clean_stream.name = "temp.png"
            try:
                image = Image.open(clean_stream)
                image.load()
            except Exception:
                # 最后的尝试：不设名字，让 PIL 盲猜
                clean_stream.seek(0)
                clean_stream.name = None 
                try:
                    image = Image.open(clean_stream)
                    image.load()
                except Exception as e:
                    st.error(f"⚠️ 无法解析图片数据。请尝试截图后上传，或转换格式。")
                    return None

        # 5. 修复旋转 (手机照片常见问题)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass 
        
        # 6. 统一转为 RGB (去除 Alpha 通道)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 7. 缩放限制内存 (这里使用的是 600px 的新常量)
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image

    except Exception as e:
        st.error(f"处理图片时发生系统错误: {e}")
        return None

# ==========================================
# 3. 模型加载逻辑 (优化版)
# ==========================================

STYLE_MODELS = {
    "✨ 马赛克 (Mosaic)": "saved_models/mosaic.pth",
    "🍬 糖果世界 (Candy)": "saved_models/candy.pth",
    "☔ 雨之公主 (Rain Princess)": "saved_models/rain_princess.pth",
    "🎨 乌德尼 (Udnie)": "saved_models/udnie.pth"
}

@st.cache_resource(max_entries=2) # 限制缓存数量，节省内存
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 utils 中定义的优化加载函数 (包含动态量化)
    return load_optimized_model(model_path, device)

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
    
    # 显式清理
    del content_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
        
    return Image.fromarray(output_tensor)

# ==========================================
# 4. 侧边栏
# ==========================================
st.sidebar.title("⚙️ 设置面板")
st.sidebar.markdown("上传图片并选择你喜欢的艺术风格。")

uploaded_file = st.sidebar.file_uploader(
    "1️⃣ 上传一张照片...", 
    type=["jpg", "jpeg", "png", "webp"], 
    help="建议上传包含人物的自拍或生活照，以体验人像保护功能。"
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
            st.image(Image.open(local_image_path), caption="效果预览：马赛克风格", width=True)
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
    # 核心修改：先安全加载图片
    content_image = load_and_resize_image(uploaded_file)
    
    # 只有当 content_image 成功变为 PIL 对象后，才渲染界面
    if content_image is not None:
        col_input, col_output = st.columns(2)
        with col_input:
            st.markdown("##### 📸 原始图像")
            st.image(content_image, width=True)

        if generate_btn:
            model_path = STYLE_MODELS[selected_style_name]
            if not os.path.exists(model_path):
                st.error(f"❌ 模型文件未找到：{model_path}。")
            else:
                with col_output:
                    st.markdown(f"##### 🎨 艺术化结果")
                    status_box = st.empty()
                    progress_bar = st.progress(0)
                    
                    # 🔒 高并发防御逻辑：排队锁
                    # 使用 FileLock 确保同一时间只有 1 个任务在进行推理
                    lock = FileLock(LOCK_FILE + ".lock")
                    
                    try:
                        # 尝试获取锁，如果排队超过 10 秒则超时
                        status_box.info("⌛ 正在排队等待服务器资源，请稍候...")
                        with lock.acquire(timeout=10):
                            
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
                            st.image(output_image, width=True)
                            
                            buf = io.BytesIO()
                            output_image.save(buf, format="JPEG", quality=95)
                            byte_im = buf.getvalue()
                            st.download_button(
                                label="📥 保存高清图片", data=byte_im,
                                file_name="art_style_result.jpg", mime="image/jpeg",
                                width=True
                            )
                            
                            # 立即回收内存
                            gc.collect()

                    except Timeout:
                        status_box.warning("⚠️ 当前服务器排队人数过多 (并发保护)，请稍等 5 秒后重试。")
                    except Exception as e:
                        status_box.error("处理出错，可能是图片过于复杂或内存不足。")
                        st.error(f"Error: {e}")
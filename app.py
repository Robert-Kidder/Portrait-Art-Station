import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import io # 用于处理图片下载流

# 导入工具库
# 确保 style_transfer_utils.py 在同一目录下
from style_transfer_utils import TransformerNet, portrait_style_transfer

# ==========================================
# 1. 页面配置与 CSS 样式注入 (UI/UX 核心)
# ==========================================
st.set_page_config(
    page_title="艺术风格迁移实验室",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS：美化界面、修复对齐问题、适配移动端
st.markdown("""
    <style>
    /* 全局字体 */
    .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* ---------------------------------------------------
       修复 1 & 2: 侧边栏按钮可见性与标题对齐修正
    --------------------------------------------------- */
    
    /* 仅隐藏 Streamlit 顶部的彩虹装饰线，保留 Header 以显示侧边栏按钮 */
    
    [data-testid="stDecoration"] {
        visibility: hidden;
    }
    
    /* 隐藏页脚和汉堡菜单 */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* 1. 让所有 H1 默认居中 (这会影响主界面标题) */
    h1 {
        font-weight: 700;
        color: #333;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* 2. 特别指定：侧边栏 (.css-...) 的 H1 必须居左 */
    [data-testid="stSidebar"] h1 {
        text-align: left;
    }
    
    /* 侧边栏标题保持默认左对齐，不需要额外写 CSS，
       因为上面的规则限定了 .main，不会影响侧边栏 */

    /* ---------------------------------------------------
       其他美化样式
    --------------------------------------------------- */
    
    /* 移动端优化：调整顶部留白 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    /* 信息框美化 */
    .stAlert {
        border-radius: 12px;
        border: none;
        background-color: #f8f9fa;
        border-left: 5px solid #11998e;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 模型定义与加载逻辑
# ==========================================

# 模型路径配置
STYLE_MODELS = {
    "✨ 马赛克 (Mosaic)": "saved_models/mosaic.pth",
    "🍬 糖果世界 (Candy)": "saved_models/candy.pth",
    "☔ 雨之公主 (Rain Princess)": "saved_models/rain_princess.pth",
    "🎨 乌德尼 (Udnie)": "saved_models/udnie.pth"
}

@st.cache_resource
def load_model(model_path):
    """
    加载模型并缓存。
    包含针对旧版 .pth 文件的 unexpected running stats 修复逻辑。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNet()
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # 🟢 修复逻辑：移除多余的 keys
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
    """全局风格迁移逻辑"""
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
    return Image.fromarray(output_tensor)

# ==========================================
# 3. 侧边栏 (Sidebar)
# ==========================================
# 标题会自动左对齐，因为我们的 CSS 只强制了主界面的 H1 居中
st.sidebar.title("⚙️ 设置面板")
st.sidebar.markdown("上传图片并选择你喜欢的艺术风格。")

# A. 图片上传区
uploaded_file = st.sidebar.file_uploader(
    "1️⃣ 上传一张照片...", 
    type=["jpg", "jpeg", "png"],
    help="建议上传包含人物的自拍或生活照，以体验人像保护功能。"
)

# B. 风格选择区
selected_style_name = st.sidebar.selectbox(
    "2️⃣ 选择艺术风格",
    list(STYLE_MODELS.keys())
)

st.sidebar.markdown("---")

# C. 创新模式开关
st.sidebar.markdown("### 🚀 创新功能")
use_portrait_mode = st.sidebar.checkbox(
    "🛡️ 人像保护模式",
    value=True,
    help="勾选后，系统将自动识别人物，仅对背景进行风格化，保留人物真实质感。"
)

# D. 提交按钮
generate_btn = st.sidebar.button("开始创作 ✨")

# ==========================================
# 4. 主界面逻辑 (Main Interface)
# ==========================================

# 页面主标题
st.title("艺术风格迁移实验室")
st.markdown("<p style='text-align: center; color: #666; margin-bottom: 30px;'>基于深度语义感知的智能风格迁移系统</p>", unsafe_allow_html=True)

if uploaded_file is None:
    # -------------------------------------------------------
    # 🏠 落地页 (Landing Page) - 未上传图片时显示
    # -------------------------------------------------------
    
    st.info("👋 欢迎体验！请点击左侧侧边栏 (电脑) 或左上角箭头 (手机) 上传图片。")

    # 布局优化：针对 470x391 像素的图片
    # 在宽屏上，使用 [3, 4, 3] 的比例，让中间的列占约 40% 宽度，避免小图被过度拉伸
    # 在手机上，st.columns 会自动垂直排列，use_container_width=True 会让图片自动填满手机宽
    col_spacer1, col_img, col_spacer2 = st.columns([3, 4, 3])
    
    with col_img:
        # 🟢 加载本地图片 mosaic.jpg
        local_image_path = "mosaic.jpg"
        if os.path.exists(local_image_path):
            st.image(
                Image.open(local_image_path),
                caption="效果预览：马赛克风格",
                use_container_width=True 
            )
        else:
            # 备用方案：如果本地图片不存在，显示文字提示或网络图
            st.warning(f"⚠️ 提示：未在当前目录下找到 '{local_image_path}'，请确保图片文件存在。")
            # 这里也可以放回之前的网络图片链接作为兜底
    
    st.markdown("---")
    
    # 功能亮点展示 (三列布局)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🛡️ 人像保护")
        st.caption("智能分割前景人物，拒绝五官乱码与变形。")
    with c2:
        st.markdown("#### ⚡ 极速推理")
        st.caption("基于深度学习的图像风格迁移技术，毫秒级生成速度。")
    with c3:
        st.markdown("#### 📱 全端适配")
        st.caption("无论手机还是电脑，随时随地开启创作。")

else:
    # -------------------------------------------------------
    # 🛠️ 工作台 (Workspace) - 图片已上传
    # -------------------------------------------------------
    content_image = Image.open(uploaded_file).convert('RGB')
    
    # 布局：手机端自动垂直排列，电脑端左右分栏
    col_input, col_output = st.columns(2)
    
    with col_input:
        st.markdown("##### 📸 原始图像")
        # 🌟 关键：use_container_width=True 确保手机端占满屏幕
        st.image(content_image, use_container_width=True)

    if generate_btn:
        model_path = STYLE_MODELS[selected_style_name]
        
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件未找到：{model_path}，请检查 saved_models 文件夹。")
        else:
            with col_output:
                st.markdown(f"##### 🎨 艺术化结果")
                
                # 占位符用于显示进度和状态
                status_box = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    if use_portrait_mode:
                        status_box.info("🔍 正在识别人物轮廓并分离背景...")
                        progress_bar.progress(30)
                        
                        # 调用核心函数
                        output_image = portrait_style_transfer(
                            content_image, 
                            model_path, 
                            use_gpu=torch.cuda.is_available()
                        )
                        progress_bar.progress(80)
                        status_box.info("🖌️ 正在进行边缘融合...")
                        
                    else:
                        status_box.info("⚡ 正在进行全局风格渲染...")
                        progress_bar.progress(50)
                        
                        # 调用全局函数
                        output_image = global_style_transfer(content_image, model_path)
                    
                    # 完成
                    progress_bar.progress(100)
                    # 清除进度条和状态文字，展示结果
                    progress_bar.empty()
                    status_box.success("✨ 生成成功！")
                    
                    # 展示结果图
                    st.image(output_image, use_container_width=True)
                    
                    # 处理下载
                    buf = io.BytesIO()
                    output_image.save(buf, format="JPEG", quality=95)
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 保存高清大图",
                        data=byte_im,
                        file_name="art_style_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True # 让下载按钮也自适应宽度
                    )
                    
                except Exception as e:
                    status_box.error("处理过程中发生错误")
                    st.error(f"Error Details: {e}")
                    # 打印控制台日志以便调试
                    import traceback
                    traceback.print_exc()
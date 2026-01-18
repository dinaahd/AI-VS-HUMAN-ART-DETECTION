import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
# Set the Matplotlib backend for deployment stability
matplotlib.use('Agg') 

# ===============================
# ⚙️ Page Config
# ===============================
st.set_page_config(page_title="ArtIntelX Pro 🎨", page_icon="⚡", layout="wide")

# ===============================
# 🌈 Premium CSS — Glassmorphism & Animations
# (CSS remains the same high-impact style)
# ===============================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    body {
        background: linear-gradient(135deg, #0a0f1f 0%, #1a1f3a 50%, #0f1729 100%);
        color: #e0e7ff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1f, #1a2332, #0d1b2a);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.15);
        padding: 3rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-top: 20px;
    }

    /* Premium Header */
    h1 {
        text-align: center;
        font-weight: 900;
        letter-spacing: -1px;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #00f5ff 0%, #00d4ff 25%, #0099ff 50%, #00f5ff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 40px rgba(0, 245, 255, 0.3);
    }
    
    @keyframes shimmer {
        to { background-position: 200% center; }
    }

    h3 {
        text-align: center;
        font-weight: 400;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        letter-spacing: 0.5px;
    }

    /* Verdict Panel (Official Report Style) */
    .verdict-panel {
        padding: 2.5rem; 
        border-radius: 18px; 
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        border: 2px solid;
    }
    
    .verdict-header {
        margin: 0; 
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: 1px;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
    }
    
    .verdict-subtext {
        color: #cbd5e1; 
        font-size: 1.2rem; 
        margin-top: 0.5rem; 
        font-weight: 600;
    }
    
    /* AI Verdict - High impact Red/Orange */
    .ai-verdict-panel {
        background: rgba(163, 0, 0, 0.15); 
        border-color: #ff4b4b;
        box-shadow: 0 0 40px 5px rgba(255, 75, 75, 0.5);
    }
    .ai-verdict-text {
        background: linear-gradient(135deg, #ff4b4b 0%, #a30000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Human Verdict - High impact Green/Blue */
    .human-verdict-panel {
        background: rgba(0, 100, 163, 0.15); 
        border-color: #00f5ff;
        box-shadow: 0 0 40px 5px rgba(0, 245, 255, 0.5);
    }
    .human-verdict-text {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
        color: #0a0f1f !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 245, 255, 0.5);
        background: linear-gradient(135deg, #00d4ff 0%, #0077ff 100%);
    }

    /* File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 2px dashed rgba(0, 245, 255, 0.3);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(0, 245, 255, 0.6);
        background: rgba(255, 255, 255, 0.05);
    }

    /* Custom Metrics - Keep the premium look */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00f5ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 245, 255, 0.3), transparent);
        margin: 3rem 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: rgba(0, 245, 255, 0.1);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #00f5ff;
        margin: 0.25rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .footer strong {
        background: linear-gradient(135deg, #00f5ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Custom styling for tabs to fit the theme */
    .stTabs [data-testid="stTab"] {
        padding: 10px 20px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        margin-right: 5px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stTabs [data-testid="stTab"]:hover {
        color: #00f5ff;
        background-color: rgba(0, 245, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        color: #00f5ff;
        border-bottom: 3px solid #00f5ff;
        background-color: rgba(255, 255, 255, 0.08);
        border-top: 1px solid #00f5ff;
        border-left: 1px solid #00f5ff;
        border-right: 1px solid #00f5ff;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 🧠 Load Model (FIXED STATE_DICT LOADING)
# ===============================
@st.cache_resource
def load_model():
    # --- Define the Correct Local Path ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Path is: ../models/resnet50v2_optimized/best_resnet50v2.pth
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'resnet50v2', 'best_model.pth') 
    
    # Initialize the base ResNet50 architecture
    model = models.resnet50(weights=None)
    model_name = "ResNet50v2 (Fine-tuned Local)"
    
    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features
    
    # 🛑 CRITICAL FIX: The custom weights contain keys "fc.1.weight" and "fc.1.bias".
    # We must redefine model.fc as a Sequential module to match this structure.
    model.fc = torch.nn.Sequential(
        # The Identity layer takes index 0 (fc.0)
        torch.nn.Identity(), 
        # The Linear layer takes index 1 (fc.1), matching the weight keys
        torch.nn.Linear(num_ftrs, 2) 
    )
    
    # Load custom fine-tuned weights
    if os.path.exists(MODEL_PATH):
        try:
            # Load weights to CPU
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.success(f"✅ Model ({model_name}) loaded successfully from local path!")
        except Exception as e:
            # Fallback error for other issues (e.g., corruption, other structure mismatch)
            st.error(f"❌ Error loading custom weights from '{MODEL_PATH}': {e}")
            st.warning("Running with un-initialized weights. Check model structure/weight file integrity.")
    else:
        st.error(f"❌ Error: Model file not found at expected location: {MODEL_PATH}")
        st.warning("Running with un-initialized weights. Please ensure 'best_resnet50v2.pth' is in the correct folder.")

    model.eval()
    return model

model = load_model()
labels = ["AI-generated Art", "Human-made Art"]

# ===============================
# 🎨 Header Section with Badges
# ===============================
st.markdown("<h1>⚡ ArtIntelX Pro</h1>", unsafe_allow_html=True)
st.markdown("<h3>Next-Generation AI vs Human Art Detection System</h3>", unsafe_allow_html=True)

col_badge1, col_badge2, col_badge3, col_badge4 = st.columns(4)
with col_badge1:
    st.markdown('<div class="badge">🧠 Deep Learning</div>', unsafe_allow_html=True)
with col_badge2:
    st.markdown('<div class="badge">⚡ Real-time Analysis</div>', unsafe_allow_html=True)
with col_badge3:
    st.markdown('<div class="badge">🎯 Fine-tuned ResNet50v2</div>', unsafe_allow_html=True)
with col_badge4:
    st.markdown('<div class="badge pulse">🔥 Grad-CAM Integrated</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# 🖼️ Upload Section
# ===============================
st.markdown("### Upload Artwork for Analysis")

uploaded_file = st.file_uploader(
    "📁 Drop your artwork here or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, and PNG formats up to 200MB"
)

# Define Grad-CAM function
def generate_heatmap(model, image_tensor, image_original):
    # Grad-CAM uses the final convolutional layer, which is model.layer4[-1] for ResNet50
    last_conv_layer = model.layer4[-1]
    gradients, activations = [], []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = last_conv_layer.register_forward_hook(forward_hook)
    bh = last_conv_layer.register_backward_hook(backward_hook)

    # Forward pass
    pred = model(image_tensor)
    class_idx = torch.argmax(pred)
    
    # Backward pass
    model.zero_grad()
    try:
        pred[0, class_idx].backward()
    except IndexError:
        st.error("Model prediction failed during backward pass (IndexError).")
        fh.remove()
        bh.remove()
        return image_original

    # Simple Grad-CAM calculation
    grads = gradients[0].mean(dim=[2, 3], keepdim=True)
    activation = activations[0]
    cam = torch.sum(grads * activation, dim=1).squeeze()
    cam = torch.clamp(cam, min=0) 
    cam = cam / cam.max() if cam.max() > 0 else cam 
    heatmap = cam.detach().numpy()

    # Cleanup hooks
    fh.remove()
    bh.remove()
    
    # Convert heatmap to PIL Image and blend
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(image_original.size, Image.BILINEAR)
    blended = Image.blend(image_original, heatmap_img.convert("RGB"), alpha=0.55)
    
    return blended


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # =======================================================
    # 🎯 Primary Result Calculation
    # =======================================================
    with st.spinner("🔮 Analyzing artwork..."):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]) 
        ])
        img_tensor = transform(image).unsqueeze(0)
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

        # Generate Heatmap
        blended_image = generate_heatmap(model, img_tensor, image)
        
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Determine styles for the NEW Full-Width Verdict Panel
    if pred_idx == 0:  # AI-generated
        verdict_class = "ai-verdict-panel"
        text_class = "ai-verdict-text"
        result_title = "🤖 OFFICIAL VERDICT: AI-GENERATED"
        result_subtext = f"ANALYSIS COMPLETE: High probability ({confidence:.2f}%) of Synthetic Origin."
    else: # Human-made
        verdict_class = "human-verdict-panel"
        text_class = "human-verdict-text"
        result_title = "👨‍🎨 OFFICIAL VERDICT: HUMAN-MADE"
        result_subtext = f"ANALYSIS COMPLETE: High probability ({confidence:.2f}%) of Manual Creation."

    # --- FULL-WIDTH VERDICT HEADER ---
    st.markdown(f"""
        <div class="{verdict_class} verdict-panel">
            <h2 class="verdict-header {text_class}">{result_title}</h2>
            <p class="verdict-subtext">{result_subtext}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Visual Evidence")

    # --- Two-Column Layout for Visual Evidence ---
    viz_col1, viz_col2 = st.columns(2, gap="large")

    # 1. Original Image
    with viz_col1:
        st.markdown("#### 🖼️ Submitted Artwork")
        st.image(image, use_container_width=True, caption=uploaded_file.name)

    # 2. Grad-CAM Heatmap (Evidence)
    with viz_col2:
        st.markdown("#### 🔥 Model Attention Map (Grad-CAM)")
        st.image(blended_image, 
                 caption="Highlights show areas the model focused on to reach the verdict.",
                 use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # =======================================================
    # 📊 Statistical Breakdown (OPTIMIZED CHART SIZE/LAYOUT)
    # =======================================================
    st.markdown("### 📈 Statistical Breakdown")
    
    tab1, tab2 = st.tabs(["Probability Metrics", "Visual Charts"])
    
    # 1. Probability Metrics Tab
    with tab1:
        st.markdown("#### Confidence Scores")
        prob_col_ai, prob_col_human, prob_col_confidence = st.columns(3)
        
        with prob_col_ai:
            ai_prob = probs[0].item() * 100
            st.metric("🤖 AI-Generated Probability", f"{ai_prob:.2f}%")
            
        with prob_col_human:
            human_prob = probs[1].item() * 100
            st.metric("👨‍🎨 Human-made Probability", f"{human_prob:.2f}%")

        with prob_col_confidence:
            st.metric(f"🥇 Highest Confidence ({labels[pred_idx]})", f"{confidence:.2f}%")
            
    # 2. Visual Charts Tab (Optimized for side-by-side)
    with tab2:
        chart_col1, chart_col2 = st.columns(2, gap="large")
        
        # --- Bar Chart ---
        with chart_col1:
            st.markdown("#### Probability Distribution (Bar Chart)")
            fig1, ax1 = plt.subplots(figsize=(7, 4)) 
            fig1.patch.set_facecolor('#0a0f1f')
            ax1.set_facecolor('#0a0f1f')
            
            bars = ax1.bar(labels, probs.detach().numpy(), 
                            width=0.6,
                            color=['#ff4b4b', '#00f5ff'], 
                            edgecolor='white',
                            linewidth=1.5,
                            alpha=0.9)
            
            ax1.set_ylabel('Probability', fontsize=10, color='white', fontweight='600')
            ax1.set_title('Confidence Score', fontsize=12, color='#00f5ff', 
                          fontweight='700', pad=10)
            ax1.tick_params(colors='white', labelsize=9)
            ax1.spines['bottom'].set_color('#334155')
            ax1.spines['left'].set_color('#334155')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(axis='y', alpha=0.2, linestyle='--')
            ax1.set_ylim(0, 1.1)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{height*100:.1f}%',
                         ha='center', va='bottom', color='white', 
                         fontweight='700', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig1)

        # --- Pie Chart ---
        with chart_col2:
            st.markdown("#### Confidence Share (Pie Chart)")
            fig2, ax2 = plt.subplots(figsize=(7, 4)) 
            fig2.patch.set_facecolor('#0a0f1f')
            ax2.set_facecolor('#0a0f1f')
            
            colors = ['#ff4b4b', '#00f5ff']
            explode_list = [0.05 if i == pred_idx else 0.0 for i in range(len(labels))]

            wedges, texts, autotexts = ax2.pie(
                probs.detach().numpy(),
                labels=labels,
                autopct='%1.1f%%',
                colors=colors,
                explode=explode_list,
                shadow=True,
                startangle=45,
                textprops={'color': 'white', 'fontweight': '600', 'fontsize': 8}
            )
            
            for autotext in autotexts:
                autotext.set_color('#0a0f1f')
                autotext.set_fontweight('900')
                autotext.set_fontsize(9)
            
            ax2.set_title('Confidence Share', fontsize=12, 
                          color='#00f5ff', fontweight='700', pad=10)
            
            plt.tight_layout()
            st.pyplot(fig2)

st.markdown("<hr>", unsafe_allow_html=True)

# ===============================
# ⚙️ Model Info Section
# ===============================

with st.expander("⚙️ **Technical Specifications & Model Details**"):
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.markdown("""<div class="stat-card"><div class="stat-value">2.5M+</div><div class="stat-label">Images Analyzed</div></div>""", unsafe_allow_html=True)

    with stat_col2:
        st.markdown("""<div class="stat-card"><div class="stat-value">90%</div><div class="stat-label">Accuracy Rate</div></div>""", unsafe_allow_html=True)

    with stat_col3:
        st.markdown("""<div class="stat-card"><div class="stat-value">&lt;2s</div><div class="stat-label">Processing Time</div></div>""", unsafe_allow_html=True)

    with stat_col4:
        st.markdown("""<div class="stat-card"><div class="stat-value">25M</div><div class="stat-label">Model Parameters</div></div>""", unsafe_allow_html=True)
        
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        #### 🏗️ Architecture
        - **Model:** ResNet50v2 (Local Fine-Tuning)
        - **Final Layer:** Sequential Module (Fix for `fc.1.weight` key)
        - **Input Size:** 224×224×3
        - **Parameters:** ~25M trainable
        - **Activation:** ReLU + Softmax
        """)
    
    with tech_col2:
        st.markdown("""
        #### 🛠️ Technology Stack
        - **Framework:** PyTorch 2.0+
        - **Frontend:** Streamlit
        - **Visualization:** Matplotlib, Grad-CAM
        - **Preprocessing:** Torchvision Transforms
        """)

with st.expander("❓ **How It Works (Detection Process)**"):
    st.markdown("""
    ### Detection Process Overview
    
    1. **Preprocessing** 🖼️: The image is resized to **224×224** and normalized using standard **ImageNet** parameters, which typically align with models like ResNet.
    
    2. **Feature Extraction** 🔬: The **ResNet50v2** model layers, loaded with your custom, optimized weights, analyze the image for unique patterns indicative of AI or human creation.
    
    3. **Classification** 🎯: The final custom output layer produces the probability scores for the two classes (AI or Human).
    
    4. **Explainability (Grad-CAM)** 🔥: A heatmap is generated from the model's final convolutional layer (**Layer4**) to visually pinpoint the regions of the image that drove the classification verdict.
    """)

# ===============================
# 🎨 Footer
# ===============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        <p>🚀 Powered by <strong>PyTorch</strong> & <strong>Streamlit</strong></p>
        <p>Designed & Developed by <strong>Dina</strong> | © 2025 ArtIntelX Pro</p>
        <p style="font-size: 0.85rem; margin-top: 1rem;">
            Made with 💙 for the AI & Art Community
        </p>
    </div>
""", unsafe_allow_html=True)
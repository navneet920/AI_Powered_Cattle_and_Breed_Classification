import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Breed Classifier",
    page_icon="🐄",
    layout="wide",
)

# ------------------------------
# CUSTOM CSS FOR ANIMATIONS & UI
# ------------------------------
custom_css = """
<style>

body {
    background: linear-gradient(135deg, #e8f1ff, #fff7e6);
}

.upload-box {
    border: 2px dashed #6c63ff;
    padding: 20px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(12px);
    transition: 0.3s;
}
.upload-box:hover {
    border: 2px solid #6c63ff;
    transform: scale(1.02);
}

.result-card {
    padding: 25px;
    background: rgba(255, 255, 255, 0.65);
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    backdrop-filter: blur(12px);
    transition: 0.35s;
}
.result-card:hover {
    transform: scale(1.02);
}

.gauge {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: conic-gradient(#6c63ff calc(var(--percentage)*1%), #e0e0e0 0%);
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:bold;
    font-size:26px;
    color:#333;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------
# TITLE
# ------------------------------
st.markdown(
    """
    <h1 style='text-align:center; font-size:45px;'>🐄 AI Cattle & Buffalo Breed Classifier</h1>
    <h4 style='text-align:center; color:#5a5a5a;'>
        Upload an image and experience an interactive prediction dashboard
    </h4>
    <br>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/bovine_breed_classifier_vgg16.h5")

model = load_model()

class_names = ['Alambadi',
 'Amritmahal',
 'Ayrshire',
 'Banni',
 'Bargur',
 'Bhadawari',
 'Brown_Swiss',
 'Dangi',
 'Deoni',
 'Gir',
 'Guernsey',
 'Hallikar',
 'Hariana',
 'Holstein_Friesian',
 'Jaffrabadi',
 'Jersey',
 'Kangayam',
 'Kankrej',
 'Kasargod',
 'Kenkatha',
 'Kherigarh',
 'Khillari',
 'Krishna_Valley',
 'Malnad_gidda',
 'Mehsana',
 'Murrah',
 'Nagori',
 'Nagpuri',
 'Nili_Ravi',
 'Nimari',
 'Ongole',
 'Pulikulam',
 'Rathi',
 'Red_Dane',
 'Red_Sindhi',
 'Sahiwal',
 'Surti',
 'Tharparkar',
 'Toda',
 'Umblachery',
 'Vechur']

# ------------------------------
# PREDICT FUNCTION
# ------------------------------
def predict_breed(img, model):
    img = img.resize((150, 150))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0) / 255.0

    pred = model.predict(img_arr)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx])
    return class_names[idx], confidence, pred[0]


# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("⚙️ Controls")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
st.sidebar.info("Upload an image to classify the breed.")

# ------------------------------
# MAIN APP SECTION
# ------------------------------
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png",'jfif'])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.spinner("🔍 Analyzing image..."):
            label, conf, all_preds = predict_breed(img, model)

        # ------------------------------
        # RESULT CARD
        # ------------------------------
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"<h2>✨ Prediction: <b>{label}</b></h2>", unsafe_allow_html=True)

        # Confidence Gauge
        st.markdown(
            f"""
            <div class="gauge" style="--percentage:{conf*100:.2f};">
                {conf*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show raw probabilities
        st.subheader("📊 Confidence Breakdown")
        st.bar_chart(
            {
                class_names[i]: all_preds[i]
                for i in range(len(class_names))
            }
        )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class='upload-box'>
            <h3 style='text-align:center;'>📤 Upload Image for Classification</h3>
            <p style='text-align:center;color:#777;'>Supports JPG, JPEG, PNG</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# FOOTER
# ------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
        Built with ❤️ by Navneet Kumar using Streamlit + TensorFlow
    </p>
    """,
    unsafe_allow_html=True
)

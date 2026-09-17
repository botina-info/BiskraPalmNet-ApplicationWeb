import streamlit as st
import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image
import cv2


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BiskraPalmNet - Grad-CAM",
    page_icon="🌴",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "biskraPalmNet_fixed.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3

LAST_CONV_LAYER = "conv2d_2"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f8f5;
        border: 1px solid #d9e5d9;
        margin-bottom: 15px;
    }

    .confidence {
        font-size: 30px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if CLASS_NAMES_PATH.exists():

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(names) == NUM_CLASSES:
            return names

    # Fallback
    return [
        "1-white-cochiniel-disease",
        "brown-leaf-spot-caused-by-mycosphaerella-tassiana",
        "date-palm-leaves-healthy"
    ]


# ============================================================
# LOAD BISKRA PALMNET
# ============================================================

@st.cache_resource
def load_biskrapalmnet():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    img_array = np.asarray(
        image,
        dtype=np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_biskrapalmnet(
    image,
    model,
    class_names
):

    img_array = preprocess_image(
        image
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    )

    predictions = predictions[0]

    if len(predictions) != NUM_CLASSES:

        raise ValueError(
            f"Unexpected model output.\n"
            f"Expected: {NUM_CLASSES}\n"
            f"Received: {len(predictions)}"
        )

    # --------------------------------------------------------
    # Handle probability/logit output
    # --------------------------------------------------------

    if (
        np.all(predictions >= 0)
        and
        np.all(predictions <= 1)
        and
        np.isclose(
            np.sum(predictions),
            1.0,
            atol=1e-3
        )
    ):

        probabilities = predictions

    else:

        probabilities = tf.nn.softmax(
            predictions
        ).numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_label,
        predicted_index,
        confidence,
        probabilities
    )


# ============================================================
# GRAD-CAM
# ============================================================


def make_gradcam_heatmap(
    image_array,
    model,
    last_conv_layer_name="conv2d_2"
):
    #"""
    #Grad-CAM for the Sequential BiskraPalmNet model.

   # The model is rebuilt as a Functional graph so that
   # TensorFlow can correctly compute gradients.
   # """

    # ========================================================
    # BUILD FUNCTIONAL GRAPH FROM THE ORIGINAL MODEL
    # ========================================================

    input_tensor = tf.keras.Input(
        shape=image_array.shape[1:],
        name="gradcam_input"
    )

    x = input_tensor
    conv_output = None

    for layer in model.layers:

        x = layer(x)

        if layer.name == last_conv_layer_name:
            conv_output = x

    # ========================================================
    # CHECK TARGET LAYER
    # ========================================================

    if conv_output is None:

        raise ValueError(
            f"Grad-CAM layer '{last_conv_layer_name}' "
            f"was not found in the model."
        )

    # ========================================================
    # CREATE FUNCTIONAL MODEL
    # ========================================================

    grad_model = tf.keras.Model(
        inputs=input_tensor,
        outputs=[
            conv_output,
            x
        ]
    )

    # ========================================================
    # GRADIENT CALCULATION
    # ========================================================

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_array,
            training=False
        )

        # Predicted class
        predicted_index = tf.argmax(
            predictions[0]
        )

        # Score of predicted class
        class_score = predictions[
            0,
            predicted_index
        ]

    # ========================================================
    # GRADIENT OF CLASS SCORE
    # ========================================================

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    if grads is None:

        raise ValueError(
            "Gradients are None. "
            "Grad-CAM could not connect the prediction "
            "to the target convolutional layer."
        )

    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # ========================================================
    # WEIGHT FEATURE MAPS
    # ========================================================

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ========================================================
    # RELU
    # ========================================================

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:

        heatmap = heatmap / max_value

    return heatmap.numpy()
# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.45
):

    original = np.asarray(
        original_image.convert("RGB")
    )

    # Resize heatmap to original image size

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    # Convert heatmap to 8-bit

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Apply color map

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay

    overlay = cv2.addWeighted(
        original,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return Image.fromarray(
        overlay
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🕹️ Control Panel")

    st.markdown("---")

    st.markdown("### 🧠 AI Model")

    st.success(
        "BiskraPalmNet Loaded"
    )

    st.write(
        f"**Model:** {MODEL_PATH.name}"
    )

    st.write(
        "**Input:** 224 × 224 × 3"
    )

    st.write(
        "**Normalization:** /255"
    )

    st.write(
        "**Classes:** 3"
    )

    st.markdown("---")

    st.markdown(
        "### 🏷️ Class Encoding"
    )

    class_names = load_class_names()

    for index, name in enumerate(
        class_names
    ):

        st.write(
            f"**{index}** → {name}"
        )

    st.markdown("---")

    st.write(
        f"**Grad-CAM Layer:** `{LAST_CONV_LAYER}`"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌴 BiskraPalmNet</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Date Palm Disease Classification '
    'and Explainable AI using Grad-CAM'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_biskrapalmnet()

except Exception as e:

    st.error(
        "❌ Error while loading BiskraPalmNet."
    )

    st.exception(e)

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.success(
    "🤖 BiskraPalmNet is connected and ready for inference."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.markdown(
    "## 📸 Step 1: Upload Date Palm Leaf"
)

uploaded_file = st.file_uploader(
    "Choose a date palm leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# DIAGNOSIS
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a date palm leaf image "
        "to start AI diagnosis."
    )

else:

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        # ----------------------------------------------------
        # Display image information
        # ----------------------------------------------------

        st.image(
            uploaded_image,
            caption="Uploaded Date Palm Leaf",
            width="stretch"
        )

        st.write(
            f"Original image size: "
            f"{uploaded_image.width} × "
            f"{uploaded_image.height} pixels"
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        (
            predicted_label,
            predicted_index,
            confidence,
            probabilities
        ) = predict_biskrapalmnet(
            uploaded_image,
            model,
            class_names
        )

        # ----------------------------------------------------
        # Display diagnosis
        # ----------------------------------------------------

        st.markdown(
            "## 🔬 Step 2: AI Diagnosis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Condition",
                predicted_label
            )

        with col2:

            st.metric(
                "AI Confidence",
                f"{confidence * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Predicted Class Index",
                predicted_index
            )

        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Model Probability Distribution"
        )

        for index, probability in enumerate(
            probabilities
        ):

            st.write(
                f"**{class_names[index]}:** "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        st.markdown(
            "## 🔍 Step 3: Explainable AI — Grad-CAM"
        )

        st.write(
            "Grad-CAM highlights image regions "
            "that contributed most strongly to "
            "the model's prediction."
        )

        # Prepare image

        processed_image = preprocess_image(
            uploaded_image
        )

        # Generate heatmap

        heatmap = make_gradcam_heatmap(
            processed_image,
            model,
            LAST_CONV_LAYER
        )

        # Create overlay

        gradcam_image = create_gradcam_overlay(
            uploaded_image,
            heatmap
        )

        # Display original + Grad-CAM

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                uploaded_image,
                caption="Original Leaf Image",
                width="stretch"
            )

        with col2:

            st.image(
                gradcam_image,
                caption="Grad-CAM Explanation",
                width="stretch"
            )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        st.info(
            "🔎 The highlighted regions represent "
            "areas that contributed most strongly "
            "to the model's prediction. "
            "Grad-CAM provides model-level visual "
            "explanation and should not be interpreted "
            "as precise disease segmentation."
        )

        # ----------------------------------------------------
        # Decision Support
        # ----------------------------------------------------

        st.markdown(
            "## 💡 Step 4: Decision Support"
        )

        if predicted_index == 0:

            recommendation = (
                "Inspect the affected palm and "
                "surrounding palms. Increase monitoring "
                "and consider appropriate integrated "
                "pest-management measures according "
                "to local agricultural guidance."
            )

        elif predicted_index == 1:

            recommendation = (
                "Inspect the affected and neighbouring "
                "palms. Monitor visible lesions and "
                "consider appropriate disease-management "
                "measures according to local agricultural "
                "guidance."
            )

        else:

            recommendation = (
                "The leaf is classified as healthy. "
                "Continue routine field inspection, "
                "preventive monitoring, irrigation "
                "management, and orchard maintenance."
            )

        st.success(
            recommendation
        )

        # ----------------------------------------------------
        # Technical Details
        # ----------------------------------------------------

        with st.expander(
            "🔎 Technical Details"
        ):

            st.write(
                f"Model file: `{MODEL_PATH.name}`"
            )

            st.write(
                "Architecture: Custom CNN — BiskraPalmNet"
            )

            st.write(
                "Input size: 224 × 224 × 3"
            )

            st.write(
                "Normalization: Pixel values / 255"
            )

            st.write(
                f"Grad-CAM target layer: `{LAST_CONV_LAYER}`"
            )

            st.write(
                f"Predicted class: {predicted_index}"
            )

            st.write(
                f"Confidence: {confidence * 100:.2f}%"
            )

    except Exception as e:

        st.error(
            "❌ Error during diagnosis or Grad-CAM generation."
        )

        st.exception(e)
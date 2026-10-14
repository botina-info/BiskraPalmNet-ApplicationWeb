
import streamlit as st

st.set_page_config(
    page_title="Date Palm Disease Guide",
    page_icon="🌴",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f8f4;
}
.block-container {
    padding-top: 2rem;
}
.guide-header {
    padding: 24px;
    background: linear-gradient(120deg, #164e35, #39845a);
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
}
.guide-header h1 {
    color: white;
}
.info-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dce7dd;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="guide-header">
    <h1>🌴 Date Palm Disease Guide</h1>
    <p>
        Learn about common date palm leaf conditions,
        their symptoms, prevention, and recommended monitoring.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This guide provides general educational information. "
    "Image-based predictions should be confirmed by field inspection "
    "when symptoms are uncertain."
)

# -----------------------------
# Disease information
# -----------------------------
diseases = {
    "White Cochineal": {
        "scientific": "Parlatoria blanchardi",
        "category": "Insect pest",
        "description": (
            "White cochineal is a scale insect that can colonize "
            "date palm leaves and other green plant surfaces."
        ),
        "symptoms": [
            "Small whitish or greyish scales on leaflets.",
            "Accumulation of scale insects on affected surfaces.",
            "Yellowing or reduced vigor in heavily affected foliage."
        ],
        "risk_factors": [
            "Insufficient monitoring of palm foliage.",
            "Conditions that favor pest persistence and spread.",
            "Movement of infested plant material between locations."
        ],
        "prevention": [
            "Inspect leaflets regularly, including less visible surfaces.",
            "Monitor changes in pest abundance over time.",
            "Avoid moving infested plant material between palms.",
            "Seek local agricultural advice when populations increase."
        ],
        "action": (
            "Inspect the palm and assess the extent of infestation. "
            "Select control measures based on local agricultural guidance."
        )
    },

    "Brown Leaf Spot": {
        "scientific": "Mycosphaerella tassiana (reported dataset label)",
        "category": "Leaf disease symptom class",
        "description": (
            "Brown leaf spot refers to brown lesions visible on leaf "
            "tissue. Similar symptoms may have different causes, so "
            "visual appearance alone cannot always confirm the pathogen."
        ),
        "symptoms": [
            "Brown or dark spots on leaflets.",
            "Lesions that may vary in size, shape, and distribution.",
            "Possible progression of visible damage over time."
        ],
        "risk_factors": [
            "Environmental conditions favorable to some leaf pathogens.",
            "Prolonged leaf wetness where relevant.",
            "Plant stress or other factors affecting palm health."
        ],
        "prevention": [
            "Inspect affected and neighboring palms regularly.",
            "Record changes in lesion appearance and distribution.",
            "Maintain appropriate irrigation and general palm care.",
            "Obtain expert diagnosis if symptoms spread or remain unclear."
        ],
        "action": (
            "Document the symptoms and arrange field assessment if "
            "the lesions progress. Confirm the cause before selecting "
            "a disease-specific treatment."
        )
    },

    "Healthy Date Palm": {
        "scientific": "Phoenix dactylifera L.",
        "category": "Healthy reference class",
        "description": (
            "This category represents leaf images labeled as healthy "
            "in the BiskraPalmNet dataset."
        ),
        "symptoms": [
            "No obvious disease symptoms in the inspected area.",
            "Leaf appearance consistent with the observed healthy reference.",
            "No visible signs necessarily does not guarantee that the "
            "entire palm is free of pests or disease."
        ],
        "risk_factors": [
            "New problems may appear after the image is captured.",
            "Some pests or diseases may not be visible in one photograph."
        ],
        "prevention": [
            "Continue routine inspection of leaves and other palm parts.",
            "Maintain suitable irrigation and agricultural practices.",
            "Photograph suspicious changes for later comparison.",
            "Repeat the assessment if new symptoms appear."
        ],
        "action": (
            "Continue routine monitoring. If new symptoms develop, "
            "capture additional images and inspect the palm in the field."
        )
    }
}

# -----------------------------
# Disease selector
# -----------------------------
st.subheader("Explore the Guide")

selected_disease = st.selectbox(
    "Select a date palm condition",
    list(diseases.keys())
)

data = diseases[selected_disease]

st.markdown("---")

st.title(selected_disease)
st.caption(data["category"])
st.write(f"**Scientific name / label:** {data['scientific']}")

st.markdown("### Overview")
st.write(data["description"])

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔎 Symptoms")
    for item in data["symptoms"]:
        st.markdown(f"- {item}")

    st.markdown("### ⚠️ Possible risk factors")
    for item in data["risk_factors"]:
        st.markdown(f"- {item}")

with col2:
    st.markdown("### 🛡️ Prevention and monitoring")
    for item in data["prevention"]:
        st.markdown(f"- {item}")

    st.markdown("### ✅ Recommended next step")
    st.success(data["action"])

# -----------------------------
# Important notice
# -----------------------------
st.markdown("---")
st.subheader("Important Notice")

st.write(
    "BiskraPalmNet classifies images into the three categories "
    "represented in its training dataset. Its prediction and confidence "
    "score do not independently establish disease severity, confirm "
    "the causal organism, or quantify the percentage of an affected palm."
)

st.write(
    "Chemical treatments should only be selected after appropriate "
    "diagnosis and consultation of current local agricultural guidance."
)

st.caption(
    "BiskraPalmNet-Web | Date Palm Disease Guide"
)

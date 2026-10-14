
import streamlit as st
import csv
import io
from datetime import date

st.set_page_config(
    page_title="Prevention & Monitoring",
    page_icon="🌴",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5f8f4;
}
.header-box {
    background: linear-gradient(120deg, #164e35, #39845a);
    padding: 25px;
    border-radius: 15px;
    color: white;
    margin-bottom: 20px;
}
.header-box h1 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>🌴 Prevention & Monitoring</h1>
    <p>
        Monitor date palm health, record field observations,
        and support preventive agricultural decisions.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This module supports field monitoring. It does not replace "
    "professional diagnosis or local agricultural advice."
)

# --------------------------------
# 1. Field inspection information
# --------------------------------
st.header("1. Field Inspection")

with st.form("monitoring_form"):
    inspection_date = st.date_input(
        "Inspection date",
        value=date.today()
    )

    col1, col2 = st.columns(2)

    with col1:
        palm_id = st.text_input(
            "Palm ID / Reference",
            placeholder="e.g. Palm-001"
        )

        farm_location = st.text_input(
            "Farm / Plot",
            placeholder="e.g. Plot A, Biskra"
        )

    with col2:
        inspector = st.text_input(
            "Inspector name (optional)"
        )

        palm_condition = st.selectbox(
            "General observed condition",
            [
                "Apparently healthy",
                "Possible pest symptoms",
                "Possible leaf disease symptoms",
                "Other visible abnormality"
            ]
        )

    st.header("2. Symptoms and Observations")

    symptoms = st.multiselect(
        "Select all observed signs",
        [
            "White or grey scale-like insects",
            "Brown or dark leaf spots",
            "Yellowing of leaflets",
            "Dry or damaged leaflets",
            "Changes in leaf appearance",
            "No visible symptoms",
            "Other symptoms"
        ]
    )

    spread = st.selectbox(
        "Observed distribution",
        [
            "Not assessed",
            "One leaflet or limited area",
            "Several leaflets",
            "Several fronds",
            "Multiple palms",
            "Unknown"
        ]
    )

    trend = st.selectbox(
        "Change since the previous inspection",
        [
            "First inspection / no comparison",
            "No obvious change",
            "Symptoms appear to be increasing",
            "Symptoms appear to be decreasing",
            "Uncertain"
        ]
    )

    notes = st.text_area(
        "Field notes",
        placeholder=(
            "Describe the observations, affected parts, "
            "and any relevant changes..."
        )
    )

    submitted = st.form_submit_button(
        "Generate Monitoring Record",
        type="primary"
    )

# --------------------------------
# 3. Monitoring guidance
# --------------------------------
st.header("3. Recommended Monitoring Actions")

st.markdown("""
- Inspect palm leaflets regularly, including their less visible surfaces.
- Compare new observations with previous photographs and field notes.
- Record the date, palm reference, visible signs, and changes over time.
- Avoid diagnosing a disease solely from one visual symptom.
- Seek advice from a qualified agricultural specialist when symptoms
  spread, persist, or are difficult to identify.
""")

if submitted:
    st.markdown("---")
    st.subheader("Inspection Summary")

    st.write(f"**Date:** {inspection_date}")
    st.write(f"**Palm reference:** {palm_id or 'Not provided'}")
    st.write(f"**Farm / plot:** {farm_location or 'Not provided'}")
    st.write(f"**Observed condition:** {palm_condition}")
    st.write(f"**Observed distribution:** {spread}")
    st.write(f"**Trend:** {trend}")
    st.write(
        "**Symptoms:** " +
        (", ".join(symptoms) if symptoms else "None selected")
    )

    if notes:
        st.write(f"**Field notes:** {notes}")

    if (
        "No visible symptoms" in symptoms
        and len(symptoms) > 1
    ):
        st.warning(
            "Review the selected symptoms: 'No visible symptoms' "
            "is selected together with other signs."
        )

    if (
        palm_condition != "Apparently healthy"
        or any(
            s != "No visible symptoms"
            for s in symptoms
        )
        or trend == "Symptoms appear to be increasing"
    ):
        st.warning(
            "Follow-up recommended: inspect the palm carefully, "
            "document the observations, and seek expert assessment "
            "if needed. These observations do not confirm a diagnosis."
        )
    else:
        st.success(
            "Continue routine monitoring and record any new changes."
        )

    st.subheader("Download Inspection Record")

    record = {
        "Inspection date": str(inspection_date),
        "Palm ID": palm_id,
        "Farm / plot": farm_location,
        "Inspector": inspector,
        "Observed condition": palm_condition,
        "Symptoms": "; ".join(symptoms),
        "Distribution": spread,
        "Trend": trend,
        "Field notes": notes
    }

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=record.keys())
    writer.writeheader()
    writer.writerow(record)

    st.download_button(
        label="Download Record (CSV)",
        data="\ufeff" + output.getvalue(),
        file_name=f"palm_monitoring_{inspection_date}.csv",
        mime="text/csv"
    )

st.caption(
    "BiskraPalmNet-Web | Prevention & Monitoring"
)

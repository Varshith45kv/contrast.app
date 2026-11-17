import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="CLAHE Contrast Enhancement", page_icon="🌗", layout="wide")

st.title("Adaptive Histogram Equalization (CLAHE) for Local Contrast Enhancement")
st.write(
    "Upload a dark or low-contrast image. CLAHE improves contrast locally, "
    "so dark and bright regions both become easier to see."
)

# Sidebar settings
st.sidebar.header("CLAHE Settings")
clip_limit = st.sidebar.slider("Clip limit (contrast strength)", 1.0, 10.0, 2.0, 0.5)
tile_size = st.sidebar.slider("Tile grid size", 2, 16, 8)
st.sidebar.write("Higher clip limit = stronger contrast. Larger tile size = wider local regions.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Method 1: Global histogram equalization (for comparison)
    global_eq = cv2.equalizeHist(gray)

    # Method 2: CLAHE (equalizes small tiles, with a clip limit)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    clahe_img = clahe.apply(gray)

    # Color version: apply CLAHE only to the lightness channel (LAB color space)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = clahe.apply(l_channel)
    color_clahe = cv2.cvtColor(cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(gray, caption="1. Original (grayscale)")
    with col2:
        st.image(global_eq, caption="2. Global histogram equalization")
    with col3:
        st.image(clahe_img, caption="3. CLAHE (local)")

    st.subheader("Color image: original vs CLAHE")
    col4, col5 = st.columns(2)
    with col4:
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original color")
    with col5:
        st.image(cv2.cvtColor(color_clahe, cv2.COLOR_BGR2RGB), caption="CLAHE color")

    st.subheader("Histograms (how pixel brightness is distributed)")
    hist_original = np.histogram(gray.flatten(), bins=256, range=(0, 256))[0]
    hist_global = np.histogram(global_eq.flatten(), bins=256, range=(0, 256))[0]
    hist_clahe = np.histogram(clahe_img.flatten(), bins=256, range=(0, 256))[0]
    chart_data = pd.DataFrame({
        "Original": hist_original,
        "Global equalization": hist_global,
        "CLAHE": hist_clahe,
    })
    st.line_chart(chart_data)

    st.subheader("Contrast measurement (standard deviation, higher = more contrast)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Original", round(float(gray.std()), 1))
    m2.metric("Global equalization", round(float(global_eq.std()), 1))
    m3.metric("CLAHE", round(float(clahe_img.std()), 1))
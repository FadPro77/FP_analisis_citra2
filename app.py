import streamlit as st
from batik_processor import preprocess_and_segment, extract_hsv_features
from knn_classifier import classify_hsv_feature
import cv2
import numpy as np
from PIL import Image

st.title("🔍 Identifikasi Motif Batik")
st.write("Gunakan gambar batik untuk mengidentifikasi jenis motif menggunakan K-Means + KNN.")

uploaded_file = st.file_uploader("Upload gambar batik", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar
    st.image(uploaded_file, caption="Gambar Diupload", use_container_width=True)

    # Simpan sementara ke disk
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Proses gambar dan ekstraksi HSV
    masked_hsv = preprocess_and_segment("temp_image.jpg")
    features = extract_hsv_features(masked_hsv)
    prediction = classify_hsv_feature(features)

    st.subheader("🎯 Hasil Prediksi:")
    st.success(f"Motif Batik: **{prediction}**")

    # Tampilkan hasil segmentasi (tanpa background)
    rgb_result = cv2.cvtColor(masked_hsv, cv2.COLOR_HSV2RGB)
    st.image(rgb_result, caption="🖼️ Hasil Segmentasi (tanpa latar)", use_container_width=True)

    # Ekstraksi dan tampilkan masing-masing channel HSV sebagai gambar grayscale
    h_channel = masked_hsv[:, :, 0]
    s_channel = masked_hsv[:, :, 1]
    v_channel = masked_hsv[:, :, 2]

    st.subheader("📊 Ekstraksi Komponen HSV:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(h_channel, caption="Hue (H)", clamp=True, channels="GRAY")
    with col2:
        st.image(s_channel, caption="Saturation (S)", clamp=True, channels="GRAY")
    with col3:
        st.image(v_channel, caption="Value (V)", clamp=True, channels="GRAY")

    # Tampilkan nilai rata-rata HSV
    st.markdown("**📈 Fitur HSV Rata-rata:**")
    st.write({
        "H": round(features[0], 2),
        "S": round(features[1], 2),
        "V": round(features[2], 2)
    })

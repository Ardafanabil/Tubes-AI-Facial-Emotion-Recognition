import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_image

# Fungsi: Menampilkan deskripsi emosi dan kriteria kualitas
def show_quality_info():
    st.markdown("""
    ## Aplikasi ini melakukan klasifikasi terhadap ekspresi wajah ke dalam beberapa emosi:
    - 😠 Marah
    - 🤢 Jijik  
    - 😨 Takut 
    - 😀 Bahagia    
    - 😢 Sedih  
    - 😲 Terkejut
    - 😐 Netral    

    ---

    ### 📷 Kriteria Kualitas Gambar untuk Pengenalan Emosi Wajah:
    | **Kategori** | **Kriteria** |
    |--------------|-------------|
    | ❌ **Buruk**  | Blur parah, wajah tidak terlihat jelas, terlalu gelap/terang, noise tinggi |
    | ⚠️ **Sedang** | Wajah terlihat tapi pencahayaan kurang optimal atau sedikit blur          |
    | ✅ **Baik**   | Wajah tajam, pencahayaan merata, posisi simetris, kontras baik             |

    ---

    ### 📊 Kriteria Akurasi Prediksi Emosi:
    | **Kategori**     | **Kriteria**  |
    |------------------|----------------|
    | ⚠️ **Rendah**      | Akurasi < 50%, hasil prediksi kemungkinan tidak akurat |
    | ℹ️ **Sedang**     | Akurasi 50% – 80%, hasil prediksi cukup baik tapi perlu dikonfirmasi secara visual |
    | ✅ **Tinggi** | Akurasi > 80%, prediksi konsisten dan sesuai ekspresi yang ditampilkan |
    """, unsafe_allow_html=True)

# Fungsi: Validasi kualitas gambar berdasarkan kontras dan kecerahan
def validate_image_quality(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()

    if contrast < 20 or brightness < 40 or brightness > 210:
        return "❌ Buruk"
    elif contrast < 40 or brightness < 80 or brightness > 180:
        return "⚠️ Sedang"
    else:
        return "✅ Baik"

# Konfigurasi halaman
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Facial Emotion Recognition App</h1>
    <h4 style='text-align: center; color: gray;'>Powered by CNN & Streamlit</h4>
""", unsafe_allow_html=True)

# Tampilkan kriteria terlebih dahulu
show_quality_info()
st.markdown("---")

# Load model CNN
model = load_model('model.h5')
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Sidebar Upload Gambar
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Layout tampilan gambar & hasil prediksi
col1, col2 = st.columns([1, 2])

# Kolom Gambar
with col1:
    st.subheader("Original Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, use_column_width=True)

        # Validasi kualitas gambar
        kualitas = validate_image_quality(image_np)
        st.markdown(f"**Kualitas Gambar:** {kualitas}")
    else:
        st.info("Please upload an image from the sidebar.")

# Kolom Hasil Prediksi
if uploaded_file is not None:
    with col2:
        st.subheader("Prediction Result")

        processed = preprocess_image(image, source='pil')
        prediction = model.predict(processed)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Hasil prediksi emosi
        st.success(f"**Detected Emotion: {class_names[predicted_class]} ({confidence:.2f}%)**")

        # Validasi Confidence
        if confidence < 50:
            st.warning("⚠️ Akurasi rendah, hasil prediksi kemungkinan tidak akurat.")
        elif confidence < 80:
            st.info("ℹ️ Akurasi sedang, hasil prediksi cukup baik tapi perlu dikonfirmasi secara visual.")
        else:
            st.success("✅ Akurasi tinggi, prediksi konsisten dan sesuai ekspresi yang ditampilkan.")

        # Grafik probabilitas
        st.markdown("### Emotion Probabilities")
        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0], color='skyblue')
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        st.pyplot(fig)

        # Output vektor mentah
        st.markdown("### Raw Output Vector")
        st.code(prediction[0])

# Footer
st.markdown("---")
st.markdown("<center>Developed for Tugas Besar Artificial Intelligence</center>", unsafe_allow_html=True)

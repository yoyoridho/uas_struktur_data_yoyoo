# app.py
# Streamlit app: Deteksi Gejala COVID-19 (Decision Tree)
# Jalankan: streamlit run app.py

import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data_gejala_covid19.csv"
MODEL_PATH = "model_decision_tree_covid.pkl"

FEATURES = [
    "Demam","Batuk","SesakNapas","SakitTenggorokan","Kelelahan",
    "Anosmia","Diare","KontakErat","SaturasiO2"
]

def train_and_save_model(data_path: str, model_path: str):
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df["LabelCOVID"]
    model = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    if os.path.exists(DATA_PATH):
        return train_and_save_model(DATA_PATH, MODEL_PATH)
    return None

def main():
    st.set_page_config(page_title="Deteksi Gejala COVID-19", page_icon="", layout="centered")
    st.title("Deteksi Gejala COVID-19 (Decision Tree)")
    st.caption("Aplikasi edukasi: hasil prediksi bukan diagnosis medis.")

    with st.sidebar:
        st.header("Navigasi")
        menu = st.radio("Pilih menu", ["Deteksi", "Tentang"])
        st.divider()
        st.write("File yang digunakan:")
        st.code(f"{DATA_PATH}\n{MODEL_PATH}")

    if menu == "Tentang":
        st.subheader("Tentang Aplikasi")
        st.write(
            "Model menggunakan algoritma Decision Tree dari scikit-learn. "
            "Dataset contoh bersifat sintetis (dibuat untuk tugas) dan dapat diganti "
            "dengan data nyata (lebih baik) agar performa lebih valid."
        )
        st.warning("Jika Anda mengganti dataset, pastikan kolomnya sama seperti FEATURES dan LabelCOVID.")
        return

    model = load_model()
    if model is None:
        st.error("Model/dataset tidak ditemukan. Pastikan file data_gejala_covid19.csv ada di folder yang sama.")
        return

    st.subheader("Masukkan Gejala")
    c1, c2 = st.columns(2)

    with c1:
        demam = st.checkbox("Demam")
        batuk = st.checkbox("Batuk")
        sesak = st.checkbox("Sesak napas")
        sakit_tenggorokan = st.checkbox("Sakit tenggorokan")

    with c2:
        kelelahan = st.checkbox("Kelelahan")
        anosmia = st.checkbox("Kehilangan penciuman (anosmia)")
        diare = st.checkbox("Diare")
        kontak = st.checkbox("Riwayat kontak erat")

    spo2 = st.slider("Saturasi oksigen (SpO2) %", min_value=85, max_value=100, value=97)

    if st.button("Prediksi"):
        input_row = pd.DataFrame([{
            "Demam": int(demam),
            "Batuk": int(batuk),
            "SesakNapas": int(sesak),
            "SakitTenggorokan": int(sakit_tenggorokan),
            "Kelelahan": int(kelelahan),
            "Anosmia": int(anosmia),
            "Diare": int(diare),
            "KontakErat": int(kontak),
            "SaturasiO2": int(spo2),
        }])

        proba = model.predict_proba(input_row)[0]
        pred = int(model.predict(input_row)[0])

        if pred == 1:
            st.error(f"Hasil: Berisiko COVID-19 (probabilitas ~ {proba[1]:.2f})")
        else:
            st.success(f"Hasil: Cenderung Non-COVID (probabilitas COVID ~ {proba[1]:.2f})")

        st.write("Data input:")
        st.dataframe(input_row, use_container_width=True)

        st.info("Saran: jika gejala berat atau SpO2 rendah, segera konsultasi tenaga kesehatan.")

if __name__ == "__main__":
    main()

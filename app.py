# -*- coding: utf-8 -*-
"""
Streamlit App (Single File, Cloud-Ready)
Deteksi Gejala COVID-19 menggunakan Decision Tree

✅ Tidak butuh file CSV/PKL eksternal (semuanya sudah di-embed).
✅ Aman untuk Streamlit Cloud (ada fallback kalau PKL tidak kompatibel).
"""

from __future__ import annotations

import base64
import gzip
import io
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# -----------------------------
# DATA EMBED (CSV + PKL optional)
# -----------------------------

_EMBED_CSV_B64GZ = """H4sIADcVaWkC/4VWTU/rQAy897fswW6z2eTIe+WAiuAA4r5IEa9KP1Ab/j8h3W7GXpcnJEBTe9eeHY+77vZx7/7E4at3L9059k/xM57dS+y3w2t3+Pg4no59PLhNt+t28d/4393heN5vo1tv46lzm+NhiP39KQ5jzvB1iuft89I9xvdu9/f57WG9YMeO3OX3z09bO1pQAi8QuzaM4Bx5+TtG8uIakdP9lE6QnsBrTI5sBJjObKczjUh54hhZTekyNoMEHySQJdjkNuHMuihJdATpvmBJdUTIpwmS7t1D71eiR+ZZvdFUPC/mfi4fTG0SgNNFAYrP4CrzCQ8XgM980SqVxFgni3fn8t0VqJivLT5DikSe8plIc7sUfKaLWnh3oRC8iOdIlm1S5hMuqizNe4gkpK7QUpP5nFkdi2coiTGSZZuV6oj1GF5rTQJTxZskr0T6TeoARKITS2wNl2Jequ7Gc8DEyUgmsibWWy7AIp/1odBnXYhevQeVFgZgVXCX5lCPR9NatthYLHuhMLJmJrmAB+rJ4o4t6rPCgiWRAIRkha2EBwmvVGdWlrN4deYEcuGAjDOj+JSzDWuGbTGxJERGkhVJpfuzLEnxWVgtuj+QzKXoW2vzBVFneuJlZv5X2bAExTIuxNAWK4FxSzHOoXF7sKgL1uarLTGEYjZhHSIlTS1sUc274GmyBtLrtLFcwN+qibTCmgLMe4Yss/xv86ybp3lFs6WwG18lDFstBqmynOUn8hsysuReOwoAAA=="""
_EMBED_PKL_B64GZ = """H4sIADcVaWkC/71Wa2wUVRTebbvdB23dAoVKiTxjKoSlQFGQym1tCZGJS4BaiIEMd2ennbHzYh5AjQ/A8KidgMJdlWgKMYb4SExVDBEFEcVXjJrURBQijY08FJAGiMYgeO50ZtndQqB/vD/2zD1zzj3fd865Z3ZdQSqW53OWXWa0SjzWlZip83yM5SRsGLxB7BENPCcaoqo0gr6easVmkdfJDnLPevIkqbTDnC6avA4WxC5oERWR2CFDk0QTlKBJ8IZJ7LCM17JJXjMFwhTYpbKosAaWNYk3WMeWMHl2NFMLWJoJ47fLqXINL7YIJtusY86EOP0v5/vcZRfRw5t5bFo6II7bJXRPbVhFTTqaIh0rSVVmDRObPGEm2WX0WFHWLMDeBsA4nccGn3mmkwA3MjBKYAkrHJ8EKhynsVjSBJxhXiyrimqqisixHPCN26UuHlbBMtCBaMQe9piakMRETLFkrY3VRA4yTuzSON3W6TpuW6pjTcvMbciwEg4QYgccL2IHlSSmtmBkBwwBa8AnvBGeVT1JE+6vh+ek2QZ6YbQwAazyFs4i7Vs2k8Wkksm3/U+QeDy+4Bos54dBJknYESxJ6hpWlrFG2u2KfoROHKAqtigyr5hsos2EbDJRK7EukOrzqJc5xjFO1fmYbEmm2A/PLmJ1nlMVw9QtznTQDmTA+DaSen+C9IPzUyJCvgf/9qC3r4A8BRp4GcsQ4kFIeiuxI0t4A7fGsYYhc9EluFU0G3mlpUXV1VYMfRpmeImXsECfg3WKasgiBu8GwA4ViTCqYuLWeTo26VG0jtgQF04nPMSLpUyPeYmS7jqnwEzYjiisapmaZRos7d6Qe41YIpT3l1SoEO4SxjB5QHScMF6YIEy088QskjU5JH0QVJi0RZhME198LXN57ed3ZarTgwY40qFvWqFCg4Oupu0mVNeH3EHgI5sACTR0xp1yqAXoYIDThmYPCvoL15xOB1rQcA5Rf5podZpEyMPvhUx1uAwYP00D9Q7DmIjQ68tyqqXAfCiF/nFuc06A0oxM5jfdW309lUJlvNKOSHyzyXKCKMHVHaI7c8TdBV160A6mACwFlWpD3lQgtL4OAncmEXtk/zjgk2zum6GyCINRaWFbVNZUWRqUmA7KRbkVFmYOrC/j20SExcIKJgRyCcgoyEbq2Xwrz3KwfERIMmNBNoFnJcilsK8CuYyeYE3LSsnAE2ZtIhZTC7cvmtlp+fDe66sCVwZ9masHnXWqeAI96mou0+3TTbWehVfffFeGMt3HNtXWVPVuH9ZxBE12VacmvNr5YntN2v9azrqa3feHcvt/irMCg/af5Mq6p6783dc7O+0fcGVRzt7jf2Fe3X2qdAxNdTUl59+uqv4wXuvhKXRlOCcPnv/RhuLgj30n0HjPvmLezrrttWn/YE7esvK3cmlt48T9Qw/Hv0WjXdUl9HPJ2gUzbpt/zTsvvIRf6UZenYZf+Vp4rSIyyPwdqhnlKlb1HFqfH6xK+0dcOcR3o9WDTnftH776yPvIy9O0s+z8EXsn33b8443ss1N//wJ5ed3tEBg5SPxdH3n+n31K15i0f7Erozl95uHvOlytzaj9GHl2pXu6Y78tut6/Ja684yb8Pz8+p+zy1qPI66stvTsLe7aPHnT/evXrnLPm/PEPLqL/OX81oaz7NzPtn6ryvkeFq7Fk3WB4w9TJ2+wN8GR6+BT2x/G7eQq78pmk+MO40DnUe2zv8r6D+1B2Pi+gbOnzrVg590DgZDfq/m7luH+LTqJlB+ZGmTd70OKNG06Ib/2CDs7+5+Tpl/9AgTtHJpc/9wn6acOUoxumfIkW7Lr00K5LZwac9+eqsmjRV3vQr69fnavuOI8eviBfjby3H/1VN3135xtnB+CJd1x8YNu+LpTaWldecuYc+ib47pCO579Ho67c/fj9+04NOP9W56XGuo9Wwo6y7neYXc3rhvPvOzAtVh2bTqxE7D9Tf5h52QsAAA=="""  # opsional, bisa gagal load jika versi sklearn beda


def _decode_b64gz_to_bytes(b64gz_text: str) -> bytes:
    if not b64gz_text.strip():
        return b""
    raw = base64.b64decode(b64gz_text.encode("ascii"))
    return gzip.decompress(raw)


def load_embedded_dataframe() -> pd.DataFrame:
    csv_bytes = _decode_b64gz_to_bytes(_EMBED_CSV_B64GZ)
    return pd.read_csv(io.BytesIO(csv_bytes))


def try_load_embedded_model() -> DecisionTreeClassifier | None:
    """
    Coba load model PKL yang di-embed.
    Jika tidak kompatibel (sering terjadi di Streamlit Cloud karena versi sklearn beda),
    fungsi ini mengembalikan None dan kita akan train ulang dari CSV.
    """
    try:
        pkl_bytes = _decode_b64gz_to_bytes(_EMBED_PKL_B64GZ)
        if not pkl_bytes:
            return None
        obj = pickle.loads(pkl_bytes)
        if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
            return obj
        return None
    except Exception:
        return None


# -----------------------------
# KONFIG APLIKASI
# -----------------------------

st.set_page_config(
    page_title="Deteksi Gejala COVID-19 (Decision Tree)",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Deteksi Gejala COVID-19 (Decision Tree)")
st.caption("Aplikasi edukasi: hasil prediksi bukan diagnosis medis. Jika gejala berat, hubungi tenaga kesehatan.")

menu = st.sidebar.radio("Navigasi", ["Deteksi", "Tentang"], index=0)

# -----------------------------
# TRAIN / LOAD MODEL (CACHE)
# -----------------------------

FEATURE_COLS = [
    "Demam",
    "Batuk",
    "SesakNapas",
    "SakitTenggorokan",
    "Kelelahan",
    "Anosmia",
    "Diare",
    "KontakErat",
    "SaturasiO2",
]
TARGET_COL = "LabelCOVID"


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    df = load_embedded_dataframe()
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing_cols}")
    return df


@st.cache_resource(show_spinner=False)
def get_model_and_metrics() -> Tuple[DecisionTreeClassifier, Dict[str, object]]:
    df = get_dataset()

    # 1) coba load pkl dulu (kalau kompatibel)
    model = try_load_embedded_model()

    # 2) fallback: train ulang dari csv
    if model is None:
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float((y_pred == y_test).mean())
        rules = export_text(model, feature_names=FEATURE_COLS)
        metrics = {
            "accuracy": acc,
            "rules": rules,
            "used_pkl": False,
        }
    else:
        rules = export_text(model, feature_names=FEATURE_COLS)
        metrics = {
            "accuracy": None,
            "rules": rules,
            "used_pkl": True,
        }

    return model, metrics


def predict_one(model: DecisionTreeClassifier, x: np.ndarray) -> Tuple[int, float]:
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0, 1])
    return pred, proba


# -----------------------------
# HALAMAN DETEKSI
# -----------------------------

if menu == "Deteksi":
    try:
        df = get_dataset()
        model, metrics = get_model_and_metrics()
    except Exception as e:
        st.error(f"Aplikasi gagal memuat dataset/model: {e}")
        st.stop()

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Input Gejala")

        c1, c2 = st.columns(2)
        with c1:
            demam = st.selectbox("Demam", ["Tidak", "Ya"], index=0)
            batuk = st.selectbox("Batuk", ["Tidak", "Ya"], index=0)
            sesak = st.selectbox("Sesak Napas", ["Tidak", "Ya"], index=0)
            tenggorokan = st.selectbox("Sakit Tenggorokan", ["Tidak", "Ya"], index=0)
            lelah = st.selectbox("Kelelahan", ["Tidak", "Ya"], index=0)

        with c2:
            anosmia = st.selectbox("Anosmia (hilang penciuman)", ["Tidak", "Ya"], index=0)
            diare = st.selectbox("Diare", ["Tidak", "Ya"], index=0)
            kontak = st.selectbox("Kontak Erat", ["Tidak", "Ya"], index=0)
            saturasi = st.number_input("Saturasi O2 (SpO2)", min_value=80, max_value=100, value=96, step=1)

        x = np.array([[
            1 if demam == "Ya" else 0,
            1 if batuk == "Ya" else 0,
            1 if sesak == "Ya" else 0,
            1 if tenggorokan == "Ya" else 0,
            1 if lelah == "Ya" else 0,
            1 if anosmia == "Ya" else 0,
            1 if diare == "Ya" else 0,
            1 if kontak == "Ya" else 0,
            int(saturasi),
        ]], dtype=float)

        if st.button("🔍 Prediksi", type="primary"):
            pred, proba = predict_one(model, x)
            if pred == 1:
                st.error(f"**Hasil: COVID (Positif)**\n\nProbabilitas: **{proba:.3f}**")
            else:
                st.success(f"**Hasil: Tidak COVID (Negatif)**\n\nProbabilitas COVID: **{proba:.3f}**")

        st.markdown("---")
        with st.expander("📄 Lihat Data (5 baris pertama)"):
            st.dataframe(df.head(), use_container_width=True)

    with right:
        st.subheader("Info Model")
        if metrics.get("accuracy") is not None:
            st.metric("Akurasi (test split)", f"{metrics['accuracy']:.3f}")
        else:
            if metrics.get("used_pkl"):
                st.info("Model dimuat dari PKL. (Akurasi test split tidak dihitung ulang di Cloud.)")
            else:
                st.info("Model dibuat ulang dari dataset.")

        with st.expander("🌳 Rules Decision Tree"):
            st.code(metrics["rules"])

        with st.expander("🖼️ Diagram Pohon (opsional)"):
            try:
                import matplotlib.pyplot as plt
                from sklearn.tree import plot_tree

                fig = plt.figure(figsize=(12, 6))
                plot_tree(
                    model,
                    feature_names=FEATURE_COLS,
                    class_names=["Tidak", "Ya"],
                    filled=True,
                    rounded=True,
                    fontsize=8,
                )
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.warning(
                    "Diagram membutuhkan matplotlib. Pastikan `matplotlib` ada di requirements.txt.\n"
                    f"Detail: {e}"
                )

# -----------------------------
# HALAMAN TENTANG
# -----------------------------
else:
    st.subheader("Tentang Aplikasi")
    st.write(
        """
Aplikasi ini dibuat untuk memenuhi tugas Decision Tree:
- Menyusun tabel data gejala
- Membuat model Decision Tree
- Implementasi program Python
- Implementasi UI di Streamlit

Catatan:
- Dataset pada contoh ini adalah data latihan/edukasi.
- Hasil prediksi bukan pengganti diagnosis medis.
"""
    )
    with st.expander("📌 Cara Jalankan Lokal (Windows)"):
        st.code("py -m pip install streamlit scikit-learn pandas numpy matplotlib\npy -m streamlit run app.py")


from __future__ import annotations

import base64
import io
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def _optional_matplotlib():
    """Import matplotlib lazily. Return (plt, error_message_or_None)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt, None
    except Exception as e:
        return None, str(e)




DATA_CSV = r"""\
Demam,Batuk,SesakNapas,SakitTenggorokan,Kelelahan,Anosmia,Diare,KontakErat,SaturasiO2,LabelCOVID
1,1,0,1,1,0,0,0,96,0
0,1,0,1,0,0,0,1,97,0
1,1,0,1,1,1,0,1,96,1
0,0,0,1,1,0,0,0,95,0
0,0,0,1,0,0,0,0,95,0
1,1,0,0,1,0,0,0,98,0
1,1,0,0,1,1,0,1,99,1
0,1,0,0,1,0,0,0,98,0
0,0,1,0,0,0,0,1,94,0
1,0,1,0,0,1,0,0,94,0
1,0,0,0,1,0,0,1,94,0
1,1,0,0,1,0,0,1,98,0
0,1,0,0,0,0,0,1,96,0
1,1,0,0,1,1,0,0,95,0
0,0,0,0,1,0,0,1,95,0
0,1,0,1,0,0,0,0,98,0
1,1,0,0,0,0,0,0,96,0
1,1,0,0,0,0,0,0,96,0
0,1,0,0,1,1,0,1,95,1
0,1,0,0,0,1,1,1,97,1
1,1,0,1,1,0,0,1,98,1
1,0,0,1,0,1,1,0,94,0
0,0,0,1,0,1,0,0,97,0
0,0,0,0,0,1,0,0,93,0
0,0,1,1,1,0,0,0,97,0
1,0,1,0,1,0,0,1,93,1
0,1,1,0,1,1,0,1,91,1
0,0,0,1,1,1,0,0,95,0
0,0,0,1,1,1,0,0,95,0
0,1,0,0,1,0,0,1,96,0
0,0,0,0,1,0,0,1,97,0
0,1,1,0,0,0,0,0,95,0
0,0,1,0,1,0,0,0,92,0
1,1,0,0,0,1,0,0,99,0
1,0,0,0,0,0,0,1,94,0
0,0,0,0,1,0,1,0,99,0
1,1,1,0,1,0,0,1,90,1
1,0,0,0,1,0,0,0,94,0
1,1,0,1,1,1,0,1,95,1
1,0,0,0,0,0,0,0,97,0
0,1,0,0,0,0,0,1,98,0
1,0,1,1,0,1,0,1,92,1
0,0,0,0,1,1,0,1,98,0
1,1,1,0,1,0,0,0,94,0
1,0,0,0,0,0,1,1,97,0
1,1,0,1,0,0,1,0,96,0
1,0,0,0,1,0,0,0,96,0
0,0,0,0,1,0,0,1,93,0
1,1,0,1,0,1,0,0,99,0
1,0,0,0,0,0,0,0,99,0
1,0,1,1,0,0,0,1,90,1
1,1,0,0,1,0,0,1,94,0
1,0,0,0,0,0,0,1,97,0
0,0,0,0,0,0,0,1,94,0
0,0,0,0,1,0,0,1,96,0
1,0,0,0,0,0,0,1,100,0
1,1,0,0,1,0,0,1,95,0
0,0,1,0,0,0,0,1,91,0
1,0,0,0,0,1,0,1,96,0
1,0,0,1,0,1,0,0,96,0
0,0,1,1,1,0,0,1,92,1
0,0,0,0,0,0,0,0,95,0
0,0,0,0,0,0,0,0,94,0
1,1,0,1,0,1,0,1,98,1
0,1,1,0,1,1,0,1,89,1
0,0,0,1,1,0,0,0,98,0
1,0,0,0,0,0,0,0,95,0
1,1,0,1,1,0,0,0,95,0
0,1,0,0,1,1,1,1,95,1
1,1,0,0,0,0,0,1,96,0
1,0,0,0,1,0,0,1,97,0
0,0,0,0,1,1,0,1,97,0
1,0,0,0,1,0,0,0,97,0
0,0,1,1,0,0,1,1,93,1
0,1,0,0,0,0,0,1,98,0
0,0,0,0,1,1,0,1,94,0
0,0,0,0,0,1,0,0,95,0
0,0,1,1,0,0,0,0,91,0
1,1,0,0,0,0,0,1,97,0
0,1,0,1,0,1,0,1,98,1
1,1,0,0,0,1,0,1,96,1
0,0,1,0,0,0,0,1,94,0
0,0,0,1,0,0,0,1,96,0
0,0,1,0,0,0,0,0,94,0
0,0,0,1,0,0,0,0,98,0
0,1,0,0,0,0,1,1,97,0
0,1,0,0,1,0,0,1,97,0
1,0,0,1,0,1,1,0,95,0
0,1,0,1,1,1,0,1,97,1
0,0,0,0,0,0,0,1,99,0
1,1,0,0,1,0,0,0,97,0
0,0,1,0,0,1,1,1,92,1
0,1,0,1,1,0,0,0,98,0
1,0,0,0,0,0,0,1,98,0
1,0,0,0,1,1,0,1,96,1
1,0,0,0,1,0,0,1,99,0
0,0,0,1,0,1,0,1,98,0
1,1,0,1,1,0,0,1,96,1
0,1,0,1,1,0,0,0,97,0
0,0,0,1,0,0,0,0,97,0
0,1,0,1,0,0,0,0,96,0
0,0,0,0,0,1,0,0,97,0
1,0,0,0,0,1,0,0,96,0
1,1,1,0,0,1,0,0,86,1
0,0,0,0,1,0,0,1,95,0
0,0,0,1,0,0,1,1,100,0
0,1,0,0,0,0,0,0,98,0
1,0,0,0,0,1,0,1,95,0
0,0,0,0,0,1,0,0,97,0
0,1,0,0,0,1,0,1,98,0
0,1,0,0,0,1,0,0,92,0
1,0,0,1,0,1,0,0,99,0
1,1,0,1,0,0,0,0,96,0
0,0,0,0,0,1,0,1,97,0
1,0,0,0,0,0,1,0,94,0
1,1,0,0,1,0,0,1,97,0
0,1,0,1,0,0,0,0,98,0
1,0,0,0,0,0,0,1,97,0
1,1,0,0,0,0,0,1,94,0
1,0,0,0,0,0,0,0,97,0

"""

MODEL_PKL_B64 = "gASVLgIAAAAAAACMFXNrbGVhcm4udHJlZS5fY2xhc3Nlc5SMFkRlY2lzaW9uVHJlZUNsYXNzaWZpZXKUk5QpgZR9lCiMCWNyaXRlcmlvbpSMBGdpbmmUjAhzcGxpdHRlcpSMBGJlc3SUjAltYXhfZGVwdGiUSwSMEW1pbl9zYW1wbGVzX3NwbGl0lEsCjBBtaW5fc2FtcGxlc19sZWFmlEsBjBhtaW5fd2VpZ2h0X2ZyYWN0aW9uX2xlYWaURwAAAAAAAAAAjAxtYXhfZmVhdHVyZXOUTowObWF4X2xlYWZfbm9kZXOUTowMcmFuZG9tX3N0YXRllEsqjBVtaW5faW1wdXJpdHlfZGVjcmVhc2WURwAAAAAAAAAAjAxjbGFzc193ZWlnaHSUjAhiYWxhbmNlZJSMCWNjcF9hbHBoYZRHAAAAAAAAAACMDW1vbm90b25pY19jc3SUTowRZmVhdHVyZV9uYW1lc19pbl+UjBNqb2JsaWIubnVtcHlfcGlja2xllIwRTnVtcHlBcnJheVdyYXBwZXKUk5QpgZR9lCiMCHN1YmNsYXNzlIwFbnVtcHmUjAduZGFycmF5lJOUjAVzaGFwZZRLCYWUjAVvcmRlcpSMAUOUjAVkdHlwZZRoHGgjk5SMAk84lImIh5RSlChLA4wBfJROTk5K/////0r/////Sz90lGKMCmFsbG93X21tYXCUiYwbbnVtcHlfYXJyYXlfYWxpZ25tZW50X2J5dGVzlEsQdWKABZXxAAAAAAAAAIwVbnVtcHkuY29yZS5tdWx0aWFycmF5lIwMX3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RLAIWUQwFilIeUUpQoSwFLCYWUaAOMBWR0eXBllJOUjAJPOJSJiIeUUpQoSwOMAXyUTk5OSv////9K/////0s/dJRiiV2UKIwFRGVtYW2UjAVCYXR1a5SMClNlc2FrTmFwYXOUjBBTYWtpdFRlbmdnb3Jva2FulIwJS2VsZWxhaGFulIwHQW5vc21pYZSMBURpYXJllIwKS29udGFrRXJhdJSMClNhdHVyYXNpTzKUZXSUYi6VdAAAAAAAAACMDm5fZmVhdHVyZXNfaW5flEsJjApuX291dHB1dHNflEsBjAhjbGFzc2VzX5RoGCmBlH2UKGgbaB5oH0sChZRoIWgiaCNoJIwCaTiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYmgqiGgrSxB1Yg3/////////////////AAAAAAAAAAABAAAAAAAAAJWeAAAAAAAAAIwKbl9jbGFzc2VzX5SMFW51bXB5LmNvcmUubXVsdGlhcnJheZSMBnNjYWxhcpSTlGg0QwgCAAAAAAAAAJSGlFKUjA1tYXhfZmVhdHVyZXNflEsJjAV0cmVlX5SMEnNrbGVhcm4udHJlZS5fdHJlZZSMBFRyZWWUk5RLCWgYKYGUfZQoaBtoHmgfSwGFlGghaCJoI2g0aCqIaCtLEHViCP//////////AgAAAAAAAACViwEAAAAAAABLAYeUUpR9lChoCUsEjApub2RlX2NvdW50lEsRjAVub2Rlc5RoGCmBlH2UKGgbaB5oH0sRhZRoIWgiaCNoJIwDVjY0lImIh5RSlChLA2goTiiMCmxlZnRfY2hpbGSUjAtyaWdodF9jaGlsZJSMB2ZlYXR1cmWUjAl0aHJlc2hvbGSUjAhpbXB1cml0eZSMDm5fbm9kZV9zYW1wbGVzlIwXd2VpZ2h0ZWRfbl9ub2RlX3NhbXBsZXOUjBJtaXNzaW5nX2dvX3RvX2xlZnSUdJR9lChoUWgkjAJpOJSJiIeUUpQoSwNoNU5OTkr/////Sv////9LAHSUYksAhpRoUmhdSwiGlGhTaF1LEIaUaFRoJIwCZjiUiYiHlFKUKEsDaDVOTk5K/////0r/////SwB0lGJLGIaUaFVoZEsghpRoVmhdSyiGlGhXaGRLMIaUaFhoJIwCdTGUiYiHlFKUKEsDaChOTk5K/////0r/////SwB0lGJLOIaUdUtASwFLEHSUYmgqiGgrSxB1YgP///8BAAAAAAAAAAQAAAAAAAAABwAAAAAAAAAAAAAAAADgP+z//////98/WgAAAAAAAAD1/////39WQAAAAAAAAAAAAgAAAAAAAAADAAAAAAAAAAgAAAAAAAAAAAAAAAAgVkA8MOOSE4vFPysAAAAAAAAA5yOinpeJPEAAAAAAAAAAAP/////////////////////+/////////wAAAAAAAADAAAAAAAAAAAABAAAAAAAAAC0tLS0tLQVAAAAAAAAAAAD//////////////////////v////////8AAAAAAAAAwAAAAAAAAAAAKgAAAAAAAABBfvz48eM5QAAAAAAAAAAABQAAAAAAAAAMAAAAAAAAAAUAAAAAAAAAAAAAAAAA4D/wRUE3b2zbPy8AAAAAAAAADu6uMDS7TkABAAAAAAAAAAYAAAAAAAAACQAAAAAAAAADAAAAAAAAAAAAAAAAAOA/2EQNB9fx3z8iAAAAAAAAAAYbRZlBkkBAAQAAAAAAAAAHAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAAAAAAAGBXQFQkuhLCTs4/HAAAAAAAAAD0P9oOeEozQAAAAAAAAAAA//////////////////////7/////////AAAAAAAAAMA8sZaYYZzTPwIAAAAAAAAAFPzKaKQbCkAAAAAAAAAAAP/////////////////////+/////////wAAAAAAAADAAAAAAAAAwDwaAAAAAAAAAHHgwIEDBzBAAAAAAAAAAAAKAAAAAAAAAAsAAAAAAAAAAAAAAAAAAAAAAAAAAADgP+iwuhR2xbU/BgAAAAAAAAAx7F9HFrQrQAAAAAAAAAAA//////////////////////7/////////AAAAAAAAAMDcVF+KL+rHPwMAAAAAAAAAoBT8ymikF0AAAAAAAAAAAP/////////////////////+/////////wAAAAAAAADAAAAAAAAAsLwDAAAAAAAAAMTDw8PDwx9AAAAAAAAAAAANAAAAAAAAABAAAAAAAAAAAQAAAAAAAAAAAAAAAADgP7DCNHAzQL8/DQAAAAAAAAARptMu5VE8QAAAAAAAAAAADgAAAAAAAAAPAAAAAAAAAAAAAAAAAAAAAAAAAAAA4D/G3DsV9Y7YPwUAAAAAAAAAiOOZBuCSHEAAAAAAAAAAAP/////////////////////+/////////wAAAAAAAADAAAAAAAAAAAACAAAAAAAAAJ47d+7cufM/AAAAAAAAAAD//////////////////////v////////8AAAAAAAAAwNxUX4ov6sc/AwAAAAAAAACgFPzKaKQXQAAAAAAAAAAA//////////////////////7/////////AAAAAAAAAMAAAAAAAACwPAgAAAAAAAAALS0tLS0tNUAAAAAAAAAAAJUwAAAAAAAAAIwGdmFsdWVzlGgYKYGUfZQoaBtoHmgfSxFLAUsCh5RoIWgiaCNoZGgqiGgrSxB1Ygb///////8BAAAAAADgPwkAAAAAAOA/g2Rp1CEI7T/j27Rc8b63PwAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAAAAXWA+vQXm0z/Tz2Ah/QzmP1i9PhBLquA/UoWC32mr3j++Ofrm6JvrPwUZF2RckME/2YIt2IItyD9Kn/RJn/TpPwAAAAAAAPA/AAAAAAAAAADvcRUQDMmmP+Ko/j5vk+4/TfBt/gqzuj/3QTKgnqnsPwAAAAAAAAAAAAAAAAAA8D9Oi/M9j7ewP5WOQRgO6e0/zAeyC4uR0D8a/CZ6OrfnPwAAAAAAAPA/AAAAAAAAAABN8G3+CrO6P/dBMqCeqew/AAAAAAAAAAAAAAAAAADwP5UgAAAAAAAAAHVijBBfc2tsZWFybl92ZXJzaW9ulIwFMS40LjKUdWIu"

TARGET_COL = "LabelCOVID"


@dataclass
class Artifacts:
    df: pd.DataFrame
    features: List[str]
    model: DecisionTreeClassifier
    accuracy: float
    cm: np.ndarray
    report: str
    rules: str


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(DATA_CSV))
    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan di dataset.")
    return df


def _train_model(df: pd.DataFrame, seed: int = 42) -> Artifacts:
    features = [c for c in df.columns if c != TARGET_COL]
    X = df[features].copy()

    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.25, random_state=seed, stratify=y.values
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=8,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    rules = export_text(model, feature_names=features)

    return Artifacts(df=df, features=features, model=model, accuracy=acc, cm=cm, report=report, rules=rules)


def _try_load_embedded_model():
    try:
        raw = base64.b64decode(MODEL_PKL_B64.encode("ascii"))
        obj = pickle.loads(raw)
        if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
            return obj
        return None
    except Exception:
        return None


def get_model_and_features(df: pd.DataFrame):
    features = [c for c in df.columns if c != TARGET_COL]

    model = _try_load_embedded_model()

    artifacts = _train_model(df, seed=42)

    if model is None:
        model = artifacts.model

    return model, features, artifacts


def make_tree_figure(model: DecisionTreeClassifier, features: List[str]):
    plt, err = _optional_matplotlib()
    if plt is None:
        raise ModuleNotFoundError(
            'matplotlib belum terpasang. Install dulu: pip install matplotlib. ' + f'Detail: {err}'
        )
    fig = plt.figure(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=features,
        class_names=["Tidak COVID", "COVID"],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.tight_layout()
    return fig


def predict_one(model: DecisionTreeClassifier, features: List[str], x_dict: Dict[str, float]):
    x = np.array([[float(x_dict[f]) for f in features]], dtype=float)
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0, 1])
    return pred, proba



def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Deteksi Gejala COVID-19", page_icon="🩺", layout="wide")

    @st.cache_resource
    def _load():
        df_local = load_dataset()
        model_local, features_local, artifacts_local = get_model_and_features(df_local)
        return df_local, model_local, features_local, artifacts_local

    df, model, features, artifacts = _load()

    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio("Pilih menu", ["Deteksi", "Tentang"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.write("✅ Single file: dataset & model sudah tertanam.")

    if menu == "Tentang":
        st.title("Tentang Aplikasi")
        st.write(
            "Aplikasi ini menggunakan **Decision Tree** untuk memprediksi label COVID-19 "
            "berdasarkan gejala. Dataset CSV dan model (opsional) sudah di-embed dalam file ini."
        )
        st.info("Catatan: hasil prediksi bukan diagnosis medis.")
        st.subheader("Contoh data (5 baris)")
        st.dataframe(df.head(), use_container_width=True)
        return

    st.title("Deteksi Gejala COVID-19 (Decision Tree)")
    st.caption("Aplikasi edukasi: hasil prediksi bukan diagnosis medis.")

    st.subheader("Input Gejala")
    inputs: Dict[str, float] = {}

    cols = st.columns(3)
    for i, f in enumerate(features):
        with cols[i % 3]:
            if "saturasi" in f.lower() or "o2" in f.lower():
                default_val = float(pd.to_numeric(df[f], errors="coerce").dropna().median())
                val = st.slider(f"{f} (angka)", min_value=80, max_value=100, value=int(round(default_val)))
                inputs[f] = float(val)
            else:
                uniq = set(pd.to_numeric(df[f], errors="coerce").dropna().unique().tolist())
                if uniq.issubset({0, 1}):
                    opt = st.radio(f, ["Tidak", "Ya"], horizontal=True)
                    inputs[f] = 1.0 if opt == "Ya" else 0.0
                else:
                    default_val = float(pd.to_numeric(df[f], errors="coerce").dropna().median())
                    val = st.number_input(f"{f} (angka)", value=default_val)
                    inputs[f] = float(val)

    if st.button("🔍 Prediksi", type="primary"):
        pred, proba = predict_one(model, features, inputs)
        if pred == 1:
            st.error(f"Prediksi: **COVID (Positif)**  | Probabilitas: **{proba:.3f}**")
        else:
            st.success(f"Prediksi: **Tidak COVID (Negatif)** | Probabilitas COVID: **{proba:.3f}**")

    with st.expander("📌 Metrik Model & Rules (Decision Tree)"):
        st.write(f"Akurasi (test split): **{artifacts.accuracy:.3f}**")
        st.write("Confusion Matrix (TN FP / FN TP):")
        st.code(str(artifacts.cm))
        st.text("Classification report:")
        st.code(artifacts.report)
        st.text("Rules (export_text):")
        st.code(artifacts.rules)

    with st.expander("🌳 Diagram Decision Tree"):
        try:
            fig = make_tree_figure(model, features)
            st.pyplot(fig, clear_figure=True)
        except ModuleNotFoundError as e:
            st.warning("Diagram butuh matplotlib. Install/deploy dengan menambahkan `matplotlib` ke requirements.\n\nDetail: " + str(e))


def main():
    import sys
    if not any("streamlit" in a.lower() for a in sys.argv):
        print("File ini adalah Streamlit app.")
        print("Jalankan dengan:")
        print("  py -m streamlit run app_single.py")
        return

    run_streamlit()


if __name__ == "__main__":
    main()

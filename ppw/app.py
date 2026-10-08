"""
Aplikasi Streamlit: Klasifikasi Judul Berita Detik (Hot, Health, Sport)
Model: Skip-gram (Word2Vec, Gensim) + Naive Bayes (GaussianNB)

Cara menjalankan:
    streamlit run app.py

File yang WAJIB ada satu folder dengan app.py ini (hasil dari notebook
skipgram_klasifikasi_berita_detik.ipynb, Bagian 10):
    - model_skipgram.model   (Word2Vec Skip-gram)
    - scaler.pkl             (StandardScaler)
    - model_naive_bayes.pkl  (GaussianNB final)
    - label_encoder.pkl      (LabelEncoder: hot/health/sport)
"""

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec

# =========================================================
# Konfigurasi halaman
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Berita Detik",
    page_icon="📰",
    layout="centered",
)

# =========================================================
# Memuat model (di-cache supaya tidak reload tiap kali ada interaksi)
# =========================================================
@st.cache_resource
def muat_model():
    model_w2v = Word2Vec.load("model_skipgram.model")
    scaler = joblib.load("scaler.pkl")
    clf = joblib.load("model_naive_bayes.pkl")
    le = joblib.load("label_encoder.pkl")
    return model_w2v, scaler, clf, le


def tokenisasi(teks: str):
    teks = str(teks).lower()
    tokens = re.findall(r"[a-zA-Z]+", teks)
    return tokens


def vektor_dokumen(tokens, model_w2v):
    vektor_kata = [model_w2v.wv[w] for w in tokens if w in model_w2v.wv]
    if len(vektor_kata) == 0:
        return np.zeros(model_w2v.vector_size), 0, len(tokens)
    return np.mean(vektor_kata, axis=0), len(vektor_kata), len(tokens)


def klasifikasi_judul(judul: str, model_w2v, scaler, clf, le):
    tokens = tokenisasi(judul)
    vektor, jumlah_dikenali, jumlah_total = vektor_dokumen(tokens, model_w2v)
    vektor_scaled = scaler.transform(vektor.reshape(1, -1))

    prediksi_encoded = clf.predict(vektor_scaled)[0]
    prediksi_label = le.inverse_transform([prediksi_encoded])[0]

    proba = clf.predict_proba(vektor_scaled)[0]
    df_proba = pd.DataFrame({
        "Kategori": le.classes_,
        "Probabilitas": proba
    }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

    return prediksi_label, df_proba, tokens, jumlah_dikenali, jumlah_total


# =========================================================
# Tampilan utama
# =========================================================
st.title("📰 Klasifikasi Judul Berita Detik")
st.markdown(
    "Masukkan judul berita, lalu sistem akan memprediksi kategorinya: "
    "**Hot** (hiburan/selebriti), **Health** (kesehatan), atau **Sport** (olahraga).\n\n"
    "Model dilatih menggunakan representasi vektor **Skip-gram (Word2Vec)** "
    "dan diklasifikasikan dengan **Naive Bayes**, berdasarkan 300 judul berita "
    "asli dari detik.com."
)

try:
    model_w2v, scaler, clf, le = muat_model()
except FileNotFoundError as e:
    st.error(
        "File model belum ditemukan. Pastikan file-file berikut ada satu folder "
        "dengan app.py ini:\n\n"
        "- model_skipgram.model\n- scaler.pkl\n- model_naive_bayes.pkl\n- label_encoder.pkl\n\n"
        f"Detail error: {e}"
    )
    st.stop()

st.divider()

judul_input = st.text_area(
    "Masukkan judul berita:",
    placeholder="Contoh: Timnas Indonesia menang telak di laga final sepak bola",
    height=100,
)

contoh_judul = {
    "Contoh (Hot)": "Artis ibu kota umumkan kehamilan anak pertama lewat media sosial",
    "Contoh (Health)": "Dokter sarankan rutin olahraga untuk jaga kesehatan jantung",
    "Contoh (Sport)": "Timnas Indonesia menang telak di laga final sepak bola",
}

kolom = st.columns(3)
for i, (label_tombol, contoh) in enumerate(contoh_judul.items()):
    if kolom[i].button(label_tombol, use_container_width=True):
        judul_input = contoh
        st.session_state["judul_terpilih"] = contoh

if "judul_terpilih" in st.session_state and not judul_input:
    judul_input = st.session_state["judul_terpilih"]

if st.button("🔍 Klasifikasikan", type="primary", use_container_width=True):
    if not judul_input or not judul_input.strip():
        st.warning("Silakan masukkan judul berita terlebih dahulu.")
    else:
        prediksi, df_proba, tokens, jumlah_dikenali, jumlah_total = klasifikasi_judul(
            judul_input, model_w2v, scaler, clf, le
        )

        emoji_kategori = {"hot": "🔥", "health": "🏥", "sport": "⚽"}
        st.success(
            f"### Prediksi Kategori: {emoji_kategori.get(prediksi, '')} **{prediksi.upper()}**"
        )

        st.subheader("Probabilitas tiap kategori")
        st.bar_chart(df_proba.set_index("Kategori"))
        st.dataframe(df_proba, use_container_width=True, hide_index=True)

        with st.expander("Detail pemrosesan teks"):
            st.write(f"**Token hasil tokenisasi:** {tokens}")
            st.write(
                f"**Kata yang dikenali model Skip-gram:** {jumlah_dikenali} dari {jumlah_total} kata"
            )
            if jumlah_dikenali == 0:
                st.warning(
                    "Tidak ada satupun kata pada judul ini yang dikenali oleh model "
                    "(mungkin semua kata baru/di luar vocabulary training). "
                    "Prediksi mungkin kurang akurat."
                )
            elif jumlah_dikenali < jumlah_total:
                st.info(
                    "Beberapa kata tidak dikenali model (di luar vocabulary training) "
                    "dan diabaikan saat menghitung vektor rata-rata."
                )

st.divider()
st.caption(
    "Model: Skip-gram (Gensim Word2Vec) + Naive Bayes (GaussianNB) · "
    "Data latih: 300 judul berita asli detik.com (100 hot + 100 health + 100 sport)"
)

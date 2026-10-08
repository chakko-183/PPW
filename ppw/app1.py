"""
Aplikasi Streamlit: Klasifikasi Berita Detik dari LINK (Hot, Health, Sport)
Model: Skip-gram (Word2Vec, Gensim) + Naive Bayes (GaussianNB)

Cara kerja:
    1. Pengguna memasukkan link artikel berita.
    2. Aplikasi mengambil JUDUL artikel dari halaman tersebut
       (meta og:title -> <h1> -> <title>). Jika halaman tidak bisa dibuka,
       judul dibentuk dari bagian akhir (slug) link, mis.
       ".../d-4965904/andrea-dian-sembuh-dari-corona" -> "andrea dian sembuh dari corona".
    3. Judul diubah menjadi vektor Skip-gram (rata-rata vektor kata), diskalakan,
       lalu diklasifikasikan oleh Naive Bayes -> hot / health / sport.

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py

File yang WAJIB ada satu folder dengan app.py ini (hasil dari notebook
klasifikasi_skipgram.ipynb):
    - model_skipgram.model   (Word2Vec Skip-gram)
    - scaler.pkl             (StandardScaler)
    - model_naive_bayes.pkl  (GaussianNB final)
    - label_encoder.pkl      (LabelEncoder: hot/health/sport)
"""

import re
import socket
import ipaddress
from urllib.parse import urlparse, urljoin, unquote

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

# =========================================================
# Konfigurasi halaman
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Berita Detik dari Link",
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


# =========================================================
# Bagian 1: dari LINK -> JUDUL
# =========================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "id,en;q=0.8",
}


def normalisasi_url(url: str) -> str:
    url = url.strip()
    if url and not re.match(r"^https?://", url, flags=re.I):
        url = "https://" + url
    return url


def host_aman(host: str) -> bool:
    """Tolak alamat internal (localhost / jaringan privat) agar aplikasi tidak
    bisa disalahgunakan untuk mengakses layanan di jaringan sendiri (SSRF)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            return False
    return True


def ambil_html(url: str, timeout: int = 10, maks_byte: int = 2_000_000,
               cek_host: bool = True):
    """Mengunduh halaman (maks. 5x redirect, maks. ~2 MB). Mengembalikan (bytes, url_akhir)."""
    for _ in range(5):
        host = urlparse(url).hostname
        if not host:
            raise ValueError("Link tidak valid.")
        if cek_host and not host_aman(host):
            raise ValueError("Alamat tidak dapat diakses (tidak ditemukan atau alamat internal).")

        resp = requests.get(url, headers=HEADERS, timeout=timeout,
                            allow_redirects=False, stream=True)
        if resp.status_code in (301, 302, 303, 307, 308):
            lokasi = resp.headers.get("Location")
            resp.close()
            if not lokasi:
                break
            url = urljoin(url, lokasi)
            continue

        resp.raise_for_status()
        konten = b""
        for potongan in resp.iter_content(65536):
            konten += potongan
            if len(konten) > maks_byte:
                break
        resp.close()
        return konten, url
    raise ValueError("Terlalu banyak pengalihan (redirect).")


def bersihkan_judul(judul: str) -> str:
    """Rapikan spasi dan buang akhiran nama situs, mis. 'Judul - detikHealth'."""
    judul = re.sub(r"\s+", " ", judul or "").strip()
    pola_akhiran = re.compile(r"\s*[-–—|]\s*[^-–—|]*detik[^-–—|]*$", flags=re.I)
    while True:
        baru = pola_akhiran.sub("", judul).strip()
        if baru == judul:
            break
        judul = baru
    return judul


def ekstrak_judul_html(konten) -> str:
    """Ambil judul artikel dari HTML: og:title -> twitter:title -> <h1> -> <title>."""
    soup = BeautifulSoup(konten, "html.parser")
    kandidat = []
    for atribut, nilai in [("property", "og:title"), ("name", "twitter:title")]:
        tag = soup.find("meta", attrs={atribut: nilai})
        if tag and tag.get("content"):
            kandidat.append(tag["content"])
    h1 = soup.find("h1")
    if h1:
        kandidat.append(h1.get_text(" ", strip=True))
    if soup.title and soup.title.string:
        kandidat.append(soup.title.string)

    for k in kandidat:
        k = bersihkan_judul(k)
        if len(k) >= 5:
            return k
    return ""


def judul_dari_slug(url: str) -> str:
    """Cadangan: bentuk judul dari bagian akhir link (slug)."""
    path = unquote(urlparse(url).path).strip("/")
    if not path:
        return ""
    slug = path.split("/")[-1]
    slug = re.sub(r"\.(html?|php|aspx?)$", "", slug, flags=re.I)
    slug = re.sub(r"[-_+]+", " ", slug).strip()
    if slug.replace(" ", "").isdigit() or len(slug) < 5:
        return ""
    return slug


def channel_detik(url: str):
    """Channel pada alamat detik (hot/health/sport), mis. health.detik.com -> 'health'."""
    host = (urlparse(url).hostname or "").lower()
    m = re.match(r"^(hot|health|sport)\.detik\.com$", host)
    return m.group(1) if m else None


def ambil_judul_dari_url(url: str, pakai_halaman: bool = True) -> dict:
    url = normalisasi_url(url)
    hasil = {"url": url, "judul": "", "sumber": "", "peringatan": "",
             "channel": channel_detik(url)}

    if pakai_halaman:
        try:
            konten, url_akhir = ambil_html(url)
            judul = ekstrak_judul_html(konten)
            hasil["channel"] = channel_detik(url_akhir) or hasil["channel"]
            if judul:
                hasil.update(judul=judul, sumber="halaman artikel (meta og:title / h1 / title)")
                return hasil
            hasil["peringatan"] = "Halaman terbuka tetapi judul tidak ditemukan."
        except Exception as e:
            hasil["peringatan"] = f"Halaman tidak bisa diambil ({e})."

    judul = judul_dari_slug(url)
    if judul:
        hasil.update(judul=judul, sumber="slug pada link")
        if hasil["peringatan"]:
            hasil["peringatan"] += " Judul dibentuk dari slug link."
    elif not hasil["peringatan"]:
        hasil["peringatan"] = "Judul tidak dapat dibentuk dari link ini."
    return hasil


# =========================================================
# Bagian 2: JUDUL -> KATEGORI (sama seperti sebelumnya)
# =========================================================
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
st.title("📰 Klasifikasi Berita Detik dari Link")
st.markdown(
    "Tempelkan **link artikel berita**, lalu sistem akan memprediksi kategorinya: "
    "**Hot** (hiburan/selebriti), **Health** (kesehatan), atau **Sport** (olahraga).\n\n"
    "Aplikasi mengambil **judul** artikel dari link tersebut, mengubahnya menjadi vektor "
    "**Skip-gram (Word2Vec)**, lalu diklasifikasikan dengan **Naive Bayes**. "
    "Model dilatih dari 300 judul berita asli detik.com."
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

contoh_link = {
    "Contoh (Hot)": "https://hot.detik.com/celeb/d-4954259/eyangnya-meninggal-kaesang-menyesal-belum-sempat-kirim-sang-pisang",
    "Contoh (Health)": "https://health.detik.com/berita-detikhealth/d-4965904/andrea-dian-sembuh-dari-corona-ini-artinya",
    "Contoh (Sport)": "https://sport.detik.com/raket/d-4851962/anthony-ginting-dan-gregoria-tersingkir-di-babak-32-besar-malaysia-masters",
}

def isi_contoh(link):
    st.session_state["url_input"] = link

url_input = st.text_input(
    "Masukkan link berita:",
    key="url_input",
    placeholder="https://sport.detik.com/.../d-123456/judul-berita-anda",
)

kolom = st.columns(3)
for i, (label_tombol, link) in enumerate(contoh_link.items()):
    kolom[i].button(label_tombol, use_container_width=True,
                    on_click=isi_contoh, args=(link,))

pakai_halaman = st.checkbox(
    "Ambil judul dari halaman artikel (butuh internet). "
    "Jika dimatikan, judul dibentuk dari slug link saja.",
    value=True,
)

if st.button("🔍 Klasifikasikan", type="primary", use_container_width=True):
    if not url_input or not url_input.strip():
        st.warning("Silakan masukkan link berita terlebih dahulu.")
    else:
        with st.spinner("Mengambil judul dari link..."):
            info = ambil_judul_dari_url(url_input, pakai_halaman)

        if not info["judul"]:
            st.error(info["peringatan"] or "Judul tidak dapat diambil dari link ini.")
        else:
            if info["peringatan"]:
                st.warning(info["peringatan"])

            prediksi, df_proba, tokens, jumlah_dikenali, jumlah_total = klasifikasi_judul(
                info["judul"], model_w2v, scaler, clf, le
            )

            st.markdown(f"**Judul yang dianalisis:** {info['judul']}")
            st.caption(f"Sumber judul: {info['sumber']}")

            emoji_kategori = {"hot": "🔥", "health": "🏥", "sport": "⚽"}
            st.success(
                f"### Prediksi Kategori: {emoji_kategori.get(prediksi, '')} **{prediksi.upper()}**"
            )

            if info["channel"]:
                if info["channel"] == prediksi:
                    st.info(f"Pembanding: link berada di channel **{info['channel']}**.detik.com, sama dengan prediksi model.")
                else:
                    st.info(f"Pembanding: link berada di channel **{info['channel']}**.detik.com, "
                            f"berbeda dengan prediksi model ({prediksi}).")
                st.caption("Channel pada alamat link hanya ditampilkan sebagai pembanding. "
                           "Model memprediksi murni dari isi judul, bukan dari alamat link.")
            elif "detik.com" not in info["url"].lower():
                st.caption("Link ini bukan dari hot/health/sport.detik.com. Model dilatih dari judul "
                           "detik.com, sehingga prediksi untuk situs lain bisa kurang akurat.")

            st.subheader("Probabilitas tiap kategori")
            st.bar_chart(df_proba.set_index("Kategori"))
            st.dataframe(df_proba, use_container_width=True, hide_index=True)

            with st.expander("Detail pemrosesan"):
                st.write(f"**Link:** {info['url']}")
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

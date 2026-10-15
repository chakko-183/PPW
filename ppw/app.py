import ipaddress
import os
import re
import socket
import time
from urllib.parse import unquote, urljoin, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Klasifikasi Berita Detik", page_icon="📰", layout="wide")

CSV = "df_berita_3kategori.csv"
EMOJI = {"hot": "🔥", "health": "🏥", "sport": "⚽"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept-Language": "id,en;q=0.8"}
CONTOH = {
    "🔥 Contoh Hot": "https://hot.detik.com/celeb/d-4954259/eyangnya-meninggal-kaesang-menyesal-belum-sempat-kirim-sang-pisang",
    "🏥 Contoh Health": "https://health.detik.com/berita-detikhealth/d-4965904/andrea-dian-sembuh-dari-corona-ini-artinya",
    "⚽ Contoh Sport": "https://sport.detik.com/raket/d-4851962/anthony-ginting-dan-gregoria-tersingkir-di-babak-32-besar-malaysia-masters",
}
# Hasil evaluasi dari notebook (dipakai jika dataset tidak tersedia)
EVAL_NOTEBOOK = {
    "akurasi": 0.6667, "kelas": ["health", "hot", "sport"],
    "cm": np.array([[16, 4, 0], [5, 11, 4], [6, 1, 13]]),
    "laporan": pd.DataFrame({"precision": [.59, .69, .76], "recall": [.80, .55, .65],
                             "f1-score": [.68, .61, .70]}, index=["health", "hot", "sport"]),
}


@st.cache_resource
def muat_model():
    return (Word2Vec.load("model_skipgram.model"), joblib.load("scaler.pkl"),
            joblib.load("model_naive_bayes.pkl"), joblib.load("label_encoder.pkl"))


try:
    w2v, scaler, clf, le = muat_model()
except FileNotFoundError as e:
    st.error(f"File model belum ditemukan satu folder dengan app2.py.\n\n{e}")
    st.stop()


# =========================================================
# 1. CRAWLING: link -> HTML -> judul
# =========================================================
def host_aman(host):
    """Tolak alamat internal (anti-SSRF)."""
    try:
        for i in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(i[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except socket.gaierror:
        return False


def ambil_html(url, timeout=10, maks=2_000_000):
    """Unduh halaman. Redirect diikuti manual agar setiap host tetap dicek."""
    mulai = time.time()
    for _ in range(6):
        host = urlparse(url).hostname
        if not host or not host_aman(host):
            raise ValueError("alamat tidak valid / tidak dapat diakses")
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=False, stream=True)
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            url = urljoin(url, r.headers["Location"])
            r.close()
            continue
        r.raise_for_status()
        isi = b""
        for potongan in r.iter_content(65536):
            isi += potongan
            if len(isi) > maks:
                break
        r.close()
        return isi, url, {"Status HTTP": r.status_code, "Waktu (detik)": round(time.time() - mulai, 2),
                          "Ukuran (KB)": len(isi) // 1024}
    raise ValueError("terlalu banyak redirect")


def bersihkan_judul(j):
    """Rapikan spasi & buang akhiran nama situs, mis. 'Judul - detikHealth'."""
    j = re.sub(r"\s+", " ", j or "").strip()
    return re.sub(r"(\s*[-–—|]\s*[^-–—|]*detik[^-–—|]*)+$", "", j, flags=re.I).strip()


def ekstrak_judul(html):
    """Urutan prioritas: og:title -> twitter:title -> <h1> -> <title>."""
    soup = BeautifulSoup(html, "html.parser")
    kandidat = []
    for nama, atribut in [("og:title", "property"), ("twitter:title", "name")]:
        tag = soup.find("meta", attrs={atribut: nama})
        if tag and tag.get("content"):
            kandidat.append((f"meta {nama}", tag["content"]))
    if soup.h1:
        kandidat.append(("<h1>", soup.h1.get_text(" ", strip=True)))
    if soup.title and soup.title.string:
        kandidat.append(("<title>", soup.title.string))
    for sumber, teks in kandidat:
        if len(bersihkan_judul(teks)) >= 5:
            return teks, bersihkan_judul(teks), sumber, kandidat
    return "", "", "", kandidat


def judul_dari_slug(url):
    """Cadangan bila halaman gagal diambil: '.../d-123/andrea-sembuh' -> 'andrea sembuh'."""
    slug = unquote(urlparse(url).path).strip("/").split("/")[-1]
    slug = re.sub(r"[-_+]+", " ", re.sub(r"\.(html?|php|aspx?)$", "", slug, flags=re.I)).strip()
    return "" if len(slug) < 5 or slug.replace(" ", "").isdigit() else slug


def channel_detik(url):
    m = re.match(r"^(hot|health|sport)\.detik\.com$", (urlparse(url).hostname or "").lower())
    return m.group(1) if m else None


def ambil_judul(url, pakai_halaman=True):
    url = url.strip()
    if url and not re.match(r"^https?://", url, flags=re.I):
        url = "https://" + url
    info = {"url": url, "channel": channel_detik(url), "judul": "", "judul_mentah": "",
            "sumber": "", "http": None, "kandidat": [], "peringatan": ""}
    if pakai_halaman:
        try:
            html, url_akhir, info["http"] = ambil_html(url)
            info["judul_mentah"], info["judul"], info["sumber"], info["kandidat"] = ekstrak_judul(html)
            info["channel"] = channel_detik(url_akhir) or info["channel"]
            if info["judul"]:
                return info
            info["peringatan"] = "Halaman terbuka tetapi judul tidak ditemukan."
        except Exception as e:
            info["peringatan"] = f"Halaman tidak bisa diambil ({e})."
    info["judul"] = info["judul_mentah"] = judul_dari_slug(url)
    info["sumber"] = "slug link"
    if info["judul"]:
        info["peringatan"] += " Judul dibentuk dari slug link."
    return info


def crawl_indeks(channel, target, maks_halaman, jeda, progres):
    """Crawling daftar berita dari https://<channel>.detik.com/indeks (seperti notebook)."""
    url, data, log = f"https://{channel}.detik.com/indeks", [], []
    for h in range(1, maks_halaman + 1):
        progres(h / maks_halaman, f"Mengambil halaman {h} ...")
        try:
            html, _, http = ambil_html(url)
        except Exception as e:
            log.append(f"Halaman {h}: GAGAL - {e}")
            break
        soup = BeautifulSoup(html, "html.parser")
        artikel = soup.find_all("article")
        log.append(f"Halaman {h}: HTTP {http['Status HTTP']}, {len(artikel)} artikel")
        for a in artikel:
            judul, link = a.find(["h2", "h3"]), a.find("a", href=True)
            if judul and link:
                data.append({"Judul Berita": judul.get_text(" ", strip=True),
                             "Tautan": urljoin(url, link["href"]), "Kategori": channel})
        nxt = soup.find("a", rel="next") or soup.find(
            "a", string=lambda s: s and s.strip().lower() == "next")
        if len({d["Tautan"] for d in data}) >= target or not nxt:
            break
        url = urljoin(url, nxt["href"])
        time.sleep(jeda)
    df = pd.DataFrame(data, columns=["Judul Berita", "Tautan", "Kategori"])
    df = df.drop_duplicates("Tautan").head(target).reset_index(drop=True)
    df.index += 1
    return df, log


# =========================================================
# 2. PREPROCESSING (harus sama persis dengan saat pelatihan)
# =========================================================
def tokenisasi(teks):
    return re.findall(r"[a-zA-Z]+", str(teks).lower())


# =========================================================
# 3. SKIP-GRAM: token -> vektor kata -> vektor judul (rata-rata)
# =========================================================
def vektor_dokumen(tokens):
    dikenali = [w for w in tokens if w in w2v.wv]
    vektor = np.mean([w2v.wv[w] for w in dikenali], axis=0) if dikenali else np.zeros(w2v.vector_size)
    return vektor, dikenali


def pasangan_skipgram(tokens):
    return [(t, tokens[j]) for i, t in enumerate(tokens)
            for j in range(max(0, i - w2v.window), min(len(tokens), i + w2v.window + 1)) if j != i]


def fitur(juduls):
    """Judul -> vektor rata-rata -> distandarkan scaler."""
    return scaler.transform(np.array([vektor_dokumen(tokenisasi(j))[0] for j in juduls]))


# =========================================================
# 4. NAIVE BAYES: log prior + log likelihood Gaussian per kelas
# =========================================================
def hitung_nb(x):
    var, mu = clf.var_, clf.theta_
    log_prior = np.log(clf.class_prior_)
    log_lik = -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mu) ** 2 / var, axis=1)
    post = log_prior + log_lik
    proba = np.exp(post - post.max())
    return pd.DataFrame({"Kategori": le.inverse_transform(clf.classes_), "Prior": clf.class_prior_,
                         "log prior": log_prior, "log likelihood": log_lik, "log posterior": post,
                         "Probabilitas": proba / proba.sum()})


def proses(url, pakai_halaman):
    info = ambil_judul(url, pakai_halaman)
    if not info["judul"]:
        return info, None
    tokens = tokenisasi(info["judul"])
    vektor, dikenali = vektor_dokumen(tokens)
    x = scaler.transform(vektor.reshape(1, -1))
    nb = hitung_nb(x)
    return info, {"tokens": tokens, "dikenali": dikenali, "vektor": vektor, "x": x, "nb": nb,
                  "prediksi": nb.loc[nb["Probabilitas"].idxmax(), "Kategori"]}


# =========================================================
# Dataset & evaluasi (meniru notebook: split 80:20 stratified, random_state=42)
# =========================================================
def ambil_dataset():
    df = st.session_state.get("df_upload")
    if df is None and os.path.exists(CSV):
        df = pd.read_csv(CSV)
    ok = df is not None and {"Judul Berita", "Kategori"} <= set(df.columns) \
        and set(df["Kategori"]) <= set(le.classes_)
    return df if ok else None


@st.cache_data(show_spinner=False)
def evaluasi(df):
    X, y = fitur(df["Judul Berita"]), le.transform(df["Kategori"])
    X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
        X, y, np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
    pred = GaussianNB().fit(X_tr, y_tr).predict(X_te)
    pca = PCA(2, random_state=42).fit(X)
    return {"akurasi": float((pred == y_te).mean()), "kelas": list(le.classes_),
            "cm": confusion_matrix(y_te, pred),
            "laporan": pd.DataFrame(classification_report(
                y_te, pred, target_names=le.classes_, output_dict=True)).T,
            "pca": pca, "xy": pca.transform(X), "kategori": df["Kategori"].values,
            "idx_te": idx_te, "y_te": y_te, "pred": pred}


def peta_pca(ev, x_baru=None):
    """Sebaran vektor judul (PCA 2D); judul pengguna ditandai besar."""
    p = pd.DataFrame(ev["xy"], columns=["PC1", "PC2"])
    p["Kategori"], p["Ukuran"] = ev["kategori"], 40
    if x_baru is not None:
        p.loc[len(p)] = [*ev["pca"].transform(x_baru)[0], "JUDUL ANDA", 500]
    st.scatter_chart(p, x="PC1", y="PC2", color="Kategori", size="Ukuran")


def sebelum_sesudah(sebelum, sesudah, a="SEBELUM", b="SESUDAH"):
    k1, k2 = st.columns(2)
    k1.info(f"**{a}**\n\n{sebelum or '-'}")
    k2.success(f"**{b}**\n\n{sesudah or '-'}")


# =========================================================
# HALAMAN 1: Klasifikasi dari link
# =========================================================
def halaman_klasifikasi():
    st.text_input("Masukkan link berita:", key="url",
                  placeholder="https://sport.detik.com/.../d-123456/judul-berita")
    for kol, (label, link) in zip(st.columns(3), CONTOH.items()):
        kol.button(label, on_click=lambda l=link: st.session_state.update(url=l))
    pakai = st.checkbox("Ambil judul dari halaman artikel (jika dimatikan: dari slug link)", True)

    if st.button("🔍 Klasifikasikan", type="primary") or st.session_state.pop("auto", False):
        if not st.session_state.get("url", "").strip():
            st.warning("Masukkan link berita terlebih dahulu.")
        else:
            with st.spinner("Crawling → preprocessing → Skip-gram → Naive Bayes ..."):
                st.session_state["hasil"] = proses(st.session_state["url"], pakai)

    if "hasil" not in st.session_state:
        return
    info, k = st.session_state["hasil"]
    if k is None:
        st.error(info["peringatan"] or "Judul tidak dapat diambil dari link ini.")
        return
    if info["peringatan"]:
        st.warning(info["peringatan"])

    nb = k["nb"].sort_values("Probabilitas", ascending=False).reset_index(drop=True)
    yakin = nb.loc[0, "Probabilitas"]
    st.success(f"## {EMOJI[k['prediksi']]} {k['prediksi'].upper()}  ·  keyakinan {yakin:.1%}")
    st.markdown(f"**Judul:** {info['judul']}")
    if not k["dikenali"]:
        st.warning("Tidak ada kata yang dikenali model; prediksi hanya berdasarkan prior kelas.")
    elif yakin < 0.5:
        st.warning("Keyakinan model rendah; judul ini kemungkinan ambigu.")

    t1, t2, t3, t4 = st.tabs(["🕷️ 1. Crawling", "🧹 2. Preprocessing", "🧠 3. Skip-gram", "🎯 4. Naive Bayes"])

    with t1:
        if info["http"]:
            for kol, (nama, nilai) in zip(st.columns(3), info["http"].items()):
                kol.metric(nama, nilai)
        sebelum_sesudah(info["judul_mentah"], info["judul"], "JUDUL MENTAH (HTML)", "JUDUL BERSIH")
        st.caption(f"Sumber judul: {info['sumber']}")
        if info["kandidat"]:
            st.dataframe(pd.DataFrame(info["kandidat"], columns=["Lokasi di HTML", "Teks"]),
                         hide_index=True)

    with t2:
        sebelum_sesudah(info["judul"], " · ".join(k["tokens"]), "SEBELUM", "SESUDAH (token)")
        st.dataframe(pd.DataFrame({
            "Langkah": ["Teks asli", "Case folding", "Cleaning + tokenizing"],
            "Hasil": [info["judul"], info["judul"].lower(), str(k["tokens"])]}), hide_index=True)
        st.dataframe(pd.DataFrame({
            "Token": k["tokens"],
            "Dikenali model?": ["✅" if w in w2v.wv else "❌ (diabaikan)" for w in k["tokens"]]}),
            hide_index=True)

    with t3:
        for kol, (nama, nilai) in zip(st.columns(3), [("Arsitektur", "Skip-gram"),
                                                      ("Dimensi vektor", w2v.vector_size),
                                                      ("Window", w2v.window)]):
            kol.metric(nama, nilai)
        if k["dikenali"]:
            st.markdown("**Pasangan (target → konteks)**")
            st.dataframe(pd.DataFrame(pasangan_skipgram(k["dikenali"]),
                                      columns=["Target", "Konteks"]).head(30), hide_index=True, height=220)
            st.markdown("**Vektor kata (8 dari 100 dimensi) → rata-rata → setelah scaler**")
            tabel = pd.DataFrame([w2v.wv[w][:8] for w in k["dikenali"]], index=k["dikenali"],
                                 columns=[f"dim_{i + 1}" for i in range(8)])
            tabel.loc["➜ RATA-RATA"] = k["vektor"][:8]
            tabel.loc["➜ SETELAH SCALER"] = k["x"][0][:8]
            st.dataframe(tabel.round(4))
            st.markdown("**Kata terdekat (cosine similarity)**")
            st.dataframe(pd.DataFrame([
                {"Kata": w, "5 terdekat": ", ".join(f"{a} ({s:.2f})" for a, s in w2v.wv.most_similar(w, topn=5))}
                for w in k["dikenali"][:8]]), hide_index=True)
        else:
            st.warning("Vektor judul adalah vektor nol.")

    with t4:
        kiri, kanan = st.columns(2)
        kiri.bar_chart(nb.set_index("Kategori")["Probabilitas"])
        kanan.dataframe(nb.round(3), hide_index=True)
        if info["channel"]:
            sama = info["channel"] == k["prediksi"]
            (st.success if sama else st.info)(
                f"Pembanding: link berada di channel **{info['channel']}**.detik.com — "
                f"{'sama dengan' if sama else 'berbeda dari'} prediksi model.")
        df = ambil_dataset()
        if df is not None:
            st.markdown("**Posisi judul Anda di antara data latih (PCA 2D)**")
            peta_pca(evaluasi(df), k["x"])


# =========================================================
# HALAMAN 2: Crawling indeks berita
# =========================================================
def halaman_crawling():
    st.subheader("🕷️ Crawling Indeks Berita")
    c1, c2, c3, c4 = st.columns(4)
    channel = c1.selectbox("Channel", ["health", "hot", "sport"])
    target = c2.number_input("Jumlah berita", 5, 200, 30, step=5)
    maks = c3.number_input("Maks. halaman", 1, 15, 3)
    jeda = c4.number_input("Jeda (detik)", 0.0, 5.0, 1.0, step=0.5)

    if st.button("🕷️ Mulai Crawling", type="primary"):
        bar = st.progress(0.0)
        st.session_state["crawl"] = crawl_indeks(
            channel, int(target), int(maks), float(jeda), lambda p, t: bar.progress(p, text=t))
        bar.empty()
    if "crawl" not in st.session_state:
        return

    df, log = st.session_state["crawl"]
    with st.expander("Log crawling", expanded=df.empty):
        st.write("\n".join(f"- {baris}" for baris in log))
    if df.empty:
        st.error("Tidak ada berita yang terambil (kemungkinan diblokir 403). Gunakan CSV di halaman Dataset.")
        return

    st.success(f"{len(df)} berita berhasil di-crawl.")
    st.dataframe(df, height=250)
    st.download_button("⬇️ Unduh CSV", df.to_csv().encode(), "hasil_crawling.csv", "text/csv")

    hasil = df.copy()
    hasil["Token (setelah preprocessing)"] = [" · ".join(tokenisasi(j)) for j in df["Judul Berita"]]
    proba = clf.predict_proba(fitur(df["Judul Berita"]))
    hasil["Prediksi"] = le.inverse_transform(clf.classes_[proba.argmax(1)])
    hasil["Keyakinan"] = proba.max(1).round(3)
    hasil["Sesuai channel?"] = np.where(hasil["Kategori"] == hasil["Prediksi"], "✅", "❌")
    st.markdown("**Sebelum → sesudah preprocessing, beserta prediksi**")
    st.dataframe(hasil.drop(columns="Tautan"), height=300)
    st.metric("Kesesuaian prediksi dengan channel", f"{(hasil['Kategori'] == hasil['Prediksi']).mean():.1%}")

    pilih = st.selectbox("Analisis lengkap satu artikel:", df.index,
                         format_func=lambda i: f"{i}. {df.loc[i, 'Judul Berita'][:90]}")
    st.button("➡️ Analisis artikel ini", on_click=lambda: st.session_state.update(
        url=df.loc[pilih, "Tautan"], halaman=MENU[0], auto=True))


# =========================================================
# HALAMAN 3: Dataset & evaluasi
# =========================================================
def halaman_dataset():
    st.subheader("📊 Dataset & Evaluasi Model")
    up = st.file_uploader("Unggah CSV (kolom: Judul Berita, Kategori)", type="csv")
    if up is not None:
        st.session_state["df_upload"] = pd.read_csv(up)
    df = ambil_dataset()

    if df is None:
        st.info(f"`{CSV}` tidak ditemukan / tidak valid. Menampilkan hasil evaluasi dari notebook.")
        ev = EVAL_NOTEBOOK
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah judul", len(df))
        c2.metric("Jumlah kategori", df["Kategori"].nunique())
        c3.metric("Kata unik", len({w for j in df["Judul Berita"] for w in tokenisasi(j)}))

        st.markdown("**Data sebelum & sesudah preprocessing**")
        contoh = df.head(10)
        st.dataframe(pd.DataFrame({
            "Judul asli": contoh["Judul Berita"].values,
            "Token": [" · ".join(tokenisasi(j)) for j in contoh["Judul Berita"]],
            "Kategori": contoh["Kategori"].values}), hide_index=True)
        st.bar_chart(df["Kategori"].value_counts())
        ev = evaluasi(df)

    st.markdown("**Evaluasi Naive Bayes (data uji 20%)**")
    st.metric("Akurasi", f"{ev['akurasi']:.1%}")
    kiri, kanan = st.columns(2)
    kiri.dataframe(pd.DataFrame(ev["cm"], index=[f"Aktual {k}" for k in ev["kelas"]],
                                columns=[f"Prediksi {k}" for k in ev["kelas"]]))
    kanan.bar_chart(ev["laporan"].loc[ev["kelas"], ["precision", "recall", "f1-score"]])
    st.dataframe(ev["laporan"].round(3))

    if df is not None:
        st.markdown("**Sebaran vektor judul (PCA 2D)**")
        peta_pca(ev)
        st.markdown("**Prediksi pada data uji**")
        uji = pd.DataFrame({"Judul": df["Judul Berita"].values[ev["idx_te"]],
                            "Aktual": le.inverse_transform(ev["y_te"]),
                            "Prediksi": le.inverse_transform(ev["pred"])})
        uji["Benar?"] = np.where(uji["Aktual"] == uji["Prediksi"], "✅", "❌")
        st.dataframe(uji, hide_index=True, height=300)


# =========================================================
# Navigasi
# =========================================================
MENU = ["🔍 Klasifikasi dari Link", "🕷️ Crawling Indeks", "📊 Dataset & Evaluasi"]
st.sidebar.title("📰 Menu")
st.sidebar.radio("Halaman", MENU, key="halaman", label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(f"**Model**\n- Skip-gram, {len(w2v.wv.index_to_key)} kata, {w2v.vector_size} dimensi\n"
                    f"- GaussianNB: {', '.join(le.classes_)}")

st.title("📰 Klasifikasi Berita Detik dari Link")
{MENU[0]: halaman_klasifikasi, MENU[1]: halaman_crawling, MENU[2]: halaman_dataset}[st.session_state["halaman"]]()

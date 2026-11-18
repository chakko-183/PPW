"""
Streamlit app: Web Crawler + PageRank antarwebsite.
Jalankan lokal :  streamlit run app.py
"""

import time
from collections import defaultdict

import pandas as pd
import streamlit as st

from crawl_pagerank import Crawler, build_graph, get_domain, normalize_url, pagerank

st.set_page_config(page_title="WebRank Explorer", page_icon="🕸️", layout="wide")

# --------------------------------------------------------------------------
# Tampilan (CSS)
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.hero {
    padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white; box-shadow: 0 10px 30px rgba(124,58,237,.25);
}
.hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -.5px;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.02rem;}
.card {
    background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.25);
    border-radius: 14px; padding: 1rem 1.2rem; height: 100%;
}
.card .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7;}
.card .value {font-size: 1.7rem; font-weight: 800; line-height: 1.25; word-break: break-all;}
.card .value.small {font-size: 1.15rem;}
.pill {
    display: inline-block; padding: .15rem .65rem; border-radius: 999px; margin: 0 .3rem .3rem 0;
    background: rgba(124,58,237,.18); border: 1px solid rgba(124,58,237,.45); font-size: .82rem;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4f46e5, #db2777); border: 0; font-weight: 700;
    border-radius: 10px; padding: .6rem 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🕸️ WebRank Explorer</h1>
  <p>Masukkan sebuah link website. Aplikasi akan menelusuri tautan-tautannya, membangun graph
  keterhubungan antarwebsite, lalu menghitung <b>PageRank</b> untuk menemukan website paling berpengaruh.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar: pengaturan
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan Crawl")
    depth = st.slider("Kedalaman (depth)", 1, 3, 2,
                      help="Seberapa jauh tautan diikuti. Makin dalam makin lama.")
    max_pages = st.slider("Maks. total halaman", 10, 150, 50, step=10)
    per_domain = st.slider("Maks. halaman per website", 1, 15, 5)
    delay = st.slider("Jeda antar request (detik)", 0.2, 3.0, 0.5, step=0.1,
                      help="Jeda sopan terhadap website yang sama.")
    max_seconds = st.slider("Batas waktu crawl (detik)", 20, 300, 90, step=10)
    respect_robots = st.checkbox("Patuhi robots.txt", value=True)
    weighted = st.checkbox("Bobot edge = jumlah tautan", value=True)
    damping = st.slider("Damping factor PageRank", 0.50, 0.99, 0.85, step=0.01)
    st.caption("Tips: mulai dengan pengaturan kecil agar cepat, lalu naikkan bertahap.")

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
c_in, c_btn = st.columns([5, 1], vertical_alignment="bottom")
with c_in:
    url_input = st.text_input("🔗 Link website", placeholder="https://www.wikipedia.org",
                              label_visibility="collapsed")
with c_btn:
    run = st.button("Analisis", type="primary", width="stretch")

st.caption("Contoh: `https://www.wikipedia.org` · `https://news.ycombinator.com` · `https://github.com`")


# --------------------------------------------------------------------------
# Proses crawl
# --------------------------------------------------------------------------
def run_analysis(url):
    norm = normalize_url(url if "://" in url else "https://" + url)
    if not norm:
        st.error("Link tidak valid. Gunakan format http:// atau https://")
        return None

    bar = st.progress(0.0, text="Memulai crawl…")

    def on_progress(level, done, total, current):
        bar.progress(done / max(total, 1),
                     text=f"Level {level}/{depth} · {done}/{total} halaman · {current[:70]}")

    crawler = Crawler(max_depth=depth, max_pages=max_pages, per_domain=per_domain,
                      delay=delay, workers=8, respect_robots=respect_robots,
                      max_seconds=max_seconds, progress=on_progress)
    t0 = time.time()
    crawler.crawl([norm])
    elapsed = time.time() - t0
    bar.empty()

    if not crawler.pages_crawled:
        st.error("Tidak ada halaman yang berhasil dibaca. Penyebab umum: website memblokir bot, "
                 "dilarang `robots.txt`, bukan halaman HTML, atau alamat tidak dapat dijangkau.")
        return None
    if not crawler.edges:
        st.warning("Halaman terbaca, tetapi tidak ditemukan tautan ke website lain.")
        return None

    G = build_graph(crawler, weighted=weighted)
    pr = pagerank(G, damping=damping)
    return {"url": norm, "seed": get_domain(norm), "G": G, "pr": pr, "elapsed": elapsed,
            "edges": dict(crawler.edges), "page_links": list(crawler.page_links),
            "titles": dict(crawler.page_titles), "crawled": dict(crawler.pages_crawled)}


if run:
    if not url_input.strip():
        st.warning("Masukkan link terlebih dahulu.")
    else:
        result = run_analysis(url_input.strip())
        st.session_state["result"] = result

res = st.session_state.get("result")
if not res:
    st.info("👆 Masukkan link lalu tekan **Analisis** untuk memulai.")
    st.stop()

G, pr, seed = res["G"], res["pr"], res["seed"]
ranking = sorted(pr.items(), key=lambda x: -x[1])
top_site, top_score = ranking[0]

# --------------------------------------------------------------------------
# Ringkasan
# --------------------------------------------------------------------------
def card(col, label, value, small=False):
    col.markdown(f'<div class="card"><div class="label">{label}</div>'
                 f'<div class="value {"small" if small else ""}">{value}</div></div>',
                 unsafe_allow_html=True)

m = st.columns(4)
card(m[0], "Website ditemukan", f"{G.number_of_nodes():,}")
card(m[1], "Halaman di-crawl", f"{sum(res['crawled'].values()):,}")
card(m[2], "Tautan antarwebsite", f"{G.number_of_edges():,}")
card(m[3], "🏆 Paling penting", top_site, small=True)
st.write("")

tab_graph, tab_rank, tab_rel, tab_dl = st.tabs(
    ["🕸️ Graph", "🏆 Peringkat PageRank", "🔗 Website Terkait", "📥 Unduh"])

# --------------------------------------------------------------------------
# Tab Graph
# --------------------------------------------------------------------------
def lerp_color(t):
    """Gradien biru muda -> ungu -> merah muda berdasarkan t di [0,1]."""
    stops = [(96, 165, 250), (167, 139, 250), (244, 114, 182)]
    t = max(0.0, min(1.0, t)) * 2
    i = min(int(t), 1)
    f = t - i
    a, b = stops[i], stops[i + 1]
    return "#%02x%02x%02x" % tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


def build_html(G, pr, seed, top_n):
    from pyvis.network import Network
    keep = {n for n, _ in sorted(pr.items(), key=lambda x: -x[1])[:top_n]} | {seed}
    H = G.subgraph(keep)
    vmax, vmin = max(pr[n] for n in H), min(pr[n] for n in H)
    span = (vmax - vmin) or 1
    net = Network(height="640px", width="100%", directed=True, bgcolor="#0e1117",
                  font_color="#e5e7eb", cdn_resources="in_line")
    for n in H.nodes():
        t = (pr[n] - vmin) / span
        is_seed = n == seed
        net.add_node(n, label=n, size=14 + 40 * t,
                     color={"background": "#fbbf24" if is_seed else lerp_color(t),
                            "border": "#ffffff" if is_seed else "#1f2937"},
                     borderWidth=3 if is_seed else 1,
                     title=f"{n}\nPageRank: {pr[n]:.5f}\nMasuk: {H.in_degree(n)} · Keluar: {H.out_degree(n)}")
    for u, v, d in H.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1), color={"color": "rgba(148,163,184,.45)"})
    net.set_options("""{
      "interaction": {"hover": true, "tooltipDelay": 80, "navigationButtons": true},
      "physics": {"solver": "forceAtlas2Based",
                  "forceAtlas2Based": {"gravitationalConstant": -70, "springLength": 130},
                  "stabilization": {"iterations": 200}},
      "edges": {"smooth": {"type": "continuous"}, "arrows": {"to": {"scaleFactor": 0.6}}}
    }""")
    return net.generate_html(notebook=False)


with tab_graph:
    n_total = G.number_of_nodes()
    top_n = n_total if n_total <= 10 else st.slider(
        "Jumlah website ditampilkan (berdasarkan PageRank tertinggi)", 5, n_total, min(n_total, 40))
    st.iframe(build_html(G, pr, seed, top_n), width="stretch", height=660)
    st.caption("🟡 = website awal yang Anda masukkan · ukuran & warna node = PageRank · "
               "anak panah = arah tautan · scroll untuk zoom, seret untuk menggeser.")

# --------------------------------------------------------------------------
# Tab Peringkat
# --------------------------------------------------------------------------
df_rank = pd.DataFrame([
    {"Peringkat": i, "Website": s, "PageRank": sc, "Tautan masuk": G.in_degree(s),
     "Tautan keluar": G.out_degree(s), "Halaman di-crawl": res["crawled"].get(s, 0)}
    for i, (s, sc) in enumerate(ranking, 1)
])

with tab_rank:
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            df_rank, hide_index=True, width="stretch", height=470,
            column_config={
                "PageRank": st.column_config.ProgressColumn(
                    "PageRank", format="%.5f", min_value=0, max_value=float(df_rank["PageRank"].max())),
            })
    with right:
        st.markdown("**Top 10 website**")
        st.bar_chart(df_rank.head(10).set_index("Website")["PageRank"], horizontal=True,
                     color="#7c3aed")
    st.success(f"**{top_site}** adalah website dengan tingkat kepentingan tertinggi "
               f"(PageRank {top_score:.5f}), karena banyak menerima tautan dari website penting lain.")

# --------------------------------------------------------------------------
# Tab Website Terkait
# --------------------------------------------------------------------------
with tab_rel:
    st.markdown(f"#### Website yang terkait dengan `{seed}`")
    out_nb = sorted(G.successors(seed), key=lambda n: -pr[n]) if seed in G else []
    in_nb = sorted(G.predecessors(seed), key=lambda n: -pr[n]) if seed in G else []

    a, b = st.columns(2)
    with a:
        st.markdown(f"**➡️ Ditautkan oleh {seed}** ({len(out_nb)})")
        if out_nb:
            st.markdown("".join(f'<span class="pill">{n}</span>' for n in out_nb[:30]),
                        unsafe_allow_html=True)
        else:
            st.caption("Tidak ada.")
    with b:
        st.markdown(f"**⬅️ Menaut ke {seed}** ({len(in_nb)})")
        if in_nb:
            st.markdown("".join(f'<span class="pill">{n}</span>' for n in in_nb[:30]),
                        unsafe_allow_html=True)
        else:
            st.caption("Tidak ditemukan pada crawl ini.")

    st.markdown("**Rincian tautan dari website awal**")
    by_dst = defaultdict(list)
    for src_url, dst_url in res["page_links"]:
        if get_domain(src_url) == seed and get_domain(dst_url) != seed:
            by_dst[get_domain(dst_url)].append((src_url, dst_url))
    if not by_dst:
        st.caption("Tidak ada tautan keluar dari website awal.")
    for dst in sorted(by_dst, key=lambda n: -pr.get(n, 0)):
        pairs = by_dst[dst]
        with st.expander(f"{dst} · {len(pairs)} tautan · PageRank {pr.get(dst, 0):.5f}"):
            for src_url, dst_url in pairs[:15]:
                title = res["titles"].get(src_url) or src_url
                st.markdown(f"- {dst_url}  \n  <small>dari: {title}</small>", unsafe_allow_html=True)
            if len(pairs) > 15:
                st.caption(f"… dan {len(pairs) - 15} tautan lainnya")

# --------------------------------------------------------------------------
# Tab Unduh
# --------------------------------------------------------------------------
with tab_dl:
    df_edges = pd.DataFrame(
        [{"sumber": u, "tujuan": v, "jumlah_tautan": c} for (u, v), c in res["edges"].items()]
    ).sort_values("jumlah_tautan", ascending=False)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ pagerank.csv", df_rank.to_csv(index=False).encode("utf-8"),
                       "pagerank.csv", "text/csv", width="stretch")
    d2.download_button("⬇️ edges.csv", df_edges.to_csv(index=False).encode("utf-8"),
                       "edges.csv", "text/csv", width="stretch")
    st.caption(f"Crawl selesai dalam {res['elapsed']:.1f} detik · seed: {res['url']}")

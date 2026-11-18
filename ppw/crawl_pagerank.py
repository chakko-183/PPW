#!/usr/bin/env python3
"""
Web Crawling dan Analisis PageRank
==================================
Alur kerja:
  1. Crawl dimulai dari satu/lebih URL seed (BFS, per level kedalaman).
  2. Setiap tautan yang ditemukan dikunjungi dan diekstraksi kembali,
     sampai batas kedalaman / jumlah halaman tercapai.
  3. Tautan antarhalaman diubah menjadi graph antarWEBSITE (node = domain,
     edge = ada tautan dari domain A ke domain B, bobot = jumlah tautan).
  4. PageRank dihitung (power iteration, dangling node ditangani).
  5. Hasil disimpan: CSV, JSON, gambar graph (PNG) dan graph interaktif (HTML).

Contoh:
  python crawler_pagerank.py https://www.wikipedia.org https://github.com \
      --depth 2 --max-pages 200 --per-domain 15 --out hasil
"""

import argparse
import csv
import ipaddress
import json
import logging
import os
import socket
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import matplotlib

matplotlib.use("Agg")  # aman untuk server tanpa display
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

log = logging.getLogger("crawler")

USER_AGENT = "LabPageRankBot/1.0 (+penelitian laboratorium; web crawling & PageRank)"

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".pdf", ".zip", ".rar", ".gz", ".tar", ".7z", ".exe", ".dmg", ".iso",
    ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wav",
    ".css", ".js", ".json", ".xml", ".rss", ".woff", ".woff2", ".ttf",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)
_MISSING = object()


# --------------------------------------------------------------------------
# Utilitas URL
# --------------------------------------------------------------------------
def normalize_url(url):
    """Buang fragment, hanya http/https, huruf kecil pada skema+host."""
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    if p.path.lower().endswith(SKIP_EXTENSIONS):
        return None
    netloc = p.netloc.lower()
    path = p.path or "/"
    return p._replace(scheme=p.scheme.lower(), netloc=netloc, path=path).geturl()


def get_domain(url):
    """Identitas 'website' = hostname (tanpa www. dan tanpa port default)."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    default = 443 if p.scheme == "https" else 80
    if p.port and p.port != default:
        host = f"{host}:{p.port}"
    return host


_public_cache = {}


def is_public_host(host):
    """False bila host mengarah ke alamat privat/loopback (cegah SSRF di server publik)."""
    host = host.split(":")[0]
    if host in _public_cache:
        return _public_cache[host]
    ok = False
    try:
        infos = socket.getaddrinfo(host, None)
        ok = all(ipaddress.ip_address(i[4][0]).is_global for i in infos)
    except (socket.gaierror, ValueError):
        ok = False
    _public_cache[host] = ok
    return ok


# --------------------------------------------------------------------------
# Crawler
# --------------------------------------------------------------------------
class Crawler:
    def __init__(self, max_depth=2, max_pages=200, per_domain=20, delay=1.0,
                 timeout=10, workers=8, respect_robots=True,
                 max_bytes=2_000_000, block_private=True, max_seconds=None,
                 progress=None):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.per_domain = per_domain
        self.delay = delay
        self.timeout = timeout
        self.workers = workers
        self.respect_robots = respect_robots
        self.max_bytes = max_bytes
        self.block_private = block_private and not os.environ.get("CRAWLER_ALLOW_PRIVATE")
        self.deadline = None
        self.max_seconds = max_seconds
        self.progress = progress  # callback(level, selesai, total, pesan)
        self.page_titles = {}

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        self._lock = threading.Lock()
        self._domain_locks = {}
        self._last_access = {}
        self._robots = {}

        # hasil
        self.edges = Counter()      # (domain_asal, domain_tujuan) -> jumlah tautan
        self.page_links = []        # (url_asal, url_tujuan)
        self.pages_crawled = Counter()  # domain -> jumlah halaman berhasil di-crawl
        self.domains = set()

    # ---- politeness -------------------------------------------------------
    def _throttle(self, domain):
        with self._lock:
            dlock = self._domain_locks.setdefault(domain, threading.Lock())
        with dlock:  # satu request per domain pada satu waktu + jeda
            wait = self.delay - (time.time() - self._last_access.get(domain, 0))
            if wait > 0:
                time.sleep(wait)
            self._last_access[domain] = time.time()

    def _allowed(self, url):
        if not self.respect_robots:
            return True
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        with self._lock:
            rp = self._robots.get(base, _MISSING)
        if rp is _MISSING:
            rp = RobotFileParser()
            try:
                r = self.session.get(base + "/robots.txt", timeout=self.timeout)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                else:
                    rp = None  # tidak ada robots.txt -> diizinkan
            except requests.RequestException:
                rp = None
            with self._lock:
                self._robots[base] = rp
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    # ---- fetch & parse ------------------------------------------------------
    def _fetch(self, url):
        """Kembalikan (url_akhir, daftar_tautan, judul) atau None bila gagal."""
        if self.deadline and time.time() > self.deadline:
            return None
        if self.block_private and not is_public_host(urlparse(url).hostname or ""):
            log.info("Alamat non-publik diblokir: %s", url)
            return None
        if not self._allowed(url):
            log.info("robots.txt melarang: %s", url)
            return None
        self._throttle(get_domain(url))
        try:
            with self.session.get(url, timeout=self.timeout, stream=True) as r:
                ctype = r.headers.get("Content-Type", "")
                if r.status_code != 200 or "html" not in ctype.lower():
                    return None
                content = r.raw.read(self.max_bytes, decode_content=True)
                encoding = r.encoding or "utf-8"
                final_url = r.url
            if self.block_private and not is_public_host(urlparse(final_url).hostname or ""):
                return None
        except requests.RequestException as e:
            log.warning("Gagal %s (%s)", url, e.__class__.__name__)
            return None

        soup = BeautifulSoup(content.decode(encoding, errors="replace"), "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        base_tag = soup.find("base", href=True)
        base = urljoin(final_url, base_tag["href"]) if base_tag else final_url
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.lower().startswith(("mailto:", "javascript:", "tel:")):
                continue
            norm = normalize_url(urljoin(base, href))
            if norm:
                links.add(norm)
        return final_url, links, title

    def _fetch_level(self, frontier, depth):
        results = {}
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._fetch, u): u for u in frontier}
            for fut in as_completed(futs):
                u = futs[fut]
                try:
                    results[u] = fut.result()
                except Exception as e:  # satu halaman rusak tidak boleh menghentikan crawl
                    log.warning("Error %s: %s", u, e)
                    results[u] = None
                done += 1
                if self.progress:
                    self.progress(depth, done, len(frontier), u)
        return [results[u] for u in frontier]

    # ---- proses utama -----------------------------------------------------
    def crawl(self, seeds):
        seen = set()
        domain_scheduled = Counter()
        frontier = []
        for s in seeds:
            n = normalize_url(s if "://" in s else "https://" + s)
            if n and n not in seen:
                seen.add(n)
                domain_scheduled[get_domain(n)] += 1
                frontier.append(n)
                self.domains.add(get_domain(n))

        total_scheduled = len(frontier)
        if self.max_seconds:
            self.deadline = time.time() + self.max_seconds
        for depth in range(self.max_depth + 1):
            if not frontier:
                break
            log.info("=== Level %d: %d halaman ===", depth, len(frontier))
            results = self._fetch_level(frontier, depth)

            next_frontier = []
            for src_url, res in zip(frontier, results):
                if res is None:
                    continue
                final_url, links, title = res
                src = get_domain(src_url)
                self.page_titles[src_url] = title
                self.pages_crawled[src] += 1
                log.info("[d=%d] %s -> %d tautan", depth, src_url, len(links))
                for link in links:
                    dst = get_domain(link)
                    self.domains.add(dst)
                    self.page_links.append((src_url, link))
                    if dst != src:  # tautan antarwebsite saja
                        self.edges[(src, dst)] += 1
                    # jadwalkan untuk level berikutnya
                    if (depth < self.max_depth and link not in seen
                            and total_scheduled < self.max_pages
                            and domain_scheduled[dst] < self.per_domain):
                        seen.add(link)
                        domain_scheduled[dst] += 1
                        total_scheduled += 1
                        next_frontier.append(link)
            frontier = next_frontier

        log.info("Selesai: %d website, %d edge antarwebsite, %d halaman di-crawl",
                 len(self.domains), len(self.edges), sum(self.pages_crawled.values()))


# --------------------------------------------------------------------------
# Graph & PageRank
# --------------------------------------------------------------------------
def build_graph(crawler, weighted=True):
    G = nx.DiGraph()
    G.add_nodes_from(crawler.domains)
    for (u, v), w in crawler.edges.items():
        G.add_edge(u, v, weight=w if weighted else 1)
    return G


def pagerank(G, damping=0.85, tol=1e-10, max_iter=200):
    """
    PageRank dengan power iteration:
        PR(v) = (1-d)/N + d * [ sum_{u->v} PR(u)*w(u,v)/W(u) + dangling/N ]
    dangling = total PR dari node tanpa tautan keluar (disebar merata).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    idx = {v: i for i, v in enumerate(nodes)}
    out_w = [sum(d.get("weight", 1) for _, _, d in G.out_edges(v, data=True)) for v in nodes]
    edges = [(idx[u], idx[v], d.get("weight", 1)) for u, v, d in G.edges(data=True)]

    pr = [1.0 / n] * n
    for it in range(1, max_iter + 1):
        dangling = sum(pr[i] for i in range(n) if out_w[i] == 0)
        base = (1 - damping) / n + damping * dangling / n
        new = [base] * n
        for u, v, w in edges:
            new[v] += damping * pr[u] * w / out_w[u]
        err = sum(abs(a - b) for a, b in zip(new, pr))
        pr = new
        if err < tol:
            log.info("PageRank konvergen pada iterasi %d (err=%.2e)", it, err)
            break
    return {nodes[i]: pr[i] for i in range(n)}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def save_results(G, pr, crawler, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    ranking = sorted(pr.items(), key=lambda x: x[1], reverse=True)

    with open(f"{out_dir}/pagerank.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "website", "pagerank", "in_degree", "out_degree", "halaman_di_crawl"])
        for i, (site, score) in enumerate(ranking, 1):
            w.writerow([i, site, f"{score:.8f}", G.in_degree(site), G.out_degree(site),
                        crawler.pages_crawled.get(site, 0)])

    with open(f"{out_dir}/edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sumber", "tujuan", "jumlah_tautan"])
        for (u, v), c in sorted(crawler.edges.items(), key=lambda x: -x[1]):
            w.writerow([u, v, c])

    with open(f"{out_dir}/page_links.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url_sumber", "url_tujuan"])
        w.writerows(crawler.page_links)

    with open(f"{out_dir}/graph.json", "w", encoding="utf-8") as f:
        json.dump({
            "nodes": [{"id": n, "pagerank": pr[n]} for n in G.nodes()],
            "edges": [{"source": u, "target": v, "weight": d["weight"]}
                      for u, v, d in G.edges(data=True)],
        }, f, indent=2)
    return ranking


def draw_png(G, pr, path, top_labels=25):
    if G.number_of_nodes() == 0:
        return
    vals = [pr[n] for n in G.nodes()]
    vmax = max(vals) or 1
    sizes = [200 + 3500 * (v / vmax) for v in vals]
    top = {n for n, _ in sorted(pr.items(), key=lambda x: -x[1])[:top_labels]}
    labels = {n: n for n in G.nodes() if n in top}

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=1.5 / (G.number_of_nodes() ** 0.5), seed=42, iterations=100)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, arrows=True, arrowsize=10,
                           edge_color="gray", node_size=sizes)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=vals,
                                   cmap=plt.cm.YlOrRd, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    fig.colorbar(nodes, ax=ax, label="PageRank", shrink=0.7)
    ax.set_title("Graph keterhubungan antarwebsite (ukuran & warna = PageRank)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_html(G, pr, path):
    try:
        from pyvis.network import Network
    except ImportError:
        log.warning("pyvis tidak terpasang; graph interaktif dilewati (pip install pyvis)")
        return
    vmax = max(pr.values()) if pr else 1
    net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff",
                  cdn_resources="in_line")
    for n in G.nodes():
        net.add_node(n, label=n, size=10 + 50 * pr[n] / vmax,
                     title=f"{n}<br>PageRank: {pr[n]:.6f}")
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1))
    net.write_html(path, notebook=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Web crawler + PageRank antarwebsite")
    ap.add_argument("seeds", nargs="*", help="URL awal (seed)")
    ap.add_argument("--seed-file", help="file teks berisi URL seed (satu per baris)")
    ap.add_argument("--depth", type=int, default=2, help="kedalaman crawl (default 2)")
    ap.add_argument("--max-pages", type=int, default=200, help="batas total halaman (default 200)")
    ap.add_argument("--per-domain", type=int, default=20, help="batas halaman per domain (default 20)")
    ap.add_argument("--delay", type=float, default=1.0, help="jeda detik antar request ke domain sama")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--unweighted", action="store_true", help="abaikan jumlah tautan (bobot=1)")
    ap.add_argument("--allow-private", action="store_true", help="izinkan localhost/IP privat (untuk uji lokal)")
    ap.add_argument("--ignore-robots", action="store_true", help="abaikan robots.txt (tidak disarankan)")
    ap.add_argument("--out", default="hasil", help="folder output (default: hasil)")
    ap.add_argument("--top", type=int, default=15, help="jumlah peringkat teratas yang ditampilkan")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.verbose is False:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
        log.propagate = False

    seeds = list(args.seeds)
    if args.seed_file:
        with open(args.seed_file, encoding="utf-8") as f:
            seeds += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not seeds:
        ap.error("berikan minimal satu URL seed atau --seed-file")

    crawler = Crawler(max_depth=args.depth, max_pages=args.max_pages,
                      per_domain=args.per_domain, delay=args.delay,
                      timeout=args.timeout, workers=args.workers,
                      respect_robots=not args.ignore_robots,
                      block_private=not args.allow_private)
    crawler.crawl(seeds)

    G = build_graph(crawler, weighted=not args.unweighted)
    if G.number_of_nodes() == 0:
        log.error("Tidak ada data yang berhasil di-crawl.")
        sys.exit(1)

    pr = pagerank(G, damping=args.damping)

    # validasi silang dengan implementasi networkx
    try:
        ref = nx.pagerank(G, alpha=args.damping, weight="weight", tol=1e-10)
        diff = max(abs(pr[n] - ref[n]) for n in pr)
        log.info("Selisih maksimum vs networkx.pagerank: %.2e", diff)
    except Exception as e:  # pragma: no cover
        log.warning("Validasi networkx dilewati: %s", e)

    ranking = save_results(G, pr, crawler, args.out)
    draw_png(G, pr, f"{args.out}/graph.png")
    draw_html(G, pr, f"{args.out}/graph.html")

    print(f"\n{'#':<4}{'Website':<45}{'PageRank':<12}{'In':<5}{'Out':<5}")
    print("-" * 71)
    for i, (site, score) in enumerate(ranking[:args.top], 1):
        print(f"{i:<4}{site:<45}{score:<12.6f}{G.in_degree(site):<5}{G.out_degree(site):<5}")
    print(f"\nWebsite terpenting: {ranking[0][0]} (PageRank {ranking[0][1]:.6f})")
    print(f"Hasil lengkap tersimpan di folder: {args.out}/")


if __name__ == "__main__":
    main()

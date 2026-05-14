"""
app.py — Klasifikasi Emosi Wajah
GLCM + Logistic Regression · Streamlit App

Cara jalankan:
    pip install streamlit scikit-image scikit-learn pillow numpy
    streamlit run app.py
"""

import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Raut Muka - Klasifikasi Emosi Wajah - Kelompok 1 Kelas A",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# CSS KUSTOM
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Card container */
.card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
/* Probabilitas bar custom */
.prob-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    font-size: 0.95rem;
}
.prob-label {
    width: 90px;
    font-weight: 600;
    text-transform: capitalize;
}
.prob-bar-bg {
    flex: 1;
    background: #f0f2f6;
    border-radius: 8px;
    height: 22px;
    overflow: hidden;
    margin: 0 10px;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}
.prob-val {
    width: 52px;
    text-align: right;
    font-weight: 700;
}
/* Prediksi utama badge */
.pred-badge {
    display: inline-block;
    padding: 0.45rem 1.2rem;
    border-radius: 50px;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
}
/* Fitur tabel */
.feat-table th {
    background: #f7f8fc;
    font-weight: 700;
}
.step-box {
    border-left: 4px solid #4e8cff;
    background: #f4f7ff;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #000;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# FUNGSI PREPROCESSING — SAMA PERSIS DENGAN NOTEBOOK
# ════════════════════════════════════════════════════════════════

def detect_and_crop_face(pil_image: Image.Image) -> tuple[Image.Image, str]:
    """
    Deteksi wajah dengan OpenCV Haar Cascade, lalu crop + sedikit padding.

    Alasan diperlukan:
        Model ditraining pada FER2013 yang sudah center-crop wajah.
        Jika user upload foto penuh (ada background, badan, dll),
        GLCM akan dihitung dari piksel non-wajah → prediksi tidak akurat.

    Strategi:
        1. Coba deteksi wajah dengan frontalface Haar Cascade
        2. Jika ditemukan → crop area wajah terbesar + padding 10%
        3. Jika tidak ditemukan → pakai gambar asli (fallback)

    Returns
    -------
    cropped_pil : Image.Image  Gambar setelah crop (atau asli jika gagal)
    status      : str          Pesan status untuk ditampilkan di UI
    """
    import cv2

    # Konversi ke grayscale numpy untuk detektor
    img_gray_np = np.array(pil_image.convert("L"))

    # Load Haar Cascade (bawaan OpenCV, tidak perlu download)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Deteksi wajah — coba beberapa scaleFactor untuk toleransi lebih luas
    faces = face_cascade.detectMultiScale(
        img_gray_np,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
    )

    # Fallback: coba dengan scaleFactor lebih agresif jika belum ketemu
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            img_gray_np,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(15, 15),
        )

    if len(faces) == 0:
        # Tidak ada wajah terdeteksi → gunakan gambar asli
        return pil_image, "Wajah tidak terdeteksi — menggunakan gambar penuh"

    # Pilih wajah terbesar (area w×h terbesar)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Tambah padding 10% di setiap sisi agar wajah tidak terpotong
    pad    = int(max(w, h) * 0.10)
    H, W   = img_gray_np.shape
    x1     = max(0, x - pad)
    y1     = max(0, y - pad)
    x2     = min(W, x + w + pad)
    y2     = min(H, y + h + pad)

    # Crop dari gambar asli (berwarna jika ada, untuk ditampilkan)
    cropped = pil_image.crop((x1, y1, x2, y2))
    n_faces = len(faces)
    msg = (
        f" {n_faces} wajah terdeteksi — crop wajah terbesar ({w}×{h} px)"
        if n_faces == 1 else
        f" {n_faces} wajah terdeteksi — crop wajah terbesar ({w}×{h} px)"
    )
    return cropped, msg


def preprocess_image(pil_image: Image.Image, img_size=(48, 48)) -> tuple:
    """
    Preprocessing pipeline yang IDENTIK dengan training di notebook.

    Tahapan:
        1. Konversi ke Grayscale ('L')
        2. Resize ke 48x48 dengan LANCZOS (kualitas terbaik)
        3. Simpan array uint8 (untuk ditampilkan)
        4. Normalisasi /255.0 → float32 [0,1]

    Catatan: Center crop wajah sudah dilakukan SEBELUM fungsi ini
    dipanggil (lihat detect_and_crop_face). FER2013 sudah pre-cropped
    saat training, jadi preprocessing inti tetap sama.

    Returns
    -------
    img_raw  : np.ndarray uint8  (48,48) — untuk ditampilkan
    img_norm : np.ndarray float32 (48,48) — untuk GLCM
    """
    img_gray    = pil_image.convert("L")
    img_resized = img_gray.resize(img_size, Image.LANCZOS)
    img_raw     = np.array(img_resized, dtype=np.uint8)
    img_norm    = img_raw.astype(np.float32) / 255.0
    return img_raw, img_norm


def quantize_image(img_norm: np.ndarray, levels: int) -> np.ndarray:
    """
    Kuantisasi float [0,1] → integer [0, levels-1].
    Rumus: level = ⌊pixel_norm x levels⌋  (di-clip agar tidak overflow)

    HARUS sama persis dengan fungsi di notebook.
    """
    return np.clip(
        np.floor(img_norm * levels).astype(np.int32), 0, levels - 1
    )


def extract_glcm_features(img_norm: np.ndarray, config: dict) -> tuple:
    """
    Ekstrak vektor fitur GLCM dari gambar grayscale.

    Parameter diambil dari config bundle (SAMA dengan saat training).
    Mengembalikan fitur terstandarisasi DAN fitur mentah (untuk tampilan).

    Returns
    -------
    feat_raw  : np.ndarray  Vektor fitur sebelum scaling
    glcm_4d   : np.ndarray  Matriks GLCM mentah (untuk visualisasi)
    """
    levels      = config["levels"]
    distances   = config["distances"]
    angles      = config["angles"]
    props       = config["glcm_props"]
    aggregation = config["aggregation"]

    img_q = quantize_image(img_norm, levels)
    glcm  = graycomatrix(
        img_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=config["symmetric"],
        normed=config["normed"],
    )

    feat_vec = []
    if aggregation == "mean":
        for prop in props:
            feat_vec.append(float(graycoprops(glcm, prop).mean()))
    else:  # concat
        for prop in props:
            feat_vec.extend(graycoprops(glcm, prop).flatten().tolist())

    return np.array(feat_vec, dtype=np.float32), glcm


def predict(img_norm: np.ndarray, bundle: dict) -> dict:
    """
    Jalankan pipeline prediksi lengkap.

    Returns dict berisi:
        predicted_class, probabilities, feature_raw,
        feature_scaled, glcm_4d
    """
    config = bundle["config"]
    scaler = bundle["scaler"]
    model  = bundle["model"]
    le     = bundle["label_encoder"]

    # Ekstraksi fitur GLCM
    feat_raw, glcm_4d = extract_glcm_features(img_norm, config)

    # Scaling (WAJIB pakai scaler yang sama dengan training)
    feat_scaled = scaler.transform(feat_raw.reshape(1, -1))

    # Prediksi
    pred_idx   = model.predict(feat_scaled)[0]
    pred_proba = model.predict_proba(feat_scaled)[0]
    pred_class = le.inverse_transform([pred_idx])[0]

    return {
        "predicted_class" : pred_class,
        "probabilities"   : dict(zip(le.classes_, pred_proba.tolist())),
        "feature_raw"     : feat_raw,
        "feature_scaled"  : feat_scaled[0],
        "glcm_4d"         : glcm_4d,
        "confidence"      : float(pred_proba.max()),
    }


# ════════════════════════════════════════════════════════════════
# LOAD BUNDLE (di-cache agar tidak reload setiap interaksi)
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def load_bundle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


BUNDLE_PATH = Path(__file__).parent / "model_bundle.pkl"


# ════════════════════════════════════════════════════════════════
# KONSTANTA TAMPILAN
# ════════════════════════════════════════════════════════════════
EMOTION_META = {
    "angry"   : {"emoji": "😠", "color": "#e74c3c", "label": "Angry"},
    "happy"   : {"emoji": "😄", "color": "#2ecc71", "label": "Happy"},
    "sad"     : {"emoji": "😢", "color": "#3498db", "label": "Sad"},
    "surprise": {"emoji": "😲", "color": "#f39c12", "label": "Surprise"},
}

PROP_META = {
    "contrast"    : ("Contrast",     "Besar = banyak perbedaan tajam antar piksel"),
    "homogeneity" : ("Homogeneity",  "Besar = piksel bertetangga sering bernilai serupa"),
    "energy"      : ("Energy",       "Besar = tekstur seragam / distribusi terkonsentrasi"),
    "correlation" : ("Correlation",  "Besar = ada pola linear konsisten antar piksel"),
}

ANGLE_LABELS = ["0°", "45°", "90°", "135°"]


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Pengaturan")

    if not BUNDLE_PATH.exists():
        st.error(" `model_bundle.pkl` tidak ditemukan!\n\n"
                 "Jalankan CELL 6.0 di notebook untuk menghasilkan file ini.")
        st.stop()

    bundle = load_bundle(str(BUNDLE_PATH))
    meta   = bundle["metadata"]
    cfg    = bundle["config"]

    st.success(" Model berhasil dimuat")

    st.markdown("### Info Model")
    st.markdown(f"""
| Parameter | Nilai |
|:----------|------:|
| Skenario | `{meta['scenario']}` |
| Level GLCM | `{cfg['levels']}` |
| Agregasi | `{cfg['aggregation']}` |
| Dimensi Fitur | `{meta['n_features']}D` |
| Akurasi Test | `{meta['acc_test']:.2%}` |
| Kelas | `{len(meta['classes'])}` |
    """)

    st.markdown("### Kelas yang Didukung")
    for cls in meta["classes"]:
        em = EMOTION_META.get(cls, {}).get("emoji", "🔹")
        st.markdown(f"- {em} **{cls.capitalize()}**")

    st.markdown("---")
    st.markdown("### Spesifikasi Preprocessing")
    st.markdown(f"""
<div class="step-box">① Face Detection (Haar Cascade)</div>
<div class="step-box">② Center Crop + Padding 10%</div>
<div class="step-box">③ Grayscale (mode 'L')</div>
<div class="step-box">④ Resize → <b>48 x 48</b> (LANCZOS)</div>
<div class="step-box">⑤ Normalisasi → <b>÷ 255.0</b> (float32)</div>
<div class="step-box">⑥ Kuantisasi → <b>{cfg['levels']} level</b></div>
<div class="step-box">⑦ GLCM (d=1, 4 sudut)</div>
<div class="step-box">⑧ StandardScaler.transform()</div>
<div class="step-box">⑨ LogisticRegression.predict()</div>
    """, unsafe_allow_html=True)

    st.markdown(f"<br><small>Diekspor: {meta['exported_at']}</small>",
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN UTAMA
# ════════════════════════════════════════════════════════════════
st.markdown("# Raut Muka - Klasifikasi Emosi Wajah - Kelompok 1 Kelas A")
st.markdown(
    "Upload foto wajah → sistem mengekstrak fitur tekstur **GLCM** "
    "→ **Logistic Regression** memprediksi emosi."
)
st.markdown("---")

# ── Upload gambar ─────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown("### Upload Gambar")
    uploaded = st.file_uploader(
        "Pilih file gambar (JPG / PNG / BMP / WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Gambar wajah frontal, pencahayaan cukup.",
    )

    # Demo jika tidak ada upload
    use_demo = False
    if not uploaded:
        st.markdown("*atau coba dengan gambar sintetis:*")
        use_demo = st.button("Gunakan Gambar Demo (Sintetis)", use_container_width=True)

    if uploaded or use_demo:
        # ── Load gambar ──────────────────────────────────────────
        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
        else:
            # Gambar sintetis sederhana menyerupai wajah
            arr = np.full((96, 96), 140, dtype=np.uint8)
            arr[5:30, 15:81]  = 210   # dahi
            arr[30:75, 8:88]  = 195   # pipi
            arr[32:40, 20:35] = 45    # mata kiri
            arr[32:40, 60:75] = 45    # mata kanan
            arr[50:58, 38:58] = 165   # hidung
            arr[68:74, 25:70] = 70    # mulut
            arr[74:80, 30:65] = 80
            import cv2 as _cv2
            arr = _cv2.GaussianBlur(arr, (5, 5), 0)
            pil_img = Image.fromarray(arr)

        # ── Step 0: Face Detection & Center Crop ─────────────────
        pil_cropped, crop_status = detect_and_crop_face(pil_img)

        # ── Step 1–4: Preprocessing ──────────────────────────────
        img_raw, img_norm = preprocess_image(pil_cropped, tuple(cfg["img_size"]))

        # Tampilkan original vs crop vs setelah preprocessing
        st.markdown("#### Gambar Asli")
        st.image(pil_img, use_container_width=True)

        # Status face detection
        if crop_status.startswith("✅"):
            st.success(crop_status)
        else:
            st.warning(crop_status)

        # Tampilkan hasil crop jika berbeda dari asli
        if pil_cropped is not pil_img:
            st.markdown("**Setelah Face Crop (Haar Cascade)**")
            st.image(pil_cropped, use_container_width=True,
                     caption="Area wajah yang akan diproses")

        col_pp1, col_pp2 = st.columns(2)
        with col_pp1:
            st.markdown("**Setelah Grayscale + Resize**")
            st.image(
                Image.fromarray(img_raw),
                caption=f"48×48 px | uint8",
                use_container_width=True,
            )
        with col_pp2:
            img_q_show = quantize_image(img_norm, cfg["levels"])
            # Normalisasi untuk tampilan
            img_q_disp = (img_q_show / (cfg["levels"] - 1) * 255).astype(np.uint8)
            st.markdown("**Setelah Kuantisasi**")
            st.image(
                Image.fromarray(img_q_disp),
                caption=f"{cfg['levels']} level | 0–{cfg['levels']-1}",
                use_container_width=True,
            )

        # ── Prediksi ─────────────────────────────────────────────
        with st.spinner("Mengekstrak fitur GLCM & memprediksi..."):
            result = predict(img_norm, bundle)

        # Simpan ke session state agar kolom kanan bisa mengakses
        st.session_state["result"]   = result
        st.session_state["img_raw"]  = img_raw
        st.session_state["img_norm"] = img_norm
        st.session_state["cfg"]      = cfg


# ── Kolom Hasil ───────────────────────────────────────────────────
with col_result:
    if "result" not in st.session_state:
        st.info("Upload gambar untuk melihat hasil prediksi.")
    else:
        result   = st.session_state["result"]
        img_norm = st.session_state["img_norm"]
        cfg_     = st.session_state["cfg"]

        pred_cls = result["predicted_class"]
        em_meta  = EMOTION_META.get(pred_cls, {"emoji":"🔹","color":"#555","label":pred_cls})

        # ── Hasil Utama ──────────────────────────────────────────
        st.markdown("### Hasil Prediksi")
        conf = result["confidence"]

        col_badge, col_conf = st.columns([1, 1])
        with col_badge:
            st.markdown(
                f"""<div style="text-align:center; padding:1.2rem;
                     background:{em_meta['color']}18;
                     border:2px solid {em_meta['color']};
                     border-radius:12px;">
                     <div style="font-size:3.5rem">{em_meta['emoji']}</div>
                     <div class="pred-badge"
                          style="background:{em_meta['color']};color:white;">
                       {em_meta['label'].upper()}
                     </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_conf:
            st.markdown(f"**Confidence**")
            st.progress(conf, text=f"{conf:.1%}")
            # Indikator keyakinan
            if conf >= 0.70:
                st.success("🟢 Model sangat yakin")
            elif conf >= 0.45:
                st.warning("🟡 Model cukup yakin")
            else:
                st.error("🔴 Model ragu-ragu\n(probabilitas tersebar merata)")

        st.markdown("---")

        # ── Distribusi Probabilitas ───────────────────────────────
        st.markdown("### Distribusi Probabilitas Semua Kelas")
        probs = result["probabilities"]

        # Urutkan dari tertinggi
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        for cls_name, prob_val in sorted_probs:
            em       = EMOTION_META.get(cls_name, {})
            color    = em.get("color", "#4e8cff")
            emoji    = em.get("emoji", "🔹")
            is_pred  = (cls_name == pred_cls)
            pct      = prob_val * 100
            bar_pct  = int(prob_val * 100)

            st.markdown(
                f"""<div class="prob-row" style="{'font-weight:700;' if is_pred else ''}">
                  <span class="prob-label">{emoji} {cls_name.capitalize()}{'  ←' if is_pred else ''}</span>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill"
                         style="width:{bar_pct}%;background:{color};
                                opacity:{'1' if is_pred else '0.6'};"></div>
                  </div>
                  <span class="prob-val" style="color:{color if is_pred else '#555'}">
                    {pct:.1f}%
                  </span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Fitur GLCM yang Diekstrak ────────────────────────────
        st.markdown("### Fitur GLCM yang Diekstrak")

        feat_raw    = result["feature_raw"]
        feat_scaled = result["feature_scaled"]
        glcm_props  = cfg_["glcm_props"]
        aggregation = cfg_["aggregation"]

        if aggregation == "mean":
            # 4 fitur — tampilkan tabel sederhana
            st.markdown("**Agregasi: Mean (4 sudut → 1 nilai per fitur)**")
            col_f1, col_f2 = st.columns(2)
            for idx, prop in enumerate(glcm_props):
                col = col_f1 if idx % 2 == 0 else col_f2
                pname, pdesc = PROP_META.get(prop, (prop, ""))
                raw_val    = feat_raw[idx]
                scaled_val = feat_scaled[idx]
                with col:
                    st.markdown(
                        f"""<div class="card" style="border-left:4px solid
                            {'#e74c3c #2ecc71 #9b59b6 #3498db'.split()[idx]}">
                          <b>{pname}</b><br>
                          <small style="color:#888">{pdesc}</small><br><br>
                          <span style="font-size:1.4rem;font-weight:700">
                            {raw_val:.5f}
                          </span>
                          <span style="color:#aaa;font-size:0.8rem">
                            &nbsp;(scaled: {scaled_val:+.3f})
                          </span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            # 16 fitur — tampilkan sebagai tabel
            st.markdown("**Agregasi: Concat (4 sudut x 4 fitur = 16 nilai)**")
            rows = []
            for p_idx, prop in enumerate(glcm_props):
                for a_idx, ang in enumerate(ANGLE_LABELS):
                    feat_idx = p_idx * 4 + a_idx
                    rows.append({
                        "Fitur"  : PROP_META.get(prop, (prop,))[0],
                        "Sudut"  : ang,
                        "Nilai"  : f"{feat_raw[feat_idx]:.5f}",
                        "Scaled" : f"{feat_scaled[feat_idx]:+.3f}",
                    })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

        st.markdown("---")

        # ── Detail Pipeline (expandable) ────────────────────────
        with st.expander("Detail Pipeline Preprocessing & GLCM", expanded=False):
            img_q = quantize_image(img_norm, cfg_["levels"])
            glcm_4d = result["glcm_4d"]

            st.markdown(f"""
**Pipeline yang dijalankan:**

0. **Face Detection** → Haar Cascade frontalface  
   → Crop wajah terbesar + padding 10%  

1. **Grayscale + Resize** → `{cfg_['img_size']}` px  
   → Min pixel: `{st.session_state['img_raw'].min()}` | Max: `{st.session_state['img_raw'].max()}`

2. **Normalisasi** → `÷ 255.0`  
   → Range: `[{img_norm.min():.4f}, {img_norm.max():.4f}]`

3. **Kuantisasi** → `⌊pixel x {cfg_['levels']}⌋`  
   → Range: `[0, {cfg_['levels']-1}]`  
   → Distribusi level: `{np.unique(img_q).tolist()[:8]}{'...' if len(np.unique(img_q))>8 else ''}`

4. **GLCM** → `graycomatrix(d=1, angles=[0°,45°,90°,135°], L={cfg_['levels']})`  
   → Shape GLCM: `{glcm_4d.shape}` (LxLx1x4 sudut)

5. **Fitur** → `{cfg_['aggregation']}` → `{len(feat_raw)}D vector`

6. **Scaling** → `StandardScaler.transform()` (μ,σ dari data training)  
   → Sebelum: `[{', '.join([f'{v:.3f}' for v in feat_raw[:4]])}...]`  
   → Sesudah: `[{', '.join([f'{v:+.3f}' for v in feat_scaled[:4]])}...]`

7. **Prediksi** → `LogisticRegression.predict_proba()` → `argmax`
            """)

            # Mini GLCM heatmap (sudut 0° saja)
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use("Agg")

                L_show  = min(cfg_["levels"], 16)
                g2d     = glcm_4d[:L_show, :L_show, 0, 0]
                fig, ax = plt.subplots(figsize=(4, 3.5))
                im = ax.imshow(g2d, cmap="plasma", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.05)
                ax.set_title(f"GLCM Sudut 0° (L={cfg_['levels']})\nΣ={g2d.sum():.4f}",
                             fontsize=9)
                ax.set_xlabel("Level j", fontsize=8)
                ax.set_ylabel("Level i", fontsize=8)
                ax.tick_params(labelsize=7)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                buf.seek(0)
                st.image(buf, caption="GLCM Heatmap Sudut 0°")
                plt.close(fig)
            except Exception:
                pass

        # ── Perbandingan Nilai Fitur per Sudut (concat) ──────────
        if aggregation == "concat":
            with st.expander("Nilai Fitur per Sudut", expanded=False):
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use("Agg")

                    fig, ax = plt.subplots(figsize=(9, 3.5))
                    prop_colors = ["#e74c3c", "#2ecc71", "#9b59b6", "#3498db"]
                    x = np.arange(4)
                    w = 0.2

                    for p_idx, (prop, pc) in enumerate(zip(glcm_props, prop_colors)):
                        vals = feat_raw[p_idx*4:(p_idx+1)*4]
                        ax.bar(x + (p_idx-1.5)*w, vals, w,
                               label=PROP_META.get(prop,(prop,))[0],
                               color=pc, alpha=0.85, edgecolor="white")

                    ax.set_xticks(x)
                    ax.set_xticklabels(ANGLE_LABELS)
                    ax.set_xlabel("Sudut")
                    ax.set_ylabel("Nilai Fitur")
                    ax.set_title("Nilai Fitur GLCM per Sudut")
                    ax.legend(fontsize=8)
                    ax.grid(axis="y", alpha=0.3, ls="--")
                    fig.tight_layout()

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    buf.seek(0)
                    st.image(buf)
                    plt.close(fig)
                except Exception:
                    pass


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<small>Model: Logistic Regression + GLCM | Dataset: FER2013 | "
    "Preprocessing: Grayscale 48x48 + Normalisasi + Kuantisasi</small>",
    unsafe_allow_html=True,
)

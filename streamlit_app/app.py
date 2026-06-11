"""
app.py — Klasifikasi Emosi Wajah
GLCM + Geometri + Logistic Regression / SVM · Streamlit App

Cara jalankan:
    pip install streamlit scikit-image scikit-learn pillow numpy opencv-python
    streamlit run app.py
"""

import io
import pickle
import warnings
from pathlib import Path

import cv2
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
.card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.prob-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    font-size: 0.95rem;
}
.prob-label {
    width: 110px;
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
.pred-badge {
    display: inline-block;
    padding: 0.45rem 1.2rem;
    border-radius: 50px;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
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
.model-card {
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.88rem;
}
.geo-feat-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


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
    "contrast"    : ("Contrast",    "Besar = banyak perbedaan tajam antar piksel"),
    "homogeneity" : ("Homogeneity", "Besar = piksel bertetangga sering bernilai serupa"),
    "energy"      : ("Energy",      "Besar = tekstur seragam / distribusi terkonsentrasi"),
    "correlation" : ("Correlation", "Besar = ada pola linear konsisten antar piksel"),
}

GEO_FEAT_META = {
    "eye_ratio"         : ("Eye Ratio",          "Keterbukaan zona mata (surprise = tinggi)"),
    "mouth_ratio"       : ("Mouth Ratio",         "Aktivitas mulut (happy = tinggi)"),
    "brow_height"       : ("Brow Height",         "Pergerakan alis (surprise = tinggi)"),
    "face_symmetry"     : ("Face Symmetry",       "Simetri kiri-kanan wajah (1 = sempurna simetris)"),
    "edge_dens_upper"   : ("Edge Density Upper",  "Kerapatan tepi area atas (alis/dahi)"),
    "edge_dens_lower"   : ("Edge Density Lower",  "Kerapatan tepi area bawah (mulut/dagu)"),
    "vert_gradient"     : ("Vert Gradient",       "Distribusi intensitas atas vs bawah"),
    "pixel_var_ratio"   : ("Pixel Var Ratio",     "Rasio variansi bawah/atas wajah"),
}

GEO_FEAT_NAMES = list(GEO_FEAT_META.keys())
ANGLE_LABELS   = ["0°", "45°", "90°", "135°"]

MODEL_META = {
    "A": {
        "name"  : "Model A — GLCM + Logistic Regression",
        "short" : "GLCM + LR",
        "color" : "#3498db",
        "icon"  : "📊",
        "desc"  : "Baseline: hanya fitur tekstur GLCM (4D)",
        "dim"   : 4,
    },
    "B": {
        "name"  : "Model B — GLCM + Geometri + Logistic Regression",
        "short" : "GLCM + Geo + LR",
        "color" : "#2ecc71",
        "icon"  : "📐",
        "desc"  : "Tambahan fitur geometri wajah (12D total)",
        "dim"   : 12,
    },
    "C": {
        "name"  : "Model C — GLCM + Geometri + SVM (RBF)",
        "short" : "GLCM + Geo + SVM",
        "color" : "#e74c3c",
        "icon"  : "🧠",
        "desc"  : "SVM kernel RBF untuk hubungan non-linear (12D)",
        "dim"   : 12,
    },
}


# ════════════════════════════════════════════════════════════════
# FUNGSI PREPROCESSING
# ════════════════════════════════════════════════════════════════

def detect_and_crop_face(pil_image: Image.Image) -> tuple:
    img_gray_np = np.array(pil_image.convert("L"))
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        img_gray_np, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            img_gray_np, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15)
        )
    if len(faces) == 0:
        return pil_image, "Wajah tidak terdeteksi — menggunakan gambar penuh"

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(max(w, h) * 0.10)
    H, W = img_gray_np.shape
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    cropped = pil_image.crop((x1, y1, x2, y2))
    n = len(faces)
    msg = f" {n} wajah terdeteksi — crop wajah terbesar ({w}×{h} px)"
    return cropped, msg


def preprocess_image(pil_image: Image.Image, img_size=(48, 48)) -> tuple:
    img_gray    = pil_image.convert("L")
    img_resized = img_gray.resize(img_size, Image.LANCZOS)
    img_raw     = np.array(img_resized, dtype=np.uint8)
    img_norm    = img_raw.astype(np.float32) / 255.0
    return img_raw, img_norm


def quantize_image(img_norm: np.ndarray, levels: int) -> np.ndarray:
    return np.clip(
        np.floor(img_norm * levels).astype(np.int32), 0, levels - 1
    )


def extract_glcm_features_raw(img_norm: np.ndarray, config: dict) -> tuple:
    """Ekstrak fitur GLCM (unscaled). Mengembalikan (feat_vec, glcm_4d)."""
    levels      = config["levels"]
    distances   = config["distances"]
    angles      = config["angles"]
    props       = config["glcm_props"]
    aggregation = config["aggregation"]

    img_q = quantize_image(img_norm, levels)
    glcm  = graycomatrix(
        img_q, distances=distances, angles=angles, levels=levels,
        symmetric=config["symmetric"], normed=config["normed"],
    )
    feat_vec = []
    if aggregation == "mean":
        for prop in props:
            feat_vec.append(float(graycoprops(glcm, prop).mean()))
    else:
        for prop in props:
            feat_vec.extend(graycoprops(glcm, prop).flatten().tolist())
    return np.array(feat_vec, dtype=np.float32), glcm


def extract_geometric_features(img_raw: np.ndarray) -> np.ndarray:
    """
    Ekstrak 8 fitur geometri wajah dari gambar uint8 48x48.
    Identik dengan fungsi di notebook (CELL 7.0).
    """
    img = img_raw.astype(np.float32)
    h, w = img.shape  # 48, 48

    upper      = img[:h//2, :]
    lower      = img[h//2:, :]
    eye_zone   = img[12:22, 8:40]
    mouth_zone = img[30:44, 10:38]
    brow_zone  = img[8:15,  8:40]

    sx   = cv2.Sobel(img_raw, cv2.CV_64F, 1, 0, ksize=3)
    sy   = cv2.Sobel(img_raw, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sx**2 + sy**2)

    # Fitur 1: eye_ratio
    eye_var_vert = eye_zone.std(axis=0).mean()
    eye_ratio    = eye_var_vert / (eye_zone.mean() + 1e-6)

    # Fitur 2: mouth_ratio
    mouth_ratio = mouth_zone.std() / (mouth_zone.mean() + 1e-6)

    # Fitur 3: brow_height
    brow_grad   = np.abs(np.diff(brow_zone.mean(axis=1))).mean()
    brow_height = brow_grad / (brow_zone.mean() + 1e-6)

    # Fitur 4: face_symmetry
    left_half     = img[:, :w//2]
    right_half    = img[:, w//2:][:, ::-1]
    diff_sym      = np.abs(left_half - right_half).mean()
    face_symmetry = 1.0 / (1.0 + diff_sym / 255.0)

    # Fitur 5 & 6: edge_density
    edge_density_upper = edge[:h//2, :].mean() / 255.0
    edge_density_lower = edge[h//2:, :].mean() / 255.0

    # Fitur 7: vertical_gradient
    vertical_gradient = (lower.mean() - upper.mean()) / 255.0

    # Fitur 8: pixel_var_ratio
    pixel_var_ratio = lower.std() / (upper.std() + 1e-6)

    return np.array([
        eye_ratio, mouth_ratio, brow_height, face_symmetry,
        edge_density_upper, edge_density_lower,
        vertical_gradient, pixel_var_ratio,
    ], dtype=np.float32)


def predict_all_models(img_raw: np.ndarray, img_norm: np.ndarray,
                        bundles: dict) -> dict:
    """
    Jalankan prediksi untuk ketiga model sekaligus.

    Returns dict dengan key 'A', 'B', 'C', masing-masing berisi:
        predicted_class, probabilities, confidence,
        feat_glcm_raw, feat_geo_raw, feat_combined_raw,
        feat_scaled, glcm_4d
    """
    results = {}

    # Fitur GLCM (unscaled) — dihitung sekali, dipakai semua model
    bundle_A = bundles["A"]
    cfg      = bundle_A["config"]
    feat_glcm, glcm_4d = extract_glcm_features_raw(img_norm, cfg)

    # Fitur Geometri (unscaled) — dihitung sekali
    feat_geo = extract_geometric_features(img_raw)

    # Fitur gabungan
    feat_combined = np.concatenate([feat_glcm, feat_geo])

    for mid, bundle in bundles.items():
        scaler = bundle["scaler"]
        model  = bundle["model"]
        le     = bundle["label_encoder"]

        # Pilih vektor fitur sesuai model
        if mid == "A":
            feat_input = feat_glcm
        else:
            feat_input = feat_combined

        feat_scaled = scaler.transform(feat_input.reshape(1, -1))
        pred_idx    = model.predict(feat_scaled)[0]
        pred_proba  = model.predict_proba(feat_scaled)[0]
        pred_class  = le.inverse_transform([pred_idx])[0]

        results[mid] = {
            "predicted_class" : pred_class,
            "probabilities"   : dict(zip(le.classes_, pred_proba.tolist())),
            "confidence"      : float(pred_proba.max()),
            "feat_glcm_raw"   : feat_glcm,
            "feat_geo_raw"    : feat_geo,
            "feat_combined_raw": feat_combined,
            "feat_scaled"     : feat_scaled[0],
            "glcm_4d"         : glcm_4d,
        }

    return results


# ════════════════════════════════════════════════════════════════
# LOAD BUNDLE (cache agar tidak reload tiap interaksi)
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def load_bundles(base_dir: str) -> dict:
    """
    Load tiga bundle model dari folder yang sama dengan app.py.

    Nama file yang diharapkan:
        model_bundle_A.pkl  — GLCM + LR
        model_bundle_B.pkl  — GLCM + Geo + LR
        model_bundle_C.pkl  — GLCM + Geo + SVM
    
    Fallback: jika hanya ada model_bundle.pkl (legacy single-model),
    bundle tsb dipakai untuk Model A, B, C semuanya agar tidak crash.
    """
    base = Path(base_dir)
    bundles = {}
    legacy  = base / "model_bundle.pkl"

    for mid in ["A", "B", "C"]:
        path = base / f"model_bundle_{mid}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                bundles[mid] = pickle.load(f)
        elif legacy.exists():
            with open(legacy, "rb") as f:
                bundles[mid] = pickle.load(f)
        else:
            bundles[mid] = None

    return bundles


BASE_DIR = Path(__file__).parent
bundles  = load_bundles(str(BASE_DIR))

# Cek ketersediaan bundle
any_loaded = any(v is not None for v in bundles.values())


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")

    if not any_loaded:
        st.error(
            "`model_bundle_A/B/C.pkl` tidak ditemukan!\n\n"
            "Jalankan CELL 7.1 di notebook untuk menghasilkan ketiga bundle,\n"
            "atau CELL 6.0 untuk bundle tunggal (legacy)."
        )
        st.stop()

    # ── Pilihan model ──────────────────────────────────────────
    st.markdown("### Pilih Model")
    selected_model = st.radio(
        "Model yang digunakan:",
        options=["A", "B", "C"],
        format_func=lambda m: f"{MODEL_META[m]['icon']}  {MODEL_META[m]['short']}",
        index=2,   # default: Model C (terbaik)
        help="Pilih model untuk prediksi. Model C (SVM) umumnya lebih akurat.",
    )

    mm = MODEL_META[selected_model]
    st.markdown(
        f"""<div class="model-card" style="background:{mm['color']}18;
             border-left:4px solid {mm['color']};">
             <b>{mm['icon']} {mm['name']}</b><br>
             <small>{mm['desc']}</small>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Info model ─────────────────────────────────────────────
    st.markdown("### Info Semua Model")
    for mid, bndl in bundles.items():
        if bndl is None:
            st.warning(f"Model {mid}: bundle tidak ditemukan")
            continue
        meta_ = bndl.get("metadata", {})
        mm_   = MODEL_META[mid]
        acc   = meta_.get("acc_test", 0)
        dim   = meta_.get("n_features", mm_["dim"])
        active = " ← aktif" if mid == selected_model else ""
        st.markdown(
            f"""<div class="model-card" style="background:{'#f8f9fa' if mid != selected_model
                else mm_['color']+'18'};border-left:3px solid {mm_['color']};">
                <b>{mm_['icon']} Model {mid}</b>{active}<br>
                <small>{mm_['short']} | {dim}D | Acc: {acc:.2%}</small>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Preprocessing steps ────────────────────────────────────
    active_bundle = bundles[selected_model]
    cfg_sidebar   = active_bundle["config"] if active_bundle else {}
    lvl           = cfg_sidebar.get("levels", "?")
    uses_geo      = selected_model in ("B", "C")
    classifier    = "SVM (RBF)" if selected_model == "C" else "Logistic Regression"

    st.markdown("---")
    st.markdown("### Pipeline Preprocessing")
    steps = [
        "① Face Detection (Haar Cascade)",
        "② Center Crop + Padding 10%",
        "③ Grayscale (mode 'L')",
        f"④ Resize → <b>48 × 48</b> (LANCZOS)",
        "⑤ Normalisasi → <b>÷ 255.0</b> (float32)",
        f"⑥ Kuantisasi → <b>{lvl} level</b>",
        "⑦ GLCM (d=1, 4 sudut) → 4D",
    ]
    if uses_geo:
        steps += [
            "⑧ Fitur Geometri Wajah → 8D",
            "⑨ Gabung GLCM + Geometri → <b>12D</b>",
            "⑩ StandardScaler.transform()",
            f"⑪ {classifier}.predict()",
        ]
    else:
        steps += [
            "⑧ StandardScaler.transform()",
            f"⑨ {classifier}.predict()",
        ]
    for s in steps:
        st.markdown(f'<div class="step-box">{s}</div>', unsafe_allow_html=True)

    if active_bundle:
        meta_s = active_bundle.get("metadata", {})
        st.markdown(
            f"<br><small>Diekspor: {meta_s.get('exported_at','—')}</small>",
            unsafe_allow_html=True,
        )

    # ── Kelas yang didukung ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Kelas yang Didukung")
    classes_ = active_bundle["metadata"].get("classes", list(EMOTION_META.keys())) if active_bundle else list(EMOTION_META.keys())
    for cls in classes_:
        em = EMOTION_META.get(cls, {}).get("emoji", "🔹")
        st.markdown(f"- {em} **{cls.capitalize()}**")


# ════════════════════════════════════════════════════════════════
# HALAMAN UTAMA
# ════════════════════════════════════════════════════════════════
st.markdown("# Raut Muka — Klasifikasi Emosi Wajah · Kelompok 1 Kelas A")
st.markdown(
    "Upload foto wajah → sistem mengekstrak fitur **GLCM** (tekstur) "
    "+ **Geometri Wajah** → model memprediksi emosi."
)

# Ringkasan tiga model
col_ma, col_mb, col_mc = st.columns(3)
for col, mid in zip([col_ma, col_mb, col_mc], ["A", "B", "C"]):
    mm_ = MODEL_META[mid]
    bndl_ = bundles[mid]
    acc_  = bndl_["metadata"].get("acc_test", 0) if bndl_ else 0
    active_border = f"border:2px solid {mm_['color']};" if mid == selected_model else ""
    with col:
        st.markdown(
            f"""<div class="model-card" style="background:{mm_['color']}12;
                 {active_border} border-radius:10px; padding:0.8rem; text-align:center;">
                 <div style="font-size:1.5rem">{mm_['icon']}</div>
                 <b>Model {mid}</b><br>
                 <small>{mm_['short']}</small><br>
                 <span style="color:{mm_['color']};font-weight:700;">
                   Acc: {acc_:.2%}
                 </span>
                 {'<br><small>← Aktif</small>' if mid == selected_model else ''}
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# UPLOAD & PREDIKSI
# ════════════════════════════════════════════════════════════════
col_upload, col_result = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown("### Upload Gambar")
    uploaded = st.file_uploader(
        "Pilih file gambar (JPG / PNG / BMP / WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Gambar wajah frontal, pencahayaan cukup.",
    )

    use_demo = False
    if not uploaded:
        st.markdown("*atau coba dengan gambar sintetis:*")
        use_demo = st.button("Gunakan Gambar Demo (Sintetis)", use_container_width=True)

    if uploaded or use_demo:
        # ── Load gambar ──────────────────────────────────────────
        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
        else:
            arr = np.full((96, 96), 140, dtype=np.uint8)
            arr[5:30, 15:81]  = 210
            arr[30:75, 8:88]  = 195
            arr[32:40, 20:35] = 45
            arr[32:40, 60:75] = 45
            arr[50:58, 38:58] = 165
            arr[68:74, 25:70] = 70
            arr[74:80, 30:65] = 80
            arr = cv2.GaussianBlur(arr, (5, 5), 0)
            pil_img = Image.fromarray(arr)

        # ── Face detection & crop ─────────────────────────────────
        pil_cropped, crop_status = detect_and_crop_face(pil_img)

        # ── Preprocessing ─────────────────────────────────────────
        cfg_main  = bundles[selected_model]["config"]
        img_raw, img_norm = preprocess_image(pil_cropped, tuple(cfg_main["img_size"]))

        # ── Tampilkan gambar ──────────────────────────────────────
        st.markdown("#### Gambar Asli")
        st.image(pil_img, use_container_width=True)

        if crop_status.startswith("✅") or "terdeteksi" in crop_status:
            st.success(crop_status) if "terdeteksi" in crop_status else st.warning(crop_status)
        else:
            st.warning(crop_status)

        if pil_cropped is not pil_img:
            st.markdown("**Setelah Face Crop**")
            st.image(pil_cropped, use_container_width=True,
                     caption="Area wajah yang akan diproses")

        col_pp1, col_pp2 = st.columns(2)
        with col_pp1:
            st.markdown("**Grayscale + Resize**")
            st.image(Image.fromarray(img_raw),
                     caption="48×48 px | uint8", use_container_width=True)
        with col_pp2:
            img_q_show = quantize_image(img_norm, cfg_main["levels"])
            img_q_disp = (img_q_show / (cfg_main["levels"] - 1) * 255).astype(np.uint8)
            st.markdown("**Setelah Kuantisasi**")
            st.image(Image.fromarray(img_q_disp),
                     caption=f"{cfg_main['levels']} level", use_container_width=True)

        # ── Prediksi semua model ──────────────────────────────────
        with st.spinner("Mengekstrak fitur & memprediksi (3 model)..."):
            all_results = predict_all_models(img_raw, img_norm, bundles)

        # Simpan ke session state
        st.session_state["all_results"]    = all_results
        st.session_state["img_raw"]        = img_raw
        st.session_state["img_norm"]       = img_norm
        st.session_state["cfg"]            = cfg_main
        st.session_state["selected_model"] = selected_model


# ════════════════════════════════════════════════════════════════
# KOLOM HASIL
# ════════════════════════════════════════════════════════════════
with col_result:
    if "all_results" not in st.session_state:
        st.info("Upload gambar untuk melihat hasil prediksi.")
    else:
        all_results    = st.session_state["all_results"]
        img_raw_s      = st.session_state["img_raw"]
        img_norm_s     = st.session_state["img_norm"]
        cfg_s          = st.session_state["cfg"]
        active_model   = st.session_state.get("selected_model", selected_model)

        result   = all_results[active_model]
        pred_cls = result["predicted_class"]
        em_meta  = EMOTION_META.get(pred_cls, {"emoji": "🔹", "color": "#555", "label": pred_cls})
        conf     = result["confidence"]
        mm_act   = MODEL_META[active_model]

        # ── Hasil utama ───────────────────────────────────────────
        st.markdown(f"### Hasil Prediksi — {mm_act['icon']} Model {active_model}")

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
            st.markdown("**Confidence**")
            st.progress(conf, text=f"{conf:.1%}")
            if conf >= 0.70:
                st.success("🟢 Model sangat yakin")
            elif conf >= 0.45:
                st.warning("🟡 Model cukup yakin")
            else:
                st.error("🔴 Model ragu-ragu\n(probabilitas tersebar merata)")

        st.markdown("---")

        # ── Perbandingan tiga model sekaligus ─────────────────────
        st.markdown("### Perbandingan Hasil Tiga Model")
        col_a, col_b, col_c = st.columns(3)
        for col, mid in zip([col_a, col_b, col_c], ["A", "B", "C"]):
            r_   = all_results[mid]
            mm_  = MODEL_META[mid]
            em_  = EMOTION_META.get(r_["predicted_class"],
                                    {"emoji": "🔹", "color": "#555"})
            is_active = (mid == active_model)
            with col:
                st.markdown(
                    f"""<div style="text-align:center; padding:0.8rem;
                         background:{mm_['color']}{'22' if is_active else '0a'};
                         border:{'2px' if is_active else '1px'} solid {mm_['color']};
                         border-radius:10px;">
                         <b>{mm_['icon']} Model {mid}</b><br>
                         <div style="font-size:2rem">{em_['emoji']}</div>
                         <b style="color:{em_['color']}">{r_['predicted_class'].capitalize()}</b><br>
                         <small>{r_['confidence']:.1%} conf</small>
                         {'<br><small><b>← Aktif</b></small>' if is_active else ''}
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Distribusi probabilitas (model aktif) ─────────────────
        st.markdown(f"### Distribusi Probabilitas — Model {active_model}")
        probs = result["probabilities"]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        for cls_name, prob_val in sorted_probs:
            em_      = EMOTION_META.get(cls_name, {})
            color    = em_.get("color", "#4e8cff")
            emoji    = em_.get("emoji", "🔹")
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

        # ── Fitur GLCM ────────────────────────────────────────────
        st.markdown("### Fitur GLCM yang Diekstrak (4D)")
        feat_glcm   = result["feat_glcm_raw"]
        glcm_props  = cfg_s["glcm_props"]
        prop_colors = ["#e74c3c", "#2ecc71", "#9b59b6", "#3498db"]

        col_f1, col_f2 = st.columns(2)
        for idx, (prop, pc) in enumerate(zip(glcm_props, prop_colors)):
            col_fx = col_f1 if idx % 2 == 0 else col_f2
            pname, pdesc = PROP_META.get(prop, (prop, ""))
            raw_val = feat_glcm[idx]
            with col_fx:
                st.markdown(
                    f"""<div class="card" style="border-left:4px solid {pc}">
                      <b>{pname}</b><br>
                      <small style="color:#888">{pdesc}</small><br><br>
                      <span style="font-size:1.3rem;font-weight:700">{raw_val:.5f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Fitur Geometri (Model B & C) ──────────────────────────
        if active_model in ("B", "C"):
            st.markdown("### Fitur Geometri Wajah (8D)")
            feat_geo = result["feat_geo_raw"]
            col_g1, col_g2 = st.columns(2)
            for idx, (fname, (flabel, fdesc)) in enumerate(GEO_FEAT_META.items()):
                col_gx = col_g1 if idx % 2 == 0 else col_g2
                val    = feat_geo[idx]
                with col_gx:
                    st.markdown(
                        f"""<div class="card" style="border-left:4px solid #9b59b6">
                          <b>{flabel}</b><br>
                          <small style="color:#888">{fdesc}</small><br><br>
                          <span style="font-size:1.2rem;font-weight:700">{val:.5f}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

        # ── Detail pipeline (expandable) ──────────────────────────
        with st.expander("Detail Pipeline Preprocessing & Fitur", expanded=False):
            img_q = quantize_image(img_norm_s, cfg_s["levels"])
            glcm_4d = result["glcm_4d"]
            uses_geo = active_model in ("B", "C")
            dim_total = 4 + (8 if uses_geo else 0)

            st.markdown(f"""
**Pipeline Model {active_model} ({mm_act['short']}):**

0. **Face Detection** → Haar Cascade frontalface → crop + padding 10%

1. **Grayscale + Resize** → `{cfg_s['img_size']}` px  
   → Min: `{img_raw_s.min()}` | Max: `{img_raw_s.max()}`

2. **Normalisasi** → `÷ 255.0`  
   → Range: `[{img_norm_s.min():.4f}, {img_norm_s.max():.4f}]`

3. **Kuantisasi** → `⌊pixel × {cfg_s['levels']}⌋`  
   → Distribusi: `{np.unique(img_q).tolist()[:10]}{'...' if len(np.unique(img_q))>10 else ''}`

4. **GLCM** → `graycomatrix(d=1, 4 sudut, L={cfg_s['levels']})` → shape `{glcm_4d.shape}`

5. **Fitur GLCM** → mean 4 sudut → **4D**: `{np.round(feat_glcm, 4).tolist()}`
{'6. **Fitur Geometri** → 8 fitur berbasis region piksel → **8D**' if uses_geo else ''}
{'7. **Gabungkan** → 4D + 8D = **12D**' if uses_geo else ''}

{'7' if not uses_geo else '8'}. **StandardScaler.transform()** → fitur terstandarisasi

{'8' if not uses_geo else '9'}. **{"SVM (RBF)" if active_model == "C" else "LogisticRegression"}.predict_proba()** → argmax → prediksi
            """)

            # Mini GLCM heatmap
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use("Agg")

                L_show = min(cfg_s["levels"], 16)
                g2d    = glcm_4d[:L_show, :L_show, 0, 0]
                fig, ax = plt.subplots(figsize=(4, 3.5))
                im = ax.imshow(g2d, cmap="plasma", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.05)
                ax.set_title(f"GLCM Sudut 0° (L={cfg_s['levels']})\nΣ={g2d.sum():.4f}",
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

            # Bar chart fitur geometri per emosi (konteks)
            if uses_geo:
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use("Agg")

                    feat_geo_cur = result["feat_geo_raw"]
                    fig2, ax2 = plt.subplots(figsize=(9, 3))
                    geo_colors = ["#9b59b6"] * 8
                    bars = ax2.barh(GEO_FEAT_NAMES, feat_geo_cur,
                                    color=geo_colors, alpha=0.8, edgecolor="white")
                    for bar, val in zip(bars, feat_geo_cur):
                        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                                 f"{val:.4f}", va="center", fontsize=8)
                    ax2.set_xlabel("Nilai Fitur", fontsize=9)
                    ax2.set_title("Fitur Geometri Wajah yang Diekstrak", fontsize=10)
                    ax2.grid(axis="x", alpha=0.3, ls="--")
                    fig2.tight_layout()
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="png", dpi=120, bbox_inches="tight")
                    buf2.seek(0)
                    st.image(buf2, caption="Profil Fitur Geometri Gambar Ini")
                    plt.close(fig2)
                except Exception:
                    pass


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<small>Model: Logistic Regression & SVM (RBF) | Fitur: GLCM + Geometri Wajah | "
    "Dataset: FER2013 | Preprocessing: Grayscale 48×48 + GLCM + Geometri</small>",
    unsafe_allow_html=True,
)

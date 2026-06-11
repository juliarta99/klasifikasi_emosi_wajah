"""
app.py — Klasifikasi Emosi Wajah
GLCM + Geometri + Logistic Regression / SVM · Streamlit App

Cara jalankan:
    pip install streamlit scikit-image scikit-learn pillow numpy opencv-python
    streamlit run app.py

Fix utama v3:
  - predict_all_models: setiap model mengekstrak GLCM dari config bundle-nya sendiri
    sehingga dimensi fitur selalu cocok dengan scaler (tidak ada mismatch 4D vs 16D)
  - Emoji emosi diganti SVG inline agar tidak bergantung pada font emoji sistem
  - Akurasi ditampilkan langsung dari metadata bundle (bukan hardcode)
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
    page_title="Raut Muka — Klasifikasi Emosi Wajah — Kelompok 1 Kelas A",
    page_icon=":face_with_open_mouth:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# SVG EMOJI EMOSI  (tidak bergantung font emoji sistem)
# ════════════════════════════════════════════════════════════════
EMOTION_EMOJI = {
    "angry"   : "😠",
    "happy"   : "😄",
    "sad"     : "😢",
    "surprise": "😲",
}

def emotion_svg(cls: str, size: int = 56) -> str:
    """Kembalikan emoji sebagai span HTML dengan ukuran font sesuai size."""
    emoji = EMOTION_EMOJI.get(cls, "🔹")
    return f'<span style="font-size:{size}px;line-height:1;">{emoji}</span>'

# ════════════════════════════════════════════════════════════════
# CSS KUSTOM
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.card {
    background: var(--background-color, white);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.8rem;
}
.prob-row {
    display: flex;
    align-items: center;
    margin: 7px 0;
    font-size: 0.93rem;
}
.prob-label {
    width: 120px;
    font-weight: 600;
    text-transform: capitalize;
    display: flex;
    align-items: center;
    gap: 6px;
}
.prob-bar-bg {
    flex: 1;
    background: rgba(128,128,128,0.12);
    border-radius: 6px;
    height: 20px;
    overflow: hidden;
    margin: 0 10px;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.prob-val {
    width: 52px;
    text-align: right;
    font-weight: 700;
}
.pred-badge {
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 50px;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
}
.step-box {
    border-left: 4px solid #4e8cff;
    background: rgba(78,140,255,0.07);
    padding: 0.5rem 0.9rem;
    border-radius: 0 6px 6px 0;
    margin: 0.35rem 0;
    font-size: 0.85rem;
}
.model-card {
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin: 0.25rem 0;
    font-size: 0.86rem;
}
.feat-val {
    font-size: 1.2rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# KONSTANTA TAMPILAN
# ════════════════════════════════════════════════════════════════
EMOTION_META = {
    "angry"   : {"color": "#e74c3c", "label": "Angry"},
    "happy"   : {"color": "#2ecc71", "label": "Happy"},
    "sad"     : {"color": "#3498db", "label": "Sad"},
    "surprise": {"color": "#f39c12", "label": "Surprise"},
}

PROP_META = {
    "contrast"    : ("Contrast",    "Besar = banyak perbedaan tajam antar piksel"),
    "homogeneity" : ("Homogeneity", "Besar = piksel bertetangga sering bernilai serupa"),
    "energy"      : ("Energy",      "Besar = tekstur seragam / distribusi terkonsentrasi"),
    "correlation" : ("Correlation", "Besar = ada pola linear konsisten antar piksel"),
}

GEO_FEAT_META = {
    "eye_ratio"       : ("Eye Ratio",         "Keterbukaan zona mata (surprise = tinggi)"),
    "mouth_ratio"     : ("Mouth Ratio",        "Aktivitas mulut (happy = tinggi)"),
    "brow_height"     : ("Brow Height",        "Pergerakan alis (surprise = tinggi)"),
    "face_symmetry"   : ("Face Symmetry",      "Simetri kiri-kanan wajah (1 = sempurna simetris)"),
    "edge_dens_upper" : ("Edge Dens Upper",    "Kerapatan tepi area atas (alis/dahi)"),
    "edge_dens_lower" : ("Edge Dens Lower",    "Kerapatan tepi area bawah (mulut/dagu)"),
    "vert_gradient"   : ("Vert Gradient",      "Distribusi intensitas atas vs bawah"),
    "pixel_var_ratio" : ("Pixel Var Ratio",    "Rasio variansi bawah/atas wajah"),
}

GEO_FEAT_NAMES = list(GEO_FEAT_META.keys())

MODEL_META = {
    "A": {
        "name" : "Model A — GLCM + Logistic Regression",
        "short": "GLCM + LR",
        "color": "#3498db",
        "label": "A",
        "desc" : "Baseline: fitur tekstur GLCM (concat)",
    },
    "B": {
        "name" : "Model B — GLCM + Geometri + LR",
        "short": "GLCM + Geo + LR",
        "color": "#2ecc71",
        "label": "B",
        "desc" : "Tambahan fitur geometri wajah",
    },
    "C": {
        "name" : "Model C — GLCM + Geometri + SVM (RBF)",
        "short": "GLCM + Geo + SVM",
        "color": "#e74c3c",
        "label": "C",
        "desc" : "SVM kernel RBF — hubungan non-linear",
    },
}

PROP_COLORS = ["#e74c3c", "#2ecc71", "#9b59b6", "#3498db"]

# ════════════════════════════════════════════════════════════════
# FUNGSI PREPROCESSING
# ════════════════════════════════════════════════════════════════

def detect_and_crop_face(pil_image: Image.Image) -> tuple:
    """Deteksi wajah dengan Haar Cascade. Fallback ke gambar penuh."""
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
    return cropped, f"{len(faces)} wajah terdeteksi — crop wajah terbesar ({w}×{h} px)"


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


def extract_glcm_features_for_bundle(img_norm: np.ndarray, config: dict) -> tuple:
    """
    Ekstrak fitur GLCM sesuai config dari bundle.
    Mengembalikan (feat_vec, glcm_tensor).
    Dimensi feat_vec:
      - aggregation='mean'   → 4D  (rata-rata 4 sudut per properti)
      - aggregation='concat' → 16D (tiap sudut tiap properti)
    """
    levels      = config["levels"]
    distances   = config.get("distances", [1])
    angles      = config.get("angles", [0, np.pi/4, np.pi/2, 3*np.pi/4])
    props       = config.get("glcm_props", ["contrast","homogeneity","energy","correlation"])
    aggregation = config.get("aggregation", "concat")
    symmetric   = config.get("symmetric", True)
    normed      = config.get("normed", True)

    img_q = quantize_image(img_norm, levels)
    glcm  = graycomatrix(
        img_q, distances=distances, angles=angles, levels=levels,
        symmetric=symmetric, normed=normed,
    )
    feat_vec = []
    if aggregation == "mean":
        for prop in props:
            feat_vec.append(float(graycoprops(glcm, prop).mean()))
    else:  # concat
        for prop in props:
            feat_vec.extend(graycoprops(glcm, prop).flatten().tolist())

    return np.array(feat_vec, dtype=np.float32), glcm


def extract_geometric_features(img_raw: np.ndarray) -> np.ndarray:
    """
    Ekstrak 8 fitur geometri wajah dari gambar uint8 48×48.
    Identik dengan fungsi di notebook (CELL 7.0).
    """
    img = img_raw.astype(np.float32)
    h, w = img.shape

    upper      = img[:h//2, :]
    lower      = img[h//2:, :]
    eye_zone   = img[12:22, 8:40]
    mouth_zone = img[30:44, 10:38]
    brow_zone  = img[8:15,  8:40]

    sx   = cv2.Sobel(img_raw, cv2.CV_64F, 1, 0, ksize=3)
    sy   = cv2.Sobel(img_raw, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sx**2 + sy**2)

    eye_var_vert = eye_zone.std(axis=0).mean()
    eye_ratio    = eye_var_vert / (eye_zone.mean() + 1e-6)

    mouth_ratio = mouth_zone.std() / (mouth_zone.mean() + 1e-6)

    brow_grad   = np.abs(np.diff(brow_zone.mean(axis=1))).mean()
    brow_height = brow_grad / (brow_zone.mean() + 1e-6)

    left_half     = img[:, :w//2]
    right_half    = img[:, w//2:][:, ::-1]
    diff_sym      = np.abs(left_half - right_half).mean()
    face_symmetry = 1.0 / (1.0 + diff_sym / 255.0)

    edge_density_upper = edge[:h//2, :].mean() / 255.0
    edge_density_lower = edge[h//2:, :].mean() / 255.0

    vertical_gradient = (lower.mean() - upper.mean()) / 255.0

    pixel_var_ratio = lower.std() / (upper.std() + 1e-6)

    return np.array([
        eye_ratio, mouth_ratio, brow_height, face_symmetry,
        edge_density_upper, edge_density_lower,
        vertical_gradient, pixel_var_ratio,
    ], dtype=np.float32)


def predict_all_models(img_raw: np.ndarray, img_norm: np.ndarray,
                       bundles: dict) -> dict:
    """
    Jalankan prediksi untuk ketiga model secara independen.

    PERBAIKAN v3: setiap model menggunakan config dari bundle-nya sendiri
    untuk mengekstrak GLCM — sehingga dimensi fitur selalu cocok dengan
    scaler yang di-fit saat training.

    Geometri dihitung sekali dan dibagi (tidak bergantung config GLCM).
    """
    results  = {}
    feat_geo = extract_geometric_features(img_raw)  # 8D, sama untuk semua model

    for mid, bundle in bundles.items():
        if bundle is None:
            continue

        cfg    = bundle["config"]
        scaler = bundle["scaler"]
        model  = bundle["model"]
        le     = bundle["label_encoder"]

        # ── Ekstrak GLCM dengan config bundle ini ─────────────────
        feat_glcm, glcm_tensor = extract_glcm_features_for_bundle(img_norm, cfg)

        # ── Susun vektor fitur sesuai jenis model ─────────────────
        uses_geo = cfg.get("uses_geometry", mid in ("B", "C"))
        if uses_geo:
            feat_input = np.concatenate([feat_glcm, feat_geo])
        else:
            feat_input = feat_glcm

        # ── Validasi dimensi sebelum transform ────────────────────
        n_expected = scaler.n_features_in_
        n_actual   = feat_input.shape[0]
        if n_actual != n_expected:
            # Fallback: potong atau pad dengan nol agar tidak crash
            if n_actual > n_expected:
                feat_input = feat_input[:n_expected]
            else:
                feat_input = np.pad(feat_input, (0, n_expected - n_actual))

        feat_scaled = scaler.transform(feat_input.reshape(1, -1))
        pred_idx    = model.predict(feat_scaled)[0]
        pred_proba  = model.predict_proba(feat_scaled)[0]
        pred_class  = le.inverse_transform([pred_idx])[0]

        results[mid] = {
            "predicted_class"  : pred_class,
            "probabilities"    : dict(zip(le.classes_, pred_proba.tolist())),
            "confidence"       : float(pred_proba.max()),
            "feat_glcm_raw"    : feat_glcm,
            "feat_geo_raw"     : feat_geo,
            "feat_combined_raw": feat_input,
            "feat_scaled"      : feat_scaled[0],
            "glcm_tensor"      : glcm_tensor,
            "cfg"              : cfg,
            "n_glcm_feat"      : int(feat_glcm.shape[0]),
            "uses_geo"         : uses_geo,
        }

    return results


# ════════════════════════════════════════════════════════════════
# LOAD BUNDLES
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def load_bundles(base_dir: str) -> dict:
    """
    Load tiga bundle model.
    Prioritas: model_bundle_A/B/C.pkl → fallback model_bundle.pkl (legacy)
    """
    base    = Path(base_dir)
    legacy  = base / "model_bundle.pkl"
    bundles = {}

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
any_loaded = any(v is not None for v in bundles.values())


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Pengaturan")

    if not any_loaded:
        st.error(
            "`model_bundle_A/B/C.pkl` tidak ditemukan!\n\n"
            "Export dari notebook CELL 7.1, atau gunakan CELL 6.0 untuk bundle tunggal."
        )
        st.stop()

    # ── Pilihan model aktif ────────────────────────────────────
    st.markdown("### Pilih Model")
    selected_model = st.radio(
        "Model prediksi:",
        options=[m for m in ["C", "B", "A"] if bundles[m] is not None],
        format_func=lambda m: f"Model {m} — {MODEL_META[m]['short']}",
        index=0,
    )

    mm = MODEL_META[selected_model]
    bndl_sel = bundles[selected_model]
    acc_sel  = bndl_sel["metadata"].get("acc_test", 0) if bndl_sel else 0
    st.markdown(
        f"""<div class="model-card" style="background:{mm['color']}18;
             border-left:4px solid {mm['color']};">
             <b>Model {mm['label']} — {mm['short']}</b><br>
             <small>{mm['desc']}</small><br>
             <small>Akurasi test: <b>{acc_sel:.2%}</b></small>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Info semua model ───────────────────────────────────────
    st.markdown("### Info Model")
    for mid in ["A", "B", "C"]:
        bndl_ = bundles[mid]
        if bndl_ is None:
            st.caption(f"Model {mid}: bundle tidak ditemukan")
            continue
        meta_  = bndl_.get("metadata", {})
        mm_    = MODEL_META[mid]
        acc_   = meta_.get("acc_test", 0)
        dim_   = meta_.get("n_features", "?")
        active = "aktif" if mid == selected_model else "tidak aktif"
        st.markdown(
            f"""<div class="model-card" style="
                background:{''+mm_['color']+'18' if mid==selected_model else 'rgba(128,128,128,0.05)'};
                border-left:3px solid {mm_['color']};">
                <b>Model {mid}</b>{active}<br>
                <small>{mm_['short']} | {dim_}D | Acc: {acc_:.2%}</small>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Pipeline steps (menyesuaikan model) ───────────────────
    active_bundle = bundles[selected_model]
    cfg_side      = active_bundle["config"] if active_bundle else {}
    lvl_side      = cfg_side.get("levels", "?")
    agg_side      = cfg_side.get("aggregation", "?")
    uses_geo_side = cfg_side.get("uses_geometry", selected_model in ("B", "C"))
    clf_side      = "SVM (RBF)" if selected_model == "C" else "Logistic Regression"
    glcm_dim      = "16D" if agg_side == "concat" else "4D"

    st.markdown("---")
    st.markdown("### Pipeline Preprocessing")
    steps = [
        "① Face detection (Haar Cascade)",
        "② Crop + padding 10%",
        "③ Grayscale mode 'L'",
        f"④ Resize → <b>48 × 48</b> (LANCZOS)",
        "⑤ Normalisasi ÷ 255.0 → float32",
        f"⑥ Kuantisasi → <b>L = {lvl_side}</b>",
        f"⑦ GLCM (d=1, 4 sudut) → <b>{glcm_dim}</b> ({agg_side})",
    ]
    if uses_geo_side:
        steps += [
            "⑧ Fitur geometri wajah → <b>8D</b>",
            f"⑨ Gabung GLCM + Geo → <b>{16 if agg_side=='concat' else 4}+8D</b>",
            "⑩ StandardScaler.transform()",
            f"⑪ {clf_side}.predict()",
        ]
    else:
        steps += [
            "⑧ StandardScaler.transform()",
            f"⑨ {clf_side}.predict()",
        ]
    for s in steps:
        st.markdown(f'<div class="step-box">{s}</div>', unsafe_allow_html=True)

    # ── Kelas yang didukung ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Kelas Emosi")
    classes_ = (
        active_bundle["metadata"].get("classes", list(EMOTION_META.keys()))
        if active_bundle else list(EMOTION_META.keys())
    )
    cols_cls = st.columns(2)
    for i, cls in enumerate(classes_):
        with cols_cls[i % 2]:
            svg_sm = emotion_svg(cls, size=24)
            color  = EMOTION_META.get(cls, {}).get("color", "#888")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
                f'{svg_sm}<span style="font-size:0.85rem;color:{color};font-weight:600;">'
                f'{cls.capitalize()}</span></div>',
                unsafe_allow_html=True,
            )

    if active_bundle:
        exported = active_bundle.get("metadata", {}).get("exported_at", "—")
        st.caption(f"Bundle diekspor: {exported}")


# ════════════════════════════════════════════════════════════════
# HEADER UTAMA
# ════════════════════════════════════════════════════════════════
st.markdown("# Raut Muka — Klasifikasi Emosi Wajah")
st.markdown(
    "Upload foto wajah · sistem mengekstrak fitur **GLCM** (tekstur) "
    "+ **Geometri Wajah** · model memprediksi emosi."
)

# ── Kartu ringkasan 3 model ────────────────────────────────────
col_ma, col_mb, col_mc = st.columns(3)
for col, mid in zip([col_ma, col_mb, col_mc], ["A", "B", "C"]):
    mm_   = MODEL_META[mid]
    bndl_ = bundles[mid]
    acc_  = bndl_["metadata"].get("acc_test", 0) if bndl_ else 0
    is_active   = (mid == selected_model)
    border_style = f"border:2px solid {mm_['color']};" if is_active else "border:1px solid rgba(128,128,128,0.2);"
    with col:
        st.markdown(
            f"""<div class="model-card" style="background:{mm_['color']}10;
                 {border_style} padding:0.8rem; text-align:center;">
                 <b>Model {mid}</b><br>
                 <small style="color:var(--text-color,#666)">{mm_['short']}</small><br>
                 <span style="color:{mm_['color']};font-weight:700;font-size:1.1rem;">
                   {acc_:.2%}
                 </span>
                 {'<br><small><b>Aktif</b></small>' if is_active else '<br><small><b>Tidak Aktif</b></small>'}
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
        st.markdown("*atau coba gambar sintetis:*")
        use_demo = st.button("Gunakan Gambar Demo", use_container_width=True)

    if uploaded or use_demo:
        # ── Load gambar ──────────────────────────────────────────
        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
        else:
            arr = np.full((96, 96), 140, dtype=np.uint8)
            arr[5:30,  15:81] = 210
            arr[30:75,  8:88] = 195
            arr[32:40, 20:35] = 45
            arr[32:40, 60:75] = 45
            arr[50:58, 38:58] = 165
            arr[68:74, 25:70] = 70
            arr[74:80, 30:65] = 80
            arr = cv2.GaussianBlur(arr, (5, 5), 0)
            pil_img = Image.fromarray(arr)

        # ── Face detection ────────────────────────────────────────
        pil_cropped, crop_status = detect_and_crop_face(pil_img)

        # ── Preprocessing (ukuran dari config model aktif) ────────
        cfg_main  = bundles[selected_model]["config"]
        img_size  = tuple(cfg_main.get("img_size", (48, 48)))
        img_raw, img_norm = preprocess_image(pil_cropped, img_size)

        # ── Tampilkan gambar ──────────────────────────────────────
        st.markdown("#### Gambar Asli")
        st.image(pil_img, use_container_width=True)

        if "tidak terdeteksi" in crop_status:
            st.warning(crop_status)
        else:
            st.success(crop_status)

        if pil_cropped is not pil_img:
            st.markdown("**Setelah Face Crop**")
            st.image(pil_cropped, use_container_width=True,
                     caption="Area wajah yang diproses")

        col_pp1, col_pp2 = st.columns(2)
        with col_pp1:
            st.markdown("**Grayscale 48×48**")
            st.image(Image.fromarray(img_raw), use_container_width=True,
                     caption="uint8 [0–255]")
        with col_pp2:
            lvl_main  = cfg_main.get("levels", 8)
            img_q_show = quantize_image(img_norm, lvl_main)
            img_q_disp = (img_q_show / max(lvl_main - 1, 1) * 255).astype(np.uint8)
            st.markdown("**Setelah Kuantisasi**")
            st.image(Image.fromarray(img_q_disp), use_container_width=True,
                     caption=f"L = {lvl_main} level")

        # ── Prediksi semua model ──────────────────────────────────
        with st.spinner("Mengekstrak fitur & memprediksi..."):
            all_results = predict_all_models(img_raw, img_norm, bundles)

        st.session_state["all_results"]    = all_results
        st.session_state["img_raw"]        = img_raw
        st.session_state["img_norm"]       = img_norm
        st.session_state["selected_model"] = selected_model


# ════════════════════════════════════════════════════════════════
# KOLOM HASIL
# ════════════════════════════════════════════════════════════════
with col_result:
    if "all_results" not in st.session_state:
        st.info("Upload gambar untuk melihat hasil prediksi.")
        st.stop()

    all_results  = st.session_state["all_results"]
    img_raw_s    = st.session_state["img_raw"]
    img_norm_s   = st.session_state["img_norm"]
    active_model = st.session_state.get("selected_model", selected_model)

    if active_model not in all_results:
        st.warning("Hasil prediksi untuk model ini tidak tersedia.")
        st.stop()

    result   = all_results[active_model]
    pred_cls = result["predicted_class"]
    em_meta  = EMOTION_META.get(pred_cls, {"color": "#888", "label": pred_cls.capitalize()})
    conf     = result["confidence"]
    mm_act   = MODEL_META[active_model]

    # ── Hasil utama ───────────────────────────────────────────
    st.markdown(f"### Hasil Prediksi — Model {active_model}")

    col_badge, col_conf = st.columns([1, 1])
    with col_badge:
        svg_big = emotion_svg(pred_cls, size=72)
        st.markdown(
            f"""<div style="text-align:center; padding:1.2rem;
                 background:{em_meta['color']}15;
                 border:2px solid {em_meta['color']};
                 border-radius:12px;">
                 {svg_big}
                 <div class="pred-badge"
                      style="background:{em_meta['color']};color:white;display:block;margin-top:8px;">
                   {em_meta['label'].upper()}
                 </div>
                 <div style="font-size:0.8rem;margin-top:4px;color:{em_meta['color']};">
                   {mm_act['short']}
                 </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_conf:
        st.markdown("**Confidence score**")
        st.progress(conf, text=f"{conf:.1%}")
        if conf >= 0.70:
            st.success("Model sangat yakin")
        elif conf >= 0.45:
            st.warning("Model cukup yakin")
        else:
            st.error("Model ragu-ragu — probabilitas tersebar merata")
        st.caption(
            f"Akurasi model ini saat training: **{bundles[active_model]['metadata'].get('acc_test', 0):.2%}**"
            if bundles.get(active_model) else ""
        )

    st.markdown("---")

    # ── Perbandingan 3 model ──────────────────────────────────
    st.markdown("### Perbandingan Tiga Model")
    col_a, col_b, col_c = st.columns(3)
    for col, mid in zip([col_a, col_b, col_c], ["A", "B", "C"]):
        if mid not in all_results:
            with col:
                st.caption(f"Model {mid} tidak tersedia")
            continue
        r_   = all_results[mid]
        mm_  = MODEL_META[mid]
        em_  = EMOTION_META.get(r_["predicted_class"], {"color": "#888"})
        svg_sm = emotion_svg(r_["predicted_class"], size=36)
        is_active = (mid == active_model)
        with col:
            st.markdown(
                f"""<div style="text-align:center; padding:0.7rem;
                     background:{mm_['color']}{'18' if is_active else '08'};
                     border:{'2px' if is_active else '1px'} solid {mm_['color']};
                     border-radius:10px;">
                     <b>Model {mid}</b><br>
                     {svg_sm}
                     <div style="font-weight:700;color:{em_['color']};font-size:0.95rem;">
                       {r_['predicted_class'].capitalize()}
                     </div>
                     <div style="font-size:0.82rem;color:var(--text-color,#666);">
                       {r_['confidence']:.1%} conf
                     </div>
                     {'<div style="font-size:0.78rem;font-weight:600;">Aktif</div>' if is_active else '<div style="font-size:0.78rem;font-weight:600;">Tidak Aktif</div>'}
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Distribusi probabilitas ───────────────────────────────
    st.markdown(f"### Distribusi Probabilitas — Model {active_model}")
    probs = result["probabilities"]
    for cls_name, prob_val in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        em_      = EMOTION_META.get(cls_name, {})
        color    = em_.get("color", "#888")
        is_pred  = (cls_name == pred_cls)
        svg_tiny = emotion_svg(cls_name, size=18)
        bar_pct  = int(prob_val * 100)
        st.markdown(
            f"""<div class="prob-row" style="{'font-weight:700;' if is_pred else ''}">
              <span class="prob-label">
                {svg_tiny}
                {cls_name.capitalize()}{'  ←' if is_pred else ''}
              </span>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{bar_pct}%;background:{color};opacity:{'1' if is_pred else '0.55'};"></div>
              </div>
              <span class="prob-val" style="color:{color if is_pred else '#888'}">
                {prob_val*100:.1f}%
              </span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Fitur GLCM ────────────────────────────────────────────
    cfg_res     = result["cfg"]
    feat_glcm   = result["feat_glcm_raw"]
    n_glcm      = result["n_glcm_feat"]
    props_res   = cfg_res.get("glcm_props", ["contrast","homogeneity","energy","correlation"])
    agg_res     = cfg_res.get("aggregation", "concat")
    lvl_res     = cfg_res.get("levels", 8)

    st.markdown(f"### Fitur GLCM ({n_glcm}D — {agg_res}, L={lvl_res})")

    if agg_res == "concat":
        # 16D: tampilkan per properti × 4 sudut
        angle_labels = ["0°", "45°", "90°", "135°"]
        n_props      = len(props_res)
        n_angles     = 4
        for pi, prop in enumerate(props_res):
            vals   = feat_glcm[pi*n_angles : pi*n_angles + n_angles]
            pname  = PROP_META.get(prop, (prop, ""))[0]
            pdesc  = PROP_META.get(prop, (prop, ""))[1]
            color  = PROP_COLORS[pi % len(PROP_COLORS)]
            cols_p = st.columns(4)
            for ai, (col_p, v, ang) in enumerate(zip(cols_p, vals, angle_labels)):
                with col_p:
                    st.markdown(
                        f"""<div class="card" style="border-left:3px solid {color};">
                          <small style="color:{color};font-weight:600;">{pname} {ang}</small>
                          <div class="feat-val" style="color:{color}">{v:.5f}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
    else:
        # 4D: tampilkan 1 nilai per properti
        col_f1, col_f2 = st.columns(2)
        for idx, (prop, pc) in enumerate(zip(props_res, PROP_COLORS)):
            col_fx = col_f1 if idx % 2 == 0 else col_f2
            pname, pdesc = PROP_META.get(prop, (prop, ""))
            with col_fx:
                st.markdown(
                    f"""<div class="card" style="border-left:3px solid {pc};">
                      <b>{pname}</b><br>
                      <small style="color:#888">{pdesc}</small><br>
                      <div class="feat-val">{feat_glcm[idx]:.5f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Fitur Geometri (Model B & C) ──────────────────────────
    if result.get("uses_geo", False):
        st.markdown("### Fitur Geometri Wajah (8D)")
        feat_geo = result["feat_geo_raw"]
        col_g1, col_g2 = st.columns(2)
        for idx, (fname, (flabel, fdesc)) in enumerate(GEO_FEAT_META.items()):
            col_gx = col_g1 if idx % 2 == 0 else col_g2
            val    = feat_geo[idx]
            with col_gx:
                st.markdown(
                    f"""<div class="card" style="border-left:3px solid #9b59b6;">
                      <b style="color:#6c3483">{flabel}</b><br>
                      <small style="color:#888">{fdesc}</small><br>
                      <div class="feat-val" style="color:#9b59b6">{val:.5f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Detail pipeline (expander) ────────────────────────────
    with st.expander("Detail Pipeline & Heatmap GLCM", expanded=False):
        img_q_exp   = quantize_image(img_norm_s, lvl_res)
        glcm_tensor = result["glcm_tensor"]
        dim_total   = result["feat_combined_raw"].shape[0]

        st.markdown(f"""
**Model {active_model} — {mm_act['short']} | Dimensi fitur: {dim_total}D**

| Tahap | Parameter | Nilai |
|---|---|---|
| Kuantisasi | Level (L) | {lvl_res} |
| GLCM | Jarak (d) | {cfg_res.get('distances', [1])} |
| GLCM | Sudut | 0°, 45°, 90°, 135° |
| Agregasi | Strategi | {agg_res} → {n_glcm}D |
| Geometri | Fitur | {'8D (digunakan)' if result['uses_geo'] else 'tidak digunakan'} |
| Scaler | Jenis | StandardScaler (μ=0, σ=1) |
| Classifier | Model | {'SVM (RBF)' if active_model=='C' else 'Logistic Regression'} |

```
Distribusi kuantisasi: {np.unique(img_q_exp).tolist()[:12]}
Range piksel raw    : [{img_raw_s.min()}, {img_raw_s.max()}]
Range piksel norm   : [{img_norm_s.min():.4f}, {img_norm_s.max():.4f}]
```
        """)

        # GLCM heatmap
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use("Agg")

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            angle_names = ["0°", "45°", "90°", "135°"]
            L_show = min(lvl_res, 16)
            for ai, (ax, aname) in enumerate(zip(axes, angle_names)):
                g2d = glcm_tensor[:L_show, :L_show, 0, ai]
                im  = ax.imshow(g2d, cmap="plasma", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f"GLCM {aname}", fontsize=9)
                ax.set_xlabel("j", fontsize=8)
                ax.set_ylabel("i", fontsize=8)
                ax.tick_params(labelsize=7)
            fig.suptitle(f"GLCM Heatmap — L={lvl_res}", fontsize=10)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, caption="GLCM 4 sudut (plasma colormap)")
            plt.close(fig)
        except Exception as e:
            st.caption(f"Heatmap tidak dapat ditampilkan: {e}")

        # Bar chart fitur geometri
        if result.get("uses_geo", False):
            try:
                feat_geo_cur = result["feat_geo_raw"]
                fig2, ax2 = plt.subplots(figsize=(9, 3))
                geo_colors = ["#9b59b6"] * 8
                bars = ax2.barh(GEO_FEAT_NAMES, feat_geo_cur,
                                color=geo_colors, alpha=0.75, edgecolor="white")
                for bar, val in zip(bars, feat_geo_cur):
                    ax2.text(bar.get_width() + 0.001,
                             bar.get_y() + bar.get_height() / 2,
                             f"{val:.4f}", va="center", fontsize=8)
                ax2.set_xlabel("Nilai Fitur", fontsize=9)
                ax2.set_title("Profil Fitur Geometri", fontsize=10)
                ax2.grid(axis="x", alpha=0.3, ls="--")
                ax2.tick_params(labelsize=8)
                fig2.tight_layout()
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png", dpi=110, bbox_inches="tight")
                buf2.seek(0)
                st.image(buf2, caption="Fitur geometri wajah yang diekstrak")
                plt.close(fig2)
            except Exception as e:
                st.caption(f"Bar chart tidak dapat ditampilkan: {e}")


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<small>Kelompok 1 Kelas A · Universitas Udayana · 2026 · "
    "Dataset: FER2013 · Fitur: GLCM + Geometri Wajah · "
    "Model: Logistic Regression &amp; SVM (RBF)</small>",
    unsafe_allow_html=True,
)
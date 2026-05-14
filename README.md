# Klasifikasi Emosi Wajah - Kelompok 1 Kelas A

Aplikasi web untuk mengklasifikasikan emosi dari foto wajah  
menggunakan **GLCM (Gray-Level Co-occurrence Matrix)** + **Logistic Regression**.

---

## Struktur Folder

```
streamlit_app/
├── app.py               ← Aplikasi Streamlit utama
├── streamlit_app        ← Model + scaler + config (dari notebook)
├── requirements.txt     ← Dependensi Python
└── README.md            ← File ini
```

---

## Cara Menjalankan

### Langkah 1 — Generate `model_bundle.pkl` dari Notebook

Buka notebook `klasifikasi_emosi_GLCM_v4_final.ipynb`, jalankan **semua cell** mulai dari BAGIAN 0 hingga **CELL 6.0** (BAGIAN 6 — Export Model).

File `model_bundle.pkl` akan muncul di folder `streamlit_app/`.

### Langkah 2 — Install Dependensi

```bash
pip install -r requirements.txt
```

### Langkah 3 — Jalankan Aplikasi

```bash
cd streamlit_app
streamlit run app.py
```

Buka browser di: `http://localhost:8501`

---

## Pipeline Preprocessing (WAJIB Sama dengan Training)

Preprocessing di app ini **identik** dengan yang dipakai saat training:

```
Gambar Upload (format apa pun)
        ↓
① Konversi Grayscale ('L')          ← PIL .convert('L')
        ↓
② Resize → 48 × 48 px (LANCZOS)     ← PIL .resize((48,48), LANCZOS)
        ↓
③ Array uint8 [0, 255]
        ↓
④ Normalisasi ÷ 255.0               ← float32 [0.0, 1.0]
        ↓
⑤ Kuantisasi ⌊pixel × L⌋           ← L dari model_bundle (misal L=16)
        ↓
⑥ graycomatrix(d=1, 4 sudut,        ← param dari model_bundle
               symmetric, normed)
        ↓
⑦ Ekstrak 4 properti GLCM
        ↓
⑧ StandardScaler.transform()        ← scaler dari model_bundle
        ↓
⑨ LogisticRegression.predict()      ← model dari model_bundle
        ↓
Prediksi Kelas + Probabilitas
```

> **Penting:** Selalu gunakan `scaler.transform()` (bukan `fit_transform()`)  
> di app. Scaler harus yang **sama** dengan saat training agar hasilnya valid.

---

## Isi `model_bundle.pkl`

```python
{
    'model'        : LogisticRegression,  # model yang sudah ditraining
    'scaler'       : StandardScaler,      # scaler yang sudah di-fit
    'label_encoder': LabelEncoder,        # angka → nama kelas
    'config': {
        'img_size'   : (48, 48),
        'levels'     : 16,                # level kuantisasi GLCM
        'distances'  : [1],
        'angles'     : [0, π/4, π/2, 3π/4],
        'aggregation': 'mean',            # atau 'concat'
        'glcm_props' : ['contrast', 'homogeneity', 'energy', 'correlation'],
        'symmetric'  : True,
        'normed'     : True,
    },
    'metadata': {
        'scenario'   : '16-mean',
        'acc_test'   : 0.4231,
        'acc_train'  : 0.4517,
        'classes'    : ['angry', 'happy', 'sad', 'surprise'],
        'n_features' : 4,
        'exported_at': '2025-...',
    }
}
```

---

## Deploy ke Streamlit Cloud

1. Push folder `streamlit_app/` ke GitHub (termasuk `model_bundle.pkl`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Hubungkan repo → set **Main file path**: `streamlit_app/app.py`
4. Deploy!

> **Catatan:** Jika `model_bundle.pkl` terlalu besar (>100 MB), gunakan Git LFS  
> atau simpan di Google Drive dan load dengan `gdown`.

---

## Troubleshooting

| Error | Penyebab | Solusi |
|:------|:---------|:-------|
| `model_bundle.pkl not found` | File belum digenerate | Jalankan CELL 6.0 di notebook |
| `ValueError: X has N features` | Scaler tidak cocok | Pastikan pakai bundle dari notebook yang sama |
| Prediksi selalu salah | Preprocessing berbeda | Cek level kuantisasi dan agregasi di bundle |
| `ModuleNotFoundError: skimage` | scikit-image belum install | `pip install scikit-image` |

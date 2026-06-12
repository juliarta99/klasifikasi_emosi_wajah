# Raut Muka App: Klasifikasi Emosi Wajah - Kelompok 1 Kelas A

Aplikasi web untuk mengklasifikasikan emosi dari foto wajah manusia (*angry, happy, sad, surprise*).

Sistem ini menggunakan **Model A (GLCM + Logistic Regression)** sebagai model utama inferensi. Selain itu, proyek ini juga mencakup eksplorasi eksperimen tambahan yang membandingkan performa model menggunakan ekstraksi fitur **Geometri** dan algoritma **Support Vector Machine (SVM)**.

**Demo Aplikasi:** [https://rautmuka.streamlit.app/](https://rautmuka.streamlit.app/)

---

## Struktur Folder

```text
├── dataset/                  ← Folder dataset gambar
├── streamlit_app/            ← Folder khusus aplikasi web Streamlit
│   ├── app.py                ← Aplikasi Streamlit utama (berisi UI & pipeline inferensi)
│   ├── model_bundle_A.pkl    ← File bundle model eksperimen A
│   ├── model_bundle_B.pkl    ← File bundle model eksperimen B
│   ├── model_bundle_C.pkl    ← File bundle model eksperimen C
│   ├── model_bundle.pkl      ← Model utama (GLCM+LR), scaler, label encoder, & config (hasil export)
│   └── requirements.txt      ← Daftar dependensi Python untuk menjalankan Streamlit
├── main.ipynb                ← Notebook eksperimen utama (ekstraksi fitur, training, & evaluasi)
├── README.md                 ← File dokumentasi ini
└── requirements.txt          ← Daftar dependensi Python untuk keseluruhan proyek (termasuk notebook)

```

*(Catatan: File `model_bundle.pkl` beserta varian model lainnya harus berada di dalam direktori `streamlit_app` yang sama dengan `app.py` agar aplikasi dapat memuat model dengan benar).*

---

## Cara Menjalankan Lokal

Pastikan Python sudah terinstal, lalu jalankan perintah berikut di terminal Anda:

**1. Install dependensi**
(Disarankan untuk masuk ke folder `streamlit_app` terlebih dahulu jika hanya ingin menjalankan aplikasinya)

```bash
pip install -r streamlit_app/requirements.txt

```

**2. Jalankan aplikasi**

```bash
cd streamlit_app
streamlit run app.py

```

Aplikasi akan otomatis terbuka di browser melalui alamat: `http://localhost:8501`

---

## Pipeline Inferensi (Model Utama)

Pipeline yang berjalan pada aplikasi web dirancang ringkas dan identik dengan proses *training* Model A:

1. **Pre-processing:** Menerima unggahan gambar → Konversi ke Grayscale (`L`) → Resize 48×48 px (LANCZOS) → Normalisasi rentang piksel [0,1].
2. **Ekstraksi Fitur (GLCM):** Kuantisasi level piksel → Ekstraksi GLCM ($d=1$, 4 sudut, *symmetric, normed*) → Mendapatkan nilai *contrast, homogeneity, energy, correlation* yang digabung (*concat*).
3. **Prediksi:** Transformasi fitur menggunakan `StandardScaler` (dari *bundle*, tanpa *fitting* ulang) → Klasifikasi probabilitas menggunakan **Logistic Regression** → Menghasilkan output label emosi beserta skor *confidence*.

*(Eksperimen menggunakan ekstraksi fitur Geometri dan pemodelan SVM tidak di-deploy ke Streamlit, namun dieksplorasi secara terpisah di dalam notebook `main.ipynb`).*

---

## Troubleshooting

| Error / Kendala | Kemungkinan Penyebab | Solusi |
| --- | --- | --- |
| `model_bundle.pkl not found` | File model belum ada atau salah lokasi | Pastikan file model hasil *export* berada dalam satu folder yang sama dengan `app.py` di dalam folder `streamlit_app`. |
| `ValueError: X has N features` | Terdapat ketidakcocokan *scaler* atau jumlah fitur | Gunakan *bundle* `.pkl` yang dihasilkan dari *notebook* yang sama dengan konfigurasi pipeline (misal: skenario 8-concat, 16D). |
| `ModuleNotFoundError: skimage` | Library `scikit-image` belum terinstal | Jalankan `pip install scikit-image` di *environment* Anda. |
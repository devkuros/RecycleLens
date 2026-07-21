# 📑 SDLC & Technical Requirement Document
## Proyek: Sistem Otomatisasi Klasifikasi Citra Sampah (Automated Waste Classification System)

**Tanggal:** Juli 2026  
**Status Dokumen:** Approved for Development (Architecture Locked)  
**Pendekatan SDLC:** AI/ML Lifecycle Standard (Iterative Waterfall & Agile Hybrid)  

---

## 📍 FASE 1: Analisis Kebutuhan & Perencanaan (Requirement Analysis & Planning)

### 1.1 Latar Belakang & Analisis Masalah
Pengelolaan sampah konvensional secara manual menghadapi kendala keterbatasan Sumber Daya Manusia (SDM), tingginya volume sampah harian, dan tingginya risiko kesalahan identifikasi (*human error*). 

Dengan mengintegrasikan teknologi *Computer Vision* dan *Machine Learning*, proyek ini bertujuan untuk **mengotomatisasi proses pemilahan sampah secara instan dan akurat** berdasarkan citra digital guna mempercepat proses daur ulang, menekan biaya operasional, dan mendukung implementasi ekonomi sirkular.

### 1.2 Tujuan Produk (Product Objectives)
* **Akurasi Tinggi:** Membangun model *deep learning* yang mampu mengklasifikasikan objek sampah ke dalam 3 kategori utama secara presisi.
* **Otomatisasi Efisien:** Menggantikan proses pemilahan manual menjadi sistem berbasis *data-driven* yang konsisten.
* **Kesiapan Integrasi:** Menghasilkan prediksi akhir yang terstruktur dan sesuai format baku untuk dikonsumsi sistem manajemen pihak ketiga / panitia.

### 1.3 Batasan Ketat & Regulasi Proyek (Constraints & Guardrails)
> ⚠️ **Aturan Ketat (Strict Rules):**
> 1. **No Data Leakage:** Dilarang keras menggunakan informasi, pola, atau label apa pun dari Data Uji selama proses pembuatan dan pelatihan model.
> 2. **Pure Visual Approach:** Model hanya boleh mengambil keputusan berdasarkan informasi visual (*image features*) pada dataset resmi.
> 3. **Data Uji Hanya untuk Inferensi:** Data Uji (1.458 citra) hanya diakses pada tahap inferensi akhir untuk menghasilkan berkas prediksi.

---

## 🏗️ FASE 2: Desain Sistem & Arsitektur (System & Architecture Design)

### 2.1 Spesifikasi Dataset & Skema Labeling
Model wajib mengenali dan mengelompokkan gambar ke dalam kategori numerik berikut:

#### Skema Kategori Sampah
| Kode Numerik | Nama Kategori | Deskripsi & Contoh Objek |
| :---: | :--- | :--- |
| **0** | **Recyclable** | Sampah non-elektronik berpotensi daur ulang (botol plastik, kaleng, kertas, kardus, kaca). |
| **1** | **Electronic** | Limbah elektronik / *e-waste* baik berfungsi/rusak (HP, laptop, keyboard, mouse, charger, kabel). |
| **2** | **Organic** | Bahan hayati yang mudah terurai alami (daun, buah, sayuran, sisa makanan, ranting). |

#### Volume Data
* **Data Latih (Train):** 26.527 citra berlabel (tersimpan terpisah di subfolder sesuai kategori).
* **Data Uji (Test):** 1.458 citra tanpa label (nama file terurut sesuai `template.csv`).

### 2.2 Desain Arsitektur Deep Learning
* **Base Model:** **EfficientNet** (Varian: **EfficientNet-B3 / B4** atau **EfficientNetV2-S**).
* **Alasan Pemilihan:** Menggunakan teknik *compound scaling* yang secara seimbang menskalakan kedalaman (*depth*), lebar (*width*), dan resolusi (*resolution*) jaringan, sehingga menghasilkan akurasi tinggi pada ekstraksi fitur visual tekstur sampah tanpa memakan beban komputasi berlebih.
* **Model Output Layer:** Softmax Layer dengan 3 neuron output (Representasi Kelas 0, 1, 2).

### 2.3 Spesifikasi Tech Stack & Hardware

| Komponen | Teknologi / Spesifikasi |
| :--- | :--- |
| **Bahasa Pemrograman** | Python 3.10+ |
| **Environment** | Google Colab Pro / Kaggle Notebooks / Jupyter |
| **Core Framework** | PyTorch (v2.0+) |
| **Model Repository** | `timm` (Torch Image Models) |
| **Image Augmentation** | `Albumentations` (v1.4+) |
| **Computer Vision** | OpenCV, Pillow (PIL) |
| **Data & Validation** | Pandas, NumPy, Scikit-Learn (`StratifiedKFold`) |
| **Akselerator Hardware** | NVIDIA GPU (Min. 8 GB VRAM, Rec. 16 GB VRAM) |

---

## 💻 FASE 3: Pengembangan & Pengkodean (Development & Implementation)

Proses enkapsulasi alur kerja sistem dibagi menjadi 3 modul utama:

```text
[Input: Raw Images] ➡️ [Modul 1: Preprocessing & Augmentation]
                                    ⬇️
                         [Modul 2: Training Engine (EfficientNet)]
                                    ⬇️
                         [Modul 3: Inferensi & Exporter] ➡️ [Output: submission.csv]


### 3.1 Modul 1: Preprocessing & Data Augmentation Pipeline
* Resize citra seragam ke ukuran $224 \times 224$ atau $384 \times 384$ piksel.
* Penerapan teknik augmentasi via `Albumentations`: *Horizontal Flip, Random Rotation, ShiftScaleRotate, Random Brightness & Contrast Adjustment*.
* Normalisasi piksel berdasarkan statistik standar ImageNet ($\mu$ dan $\sigma$).

### 3.2 Modul 2: Training Engine & Fine-Tuning
* Implementasi *Transfer Learning* dengan bobot awal (*pre-trained weights*) ImageNet.
* Penggunaan teknik **Stratified 5-Fold Cross-Validation** untuk membagi 26.527 data latih secara presisi dan seimbang.
* Optimasi pelatihan menggunakan *Optimizer* AdamW, *Loss Function* Cross-Entropy, serta *Learning Rate Scheduler* (Cosine Annealing).

### 3.3 Modul 3: Inferensi & Prediction Exporter
* Membaca 1.458 Data Uji terurut sesuai berkas `template.csv`.
* Menjalankan inferensi dan *Ensembling* (rata-rata prediksi dari 5 model fold).
* Memetakan probabilitas tertinggi (*Argmax*) menjadi label numerik (`0`, `1`, `2`).

---

## 🧪 FASE 4: Pengujian & Validasi (Testing & Quality Assurance)

### 4.1 Pengujian Performa Model (Validation Check)
Performa model diuji secara internal pada *Validation Set* setiap *fold* menggunakan metrik **Makro F1-Score**:

$$\text{F1}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} \text{F1}_i$$

* **Target Kriteria Lolos:** Makro F1-Score Lokal $\ge 92\%$.

### 4.2 Pengujian Integritas Output (Submission Validation)
Sebelum file diserahkan, sistem QA wajib memverifikasi hal berikut:
* [x] Jumlah baris pada `submission.csv` **tepat 1.458 baris** (tidak kurang/lebih).
* [x] Urutan ID/Nama file pada file output **100% identik** dengan `template.csv`.
* [x] Tipe data label merupakan **bilangan bulat (integer)** dengan rentang nilai $[0, 1, 2]$.
* [x] Tidak ada nilai hampa (*Null/NaN*) pada file hasil.

---

## 🚀 FASE 5: Pengiriman & Penyerahan (Deployment & Deliverables)

### 5.1 Deliverables (Produk Akhir)
1. **File `submission.csv`:** Berkas CSV hasil prediksi Data Uji yang siap diunggah ke *leaderboard* kompetisi.
2. **Source Code / Notebook:** File `.ipynb` atau `.py` yang terstruktur, bersih dari *hardcode*, terkomentar dengan baik, serta siap dijalankan ulang.
3. **Model Weights Artifacts:** File bobot model tersimpan (`.pth` / `.pt`) hasil pelatihan 5-fold.

---

## 🔄 FASE 6: Pemeliharaan & Replikasi (Maintenance & Reproducibility)

* **Penguncian Seed (Reproducibility):** Menetapkannya (*setting random seed*) pada PyTorch, NumPy, dan Python standar untuk menjamin hasil eksperimen dapat direplikasi $100\%$ secara identik.
* **Dokumentasi Log Eksperimen:** Pencatatan *hyperparameter* (batch size, learning rate, jenis varian EfficientNet) untuk memudahkan *tuning* lanjutan jika diperlukan.
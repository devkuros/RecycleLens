# Automated Waste Classification System

Sistem klasifikasi citra sampah otomatis (3 kelas) menggunakan **EfficientNetV2-S** + Stratified 5-Fold Cross-Validation, sesuai SDLC proyek.

| Kode | Kategori |
| :---: | :--- |
| 0 | Recyclable |
| 1 | Electronic |
| 2 | Organic |

## Dari clone sampai running

### 1. Prasyarat

- Python **3.10+**
- Git
- GPU NVIDIA opsional (CPU tetap bisa, gunakan mode cepat)

### 2. Clone repository

Ganti `<REPO_URL>` dengan URL Git repository Anda:

```bash
git clone <REPO_URL>
cd waste-python
```

### 3. Install dependencies

```bash
python -m venv .venv
```

Aktifkan virtual environment:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Lalu install paket proyek (termasuk dependency development untuk pytest):

```bash
pip install -U pip
pip install -e ".[dev]"
```

### 4. Siapkan dataset

Citra **tidak** ikut di-commit. Letakkan dataset resmi sesuai layout di bagian [Data layout](#data-layout):

- Isi `data/train/` (citra berlabel per kelas)
- Isi `data/test/` (citra uji tanpa label)
- Pastikan `data/template.csv` ada (sudah termasuk di repository)

### 5. Verifikasi instalasi

```bash
pytest
```

Jika semua test lolos, environment siap dipakai.

### 6. Train dan predict (mode cepat)

Untuk menjalankan pipeline pertama kali (ramah CPU):

```bash
python scripts/train.py --config configs/fast.yaml
python scripts/predict.py --config configs/fast.yaml
```

Hasil yang diharapkan:

- Checkpoint di `outputs/models/` (mis. `fold_0.pth`, `fold_1.pth`)
- Submission di `outputs/submissions/submission.csv`

Untuk training penuh (5-fold) dan inferensi final, lihat bagian Training dan Inferensi di bawah.

## Data layout

Letakkan dataset resmi di folder berikut (citra tidak di-commit):

```
data/
├── train/
│   ├── 0_Recyclable/   # atau 0/ / Recyclable/
│   ├── 1_Electronic/   # atau 1/ / Electronic/
│   └── 2_Organic/      # atau 2/ / Organic/
├── test/    # 1.458 citra tanpa label (nama mengikuti id di template)
└── template.csv   # kolom: id,predicted
```

Output prediksi mengisi kolom `predicted` dengan nilai `0`, `1`, atau `2`.

## Training

Full run (5-fold, kualitas final):

```bash
python scripts/train.py --config configs/default.yaml
```

Mode cepat di CPU (2-fold, subset, resolusi lebih kecil — untuk iterasi):

```bash
python scripts/train.py --config configs/fast.yaml
```

Checkpoint per fold disimpan di `outputs/models/fold_{0..4}.pth` (atau `fold_0/1` untuk mode cepat). Early stopping menghentikan fold jika macro F1 validasi tidak naik.

## Inferensi & submission

Memakai semua checkpoint `fold_*.pth` yang ada di `outputs/models/` (1 fold atau ensemble penuh).

```bash
python scripts/predict.py --config configs/default.yaml
```

Mode cepat (batch lebih besar; cocok setelah train dengan `fast.yaml`):

```bash
python scripts/predict.py --config configs/fast.yaml
```

Menghasilkan `outputs/submissions/submission.csv` lalu menjalankan QA validasi.

## Tests

```bash
pytest
```

## Guardrails

- Data uji hanya dipakai saat inferensi (`predict.py`).
- Tidak ada akses label test selama training.
- Seed dikunci untuk reproduktifitas.

# แมลงวันอ่านภาษาไทยได้ไหม? (thai-flyvec)

**Can a Fruit Fly Read Thai? Sparse Binary Word Embeddings for Thai Inspired by the Fruit Fly Brain**

โครงงานวิชา คพ.374 การประมวลภาษาธรรมชาติ (CS374) ภาค 1/2569

นำ FlyVec (Liang et al., ICLR 2021) มาเทรนกับภาษาไทย แล้วเทียบกับ word2vec และ fastText บนคลังข้อความ คลังศัพท์ และชุดทดสอบเดียวกัน

## คำถามวิจัย

1. FlyVec ที่เทรนจากคลังข้อความไทย ได้คะแนนความคล้ายคลึงของคำใกล้เคียง word2vec และ fastText แค่ไหน
2. เมื่อใช้จำแนกความรู้สึกข้อความไทย ความแม่นยำลดลงเท่าไร เทียบกับหน่วยความจำและเวลาเทรนที่ประหยัดได้
3. ความยาว hash (k) เท่าไรเหมาะกับภาษาไทย

## Workflow

| ขั้น | งาน | ไฟล์ |
|---|---|---|
| 1 | เก็บและทำความสะอาด Thai Wikipedia (regex, normalize) | `scripts/download_data.py`, `src/thai_flyvec/preprocess.py` |
| 2 | ตัดคำ (PyThaiNLP newmm), ตัดประโยค, คลังศัพท์ 20,000 คำ | `scripts/preprocess_corpus.py`, `src/thai_flyvec/preprocess.py` |
| 3 | สร้าง input แบบบริบท + คำเป้าหมาย (หน้าต่าง w คำ) | `src/thai_flyvec/inputs.py` |
| 4 | เทรน FlyVec ด้วย NumPy (K = 400, k = 4, 8, 16, 32, 64) | `scripts/train_flyvec.py`, `src/thai_flyvec/flyvec.py` |
| 5 | Baseline: word2vec, fastText (gensim, 300 มิติ) + word2vec ไบนารี | `scripts/train_baselines.py`, `src/thai_flyvec/baselines.py` |
| 6 | ประเมินผล | `scripts/run_experiments.py`, `src/thai_flyvec/evaluate.py` |
| 7 | ตารางและกราฟผลลัพธ์ | `scripts/summarize_results.py`, `src/thai_flyvec/report.py`, `notebooks/results.ipynb` |
| 8 | Demo Streamlit | `app/streamlit_app.py` |

### การประเมินผล

| ด้านที่ประเมิน | ชุดข้อมูล | ตัวชี้วัด |
|---|---|---|
| ความคล้ายคลึงของคำ | TH-WordSim-353, TH-SimLex-999 | Spearman ρ และสัดส่วนคำ OOV |
| จำแนกความรู้สึก | wisesight_sentiment (4 คลาส) | accuracy และ macro-F1 (logistic regression บนเวกเตอร์ประโยค) |
| ทรัพยากร | คลังข้อความเดียวกัน | เวลาเทรน (นาที) และขนาดโมเดล (MB) |

## โครงสร้าง

```
thai-flyvec/
├── app/streamlit_app.py      # demo
├── data/                     # ข้อมูลดิบ/ที่ประมวลแล้ว (ไม่ commit) ดู data/README.md
├── docs/decisions.md         # การตัดสินใจเชิงออกแบบ
├── models/                   # โมเดลที่เทรนแล้ว (ไม่ commit)
├── notebooks/results.ipynb   # สรุปผลตอบคำถามวิจัย
├── results/                  # ผลดิบ (CSV), ตาราง และกราฟ
├── scripts/                  # คำสั่งรันแต่ละขั้น
├── src/thai_flyvec/          # โค้ดหลัก
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

ต้องใช้ Python 3.11 ขึ้นไป

## รันทั้ง pipeline

```bash
python scripts/download_data.py          # Wikipedia, ชุดทดสอบ, wisesight → data/
python scripts/preprocess_corpus.py      # ~10 ล้านโทเคน → data/processed/
python scripts/train_flyvec.py           # (ไม่บังคับ) เทรน 1 รอบ แล้วดูคำใกล้เคียงของคำทดสอบ 10 คำ
python scripts/run_experiments.py        # w = 5, 11, 15 × seed 0, 1, 2 → models/, results/*.csv
python scripts/summarize_results.py      # ตาราง results/summary.md และกราฟ results/figures/
streamlit run app/streamlit_app.py       # demo
```

ทุกสคริปต์มี `--help` เวลาโดยประมาณบน CPU 4 คอร์: ตัดคำ ~5 นาที, FlyVec ~20 นาทีต่อรอบ (3 epoch) รวมการทดลองทั้งหมดหลายชั่วโมง ใช้ `run_experiments.py --reuse` เพื่อใช้โมเดลที่เทรนไว้แล้ว และ `--w 11 --seeds 0` เพื่อรันเฉพาะบางค่า

## สถานะ

- โค้ดครบทุกขั้นพร้อมเทสต์ (`pytest`) และทดสอบ pipeline ทั้งสายบนคลังข้อความขนาดเล็กแล้ว
- ยังไม่ได้รันการทดลองบนคลังข้อความจริง ผลลัพธ์และรายงานจะตามมา
- ก่อนรันการทดลองหลัก ให้ลอง learning rate ของ FlyVec ตามที่ระบุใน [docs/decisions.md](docs/decisions.md) D10

แผนงานแบ่งเป็น [milestones](https://github.com/duckxy166/thai-flyvec/milestones) M0–M7 ตามขั้นใน workflow ทุกงานทำผ่าน issue → branch → PR ดู [CONTRIBUTING.md](CONTRIBUTING.md)

การตัดสินใจเชิงออกแบบที่กระทบผลทดลองอยู่ใน [docs/decisions.md](docs/decisions.md)

## เอกสารอ้างอิง

1. Liang, Y., Ryali, C. K., Hoover, B., Grinberg, L., Navlakha, S., Zaki, M. J., & Krotov, D. (2021). [Can a Fruit Fly Learn Word Embeddings?](https://arxiv.org/abs/2101.06887) ICLR 2021. โค้ดและโมเดล: [FlyVec](https://pypi.org/project/flyvec)
2. Netisopakul, P., Wohlgenannt, G., & Pulich, A. (2019). [Word Similarity Datasets for Thai: Construction and Evaluation](https://doaj.org/article/7f756123053e4f89b20baa7af6e5da71). IEEE Access, 7.
3. Netisopakul, P., Wohlgenannt, G., Pulich, A., & Hlaing, Z. Z. (2021). [Improving the state-of-the-art in Thai semantic similarity using distributional semantics and ontological information](https://doaj.org/article/a05d1995d8644219bf3b43bde4dd7461). PLoS ONE, 16(2).
4. Suriyawongkul, A., Chuangsuwanich, E., Chormai, P., & Polpanumas, C. (2019). [PyThaiNLP/wisesight-sentiment](https://huggingface.co/datasets/pythainlp/wisesight_sentiment).
5. Dasgupta, S., Stevens, C. F., & Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. Science, 358(6364), 793–796.

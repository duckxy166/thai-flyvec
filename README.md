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
| 1 | เก็บและทำความสะอาด Thai Wikipedia (regex, normalize) | `src/thai_flyvec/preprocess.py` |
| 2 | ตัดคำ (PyThaiNLP newmm), ตัดประโยค, คลังศัพท์ 20,000 คำ | `src/thai_flyvec/preprocess.py` |
| 3 | สร้าง input แบบบริบท + คำเป้าหมาย (หน้าต่าง w คำ) | `src/thai_flyvec/inputs.py` |
| 4 | เทรน FlyVec ด้วย NumPy (K = 400, k = 4, 8, 16, 32, 64) | `src/thai_flyvec/flyvec.py` |
| 5 | Baseline: word2vec, fastText (gensim, 300 มิติ) + word2vec ไบนารี | `src/thai_flyvec/baselines.py` |
| 6 | ประเมินผล | `src/thai_flyvec/evaluate.py` |
| 7 | Demo Streamlit | `app/streamlit_app.py` |

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
├── data/                     # ข้อมูลดิบ/ที่ประมวลแล้ว (ไม่ commit)
├── models/                   # โมเดลที่เทรนแล้ว (ไม่ commit)
├── notebooks/                # สมุดทดลอง
├── results/                  # ตารางและกราฟผลลัพธ์
├── src/thai_flyvec/          # โค้ดหลัก
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## สถานะ

Skeleton — ฟังก์ชันทุกตัวยังเป็น `NotImplementedError` พร้อม TODO ตาม workflow

## เอกสารอ้างอิง

1. Liang, Y., Ryali, C. K., Hoover, B., Grinberg, L., Navlakha, S., Zaki, M. J., & Krotov, D. (2021). [Can a Fruit Fly Learn Word Embeddings?](https://arxiv.org/abs/2101.06887) ICLR 2021. โค้ดและโมเดล: [FlyVec](https://pypi.org/project/flyvec)
2. Netisopakul, P., Wohlgenannt, G., & Pulich, A. (2019). [Word Similarity Datasets for Thai: Construction and Evaluation](https://doaj.org/article/7f756123053e4f89b20baa7af6e5da71). IEEE Access, 7.
3. Netisopakul, P., Wohlgenannt, G., Pulich, A., & Hlaing, Z. Z. (2021). [Improving the state-of-the-art in Thai semantic similarity using distributional semantics and ontological information](https://doaj.org/article/a05d1995d8644219bf3b43bde4dd7461). PLoS ONE, 16(2).
4. Suriyawongkul, A., Chuangsuwanich, E., Chormai, P., & Polpanumas, C. (2019). [PyThaiNLP/wisesight-sentiment](https://huggingface.co/datasets/pythainlp/wisesight_sentiment).
5. Dasgupta, S., Stevens, C. F., & Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. Science, 358(6364), 793–796.

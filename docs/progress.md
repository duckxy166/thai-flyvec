# บันทึกความคืบหน้า

บันทึกผลทดลองระหว่างทาง ทั้งที่ยังไม่ใช่ผลหลักของรายงาน เพื่อไม่ให้หายไปกับ environment ชั่วคราว

## 2569-11-21 เพิ่ม baseline GloVe (w = 11, 3 seed)

รัน `python scripts/run_glove.py --models-dir models/lr0.3 --w 11 --seeds 0 1 2 --epochs 50 --workers 8` บนเครื่อง Windows (28 คอร์, Python 3.13.14, NumPy 2.5.4, `results/analysis/glove_machine.json`) เทรน GloVe 300 มิติ (`x_max = 100`, `alpha = 0.75`, AdaGrad `lr = 0.05`, 50 รอบ) บนคลังข้อความและคลังศัพท์ 20,000 คำชุดเดียวกัน (`w = 11` ได้คู่การเกิดร่วม 10,569,983 คู่) `pytest` ผ่าน 86 ข้อ และเก็บผลแยกใน `results/analysis/` โดยไม่เขียนทับ `results/raw_results.csv`

- ทรัพยากร: เทรนเฉลี่ย 2.03 ± 0.03 นาทีต่อ seed (2.06 / 2.01 / 2.03 นาที, เคอร์เนล C/OpenMP 8 thread), ไฟล์โมเดล `.npz` ขนาด 22.89 MB
- ผลหลักบนชุดคู่คำและ test เดียวกัน (ค่าเฉลี่ย ± SD ของ 3 seed): TH-WordSim-353 ρ = 0.496 ± 0.009, TH-SimLex-999 ρ = 0.442 ± 0.003, wisesight accuracy = 0.620 ± 0.003, wisesight macro-F1 = 0.413 ± 0.005 (`results/analysis/glove_summary.md`)
- ผลวิเคราะห์ #58 (`results/analysis/summary.md`): H1 SimLex 0.330 ± 0.003 (คำหายาก) / 0.455 ± 0.006 (คำกลาง) / 0.576 ± 0.002 (คำพบบ่อย), H2 wisesight macro-F1 0.417 / 0.379 / 0.363 / 0.458 ตามกลุ่ม OOV, H3 prachathai-67k macro-F1 0.462 ± 0.002 และ micro-F1 0.587 ± 0.001

## 2569-11-21 วิเคราะห์ว่า FlyVec เก่งอะไร (issue #58)

รัน `python scripts/run_analysis.py --workers 12` บนเครื่อง Windows (28 คอร์, Python 3.13, ติดตั้งตาม `requirements-lock.txt`) ใช้โมเดลใน `models/lr0.3/` ไม่เทรนใหม่ ตัดคำ prachathai-67k (67,889 ข่าว) ใช้ 3.1 นาทีด้วย 24 process ทั้ง 3 seed ใช้เวลารวมราว 35 นาที `pytest` ผ่าน 80 ข้อ

- ตรวจไปป์ไลน์: macro-F1 wisesight รวมของ seed 0 ตรงกับ `results/runs/lr0.3/raw_results.csv` ทุกแถว (ต่างกัน 0.0)
- ผลสรุปและเหตุผลของการออกแบบอยู่ใน D11 ตารางเต็มอยู่ใน `results/analysis/summary.md` กราฟอยู่ใน `results/figures/analysis_*.png`

## 2569-11-17 รันการทดลองหลักใหม่ที่ lr 0.3

รัน w = 5, 11, 15 × seed 0, 1, 2 (9 การตั้งค่า) ด้วย lr 0.3 ซึ่งเป็นค่าเริ่มต้นใหม่ของโค้ด บน `main` หลัง #42 ใน environment บนคลาวด์ (Linux, 4 คอร์, Python 3.13, NumPy 2.5.3, gensim 4.4.0) คลังข้อความที่สร้างใหม่ตรงกับเดิม (22,475 บทความ) และ `pytest` ผ่าน 72 ข้อ

- รันพร้อมกันครั้งละ 4 การตั้งค่า (`OPENBLAS_NUM_THREADS=1`, `run_experiments.py --w W --seeds S --results-dir results/runs/wW_sS`) แล้วรวมเป็น `results/raw_results.csv` และ `results/resources.csv` ใช้เวลารวมราว 36 นาที ไม่มี error ผลรวม 1,341 แถว
- เทรนทุกโมเดลใหม่ทั้งหมด (ไม่ใช้ `--reuse`) เวลาเทรนเฉลี่ย FlyVec 2.7 / 6.8 / 9.6 นาที, word2vec 1.6 / 2.1 / 2.1 นาที, fastText 3.1 / 4.1 / 2.4 นาที ที่ w = 5 / 11 / 15
- ผล FlyVec ที่ w = 11 ตรงกับรอบปรับ lr (D10) ทุกค่า
- ข้อสรุปที่เปลี่ยนจาก lr 1.0: FlyVec ดีกว่า word2vec แบบไบนารีที่ k ≤ 8 (เดิมเฉพาะ k = 4 บน WordSim), k ที่แนะนำเปลี่ยนจาก 32–64 เป็น 16–32, เวลาเทรนนานกว่า word2vec 1.7–4.6 เท่า (เดิมหลายสิบเท่า เพราะ implementation เดิม)
- ผลเดิมที่ lr 1.0 ย้ายไป `results/archive/lr1.0/` ตาราง กราฟ และ `notebooks/results.ipynb` สร้างใหม่แล้ว

## 2569-11-13 ปรับ learning rate ของ FlyVec (#33)

ลอง lr 0.3, 1.0, 3.0 ที่ w = 11 × seed 0, 1, 2 (9 รอบ) บน `main` หลัง #36 ใน environment บนคลาวด์ (Linux, 4 คอร์, Python 3.13) คลังข้อความที่สร้างใหม่ตรงกับเดิม (22,475 บทความ, 10,000,129 โทเคน) และ `pytest` ผ่าน 72 ข้อ

- เทรน word2vec และ fastText ครั้งเดียวต่อ seed ไว้ที่ `models/base` (`train_baselines.py --models-dir models/base`) แล้ว hard link ไปที่ `models/lrLR` ก่อนรัน `run_experiments.py --w 11 --seeds S --lr LR --reuse --models-dir models/lrLR --results-dir results/runs/lrLR_sS` `--reuse` จึงข้าม baseline และเทรนเฉพาะ FlyVec baseline ทุก lr เป็นชุดเดียวกัน
- รันพร้อมกันครั้งละ 4 รอบ (`OPENBLAS_NUM_THREADS=1`) ใช้เวลารวมราว 20 นาที (baseline อีกราว 9 นาที) ไม่มี error FlyVec เทรน 4.8–7.6 นาทีต่อรอบ
- lr 1.0 seed 0 ได้ energy −6,914 → −7,607 → −8,188 เท่ากับรอบที่เริ่มบนเครื่อง Windows ก่อนหน้า (ซึ่งหยุดกลางคันเพราะเครื่องช้า) และใกล้กับ `results/` เดิม (−6,911 → −7,607 → −8,205)
- สรุป: **lr 0.3 ดีที่สุด** ρ เฉลี่ยบน k: WordSim 0.405 ± 0.015 / 0.357 ± 0.011 / 0.295 ± 0.023 และ SimLex 0.360 ± 0.008 / 0.321 ± 0.020 / 0.282 ± 0.018 สำหรับ lr 0.3 / 1.0 / 3.0 wisesight ไม่ต่างกัน ตารางเต็มและข้อสังเกตอยู่ใน D10 สร้างซ้ำได้ด้วย `python scripts/compare_lr.py`
- ยังไม่ได้รันการทดลองหลักใหม่ที่ lr 0.3 ผลใน `results/` จึงยังเป็นของ lr 1.0 (รันแล้ว 2569-11-17 ดูด้านบน)

## 2569-11-04 การทดลองครบทุก w และ seed

รัน w = 5, 11, 15 × seed 0, 1, 2 (9 การตั้งค่า) ด้วย lr 1.0 การตั้งค่า w = 11, seed 0 คือรอบแรกด้านล่าง อีก 8 การตั้งค่ารันพร้อมกันครั้งละ 4 การตั้งค่าบนเครื่อง 4 คอร์ (`run_experiments.py --w W --seeds S --results-dir results/runs/wW_sS` แล้วรวมเป็น `results/raw_results.csv` และ `results/resources.csv`) ใช้เวลารวมราว 4 ชั่วโมง 40 นาที

- ทุกการตั้งค่าจบโดยไม่มี error ผลรวม 1,341 แถว
- ตารางค่าเฉลี่ย ± SD อยู่ใน `results/summary.md` กราฟอยู่ใน `results/figures/` และ notebook `notebooks/results.ipynb` รันพร้อมผลลัพธ์แล้ว
- การวิเคราะห์และคำตอบของคำถามวิจัยอยู่ใน `docs/report.md`
- เวลาเทรนเฉลี่ย (w = 5 / 11 / 15): FlyVec 52.4 / 56.5 / 58.4 นาที, word2vec 1.1 / 2.4 / 3.9 นาที, fastText 2.4 / 4.4 / 7.3 นาที เวลาของ 8 การตั้งค่าที่รันพร้อมกันรวมผลของการแย่ง CPU

## 2569-11-04 คลังข้อความจริงและการทดลองแรก

### คลังข้อความ

รัน `scripts/download_data.py` และ `scripts/preprocess_corpus.py` ด้วยค่าเริ่มต้นสำเร็จ (`data/processed/stats.json`)

| รายการ | ค่า |
|---|---|
| บทความที่ใช้ | 22,475 |
| โทเคน | 10,000,129 |
| ประโยค | 1,665,497 |
| คลังศัพท์ | 20,000 คำ |
| สัดส่วนโทเคนนอกคลังศัพท์ | 5.7% |
| เวลาตัดคำ | ~1 นาที (4 คอร์) |

คำที่พบบ่อยที่สุด: ใน, ที่, และ, ของ, เป็น, มี, การ, ได้, โดย, พ.ศ.

ชุดทดสอบ: TH-WordSim-353 353 คู่, TH-SimLex-999 999 คู่, wisesight train/validation/test 21,628 / 2,404 / 2,671 ข้อความ

### การทดลองแรก (w = 11, seed 0)

`python scripts/run_experiments.py --w 11 --seeds 0` แล้ว `python scripts/summarize_results.py --w 11 --k 64` (ตารางเต็มอยู่ใน `results/summary.md`) ใช้ lr 1.0 ค่าเริ่มต้นในขณะนั้น (ภายหลังปรับเป็น 0.3 ตาม D10) และมี seed เดียว จึงยังเป็นผลเบื้องต้น

| โมเดล | ρ SimLex | ρ WordSim | wisesight acc | wisesight macro-F1 | เวลาเทรน (นาที) | ขนาด (MB) |
|---|---|---|---|---|---|---|
| FlyVec k = 4 | 0.301 | 0.384 | 0.593 | 0.400 | 53.5 | 61.0 (รหัส 0.95) |
| FlyVec k = 16 | 0.310 | 0.321 | 0.620 | 0.423 | | |
| FlyVec k = 32 | 0.312 | 0.340 | 0.631 | 0.431 | | |
| FlyVec k = 64 | 0.320 | 0.395 | 0.624 | 0.432 | | |
| word2vec ไบนารี k = 16 | 0.461 | 0.438 | 0.630 | 0.431 | | (รหัส 0.72) |
| word2vec ไบนารี k = 64 | 0.505 | 0.562 | 0.623 | 0.430 | | |
| word2vec | 0.513 | 0.588 | 0.633 | 0.422 | 1.1 | 23.6 |
| fastText | 0.546 | 0.654 | 0.625 | 0.417 | 1.7 | 252.5 |
| fastText + subword (ทุกคู่คำ) | 0.534 | 0.583 | 0.624 | 0.417 | | |

- ρ คิดบนคู่คำที่ทุกโมเดลรู้จัก: SimLex 743 จาก 999 คู่ (OOV 25.6%), WordSim 212 จาก 353 คู่ (OOV 39.9%) แถว fastText + subword คิดบนทุกคู่
- wisesight: โทเคนนอกคลังศัพท์ 12.1% ของชุด test และรวมทุกชุดมี 445 ข้อความที่ไม่มีคำในคลังศัพท์เลย (ใช้เวกเตอร์ศูนย์)
- FlyVec: energy ต่อหน้าต่างลดลงทุก epoch (−6,911 → −7,607 → −8,205) เทรน 3 epoch บน 9,425,376 หน้าต่าง ใช้ CPU คอร์เดียว เฉลี่ย ~18 นาทีต่อ epoch (`fit` เป็น NumPy ล้วน) ส่วน gensim ใช้ 4 worker เวลาเทรนจึงยังเทียบกันตรงๆ ไม่ได้

ข้อสังเกต

1. ความคล้ายของคำ: FlyVec ต่ำกว่า word2vec และ word2vec ไบนารีที่ k ≥ 8 ทุกค่า ที่ k = 4 FlyVec ดีกว่า word2vec ไบนารี (WordSim 0.384 กับ 0.220)
2. จำแนกความรู้สึก: ทุกโมเดลอยู่ในช่วง macro-F1 0.39–0.43 FlyVec k = 32, 64 สูงสุด แต่ต่างจาก baseline ไม่ถึง 0.015 ต้องรอหลาย seed
3. คำใกล้เคียงของ FlyVec (k = 16) เริ่มเข้าเค้า เช่น ฟุตบอล → แชมป์, เล่น, ชนะ; ดนตรี → วงการ, เพลง, การแสดง; รถยนต์ → suv, โตโยต้า แต่ยังมีคำเชื่อม (ซึ่ง, จาก, การ) ปน

## การทดลองเสริม: โมเดล Transformer สำเร็จรูป (ไม่ได้ commit โค้ด)

ทำใน environment บนคลาวด์ ซึ่ง `download.pytorch.org` ถูกบล็อก จึงติดตั้ง torch จาก PyPI (`pip install torch transformers sentencepiece protobuf`, venv ~5.5 GB เพราะมี CUDA ติดมา) ได้ torch 2.15.0 และ transformers 5.22.1 รันบน CPU

1. **sentence-transformers/all-MiniLM-L6-v2**: mean pooling + L2 normalize ได้เวกเตอร์ 384 มิติ ประโยคเรื่องทำนาสองประโยคได้ cosine 0.60 คู่อื่นต่ำกว่า 0.12
2. **airesearch/wangchanberta-base-att-spm-uncased** (`main`, fill-mask): ผลเพี้ยน ใช้ไม่ได้ เช่น "กรุงเทพเป็นเมืองหลวงของประเทศ`<mask>`" ได้ "นม" และทายโทเคนของประโยคตัวเองคืนได้ไม่เกิน 12%
   - ตัดสาเหตุแล้ว: checksum ตรง, `.bin` กับ `.safetensors` เหมือนกันทุกค่า, token ID ตรงกับ SentencePiece (+4 offset), transformers 4.46 ได้ผลเดียวกัน, tokenizer จาก branch `mlm@hotfix-meta-symbol` ไม่ช่วย
   - สรุป: น่าจะเป็นที่ checkpoint ที่เผยแพร่ ถ้าต้องการ fill-mask ภาษาไทยให้ลอง `xlm-roberta-base`
3. **WangchanBERTa revision `finetuned@wisesight_sentiment.fix_tokenizer`** (commit `a6f440e8855308764785d0b665cd0e57dd8dc171`): ถูก 5/6 ประโยค ความมั่นใจ 0.93–0.99 พลาดแค่ "ร้านนี้เปิดกี่โมงครับ" (ควรเป็น question ได้ neutral)
   - วิธีใช้: `CamembertTokenizer` + `AutoModelForSequenceClassification`, แทนช่องว่างด้วย `<_>` ก่อน tokenize, label: pos / neu / neg / q
   - โมเดลนี้ fine-tune บน wisesight เอง จึงเทียบตรงกับ FlyVec (unsupervised) ไม่ได้ ใช้เป็นเพดานอ้างอิงได้เท่านั้น

# บันทึกความคืบหน้า

บันทึกผลทดลองระหว่างทาง ทั้งที่ยังไม่ใช่ผลหลักของรายงาน เพื่อไม่ให้หายไปกับ environment ชั่วคราว

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

กำลังรัน `python scripts/run_experiments.py --w 11 --seeds 0` (lr 1.0 ค่าเริ่มต้น ยังไม่ได้ปรับตาม D10) ผลจะบันทึกที่นี่

## การทดลองเสริม: โมเดล Transformer สำเร็จรูป (ไม่ได้ commit โค้ด)

ทำใน environment บนคลาวด์ ซึ่ง `download.pytorch.org` ถูกบล็อก จึงติดตั้ง torch จาก PyPI (`pip install torch transformers sentencepiece protobuf`, venv ~5.5 GB เพราะมี CUDA ติดมา) ได้ torch 2.15.0 และ transformers 5.22.1 รันบน CPU

1. **sentence-transformers/all-MiniLM-L6-v2**: mean pooling + L2 normalize ได้เวกเตอร์ 384 มิติ ประโยคเรื่องทำนาสองประโยคได้ cosine 0.60 คู่อื่นต่ำกว่า 0.12
2. **airesearch/wangchanberta-base-att-spm-uncased** (`main`, fill-mask): ผลเพี้ยน ใช้ไม่ได้ เช่น "กรุงเทพเป็นเมืองหลวงของประเทศ`<mask>`" ได้ "นม" และทายโทเคนของประโยคตัวเองคืนได้ไม่เกิน 12%
   - ตัดสาเหตุแล้ว: checksum ตรง, `.bin` กับ `.safetensors` เหมือนกันทุกค่า, token ID ตรงกับ SentencePiece (+4 offset), transformers 4.46 ได้ผลเดียวกัน, tokenizer จาก branch `mlm@hotfix-meta-symbol` ไม่ช่วย
   - สรุป: น่าจะเป็นที่ checkpoint ที่เผยแพร่ ถ้าต้องการ fill-mask ภาษาไทยให้ลอง `xlm-roberta-base`
3. **WangchanBERTa revision `finetuned@wisesight_sentiment.fix_tokenizer`** (commit `a6f440e8855308764785d0b665cd0e57dd8dc171`): ถูก 5/6 ประโยค ความมั่นใจ 0.93–0.99 พลาดแค่ "ร้านนี้เปิดกี่โมงครับ" (ควรเป็น question ได้ neutral)
   - วิธีใช้: `CamembertTokenizer` + `AutoModelForSequenceClassification`, แทนช่องว่างด้วย `<_>` ก่อน tokenize, label: pos / neu / neg / q
   - โมเดลนี้ fine-tune บน wisesight เอง จึงเทียบตรงกับ FlyVec (unsupervised) ไม่ได้ ใช้เป็นเพดานอ้างอิงได้เท่านั้น

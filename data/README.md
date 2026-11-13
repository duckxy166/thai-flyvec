# data/

ไม่ commit ข้อมูลลง git — ดาวน์โหลดด้วย

```bash
python scripts/download_data.py            # ทั้งหมด (ข้ามส่วนที่มีไฟล์แล้ว, --force เพื่อโหลดใหม่)
python scripts/preprocess_corpus.py        # ล้าง ตัดคำ และสร้างคลังศัพท์
```

| ที่อยู่ | เนื้อหา | แหล่งที่มา |
|---|---|---|
| `raw/thwiki_subset.jsonl` | บทความ Thai Wikipedia ที่สลับลำดับด้วย seed 0 รวม ~80 ล้านอักขระ (`id`, `title`, `text`) | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.th` |
| `raw/thwiki_articles.txt` | รายชื่อบทความที่ดาวน์โหลด (`id<TAB>title`) | |
| `raw/wisesight/{train,validation,test}.jsonl` | ข้อความและป้ายความรู้สึก (`texts`, `category`: 0 บวก, 1 กลาง, 2 ลบ, 3 คำถาม) | [`pythainlp/wisesight_sentiment`](https://huggingface.co/datasets/pythainlp/wisesight_sentiment) (CC0) |
| `eval/th-wordsim-353.csv`, `eval/th-simlex-999.csv` | คู่คำและคะแนน 0–10 (`คำ1,คำ2,คะแนน` ไม่มี header) | [gwohlgen/thai_word_similarity](https://github.com/gwohlgen/thai_word_similarity) (Netisopakul et al., 2019) |
| `processed/corpus.npz` | ดัชนีคำของทุกโทเคน (`ids`, คำนอกคลังศัพท์ = -1) และจุดเริ่มประโยค (`offsets`) | `preprocess_corpus.py` |
| `processed/vocab.tsv` | คลังศัพท์ 20,000 คำ เรียงตามความถี่ (`คำ<TAB>จำนวน`) | `preprocess_corpus.py` |
| `processed/articles_used.txt` | บทความที่ใช้จริงจนครบ 10 ล้านโทเคน | `preprocess_corpus.py` |
| `processed/stats.json` | จำนวนประโยค โทเคน และสัดส่วนโทเคน OOV | `preprocess_corpus.py` |
| `processed/wisesight_tokens.json` | wisesight ที่ตัดคำแล้ว (cache) | `run_experiments.py` |

ต้องเข้าถึง `huggingface.co`, CDN ของ Hugging Face (`*.hf.co`) และ `raw.githubusercontent.com` ได้

# การมีส่วนร่วม

ขอบคุณที่สนใจร่วมพัฒนา Thai FlyVec

## สำหรับผู้ร่วมพัฒนาจากภายนอก

1. เปิด issue เพื่ออธิบายปัญหาหรือสิ่งที่ต้องการเปลี่ยนก่อน สำหรับการเปลี่ยนแปลงที่มีผลต่อการทดลอง โปรดระบุเหตุผลและอ้างอิงข้อที่เกี่ยวข้องใน [docs/decisions.md](docs/decisions.md)
2. fork repository แล้วสร้าง branch จาก `main`
3. แก้ไขโค้ดพร้อมเพิ่มหรือปรับเทสต์ แล้วตรวจสอบว่า `pytest -q` ผ่านทั้งหมด
4. เปิด pull request ที่อธิบายสิ่งที่เปลี่ยนและวิธีตรวจสอบ พร้อมใส่ `Closes #<เลข issue>`

โค้ดที่ส่งเข้ามาจะเผยแพร่ภายใต้สัญญาอนุญาต [MIT](LICENSE) เช่นเดียวกับโค้ดส่วนอื่นของ repository

## Workflow ของผู้ดูแล

ทุกงานเริ่มจาก issue และเข้าสู่ `main` ผ่าน PR เท่านั้น

1. เลือก issue จาก [milestones](https://github.com/duckxy166/thai-flyvec/milestones) ตามลำดับ M0 → M7
2. แตก branch จาก `main` ล่าสุด ตั้งชื่อ `<ประเภท>/<เลข issue>-<คำอธิบายสั้น>`

   ```bash
   git switch main
   git pull
   git switch -c feat/7-flyvec-fit
   ```

   ประเภท: `feat` (ฟีเจอร์), `fix` (แก้บั๊ก), `chore` (โครงสร้าง/เครื่องมือ), `docs`, `exp` (การทดลอง)
3. commit แล้ว push และเปิด PR ที่มี `Closes #<เลข issue>` ใน description

   ```bash
   git push -u origin HEAD
   gh pr create --fill
   ```

4. รอ CI (`pytest`) เขียว แล้ว squash merge

   ```bash
   gh pr merge --squash
   ```

   GitHub ลบ branch บน remote ให้อัตโนมัติ (`delete_branch_on_merge`) และ issue ปิดเองจาก `Closes #`
5. เก็บกวาด branch ในเครื่อง

   ```bash
   git switch main
   git pull
   git branch -D feat/7-flyvec-fit
   ```

   ต้องใช้ `-D` เพราะ squash merge สร้าง commit ใหม่บน `main` git จึงมองว่า branch ในเครื่องยังไม่ถูก merge และ `-d` จะไม่ยอมลบ ลบเฉพาะหลังจาก PR merge แล้วเท่านั้น

   repo นี้ตั้ง `fetch.prune true` ไว้ใน `.git/config` ของเครื่องนี้แล้ว ถ้า clone ใหม่ให้รัน `git config fetch.prune true`

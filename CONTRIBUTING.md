# Workflow

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
   git branch -d feat/7-flyvec-fit
   ```

   repo นี้ตั้ง `fetch.prune true` ไว้ใน `.git/config` ของเครื่องนี้แล้ว ถ้า clone ใหม่ให้รัน `git config fetch.prune true`

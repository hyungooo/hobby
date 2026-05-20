# 0522 High School Talk

고등학교 저학년 대상 20분 발표 자료입니다.

## 발표 자료

- `slides.html` — 발표용 HTML 슬라이드, 키보드 ←/→ 이동, `F` 전체화면, `P` 인쇄/PDF
- `slides.pdf` — PDF 버전
- `speaker-notes.md` — 슬라이드별 발표 멘트 초안
- `design-references.md` — Lazyweb MCP로 확인한 디자인 레퍼런스 요약

## 실행

WSL에서는 아래 스크립트를 추천합니다. 로컬 HTTP 서버를 띄우고 Windows 브라우저에서 엽니다.

```bash
./open-slides.sh
```

직접 열려면:

```bash
python3 -m http.server 8522 --bind 127.0.0.1
# Windows 브라우저에서 http://127.0.0.1:8522/slides.html 접속
```

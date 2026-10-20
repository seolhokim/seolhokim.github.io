#!/usr/bin/env python3
"""Build the content inventory (_seo/inventory.md) from _posts front matter. Standard library only.
Usage: python3 -I _seo/make_inventory.py  (run from the site root)"""
import re, sys
from pathlib import Path
from urllib.parse import quote

root = Path(__file__).resolve().parent.parent
posts = sorted((root / "_posts").glob("*.md"))
rows = []
for p in posts:
    text = p.read_text(encoding="utf-8")
    m = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", text, re.S)
    fm = m.group(1) if m else ""
    def get(key):
        mm = re.search(rf"^{key}:\s*(.*)$", fm, re.M)
        return mm.group(1).strip().strip('"').strip() if mm else ""
    title = get("title")
    desc = get("description")
    cats = re.findall(r"^categories:\s*\n((?:\s+-\s*.*\n?)+)", fm, re.M)
    cat_list = [c.strip("- ").strip() for c in cats[0].strip().splitlines()] if cats else []
    tags = get("tags")
    date, slug = p.name[:10], p.name[11:-3]
    y, mo, d = date.split("-")
    cat_path = "/".join(quote(c.lower()) for c in cat_list)
    url = f"/{cat_path}/{y}/{mo}/{d}/{slug}.html" if cat_path else f"/{y}/{mo}/{d}/{slug}.html"
    body = text[m.end():] if m else text
    rows.append(dict(url=url, title=title, cat=", ".join(cat_list), tags=tags, date=date,
                     desc_len=len(desc), chars=len(body), has_img="Y" if "![" in body else ""))

out = ["# 콘텐츠 인벤토리 — " + __import__("datetime").date.today().isoformat(),
       "", f"글 {len(rows)}건. description 있음 {sum(1 for r in rows if r['desc_len'])}건 · 이미지 있음 {sum(1 for r in rows if r['has_img'])}건.",
       "'대상 질문'·'노출'·'상태' 열은 GSC/서치어드바이저 데이터로 채운다 (references/content.md 0절).", "",
       "| # | 발행일 | 제목 | 카테고리 | 태그 | 본문 글자수 | 이미지 | desc 길이 | 대상 질문 | 최근 28일 노출/클릭 | 상태 | URL |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(sorted(rows, key=lambda r: r["date"], reverse=True), 1):
    out.append(f"| {i} | {r['date']} | {r['title']} | {r['cat']} | {r['tags']} | {r['chars']} | {r['has_img']} | {r['desc_len']} |  |  | published | {r['url']} |")
(root / "_seo" / "inventory.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"inventory: {len(rows)} posts, {sum(1 for r in rows if r['desc_len'])} with description")

#!/usr/bin/env bash
# Crawler-eye verification of the live site (run AFTER pushing; GitHub Pages takes ~1-3 min to rebuild).
# Usage: bash _seo/verify_live.sh [https://seolhokim.github.io]
S="${1:-https://seolhokim.github.io}"
P="$S/reinforcement%20learning/2022/01/08/gmlp.html"
c() { curl -sL --max-time 20 "$@"; }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$1"; }
echo "== $S =="
echo "robots.txt      : $(code $S/robots.txt)  ua-groups=$(c $S/robots.txt | grep -c '^User-agent')  sitemap-line=$(c $S/robots.txt | grep -c '^Sitemap')"
echo "llms.txt        : $(code $S/llms.txt)  post-entries=$(c $S/llms.txt | grep -c '^- \[')"
echo "sitemap.xml     : $(code $S/sitemap.xml)  urls=$(c $S/sitemap.xml | grep -c '<loc>')  pages=$(c $S/sitemap.xml | grep '<loc>' | grep -vcE '/20[0-9]{2}/')"
echo "feed.xml        : $(code $S/feed.xml)"
echo "404 check       : $(code $S/no-such-page-xyz) (expect 404)"
echo "docs noindex    : $(c $S/docs/ | grep -c 'name=\"robots\" content=\"noindex') (expect 1)"
echo "-- home --"
c $S/ | grep -oE '<meta name="description" content="[^"]{0,80}' | head -1
c $S/ | grep -oE '"sameAs":\[[^]]*\]'
echo "-- sample post: $P --"
H=$(c "$P")
echo "html lang       : $(echo "$H" | grep -oE '<html lang="[^"]*"')"
echo "description     : $(echo "$H" | grep -oE '<meta name="description" content="[^"]{0,90}' | head -1)"
echo "og:locale       : $(echo "$H" | grep -oE 'og:locale" content="[^"]*"')"
echo "og:image        : $(echo "$H" | grep -c 'og:image')  (expect 1)"
echo "meta author     : $(echo "$H" | grep -oE '<meta name="author" content="[^"]*"')"
echo "post-lead       : $(echo "$H" | grep -c 'class="post-lead"') (expect 1)"
echo "LD author       : $(echo "$H" | grep -oE '"author":\{[^}]*\}')"
echo "LD publisher    : $(echo "$H" | grep -oE '"publisher":\{.{0,160}' | grep -oE '"name":"[^"]*"' | head -1)"
echo "robots meta     : $(echo "$H" | grep -oiE '<meta[^>]*robots[^>]*>' || echo none) (expect none)"

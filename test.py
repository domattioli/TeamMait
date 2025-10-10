import requests
from bs4 import BeautifulSoup

def draw_doc_grid(url: str) -> None:
    r = requests.get(url, timeout=10); r.raise_for_status()
    s = BeautifulSoup(r.text, "html.parser")
    t = s.find("table")
    if not t: raise ValueError("no table")
    h = [x.get_text(strip=True).lower() for x in t.find_all("th")] or ["x", "ch", "y"]
    rows = t.find_all("tr")[1:]
    pts = []
    for r0 in rows:
        c = [z.get_text(strip=True) for z in r0.find_all(["td", "th"])]
        if len(c) < 3: continue
        try: x, y = int(c[0]), int(c[2])
        except: continue
        pts.append((x, y, c[1][:1] or " "))
    if not pts: raise ValueError("no coords")
    mx, my = max(p[0] for p in pts), max(p[1] for p in pts)
    g = [[" " for _ in range(mx + 1)] for _ in range(my + 1)]
    for x, y, ch in pts: g[y][x] = ch
    for r1 in g: print("".join(r1))

if __name__ == "__main__":
    draw_doc_grid("https://docs.google.com/document/d/e/2PACX-1vRPzbNQcx5UriHSbZ-9vmsTow_R6RRe7eyAU60xIF9Dlz-vaHiHNO2TKgDi7jy4ZpTpNqM7EvEcfr_p/pub")

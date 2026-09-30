"""Build docs/img/logo.svg (mark + wordmark) and docs/img/mark.svg. Text is converted to paths (Noto Sans).

    python3 docs/make_logo.py        (system python with matplotlib)

Mark: one state node fans out to three option nodes; the chosen option is filled.
Colors switch with prefers-color-scheme, so one file works on light and dark backgrounds.
"""
import os

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

INK, INK_DARK = "#0f172a", "#e2e8f0"
MUTED, MUTED_DARK = "#64748b", "#94a3b8"
ACCENT = "#2563eb"
ACCENT_DARK = "#60a5fa"
GAP = 9                # space between "open" and "decider"
REG = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"

STYLE = f"""<style>
.ink{{fill:{INK}}} .muted{{fill:{MUTED}}} .acc{{fill:{ACCENT}}} .line{{stroke:{MUTED};fill:none}}
.ring{{stroke:{MUTED};fill:none}}
@media (prefers-color-scheme: dark){{.ink{{fill:{INK_DARK}}} .muted{{fill:{MUTED_DARK}}} .acc{{fill:{ACCENT_DARK}}}
.line,.ring{{stroke:{MUTED_DARK}}}}}
</style>"""


def text_d(s, font, size, x, y):
    """SVG path data for text s with baseline at (x, y); y grows downward in SVG."""
    tp = TextPath((0, 0), s, size=size, prop=FontProperties(fname=font))
    out = []
    for verts, code in tp.iter_segments(curves=True, simplify=False):
        pts = [(x + verts[i], y - verts[i + 1]) for i in range(0, len(verts), 2)]
        f = lambda p: f"{p[0]:.2f} {p[1]:.2f}"
        if code == Path.MOVETO:
            out.append("M" + f(pts[0]))
        elif code == Path.LINETO:
            out.append("L" + f(pts[0]))
        elif code == Path.CURVE3:
            out.append("Q" + " ".join(map(f, pts)))
        elif code == Path.CURVE4:
            out.append("C" + " ".join(map(f, pts)))
        elif code == Path.CLOSEPOLY:
            out.append("Z")
    return "".join(out), tp.get_extents().width


def mark(ox, oy, s=1.0):
    """The mark in a 64x64 box at (ox, oy), scaled by s."""
    P = lambda x, y: (ox + x * s, oy + y * s)
    src = P(12, 32)
    opts = [P(52, 12), P(52, 32), P(52, 52)]
    lines = []
    for (x, y) in opts:
        c1 = (src[0] + 20 * s, src[1]); c2 = (x - 18 * s, y)
        lines.append(f'<path class="line" stroke-width="{3 * s:.2f}" stroke-linecap="round" '
                     f'd="M{src[0]:.2f} {src[1]:.2f}C{c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {x - 7 * s:.2f} {y:.2f}"/>')
    nodes = [f'<circle class="ink" cx="{src[0]:.2f}" cy="{src[1]:.2f}" r="{7 * s:.2f}"/>']
    for i, (x, y) in enumerate(opts):
        if i == 1:
            nodes.append(f'<circle class="acc" cx="{x:.2f}" cy="{y:.2f}" r="{7 * s:.2f}"/>')
        else:
            nodes.append(f'<circle class="ring" stroke-width="{3 * s:.2f}" cx="{x:.2f}" cy="{y:.2f}" r="{5.5 * s:.2f}"/>')
    return "".join(lines + nodes)


def main():
    os.makedirs("docs/img", exist_ok=True)
    size, base = 44, 50
    d1, w1 = text_d("open", REG, size, 84, base)
    d2, w2 = text_d("decider", BOLD, size, 84 + w1 + GAP, base)
    W = int(84 + w1 + GAP + w2 + 8)
    logo = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 64" width="{W * 5 // 4}" height="80" '
            f'role="img" aria-label="OpenDecider">{STYLE}<title>OpenDecider</title>{mark(0, 0)}'
            f'<path class="muted" d="{d1}"/><path class="ink" d="{d2}"/></svg>')
    open("docs/img/logo.svg", "w").write(logo)
    m = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" '
         f'aria-label="OpenDecider">{STYLE}{mark(0, 0)}</svg>')
    open("docs/img/mark.svg", "w").write(m)
    print(f"docs/img/logo.svg ({len(logo) // 1024} KB), docs/img/mark.svg")


if __name__ == "__main__":
    main()

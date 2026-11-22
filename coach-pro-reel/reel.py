# Coach Pro — promo reel, 1080x1920 @30fps, drawn frame by frame.
import math, subprocess, sys, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, DUR = 1080, 1920, 30, 35.0
F = "/tmp/reel/fonts/"
FONTS = {
    "black": F + "Tajawal-Black.ttf", "xbold": F + "Tajawal-ExtraBold.ttf",
    "bold": F + "Tajawal-Bold.ttf", "med": F + "Tajawal-Medium.ttf",
    "lal": F + "Lalezar.ttf",
    "inter": "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf",
    "interb": "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf",
}
NAVY = (16, 40, 28); NAVY2 = (30, 80, 56); CARD = (31, 84, 58)
LIME = (242, 183, 5); GOLD = (160, 222, 184); CORAL = (255, 90, 95)
WHITE = (247, 250, 248); GREY = (186, 210, 196); US = (247, 250, 248)
PITCH = (47, 106, 71)

# ---------- easing ----------
def cl(x): return 0.0 if x < 0 else 1.0 if x > 1 else x
def pr(t, a, b): return cl((t - a) / (b - a))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4*x*x*x if x < .5 else 1 - (-2*x + 2) ** 3 / 2
def eback(x, s=1.7):
    x = cl(x); return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2
def lerp(a, b, x): return a + (b - a) * x
def win(t, t0, t1, fin=.35, fout=.25):
    """alpha + in-progress for an element living in [t0,t1]"""
    if t < t0 or t > t1: return 0.0, 0.0
    a = eo(pr(t, t0, t0 + fin)) * (1 - eio(pr(t, t1 - fout, t1)))
    return a, eo(pr(t, t0, t0 + fin))

# ---------- sprites ----------
@functools.lru_cache(None)
def font(key, size):
    return ImageFont.truetype(FONTS[key], size, layout_engine=ImageFont.Layout.RAQM)

@functools.lru_cache(None)
def text(s, key, size, color=WHITE, pad=12):
    f = font(key, size)
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    rtl = any("؀" <= c <= "ۿ" for c in s)
    kw = {"direction": "rtl"} if rtl else {}
    x0, y0, x1, y1 = d.textbbox((0, 0), s, font=f, **kw)
    im = Image.new("RGBA", (x1 - x0 + 2*pad, y1 - y0 + 2*pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - x0, pad - y0), s, font=f, fill=color + (255,), **kw)
    return np.array(im)

@functools.lru_cache(None)
def rrect(w, h, r, fill, border=None, bw=0, ss=3):
    im = Image.new("RGBA", (w*ss, h*ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w*ss-1, h*ss-1), r*ss, fill=(fill + (255,)) if fill else None,
                        outline=(border + (255,)) if border else None, width=bw*ss)
    return np.array(im.resize((w, h), Image.LANCZOS))

@functools.lru_cache(None)
def disc(r, fill, ring=None, rw=0, ss=4):
    s = (2*r + 4) * ss
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = s / 2
    if ring:
        d.ellipse((c - r*ss, c - r*ss, c + r*ss, c + r*ss), fill=ring + (255,))
        rr = (r - rw) * ss
        d.ellipse((c - rr, c - rr, c + rr, c + rr), fill=fill + (255,))
    else:
        d.ellipse((c - r*ss, c - r*ss, c + r*ss, c + r*ss), fill=fill + (255,))
    return np.array(im.resize((2*r + 4, 2*r + 4), Image.LANCZOS))

@functools.lru_cache(None)
def glow(r, color, strength=0.9):
    y, x = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    d = np.sqrt(x*x + y*y) / r
    a = np.clip(1 - d, 0, 1) ** 2 * 255 * strength
    out = np.zeros((2*r+1, 2*r+1, 4), np.uint8)
    out[..., :3] = color; out[..., 3] = a.astype(np.uint8)
    return out

def blit(fr, spr, cx, cy, a=1.0, scale=1.0, anchor="c", clipw=None):
    if a <= 0.003: return
    if scale != 1.0:
        nw, nh = max(1, int(spr.shape[1]*scale)), max(1, int(spr.shape[0]*scale))
        spr = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    if clipw is not None:  # reveal from the right (RTL wipe): keep right fraction
        k = int(w * cl(clipw))
        if k <= 0: return
        spr = spr[:, w - k:]
        offx = w - k
    else:
        offx = 0
    if anchor == "c": x0, y0 = int(cx - w/2) + offx, int(cy - h/2)
    elif anchor == "r": x0, y0 = int(cx - w) + offx, int(cy - h/2)
    elif anchor == "l": x0, y0 = int(cx) + offx, int(cy - h/2)
    hh, ww = spr.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
    if X1 <= X0 or Y1 <= Y0: return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)
    al = s[..., 3:4] / 255.0 * a
    roi = fr[Y0:Y1, X0:X1].astype(np.float32)
    fr[Y0:Y1, X0:X1] = (roi * (1 - al) + s[..., :3] * al).astype(np.uint8)

def tx(fr, t, s, key, size, cx, cy, t0, t1, color=WHITE, anim="up", anchor="c", fin=.4):
    a, p = win(t, t0, t1, fin=fin)
    if a <= 0: return
    spr = text(s, key, size, color)
    if anim == "up":
        blit(fr, spr, cx, cy + (1 - p) * 40, a, anchor=anchor)
    elif anim == "pop":
        blit(fr, spr, cx, cy, a, scale=lerp(.6, 1, eback(pr(t, t0, t0 + .45))), anchor=anchor)
    elif anim == "wipe":
        blit(fr, spr, cx, cy, a, anchor=anchor, clipw=eio(pr(t, t0, t0 + .45)))

def darken(fr, x0, y0, x1, y1, a, color=NAVY):
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if a <= 0 or x1 <= x0 or y1 <= y0: return
    roi = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)

# ---------- background ----------
def make_bg():
    y = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array([30, 76, 52.]), np.array([15, 42, 28.])
    bg = top * (1 - y) + bot * y + np.zeros((1, W, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W/2) / W) ** 2 + ((yy - H*0.42) / H) ** 2)
    bg *= (1.08 - 0.55 * r)[..., None]
    rng = np.random.default_rng(3)
    bg += rng.normal(0, 2.2, (H, W, 1))
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    for gx in range(0, W, 90):  # faint grid like a coaching notebook
        bg[:, gx:gx+1] = np.clip(bg[:, gx:gx+1].astype(int) + 6, 0, 255)
    for gy in range(0, H, 90):
        bg[gy:gy+1] = np.clip(bg[gy:gy+1].astype(int) + 6, 0, 255)
    return bg
BG = make_bg()

# ---------- tactical board ----------
BW, BH, BM = 960, 854, 30
SC = (BW - 2*BM) / 68.0
def bl(px, py):  # pitch metres -> board-local px  (we attack upward, view y 45..105)
    return BM + px * SC, BM + (105 - py) * SC

def make_board():
    ss = 2
    im = Image.new("RGBA", (BW*ss, BH*ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, BW*ss-1, BH*ss-1), 40*ss, fill=PITCH + (255,))
    # mowing stripes
    stripe = Image.new("RGBA", im.size, (0, 0, 0, 0)); ds = ImageDraw.Draw(stripe)
    for i in range(12):
        if i % 2: ds.rectangle((0, i*BH*ss/12, BW*ss, (i+1)*BH*ss/12), fill=(255, 255, 255, 9))
    mask = Image.new("L", im.size, 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, BW*ss-1, BH*ss-1), 40*ss, fill=255)
    im = Image.composite(Image.alpha_composite(im, stripe), im, mask); d = ImageDraw.Draw(im)
    C = (255, 255, 255, 150); lw = 3*ss
    P = lambda px, py: tuple(v*ss for v in bl(px, py))
    def rect(x0, y0, x1, y1):
        a, b = P(x0, y1), P(x1, y0); d.rectangle((a[0], a[1], b[0], b[1]), outline=C, width=lw)
    rect(0, 45 - 3, 68, 105)                      # touchlines + goal line
    d.line((P(0, 52.5), P(68, 52.5)), fill=C, width=lw)   # halfway
    cx, cy = P(34, 52.5); r = 9.15*SC*ss
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=C, width=lw)
    rect(13.84, 88.5, 54.16, 105); rect(24.84, 99.5, 43.16, 105)
    sx, sy = P(34, 94); d.ellipse((sx-5*ss, sy-5*ss, sx+5*ss, sy+5*ss), fill=C)
    r = 9.15*SC*ss
    d.arc((sx - r, sy - r, sx + r, sy + r), 37, 143, fill=C, width=lw)
    g0, g1 = P(30.34, 105), P(37.66, 105)
    d.rectangle((g0[0], g0[1] - 1.8*SC*ss, g1[0], g0[1]), outline=(255, 255, 255, 200), width=lw)
    # clip to rounded board
    out = Image.new("RGBA", im.size, (0, 0, 0, 0)); out.paste(im, (0, 0), mask)
    sh = Image.new("RGBA", (BW*ss + 80*ss, BH*ss + 80*ss), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((40*ss, 52*ss, BW*ss + 40*ss, BH*ss + 52*ss), 40*ss, fill=(0, 0, 0, 150))
    sh = sh.filter(ImageFilter.GaussianBlur(26*ss))
    sh.alpha_composite(out, (40*ss, 40*ss))
    return np.array(sh.resize(((BW + 80), (BH + 80)), Image.LANCZOS))
BOARD = make_board()

# player keyframes: id -> list of (t, px, py)
OPP0 = {"GK": (34, 103), "D1": (22, 93), "D2": (30, 93), "D3": (38, 93), "D4": (46, 93),
        "M1": (20, 84), "M2": (29, 84), "M3": (39, 84), "M4": (48, 84), "F1": (28, 72), "F2": (40, 72)}
OPP1 = {"GK": (33, 103), "D1": (11, 92), "D2": (25, 93), "D3": (35, 93), "D4": (44, 93),
        "M1": (12, 83), "M2": (25, 84), "M3": (36, 84), "M4": (46, 84), "F1": (26, 72), "F2": (38, 72)}
US0 = {"CB1": (26, 55), "CB2": (42, 55), "LB": (8, 68), "RB": (60, 68), "DM": (34, 63),
       "C8L": (26, 77), "C8R": (42, 77), "LW": (20, 88), "RW": (48, 88), "ST": (34, 92)}
SHIFT = (16.9, 18.0); RUN = (18.1, 19.1)

def ppos(team, k, t):
    if team == "o":
        x = eio(pr(t, *SHIFT)); a, b = OPP0[k], OPP1[k]
        return lerp(a[0], b[0], x), lerp(a[1], b[1], x)
    if k == "C8L":
        x = eio(pr(t, *RUN)); return lerp(26, 18, x), lerp(77, 91, x)
    if k == "NW": return 4, 89
    return US0[k]

def board_xf(t):
    """returns (alpha, scale, left, top) for the board on screen"""
    a = eo(pr(t, 3.6, 4.3)) * (1 - eio(pr(t, 21.8, 22.4)))
    s = lerp(.92, 1, eo(pr(t, 3.6, 4.4)))
    sh = eio(pr(t, 10.0, 10.6)) * (1 - eio(pr(t, 15.4, 16.0)))
    s = lerp(s, .66, sh)
    top = lerp(400, 372, sh)
    left = 540 - BW*s/2
    return a, s, left, top

def draw_arrow(fr, pts, prog, color, th, dashed=False, head=True):
    if prog <= 0: return
    pts = np.array(pts, np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = seg.sum(); target = L * prog
    out = [pts[0]]; acc = 0
    for i, sl in enumerate(seg):
        if acc + sl >= target:
            out.append(pts[i] + (pts[i+1] - pts[i]) * ((target - acc) / max(sl, 1e-6))); break
        out.append(pts[i+1]); acc += sl
    out = np.array(out)
    if dashed:
        # walk the polyline in dash chunks
        dl, gl, carry, on = 22, 14, 0.0, True
        for i in range(len(out) - 1):
            p, q = out[i], out[i+1]; sl = np.linalg.norm(q - p); pos = 0
            while pos < sl:
                step = min((dl if on else gl) - carry, sl - pos)
                a = p + (q - p) * (pos / sl); b = p + (q - p) * ((pos + step) / sl)
                if on: cv2.line(fr, tuple(int(v) for v in a), tuple(int(v) for v in b), color, th, cv2.LINE_AA)
                pos += step; carry += step
                if carry >= (dl if on else gl) - 1e-3: carry = 0; on = not on
    else:
        cv2.polylines(fr, [out.astype(np.int32)], False, color, th, cv2.LINE_AA)
    if head and len(out) > 1:
        e = out[-1]; d = out[-1] - out[-2]; n = np.linalg.norm(d)
        if n > 1e-3:
            d /= n; nrm = np.array([-d[1], d[0]]); hl = th * 4.5
            tri = np.array([e + d*hl*.4, e - d*hl + nrm*hl*.6, e - d*hl - nrm*hl*.6], np.int32)
            cv2.fillPoly(fr, [tri], color, cv2.LINE_AA)

def bez(p0, p1, p2, n=30):
    return [((1-u)**2*p0[0] + 2*(1-u)*u*p1[0] + u*u*p2[0], (1-u)**2*p0[1] + 2*(1-u)*u*p1[1] + u*u*p2[1])
            for u in np.linspace(0, 1, n)]

def draw_board(fr, t):
    a, s, left, top = board_xf(t)
    if a <= 0: return
    blit(fr, BOARD, left - 40*s + BOARD.shape[1]*s/2, top - 40*s + BOARD.shape[0]*s/2, a, scale=s)
    S = lambda px, py: (left + bl(px, py)[0]*s, top + bl(px, py)[1]*s)
    R = int(22 * s)

    # S4 overlays under players
    za, _ = win(t, 16.2, 17.4, .3, .3)
    if za > 0:  # congested centre
        cx, cy = S(34, 86)
        g = glow(int(260*s), CORAL, .55); blit(fr, g, cx, cy, za)
    ga, _ = win(t, 17.6, 21.8, .4, .3)
    if ga > 0:  # opened half-space
        x0, y0 = S(13.5, 97); x1, y1 = S(23, 85.5)
        pulse = .55 + .25 * math.sin(t * 7)
        ov = fr.copy()
        cv2.rectangle(ov, (int(x0), int(y0)), (int(x1), int(y1)), LIME, -1, cv2.LINE_AA)
        m = ga * pulse * .35
        y0i, y1i, x0i, x1i = int(y0), int(y1), int(x0), int(x1)
        fr[y0i:y1i, x0i:x1i] = (fr[y0i:y1i, x0i:x1i] * (1 - m) + ov[y0i:y1i, x0i:x1i] * m).astype(np.uint8)
        cv2.rectangle(fr, (x0i, y0i), (x1i, y1i), LIME, max(2, int(3*s)), cv2.LINE_AA)

    # arrows
    if t > 16.9 and t < 21.8:
        p = eio(pr(t, 16.9, 17.8)); aa = 1 - eio(pr(t, 21.4, 21.8))
        if aa > 0:
            a0, a1 = S(22, 94.5), S(11.5, 94.5)
            draw_arrow(fr, [a0, a1], p, CORAL, max(2, int(4*s)))
    if t > RUN[0] and t < 21.8:
        p = eio(pr(t, RUN[0], RUN[1]))
        draw_arrow(fr, bez(S(26, 77), S(19, 82), S(18, 89.7)), p, LIME, max(2, int(5*s)), dashed=True)
    if t > 19.2 and t < 21.8:
        p = eo(pr(t, 19.2, 19.7))
        draw_arrow(fr, [S(5.5, 89.3), S(16.5, 90.8)], p, WHITE, max(2, int(5*s)))

    # players pop in
    for i, k in enumerate(OPP0):
        tp = 4.25 + i * 0.045
        pa = eo(pr(t, tp, tp + .25)) * a
        if pa <= 0: continue
        sc = eback(pr(t, tp, tp + .35))
        x, y = S(*ppos("o", k, t))
        blit(fr, disc(R, CORAL, (120, 30, 40), max(2, int(3*s))), x, y + 3*s, pa*.35, scale=sc)
        blit(fr, disc(R, CORAL, (255, 200, 200), max(2, int(3*s))), x, y, pa, scale=sc)
    for i, k in enumerate(US0):
        tp = 4.75 + i * 0.045
        pa = eo(pr(t, tp, tp + .25)) * a
        if k == "LW": pa *= 1 - eio(pr(t, 16.3, 16.7))
        if pa <= 0: continue
        sc = eback(pr(t, tp, tp + .35))
        if k == "LW": sc *= lerp(1, .5, eio(pr(t, 16.3, 16.7)))
        x, y = S(*ppos("u", k, t))
        col = LIME if (k == "C8L" and t > RUN[0] - .2) else US
        blit(fr, disc(R, col, (20, 50, 34), max(2, int(3*s))), x, y, pa, scale=sc)
    # new winger
    na = eo(pr(t, 16.6, 16.95)) * a
    if na > 0:
        x, y = S(4, 89)
        blit(fr, glow(int(70*s), LIME, .8), x, y, na * (.6 + .4*math.sin(t*6)))
        blit(fr, disc(R, LIME, NAVY, max(2, int(4*s))), x, y, na, scale=eback(pr(t, 16.6, 17.0)))
        lab = text("الجناح الجديد", "xbold", 30, NAVY)
        tag = rrect(lab.shape[1] + 16, 52, 26, LIME)
        ty = y - 54*s
        blit(fr, tag, x + 92*s, ty, na * (1 - pr(t, 17.4, 17.7)), scale=s)
        blit(fr, lab, x + 92*s, ty, na * (1 - pr(t, 17.4, 17.7)), scale=s)
    # sub-out ring
    oa, _ = win(t, 16.2, 16.9, .15, .3)
    if oa > 0:
        x, y = S(20, 88); rr = int(lerp(R, R*2.6, eo(pr(t, 16.2, 16.9))))
        cv2.circle(fr, (int(x), int(y)), rr, LIME, 3, cv2.LINE_AA)

    # formation tags
    for lbl, (px, py), col, fg, t0 in (("4-4-2", (57, 78), CORAL, WHITE, 6.85), ("4-3-3", (55, 58), US, NAVY, 7.0)):
        fa, _ = win(t, t0, 10.0, .35, .3)
        fa *= a
        if fa <= 0: continue
        txt = text(lbl, "interb", 34, fg); tag = rrect(txt.shape[1] + 24, 58, 29, col)
        x, y = S(px, py); sc = s * eback(pr(t, t0, t0 + .4))
        hl = win(t, 6.8, 7.75, .2, .2)[0]
        if hl > 0: blit(fr, glow(int(120*s), col, .7), x, y, hl * fa)
        blit(fr, tag, x, y, fa, scale=sc); blit(fr, txt, x, y, fa, scale=sc)

# ---------- header: case tag + scoreboard ----------
def draw_header(fr, t):
    a = eo(pr(t, 3.9, 4.4)) * (1 - eio(pr(t, 21.8, 22.3)))
    if a <= 0: return
    pill_t = text("حالة تكتيكية  ·  رقم 27", "bold", 34, LIME)
    pill = rrect(pill_t.shape[1] + 30, 64, 32, NAVY2, LIME, 2)
    blit(fr, pill, 540, 128, a); blit(fr, pill_t, 540, 128, a)
    sb = rrect(820, 140, 34, CARD, (74, 128, 98), 2)
    y = 262
    blit(fr, sb, 540, y, a)
    secs = int(max(0, t - 4.0))
    clock = f"78:{secs:02d}"
    mb = rrect(220, 104, 26, LIME)
    hl = win(t, 6.0, 6.85, .2, .2)[0]
    if hl: blit(fr, glow(170, LIME, .9), 540, y, hl * a)
    blit(fr, mb, 540, y, a)
    blit(fr, text(clock, "inter", 56, NAVY), 540, y + 1, a)
    hs = win(t, 5.2, 6.0, .2, .2)[0]
    if hs:
        blit(fr, glow(80, CORAL, .8), 390, y, hs * a); blit(fr, glow(80, WHITE, .5), 690, y, hs * a)
    # right = your team (RTL reading order)
    blit(fr, text("فريقك", "bold", 36, GREY), 905, y, a, anchor="r")
    blit(fr, text("0", "inter", 76, WHITE), 690, y + 2, a)
    blit(fr, text("الخصم", "bold", 36, GREY), 175, y, a, anchor="l")
    blit(fr, text("1", "inter", 76, CORAL), 390, y + 2, a)

# ---------- bench ----------
def draw_bench(fr, t):
    a, p = win(t, 7.7, 10.1, .4, .3)
    if a <= 0: return
    blit(fr, text("على الدكة", "black", 40, GOLD), 540, 1318, a)
    y = 1405
    chips = [("جناح سريع", 845), ("مهاجم طويل", 540), ("لاعب ارتكاز", 235)]
    for i, (lbl, x) in enumerate(chips):
        ca = a * eo(pr(t, 7.9 + i*.12, 8.25 + i*.12))
        sc = eback(pr(t, 7.9 + i*.12, 8.3 + i*.12))
        tt = text(lbl, "bold", 38, WHITE); bx = rrect(290, 86, 43, CARD, GOLD, 2)
        blit(fr, bx, x, y, ca, scale=sc)
        blit(fr, disc(13, US, (20, 50, 34), 2), x + 112, y, ca, scale=sc)
        blit(fr, tt, x - 18, y, ca, scale=sc)

# ---------- caption strip ----------
def cap(fr, t, s, t0, t1, color=WHITE, size=76, y=1600, key="black"):
    a, p = win(t, t0, t1, .35, .25)
    if a <= 0: return
    blit(fr, text(s, key, size, color), 540, y + (1 - p) * 34, a)

def cap_small(fr, t, s, t0, t1, color=GREY, size=40, y=1500):
    a, p = win(t, t0, t1, .35, .25)
    if a > 0: blit(fr, text(s, "bold", size, color), 540, y + (1 - p) * 20, a)

# ---------- option cards ----------
OPTS = [("أ", "مهاجم طويل… وكرات عرضية"), ("ب", "جناح سريع يثبت على الخط"), ("ج", "ارتكاز… وحافظ على التوازن")]
def draw_options(fr, t):
    if t < 10.0 or t > 15.8: return
    out = eio(pr(t, 15.1, 15.6))
    tx(fr, t, "قرارك؟", "black", 96, 540, 1030, 10.2, 15.4, LIME, anim="pop")
    # decision timer bar
    ba, _ = win(t, 11.0, 13.4, .3, .2)
    if ba > 0:
        bw = int(640 * (1 - pr(t, 11.2, 13.25)))
        blit(fr, rrect(640, 12, 6, (60, 110, 82)), 540, 1118, ba)
        if bw > 12: blit(fr, rrect(bw, 12, 6, GOLD), 540 + (640 - bw)/2, 1118, ba)
    pick = 13.3
    for i, (L, s) in enumerate(OPTS):
        t0 = 10.45 + i * .18
        a = eo(pr(t, t0, t0 + .4)) * (1 - out)
        if a <= 0: continue
        y = 1235 + i * 175 + (1 - eo(pr(t, t0, t0 + .45))) * 80 + out * 60
        sel = i == 1 and t >= pick
        dim = (i != 1) and t >= pick + .1
        fa = a * (1 - .55 * eo(pr(t, pick + .1, pick + .5))) if dim else a
        if sel:
            k = eo(pr(t, pick, pick + .25))
            sc = 1 + .04 * math.sin(pr(t, pick, pick + .35) * math.pi)
            blit(fr, glow(330, LIME, .5), 540, y, a * k)
            blit(fr, rrect(960, 150, 32, CARD, (74, 128, 98), 2), 540, y, fa * (1 - k), scale=sc)
            blit(fr, rrect(960, 150, 32, LIME), 540, y, a * k, scale=sc)
            blit(fr, text(s, "black", 46, NAVY), 900 - 18, y, a, anchor="r")
            blit(fr, disc(40, NAVY), 960, y, a)
            blit(fr, text(L, "black", 44, LIME), 960, y + 2, a)
            # check mark
            ck = eo(pr(t, pick + .15, pick + .45))
            if ck > 0:
                p0, p1, p2 = (110, y + 2), (138, y + 30), (190, y - 26)
                pts = [p0, p1, p2]
                draw_arrow(fr, pts, ck, NAVY, 9, head=False)
        else:
            blit(fr, rrect(960, 150, 32, CARD, (74, 128, 98), 2), 540, y, fa)
            blit(fr, text(s, "bold", 46, WHITE), 900 - 18, y, fa, anchor="r")
            blit(fr, disc(40, LIME, NAVY2, 0) if False else disc(40, (52, 104, 76)), 960, y, fa)
            blit(fr, text(L, "black", 44, LIME), 960, y + 2, fa)
    # tap ripple
    if pick - .05 < t < pick + .6:
        k = pr(t, pick - .05, pick + .6)
        cv2.circle(fr, (250, 1410), int(20 + 120 * eo(k)), WHITE, 4, cv2.LINE_AA)
        if k < .4: blit(fr, disc(26, WHITE), 250, 1410, (1 - k / .4) * .8)

# ---------- scenarios (when does it change) ----------
def draw_change(fr, t):
    if t < 22.0 or t > 27.6: return
    end = 27.5
    tx(fr, t, "متى يتغيّر القرار؟", "black", 84, 540, 430, 22.2, end, WHITE, anim="pop")
    tx(fr, t, "غيّر معطى واحدًا فقط…", "bold", 42, 540, 540, 22.6, end, GREY)
    rows = [("إذا كان التعادل يكفيك", "لاعب ارتكاز", 23.2, 760),
            ("إذا كان قلبا دفاعه قصيرين", "مهاجم طويل", 24.1, 960)]
    for cond, ans, t0, y in rows:
        a, p = win(t, t0, end, .4, .3)
        if a <= 0: continue
        yy = y + (1 - p) * 50
        blit(fr, rrect(960, 160, 34, CARD, (74, 128, 98), 2), 540, yy, a)
        blit(fr, text(cond, "bold", 42, WHITE), 1000, yy, a, anchor="r")
        k = eo(pr(t, t0 + .45, t0 + .9))
        if k > 0 and a > .6: draw_arrow(fr, [(470, yy), (lerp(470, 405, k), yy)], k, GREY, 5)
        ca = a * k
        at = text(ans, "black", 40, NAVY); ab = rrect(at.shape[1] + 30, 76, 38, GOLD)
        blit(fr, ab, 220, yy, ca, scale=lerp(.6, 1, eback(pr(t, t0 + .5, t0 + .9))))
        blit(fr, at, 220, yy, ca, scale=lerp(.6, 1, eback(pr(t, t0 + .5, t0 + .9))))
    # the line
    a, p = win(t, 25.2, end, .35, .3)
    if a > 0:
        blit(fr, text("الجواب يتغيّر", "black", 104, WHITE), 540, 1260 + (1 - p) * 30, a)
        # strike-through accent on "changes"
    a2, p2 = win(t, 25.95, end, .45, .3)
    if a2 > 0:
        blit(fr, text("أما طريقة القراءة فثابتة", "black", 84, LIME), 540, 1400, a2, clipw=eio(pr(t, 25.95, 26.45)))
        uw = int(700 * eio(pr(t, 26.35, 26.8)))
        if uw > 4: blit(fr, rrect(uw, 10, 5, LIME), 540 + (700 - uw)/2 - 0, 1478, a2)

# ---------- value ----------
def draw_value(fr, t):
    if t < 27.5 or t > 31.1: return
    end = 31.0
    tx(fr, t, "من كل حالة تخرج", "bold", 56, 540, 520, 27.6, end, GREY)
    tx(fr, t, "نسخة أفضل منك", "black", 120, 540, 660, 27.85, end, LIME, anim="pop")
    items = [("أسرع في القراءة", 28.5), ("أوضح في القرار", 29.0), ("أكثر ثقة أمام لاعبيك", 29.5)]
    for i, (s, t0) in enumerate(items):
        a, p = win(t, t0, end, .35, .3)
        if a <= 0: continue
        y = 930 + i * 190; xo = (1 - p) * 60
        blit(fr, rrect(900, 150, 34, CARD, (74, 128, 98), 2), 540 - xo * 0, y, a)
        sc = eback(pr(t, t0 + .1, t0 + .45))
        blit(fr, disc(42, LIME), 910, y, a, scale=max(.01, sc))
        if sc > .3:
            ck = [(892, y + 2), (906, y + 16), (930, y - 14)]
            draw_arrow(fr, ck, eo(pr(t, t0 + .25, t0 + .5)), NAVY, 7, head=False)
        blit(fr, text(s, "black", 58, WHITE), 838 + xo, y, a, anchor="r")

# ---------- logo / outro ----------
@functools.lru_cache(None)
def logo_icon(sz=230):
    ss = 4; S = sz * ss
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, S-1, S-1), 56*ss, fill=LIME + (255,))
    n = NAVY + (255,); w = 13*ss
    # mini board: half-line + box, two dots, an arrow = "read, then decide"
    d.rounded_rectangle((40*ss, 40*ss, S - 40*ss, S - 40*ss), 22*ss, outline=n, width=w//2)
    d.line((40*ss, S/2, S - 40*ss, S/2), fill=n, width=w//2)
    d.ellipse((70*ss, 140*ss, 100*ss, 170*ss), fill=n)
    d.ellipse((130*ss, 62*ss, 160*ss, 92*ss), fill=n)
    pts = [(98*ss, 142*ss), (122*ss, 118*ss), (140*ss, 98*ss)]
    d.line(pts, fill=n, width=w//2, joint="curve")
    return np.array(im.resize((sz, sz), Image.LANCZOS))

def make_logo():
    im = np.array(Image.open("/home/user/skills/coach-pro-reel/assets/logo.png").convert("RGB")).astype(np.float32)
    bg = im[5, 5].copy()
    im = im[236:950, 296:910]
    d = np.abs(im - bg).max(2)
    a = np.clip(d / 70.0, 0, 1)
    col = (im - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 1e-3)
    col = np.clip(col, 0, 255)
    out = np.zeros(im.shape[:2] + (4,), np.uint8)
    out[..., :3] = col.astype(np.uint8); out[..., 3] = (a * 255).astype(np.uint8)
    return out
LOGO = make_logo()

def draw_outro(fr, t):
    if t < 30.9: return
    end = DUR + 1
    a = eo(pr(t, 31.0, 31.5))
    sc = eback(pr(t, 31.0, 31.6), 1.3)
    blit(fr, glow(460, LIME, .16), 540, 800, a)
    blit(fr, LOGO, 540, 790, a, scale=max(.01, lerp(.7, 1.05, sc)))
    tx(fr, t, "منصة عربية لتطوير المدرب نفسه", "bold", 48, 540, 1235, 31.8, end, GREY)
    tx(fr, t, "لا لتزويده بتمارين جاهزة", "bold", 48, 540, 1300, 31.95, end, GREY)
    ca, p = win(t, 32.4, end, .4, .2)
    if ca > 0:
        ct = text("ابدأ أول حالة الآن", "black", 52, NAVY)
        pulse = 1 + .03 * math.sin((t - 32.4) * 4)
        sc = lerp(.7, 1, eback(pr(t, 32.4, 32.85))) * pulse
        blit(fr, rrect(560, 110, 55, LIME), 540, 1490, ca, scale=sc)
        blit(fr, ct, 540, 1490, ca, scale=sc)

# ---------- hook ----------
def draw_hook(fr, t):
    if t > 3.8: return
    end = 3.55
    # chalk line
    k = eio(pr(t, 0.0, 0.5)) * (1 - eio(pr(t, 3.2, 3.55)))
    if k > 0:
        x1 = int(lerp(980, 100, eio(pr(t, 0, .5))))
        cv2.line(fr, (980, 640), (x1, 640), LIME, 6, cv2.LINE_AA)
    a, p = win(t, 0.15, end, .3, .25)
    if a > 0:
        blit(fr, text("الدقيقة", "bold", 60, GREY), 540, 760, a)
        m = int(round(lerp(60, 78, eo(pr(t, 0.2, 1.0)))))
        blit(fr, text(f"{m}'", "inter", 230, WHITE), 540, 930, a, scale=lerp(.85, 1, eback(pr(t, .15, .55))))
    a, p = win(t, 1.0, end, .3, .25)
    if a > 0:
        tt = text("متأخر بهدف", "black", 58, WHITE); bb = rrect(tt.shape[1] + 40, 92, 46, CORAL)
        sc = lerp(.6, 1, eback(pr(t, 1.0, 1.35)))
        blit(fr, bb, 540, 1100, a, scale=sc); blit(fr, tt, 540, 1100, a, scale=sc)
    tx(fr, t, "والدكة تنتظر إشارتك", "bold", 56, 540, 1225, 1.5, end, GREY)
    a, p = win(t, 2.1, end, .3, .25)
    if a > 0:
        blit(fr, text("ماذا ستفعل؟", "black", 128, LIME), 540, 1400, a, scale=lerp(1.25, 1, eo(pr(t, 2.1, 2.4))))

# ---------- transitions ----------
WIPES = [3.45, 9.95, 15.3, 21.85, 27.35, 30.85]
def draw_wipes(fr, t):
    for w0 in WIPES:
        k = (t - w0) / .55
        if 0 <= k <= 1:
            for off, col, wd in ((0, LIME, 170), (-.12, WHITE, 26)):
                kk = eio(cl(k + off))
                cx = lerp(-500, W + 500, kk)
                pts = np.array([[cx - wd, H + 100], [cx + wd, H + 100], [cx + wd + 700, -100], [cx - wd + 700, -100]], np.int32)
                pts[:, 0] -= 350
                cv2.fillPoly(fr, [pts], col, cv2.LINE_AA)

# ---------- S2 / S4 captions ----------
def draw_captions(fr, t):
    cap_small(fr, t, "اقرأ اللوح كما تقرؤه على خط الملعب", 4.6, 5.15, GREY, 42, 1560)
    cap(fr, t, "النتيجة", 5.2, 6.0)
    cap(fr, t, "الدقيقة", 6.0, 6.85)
    cap(fr, t, "شكل الفريقين", 6.85, 7.75)
    cap(fr, t, "ومن على الدكة", 7.75, 8.85, y=1600)
    cap(fr, t, "اقرأ… ثم قرّر", 8.9, 9.95, LIME, 92)
    # why
    cap_small(fr, t, "لماذا هذا القرار؟", 16.0, 21.8, LIME, 44, 1380)
    cap(fr, t, "الخصم متكتّل في العمق", 16.15, 17.5, size=66, y=1520)
    cap(fr, t, "الجناح على الخط يجبره على التوسّع", 17.5, 18.95, size=58, y=1520)
    cap(fr, t, "فتنفتح المساحة بين الظهير وقلب الدفاع", 18.95, 20.4, size=54, y=1520)
    cap(fr, t, "العرض يصنع العمق", 20.4, 21.8, LIME, 84, 1525)
    cap_small(fr, t, "وهنا لا تنتهي الحالة…", 20.7, 21.8, GREY, 40, 1640)

SPLIT, STRETCH = 15.0, 1.45
OUT_DUR = SPLIT + (DUR - SPLIT) * STRETCH
def remap(T):  # output time -> internal time
    return T if T <= SPLIT else SPLIT + (T - SPLIT) / STRETCH

def frame(t):
    fr = BG.copy()
    draw_hook(fr, t)
    draw_board(fr, t)
    draw_header(fr, t)
    draw_bench(fr, t)
    draw_captions(fr, t)
    draw_options(fr, t)
    draw_change(fr, t)
    draw_value(fr, t)
    draw_outro(fr, t)
    draw_wipes(fr, t)
    fo = pr(t, DUR - .5, DUR)
    if fo > 0: fr = (fr * (1 - fo)).astype(np.uint8)
    fi = 1 - pr(t, 0, .2)
    if fi > 0: fr = (fr * (1 - fi)).astype(np.uint8)
    return fr

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        for ts in sys.argv[2:]:
            Image.fromarray(frame(remap(float(ts)))).save(f"/tmp/reel/prev_{ts}.jpg", quality=88)
        sys.exit()
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reel/video.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                          "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    n = int(OUT_DUR * FPS)
    for i in range(n):
        p.stdin.write(frame(remap(i / FPS)).tobytes())
        if i % 150 == 0: print(i, "/", n, flush=True)
    p.stdin.close(); p.wait()

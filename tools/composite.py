# Composite the workshop photo onto the brand backdrop.
# usage: python3 tools/composite.py <source.jpg> <rembg-birefnet-general-cutout.png> <out.jpg>
import sys, numpy as np
from PIL import Image, ImageFilter
src_p, cut_p, out_p = sys.argv[1:4]
I = np.asarray(Image.open(src_p).convert("RGB")).astype(np.float32) / 255
A = np.asarray(Image.open(cut_p).convert("RGBA"))[..., 3].astype(np.float32) / 255
H, W = A.shape
A[:, :178] = 0
A[400:, :215] = 0                                   # hanging bag / foreground arm
y0, y1, x0, x1 = 330, 470, 480, 500              # glass-edge sliver by the first woman's arm
reg = I[y0:y1, x0:x1].mean(2)
A[y0:y1, x0:x1][reg > .45] = 0

def blur(a, r):
    im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r))).astype(np.float32) / 255
def blur3(a, r):
    return np.dstack([blur(a[..., i], r) for i in range(3)])

# clean plate: background colour estimated around the subject (normalised convolution)
m = (A < .05).astype(np.float32)
B = blur3(I * m[..., None], 18) / np.maximum(blur(m, 18), 1e-3)[..., None]
# key out blown-out window light caught in the last woman's hair
lum0 = I.mean(2); sat0 = I.max(2) - I.min(2)
band = (blur(A, 10) < .985) & (A > 0)
wk = np.clip((lum0 - .58) / .18, 0, 1) * np.clip((.16 - sat0) / .1, 0, 1)
reg = np.zeros_like(A, bool); reg[:, 1330:] = True
wk[:205] *= .55
core = np.asarray(Image.fromarray((A * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(21))).astype(np.float32) / 255
A = np.where(band & reg, np.maximum(A * (1 - wk), core), A)
A = np.where(reg, blur(A, 1.2), A)
# tighten the matte a touch, then recover true foreground colour on the edges
A = np.clip((A - .2) / .8, 0, 1)
a = A[..., None]
F = np.where(a > .02, np.clip((I - (1 - a) * B) / np.maximum(a, .25), 0, 1), I)
F = np.where(a > .97, I, F)

# ---- backdrop (canvas 1500×682; subject offset) ----
CW, CH, ox, oy = 1500, 630, -110, 62
yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
def hexc(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
top, mid = hexc("#1e0b45"), hexc("#3b1a85")
t = (yy / CH)[..., None]
bg = top * (1 - t) + mid * t
def glow(cx, cy, rx, ry, col, k):
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return (np.exp(-d * 2.2) * k)[..., None] * col
bg = bg + glow(760, 290, 760, 400, hexc("#b9a2ff"), .55) + glow(1230, 90, 420, 260, hexc("#f7cf5d"), .22) + glow(170, 520, 380, 260, hexc("#7c4dff"), .35)
# soft bokeh in the back
rng = np.random.default_rng(4)
bok = np.zeros((CH, CW, 3), np.float32)
for _ in range(26):
    cx, cy, r = rng.uniform(0, CW), rng.uniform(0, CH * .62), rng.uniform(14, 46)
    col = hexc(rng.choice(["#fff3c4", "#f7cf5d", "#d9c8ff", "#ffffff"]))
    disk = (((xx - cx) ** 2 + (yy - cy) ** 2) < r * r).astype(np.float32)
    bok += disk[..., None] * col * rng.uniform(.08, .22)
bg = bg + blur3(np.clip(bok, 0, 1), 9)
vig = 1 - .35 * (((xx - CW / 2) / (CW * .62)) ** 2 + ((yy - CH * .45) / (CH * .9)) ** 2)
bg = np.clip(bg * vig[..., None], 0, 1)

# place subject on canvas
Fc = np.zeros((CH, CW, 3), np.float32); Ac = np.zeros((CH, CW), np.float32)
sx0, sy0 = max(0, -ox), 0
cx0, cy0 = max(0, ox), oy
w = min(W - sx0, CW - cx0); h = min(H, CH - cy0)
Fc[cy0:cy0 + h, cx0:cx0 + w] = F[sy0:sy0 + h, sx0:sx0 + w]
Ac[cy0:cy0 + h, cx0:cx0 + w] = A[sy0:sy0 + h, sx0:sx0 + w]

# grade subject into the scene: gentle violet in the shadows + lilac key from the backdrop
lum = Fc.mean(2, keepdims=True)
Fc = Fc * .93 + hexc("#2a1260") * (1 - lum) * .07 + hexc("#d9c8ff") * lum * .04
# light wrap: backdrop light bleeding over the subject's edges
inner = blur(Ac, 7)
wrap = np.clip((1 - inner) * Ac * 1.6, 0, 1)[..., None]
Fc = Fc * (1 - .3 * wrap) + blur3(bg, 22) * .3 * wrap
# soft ambient shadow behind them, so they sit in the space
sh = blur(Ac, 28)[..., None]
bg = bg * (1 - .28 * sh)

ac = Ac[..., None]
out = Fc * ac + bg * (1 - ac)
out = out + rng.normal(0, .014, out.shape).astype(np.float32)     # shared grain
Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_p, quality=92)

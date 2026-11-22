import numpy as np, wave
SR = 44100; DUR = 35.0; N = int(SR * DUR)
rng = np.random.default_rng(7)
L = np.zeros(N); R = np.zeros(N)

def add(sig, t, gain=1.0, pan=0.0):
    i = int(t * SR); j = min(N, i + len(sig))
    if j <= i: return
    s = sig[:j - i] * gain
    L[i:j] += s * (1 - max(0, pan)); R[i:j] += s * (1 + min(0, pan))

def env(n, a, d):
    e = np.ones(n); ai = int(a * SR); di = int(d * SR)
    if ai: e[:ai] = np.linspace(0, 1, ai)
    if di: e[-di:] *= np.linspace(1, 0, di)
    return e

def lp(x, k):  # simple one-pole low-pass, k in (0,1)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)): acc += k * (x[i] - acc); y[i] = acc
    return y

def tt(d): return np.arange(int(d * SR)) / SR

# ---- sounds ----
def kick():
    t = tt(.35); f = 48 + 90 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
def hat():
    t = tt(.06); n = rng.standard_normal(len(t)); n = n - lp(n, .35)
    return n * np.exp(-t * 70) * .5
def clap():
    t = tt(.18); n = rng.standard_normal(len(t)); n = n - lp(n, .15)
    return n * np.exp(-t * 22) * .5
def tick():
    t = tt(.05); return (np.sin(2*np.pi*2200*t) + .5*np.sin(2*np.pi*3300*t)) * np.exp(-t * 120) * .5
def whoosh(d=.55, up=True):
    t = tt(d); n = rng.standard_normal(len(t))
    k = np.linspace(.03, .45, len(t)) if up else np.linspace(.45, .03, len(t))
    y = np.empty_like(n); acc = 0
    for i in range(len(n)): acc += k[i] * (n[i] - acc); y[i] = acc
    return y * np.sin(np.pi * t / d) ** 2 * 1.3
def pop(f0=900):
    t = tt(.12); f = f0 * np.exp(-t * 18) + 200
    return np.sin(2*np.pi*np.cumsum(f)/SR) * np.exp(-t * 30)
def click():
    t = tt(.04); return rng.standard_normal(len(t)) * np.exp(-t * 200) * .6 + np.sin(2*np.pi*1500*t)*np.exp(-t*150)*.5
def chime(notes=(880, 1108.7, 1318.5)):
    out = np.zeros(int(1.2 * SR))
    for i, f in enumerate(notes):
        t = tt(1.2 - i * .07); s = (np.sin(2*np.pi*f*t) + .3*np.sin(4*np.pi*f*t)) * np.exp(-t * 4)
        o = int(i * .07 * SR); out[o:o+len(s)] += s * .35
    return out
def impact():
    t = tt(1.4); f = 40 + 60 * np.exp(-t * 6)
    b = np.sin(2*np.pi*np.cumsum(f)/SR) * np.exp(-t * 3)
    n = rng.standard_normal(len(t)); n = lp(n, .08) * np.exp(-t * 5) * 2
    return b + n
def riser(d):
    t = tt(d); f = 200 * (4 ** (t / d))
    s = np.sin(2*np.pi*np.cumsum(f)/SR) * .15 + lp(rng.standard_normal(len(t)), .2) * .5
    return s * (t / d) ** 2

# ---- music bed: 100 bpm, A minor, Am-F-C-G ----
BPM = 100; beat = 60 / BPM
chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (130.81, 196.0, 261.63 * 1), (196.0, 246.94, 293.66)]
bass = [110.0, 87.31, 65.41, 98.0]
def pad(freqs, d):
    t = tt(d); s = np.zeros(len(t))
    for f in freqs:
        for det in (-0.6, 0.6):
            ph = 2*np.pi*(f + det)*t
            s += (2 * ((ph / (2*np.pi)) % 1) - 1) * .5  # saw
    s = lp(s, .04)
    return s * env(len(t), .3, .5) * .08
bar = 4 * beat
t0 = 0.0; i = 0
while t0 < DUR:
    c = chords[i % 4]
    g = .55 if t0 < 3.6 else 1.0
    add(pad(c, bar + .4), t0, g)
    bt = tt(bar); bs = np.sin(2*np.pi*bass[i % 4]*bt) * .9 + np.sin(4*np.pi*bass[i % 4]*bt) * .2
    bs *= env(len(bt), .02, .2) * .16
    if t0 >= 3.4: add(bs, t0)
    t0 += bar; i += 1

def drop(t): return 25.1 <= t < 26.0 or t >= 34.0
b = 0; t = 3.6
while t < DUR:
    if not drop(t):
        if b % 4 in (0, 2) or (t > 15.4 and b % 2 == 0): add(kick(), t, .7)
        add(hat(), t + beat / 2, .35, .2)
        if b % 4 in (1, 3): add(clap(), t, .3, -.1)
        if t > 27.4: add(hat(), t, .22, -.2)
    t += beat; b += 1
# clock ticks in the hook and under the decision timer
for k in range(8): add(tick(), .2 + k * .45, .55)
for k in range(9): add(tick(), 11.2 + k * .25, .45 + k * .03)

# ---- sfx on the picture ----
for w in [3.45, 9.95, 15.3, 21.85, 27.35, 30.85]: add(whoosh(), w - .05, .55)
add(impact(), 0.15, .7); add(pop(700), 1.0, .5); add(impact(), 2.1, .5)
for k in range(11): add(pop(1100), 4.25 + k * .045, .12, .3)
for k in range(10): add(pop(800), 4.75 + k * .045, .12, -.3)
for tt_ in (5.2, 6.0, 6.85, 7.75): add(click(), tt_, .5)
for k in range(3): add(pop(1000), 7.9 + k * .12, .35)
add(riser(1.0), 8.9, .5)
add(pop(600), 10.2, .5)
for k in range(3): add(whoosh(.3), 10.45 + k * .18, .25)
add(click(), 13.3, .9); add(chime(), 13.35, .6)
add(pop(900), 16.6, .5); add(whoosh(.8, False), 16.9, .35); add(whoosh(.9), 18.1, .3); add(pop(1300), 19.2, .4)
add(chime((659.25, 830.6, 987.8)), 20.4, .4)
add(pop(600), 22.2, .45); add(pop(900), 23.2, .35); add(pop(900), 24.1, .35)
add(impact(), 25.2, .8); add(chime((523.25, 659.25, 783.99)), 25.95, .45)
for k, tt_ in enumerate((28.5, 29.0, 29.5)): add(pop(900 + k*150), tt_, .45)
add(impact(), 31.0, .6); add(chime((440, 554.37, 659.25, 880)), 31.3, .5); add(click(), 32.5, .5)

mix = np.stack([L, R], 1)
fade = np.ones(N); fl = int(1.2 * SR); fade[-fl:] = np.linspace(1, 0, fl)
mix *= fade[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
pcm = (mix * 32767).astype(np.int16)
with wave.open("/tmp/reel/audio.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("audio ok")

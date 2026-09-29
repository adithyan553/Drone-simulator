"""Drone Flight Training Simulator  -  CITY EDITION
Python + Ursina (Panda3D).   Run:  python main.py

World: procedural city (roads, sidewalks, towers, houses, trees, street lights),
traffic with working traffic lights, pedestrians, real-time shadows, day / sunset /
night, wind + gusts, battery, altitude-hold assist, minimap, rotor sound.

Graphics quality knobs are just below (SHADOWS / SHADOW_RES / counts).
"""
import math
import os
import random
import struct
import wave
from pathlib import Path

try:  # anti-aliasing (must be set before the window is created)
    from panda3d.core import loadPrcFileData
    loadPrcFileData('', 'framebuffer-multisample 1\nmultisamples 4\n')
    loadPrcFileData('', 'sync-video 1\n')
except Exception:
    pass

from ursina import *
from PIL import Image, ImageDraw

try:
    from ursina.shaders import lit_with_shadows_shader as _LIT_SHADER
    from ursina.shaders import unlit_shader as _UNLIT_SHADER
except Exception:
    _LIT_SHADER = _UNLIT_SHADER = None

# ----------------------------------------------------------------------------
# QUALITY SETTINGS  (lower these if your laptop struggles)
# ----------------------------------------------------------------------------
USE_LIT = True          # False = safe mode: no lighting shaders at all (flat but always works)
SHADOWS = True          # real-time sun shadows  (False = much faster)
SHADOW_RES = 4096       # 2048 for weaker GPUs
CARS_PER_LANE = 2       # 32 lanes  ->  64 cars
NUM_PEDS = 56
NUM_RINGS = 8

# ----------------------------------------------------------------------------
# Flight constants
# ----------------------------------------------------------------------------
G = 9.81
MAX_THRUST_G = 2.0          # full throttle = 2 g  ->  hover at 50 %
DRAG = 0.5
MAX_TILT = 30.0
ATTITUDE_RESPONSE = 6.0
YAW_RATE = 100.0
THROTTLE_RATE = 0.6
GROUND_OFF = 0.28           # drone centre height above the surface it rests on
CRASH_SPEED = 4.5
CRASH_TILT = 15.0
DRONE_RADIUS = 0.45
MAX_ALT = 150.0
PAD_HALF = 4.0
RING_RADIUS = 4.0

# ----------------------------------------------------------------------------
# City layout constants
# ----------------------------------------------------------------------------
EXT = 200.0                                   # city half-size (m)
ROADS = [-111.0, -37.0, 37.0, 111.0]          # road centre lines (both axes)
HALF_ROAD = 7.0
ROAD_Y, MARK_Y, SLAB_H = 0.06, 0.10, 0.18
TILE_W, TILE_H = 16.0, 14.4                   # facade texture = 4 bays x 4 floors
LANE_OFF = (1.8, 5.0)

PTS = [-EXT]
for _c in ROADS:
    PTS += [_c - HALF_ROAD, _c + HALF_ROAD]
PTS.append(EXT)
BLOCK_IV = [(PTS[2 * i], PTS[2 * i + 1]) for i in range(len(ROADS) + 1)]

CAM_NAMES = ['CHASE', 'FPV', 'ORBIT']
rnd = random.Random(11)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def hsv(h, s, v, a=1):
    return color.hsv(h, s, clamp(v, 0, 1), a)


def flat_dist(ax, az, bx, bz):
    return math.hypot(ax - bx, az - bz)


# ----------------------------------------------------------------------------
# App / window
# ----------------------------------------------------------------------------
app = Ursina()
window.title = 'Drone Flight Training Simulator - City Edition'
window.borderless = False
window.exit_button.visible = False
window.fps_counter.enabled = True
camera.fov = 80
camera.clip_plane_near = 0.5
sky = Sky()

LIT = None
sun = ambient = None
if _LIT_SHADER is not None and USE_LIT:
    try:
        sun = DirectionalLight(shadows=SHADOWS, shadow_map_resolution=Vec2(SHADOW_RES, SHADOW_RES))
        ambient = AmbientLight()
        LIT = _LIT_SHADER
    except Exception as ex:
        print('Lighting unavailable, using baked shading only:', ex)
        sun = ambient = None
        LIT = None


def WE(**kw):
    """World entity (lit shader when available)."""
    if LIT is not None and 'shader' not in kw:
        kw['shader'] = LIT
    return Entity(**kw)


def UE(**kw):
    """Unlit (self-illuminated) entity."""
    if _UNLIT_SHADER is not None and 'shader' not in kw:
        kw['shader'] = _UNLIT_SHADER
    return Entity(**kw)


# ----------------------------------------------------------------------------
# Mesh builder (everything static is merged into a few big meshes = fast)
# ----------------------------------------------------------------------------
class MB:
    def __init__(self):
        self.v, self.uv, self.n, self.c = [], [], [], []

    def tri(self, p, n, c, uv=None):
        self.v.extend(p)
        self.n.extend(n)
        self.c.extend(c)
        self.uv.extend(uv or ((0, 0), (0, 0), (0, 0)))

    def quad(self, a, b, c, d, n, col, uv=None):
        if uv is None:
            uv = ((0, 0), (1, 0), (1, 1), (0, 1))
        cc = (col, col, col)
        nn = (n, n, n)
        self.tri((a, b, c), nn, cc, (uv[0], uv[1], uv[2]))
        self.tri((a, c, d), nn, cc, (uv[0], uv[2], uv[3]))

    def build(self):
        return Mesh(vertices=[Vec3(*p) for p in self.v],
                    uvs=[tuple(u) for u in self.uv],
                    normals=[Vec3(*n) for n in self.n],
                    colors=list(self.c), mode='triangle')


def add_box(mb, cx, cy, cz, sx, sy, sz, col, yaw=0.0, parts='all', uvt=None, uoff=0.0, bottom=False):
    hx, hy, hz = sx * .5, sy * .5, sz * .5
    a = math.radians(yaw)
    ex = (math.cos(a), 0.0, -math.sin(a))
    ez = (math.sin(a), 0.0, math.cos(a))
    nex = (-ex[0], 0.0, -ex[2])
    nez = (-ez[0], 0.0, -ez[2])

    def P(q):
        lx, ly, lz = q
        return (cx + ex[0] * lx + ez[0] * lz, cy + ly, cz + ex[2] * lx + ez[2] * lz)

    faces = (
        (ex, ((hx, -hy, -hz), (hx, -hy, hz), (hx, hy, hz), (hx, hy, -hz)), .86, 'w', sz, sy),
        (nex, ((-hx, -hy, hz), (-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz)), .86, 'w', sz, sy),
        (ez, ((hx, -hy, hz), (-hx, -hy, hz), (-hx, hy, hz), (hx, hy, hz)), .74, 'w', sx, sy),
        (nez, ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz)), .74, 'w', sx, sy),
        ((0, 1, 0), ((-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz)), 1.0, 't', sx, sz),
        ((0, -1, 0), ((-hx, -hy, hz), (hx, -hy, hz), (hx, -hy, -hz), (-hx, -hy, -hz)), .45, 'b', sx, sz),
    )
    for n, cs, shade, kind, l1, l2 in faces:
        if kind == 'b' and not bottom:
            continue
        if parts == 'walls' and kind != 'w':
            continue
        if parts == 'top' and kind != 't':
            continue
        uv = None
        if uvt:
            u1, v1 = l1 / uvt[0], l2 / uvt[1]
            uv = ((uoff, 0), (uoff + u1, 0), (uoff + u1, v1), (uoff, v1))
        mb.quad(P(cs[0]), P(cs[1]), P(cs[2]), P(cs[3]), n, hsv(col[0], col[1], col[2] * shade), uv)


def add_ellipsoid(mb, cx, cy, cz, rx, ry, rz, col, seg=8, rings=5, k=0.35):
    grid = []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        row = []
        for j in range(seg + 1):
            th = 2 * math.pi * j / seg
            nx, ny, nz = math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th)
            v = col[2] * (1 - k + k * (ny * .5 + .5) * 1.6)
            row.append(((cx + nx * rx, cy + ny * ry, cz + nz * rz), (nx, ny, nz), hsv(col[0], col[1], v)))
        grid.append(row)
    for i in range(rings):
        for j in range(seg):
            a, b, c, d = grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]
            mb.tri((a[0], b[0], c[0]), (a[1], b[1], c[1]), (a[2], b[2], c[2]))
            mb.tri((a[0], c[0], d[0]), (a[1], c[1], d[1]), (a[2], c[2], d[2]))


def add_torus(mb, R, r, col, seg=40, tube=8):
    def pt(i, j):
        u, w = 2 * math.pi * i / seg, 2 * math.pi * j / tube
        cu, su, cw, sw = math.cos(u), math.sin(u), math.cos(w), math.sin(w)
        return ((R + r * cw) * cu, (R + r * cw) * su, r * sw), (cw * cu, cw * su, sw)
    c = hsv(*col)
    for i in range(seg):
        for j in range(tube):
            a, b, cc, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            mb.tri((a[0], b[0], cc[0]), (a[1], b[1], cc[1]), (c, c, c))
            mb.tri((a[0], cc[0], d[0]), (a[1], cc[1], d[1]), (c, c, c))


def add_pyramid(mb, cx, cy, cz, sx, sz, h, col, yaw=0.0):
    a = math.radians(yaw)
    ex = (math.cos(a), 0.0, -math.sin(a))
    ez = (math.sin(a), 0.0, math.cos(a))

    def P(lx, ly, lz):
        return (cx + ex[0] * lx + ez[0] * lz, cy + ly, cz + ex[2] * lx + ez[2] * lz)
    hx, hz = sx / 2, sz / 2
    base = [P(-hx, 0, -hz), P(hx, 0, -hz), P(hx, 0, hz), P(-hx, 0, hz)]
    apex = P(0, h, 0)
    for i in range(4):
        p0, p1 = base[i], base[(i + 1) % 4]
        u = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        w = (apex[0] - p0[0], apex[1] - p0[1], apex[2] - p0[2])
        n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        mid = ((p0[0] + p1[0]) / 2 - cx, 0, (p0[2] + p1[2]) / 2 - cz)
        if n[0] * mid[0] + n[2] * mid[2] < 0:
            n = (-n[0], -n[1], -n[2])
        ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
        n = (n[0] / ln, n[1] / ln, n[2] / ln)
        c = hsv(col[0], col[1], col[2] * (0.75 + 0.12 * i))
        mb.tri((p0, p1, apex), (n, n, n), (c, c, c))


def rect2d(mb, x0, z0, x1, z1, col):
    c = hsv(*col)
    mb.quad((x0, z0, 0), (x1, z0, 0), (x1, z1, 0), (x0, z1, 0), (0, 0, -1), c)


def mesh_entity(mb, tex=None, lit=True, **kw):
    e = Entity(model=mb.build(), double_sided=True, **kw)
    if tex is not None:
        e.texture = tex
    if lit and LIT is not None:
        e.shader = LIT
    elif not lit and _UNLIT_SHADER is not None:
        e.shader = _UNLIT_SHADER
    return e


# ----------------------------------------------------------------------------
# Procedural textures (PIL)
# ----------------------------------------------------------------------------
def _c8(v):
    return int(clamp(v, 0, 255))


def make_textures():
    r = random.Random(3)

    def noise(size, base, var):
        data = []
        for _ in range(size * size):
            d = r.randint(-var, var)
            data.append((_c8(base[0] + d), _c8(base[1] + d), _c8(base[2] + d)))
        im = Image.new('RGB', (size, size))
        im.putdata(data)
        return im

    def mk(img):
        t = Texture(img)
        try:
            t.filtering = 'mipmap'
        except Exception:
            pass
        return t

    asphalt = noise(256, (66, 66, 70), 11)
    grass = noise(256, (84, 132, 62), 16)
    side = noise(128, (178, 176, 170), 5)
    d = ImageDraw.Draw(side)
    d.rectangle([0, 0, 127, 127], outline=(146, 144, 138))
    d.line([(64, 0), (64, 127)], fill=(158, 156, 150))
    d.line([(0, 64), (127, 64)], fill=(158, 156, 150))

    def facade(night):
        im = Image.new('RGB', (512, 512), (38, 40, 52) if night else (236, 233, 225))
        dr = ImageDraw.Draw(im)
        fr = random.Random(5)
        for fy in range(4):
            for fx in range(4):
                x0, y0 = fx * 128, fy * 128
                lit = fr.random() < 0.45
                if night:
                    glass = (255, 214, 128) if lit else (20, 24, 36)
                else:
                    tone = fr.randint(-10, 12)
                    glass = (52 + tone, 74 + tone, 100 + tone)
                dr.rectangle([x0 + 22, y0 + 26, x0 + 106, y0 + 96], fill=glass)
                if not night:
                    dr.rectangle([x0 + 22, y0 + 26, x0 + 106, y0 + 50],
                                 fill=(glass[0] + 28, glass[1] + 32, glass[2] + 34))
                    dr.rectangle([x0 + 18, y0 + 96, x0 + 110, y0 + 101], fill=(205, 203, 196))
                    dr.line([(x0 + 64, y0 + 26), (x0 + 64, y0 + 96)], fill=(190, 190, 186), width=2)
                    dr.rectangle([x0, y0 + 118, x0 + 127, y0 + 127], fill=(214, 211, 203))
                else:
                    dr.line([(x0 + 64, y0 + 26), (x0 + 64, y0 + 96)], fill=(30, 32, 44), width=2)
        return im

    return {
        'asphalt': mk(asphalt), 'grass': mk(grass), 'side': mk(side),
        'day': mk(facade(False)), 'night': mk(facade(True)),
    }


TEX = make_textures()

# ----------------------------------------------------------------------------
# Sky extras: sun/moon disc, drifting clouds, distant hills
# ----------------------------------------------------------------------------
sun_disc = UE(model='sphere', scale=70, color=hsv(50, .15, 1))

hills_mb = MB()
for _i in range(30):
    _a = 2 * math.pi * _i / 30 + rnd.uniform(-.05, .05)
    _r = rnd.uniform(950, 1150)
    add_ellipsoid(hills_mb, math.sin(_a) * _r, -40, math.cos(_a) * _r,
                  rnd.uniform(220, 380), rnd.uniform(90, 200), rnd.uniform(220, 380),
                  (215, .25, .55), seg=10, rings=5, k=0.15)
hills = mesh_entity(hills_mb, lit=False)

cloud_mb = MB()
_period = 480.0
_pat = []
for _ in range(10):
    _pat.append((rnd.uniform(0, _period), rnd.uniform(-520, 520), rnd.uniform(150, 200), rnd.uniform(26, 48)))
for _off in (-960, -480, 0, 480, 960):
    for (_x, _z, _y, _r) in _pat:
        for _k in range(4):
            add_ellipsoid(cloud_mb, _x + _off + rnd.uniform(-_r, _r), _y + rnd.uniform(-4, 4),
                          _z + rnd.uniform(-_r * .6, _r * .6),
                          _r * rnd.uniform(.9, 1.4), _r * .32, _r * rnd.uniform(.6, .9),
                          (0, 0, .98), seg=8, rings=4, k=0.15)
clouds = mesh_entity(cloud_mb, lit=False)

# ----------------------------------------------------------------------------
# Ground, roads, markings
# ----------------------------------------------------------------------------
WE(model='plane', scale=(2600, 1, 2600), texture=TEX['grass'], texture_scale=(325, 325),
   color=hsv(0, 0, 1.0))

asph, marks = MB(), MB()
WHITE, YELLOW = (0, 0, .93), (48, .9, .95)


def _flat(mb, x0, z0, x1, z1, y, col, uv=None):
    mb.quad((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), (0, 1, 0), hsv(*col), uv)


def road_rect(axis, c, a0, a1, b0, b1):
    """axis 'x': road runs along x at z=c;  axis 'z': along z at x=c.   b = across-road offset."""
    if axis == 'x':
        return a0, c + b0, a1, c + b1
    return c + b0, a0, c + b1, a1


def mark(axis, c, a0, a1, b0, b1, col):
    x0, z0, x1, z1 = road_rect(axis, c, a0, a1, b0, b1)
    _flat(marks, min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1), MARK_Y, col)


for axis in ('x', 'z'):
    for c in ROADS:
        for i in range(len(PTS) - 1):
            s0, s1 = PTS[i], PTS[i + 1]
            inter = (i % 2 == 1)
            if inter and axis == 'z':
                continue
            x0, z0, x1, z1 = road_rect(axis, c, s0, s1, -HALF_ROAD, HALF_ROAD)
            x0, x1 = min(x0, x1), max(x0, x1)
            z0, z1 = min(z0, z1), max(z0, z1)
            _flat(asph, x0, z0, x1, z1, ROAD_Y, (0, 0, 1),
                  ((x0 / 8, z0 / 8), (x1 / 8, z0 / 8), (x1 / 8, z1 / 8), (x0 / 8, z1 / 8)))
            if inter:
                continue
            first, last = (i == 0), (i == len(PTS) - 2)
            a0 = s0 + (0 if first else 11.5)
            a1 = s1 - (0 if last else 11.5)
            mark(axis, c, a0, a1, -0.22, -0.07, YELLOW)
            mark(axis, c, a0, a1, 0.07, 0.22, YELLOW)
            for sgn in (-1, 1):
                mark(axis, c, a0, a1, sgn * 6.75, sgn * 6.9, WHITE)
                p = a0
                while p + 3 < a1:
                    mark(axis, c, p, p + 3, sgn * 3.45, sgn * 3.6, WHITE)
                    p += 8
            for edge, end in ((0, s0), (1, s1)):
                if (edge == 0 and first) or (edge == 1 and last):
                    continue
                sg = 1 if edge == 0 else -1     # direction pointing into the segment
                base = end
                # zebra crossing
                for k in range(-4, 5):
                    b = k * 1.45
                    p0, p1 = sorted((base + sg * 0.8, base + sg * 3.3))
                    mark(axis, c, p0, p1, b - 0.35, b + 0.35, WHITE)
                # stop line for the traffic that arrives at this end
                p0, p1 = sorted((base + sg * 3.8, base + sg * 4.4))
                # traffic heading toward the intersection (edge 0 = start side, arrives moving -)
                if axis == 'x':
                    bb = (-6.8, -0.4) if edge == 0 else (0.4, 6.8)
                else:
                    bb = (0.4, 6.8) if edge == 0 else (-6.8, -0.4)
                mark(axis, c, p0, p1, bb[0], bb[1], WHITE)

mesh_entity(asph, tex=TEX['asphalt'])
mesh_entity(marks)

# ----------------------------------------------------------------------------
# City blocks: sidewalks, buildings, houses, park, landing pads
# ----------------------------------------------------------------------------
walls_mb, roofs_mb, slab_mb, park_mb = MB(), MB(), MB(), MB()
trees_mb, poles_mb, lamps_mb, beacon_mb, pads_mb = MB(), MB(), MB(), MB(), MB()
BUILDINGS = []          # (x, z, w, d, h)
HOUSES = []             # (x, z, w, d) helper for tree placement
TOWER = (74.0, 74.0, 26.0, 26.0, 50.4)
PADS = [('HOME', Vec3(0, 0, 0)), ('PAD B', Vec3(TOWER[0], TOWER[4], TOWER[1]))]


def add_building(cx, cz, w, d, h, tint, details=True):
    hw = h - 0.6
    add_box(walls_mb, cx, hw / 2, cz, w, hw, d, tint, parts='walls', uvt=(TILE_W, TILE_H),
            uoff=rnd.choice((0, .25, .5, .75)))
    add_box(roofs_mb, cx, h - 0.3, cz, w + 0.5, 0.6, d + 0.5, (30, .05, .34))
    if details:
        for _ in range(rnd.randint(1, 3)):
            add_box(roofs_mb, cx + rnd.uniform(-w / 3, w / 3), h + 0.6, cz + rnd.uniform(-d / 3, d / 3),
                    2.2, 1.2, 1.6, (0, 0, .62), yaw=rnd.choice((0, 90)))
        if h > 40:
            add_box(roofs_mb, cx + w / 4, h + 4.5, cz - d / 4, 0.25, 9, 0.25, (0, 0, .5))
            add_box(beacon_mb, cx + w / 4, h + 9.2, cz - d / 4, 0.5, 0.5, 0.5, (0, .9, 1))
    BUILDINGS.append((cx, cz, w, d, h))


def add_helipad(x, y, z, col, tall_pole=True):
    add_box(pads_mb, x, y + .05, z, 9, .1, 9, col)
    for sx in (-1, 1):
        add_box(pads_mb, x, y + .11, z + sx * 3.7, 7.6, .02, .4, (0, 0, .95))
        add_box(pads_mb, x + sx * 3.7, y + .11, z, .4, .02, 7.6, (0, 0, .95))
        add_box(pads_mb, x + sx * 1.2, y + .11, z, .5, .02, 4, (0, 0, .95))
    add_box(pads_mb, x, y + .11, z, 2.4, .02, .5, (0, 0, .95))
    if tall_pole:
        add_box(beacon_mb, x + 4.2, y + 7, z + 4.2, 0.3, 14, 0.3, (col[0], .9, 1))
    for sx in (-1, 1):
        for sz in (-1, 1):
            add_box(beacon_mb, x + sx * 4.3, y + .5, z + sz * 4.3, 0.35, 1.0, 0.35, (0, .95, .95))


# --- park in the centre (block 2,2) with plaza + HOME pad
add_box(park_mb, 0, SLAB_H / 2, 0, 60, SLAB_H, 60, (0, 0, 1), uvt=(8, 8))
add_box(slab_mb, 0, SLAB_H + .03, 0, 24, .06, 24, (0, 0, 1), uvt=(4, 4))
add_helipad(0, SLAB_H + .06, 0, (205, .8, .9))

# --- city blocks (3 x 3 inner blocks except centre)
for i in range(1, 4):
    for j in range(1, 4):
        if (i, j) == (2, 2):
            continue
        x0, x1 = BLOCK_IV[i]
        z0, z1 = BLOCK_IV[j]
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        add_box(slab_mb, cx, SLAB_H / 2, cz, x1 - x0, SLAB_H, z1 - z0, (0, 0, 1), uvt=(4, 4))
        if (i, j) == (3, 3):                                   # helipad tower
            add_building(cx, cz, 44, 44, 7.2, (30, .08, .9), details=False)
            add_building(TOWER[0], TOWER[1], TOWER[2], TOWER[3], TOWER[4], (205, .12, .85), details=False)
            add_helipad(TOWER[0], TOWER[4], TOWER[1], (300, .7, .8))
            continue
        hue = rnd.choice((30, 40, 200, 210, 0, 25))
        if rnd.random() < 0.15:
            h = round(rnd.uniform(28, 60) / 3.6) * 3.6
            add_building(cx, cz, 44, 44, h, (hue, rnd.uniform(0, .2), rnd.uniform(.7, .95)))
        else:
            for sx in (-1, 1):
                for sz in (-1, 1):
                    if rnd.random() < 0.10:
                        continue
                    w = rnd.choice((16, 20, 24))
                    d = rnd.choice((16, 20, 24))
                    h = round(rnd.uniform(14, 62) / 3.6) * 3.6
                    hue = rnd.choice((30, 40, 200, 210, 0, 25))
                    add_building(cx + sx * 14, cz + sz * 14, w, d, h,
                                 (hue, rnd.uniform(0, .22), rnd.uniform(.66, .95)))

# --- suburb: 16 outer blocks -> houses
for i in range(5):
    for j in range(5):
        if 1 <= i <= 3 and 1 <= j <= 3:
            continue
        x0, x1 = BLOCK_IV[i]
        z0, z1 = BLOCK_IV[j]
        placed = []
        for _ in range(rnd.randint(3, 5)):
            for _try in range(40):
                w, d = rnd.choice((8, 12)), rnd.choice((8, 12))
                x = rnd.uniform(x0 + 10, x1 - 10)
                z = rnd.uniform(z0 + 10, z1 - 10)
                if all(abs(x - p[0]) > (w + p[2]) / 2 + 8 or abs(z - p[1]) > (d + p[3]) / 2 + 8 for p in placed):
                    placed.append((x, z, w, d))
                    hue = rnd.choice((30, 45, 15, 0, 200))
                    add_box(walls_mb, x, 3.3, z, w, 6.6, d, (hue, rnd.uniform(.05, .25), rnd.uniform(.75, .95)),
                            parts='walls', uvt=(TILE_W, TILE_H), uoff=rnd.choice((0, .25, .5)))
                    add_pyramid(roofs_mb, x, 6.6, z, w + 1.6, d + 1.6, 3.0,
                                (rnd.choice((8, 15, 210)), .45, .5), yaw=rnd.choice((0, 90)))
                    BUILDINGS.append((x, z, w, d, 6.6))
                    HOUSES.append((x, z, w, d))
                    break


def tree(x, z, s=1.0):
    add_box(trees_mb, x, 1.1 * s, z, 0.4 * s, 2.2 * s, 0.4 * s, (25, .6, .35))
    add_ellipsoid(trees_mb, x, 3.6 * s, z, 1.9 * s, 2.3 * s, 1.9 * s,
                  (rnd.uniform(85, 130), rnd.uniform(.5, .75), rnd.uniform(.42, .62)), seg=7, rings=4)


def near_building(x, z, m=3.0):
    for bx, bz, w, d, h in BUILDINGS:
        if abs(x - bx) < w / 2 + m and abs(z - bz) < d / 2 + m:
            return True
    return False


# street trees + street lights
LAMP_POS = []
for axis in ('x', 'z'):
    for c in ROADS:
        p = -EXT + 8
        while p < EXT - 5:
            if not any(abs(p - cc) < 13 for cc in ROADS):
                for sgn in (-1, 1):
                    b = sgn * (8.5 + rnd.uniform(-.3, .3))
                    tx, tz = (p, c + b) if axis == 'x' else (c + b, p)
                    if not near_building(tx, tz, 1.0):
                        tree(tx, tz, rnd.uniform(.85, 1.15))
            p += 18
        p = -EXT + 20
        while p < EXT - 10:
            if not any(abs(p - cc) < 13 for cc in ROADS):
                for sgn in (-1, 1):
                    b = sgn * 7.7
                    lx, lz = (p, c + b) if axis == 'x' else (c + b, p)
                    add_box(poles_mb, lx, 4.2, lz, 0.22, 8.4, 0.22, (0, 0, .35))
                    ax_, az_ = (0, -sgn * 1.4) if axis == 'x' else (-sgn * 1.4, 0)
                    add_box(poles_mb, lx + ax_, 8.3, lz + az_, 0.2 + abs(ax_) * 2.4, 0.15, 0.2 + abs(az_) * 2.4, (0, 0, .35))
                    add_box(lamps_mb, lx + ax_ * 1.9, 8.15, lz + az_ * 1.9, 0.6, 0.18, 0.6, (45, .3, 1))
            p += 30

for _ in range(34):                               # park trees
    x, z = rnd.uniform(-27, 27), rnd.uniform(-27, 27)
    if abs(x) > 14 or abs(z) > 14:
        tree(x, z, rnd.uniform(1.0, 1.5))
for i in range(5):                                # suburb trees
    for j in range(5):
        if 1 <= i <= 3 and 1 <= j <= 3:
            continue
        x0, x1 = BLOCK_IV[i]
        z0, z1 = BLOCK_IV[j]
        for _ in range(9):
            x, z = rnd.uniform(x0 + 5, x1 - 5), rnd.uniform(z0 + 5, z1 - 5)
            if not near_building(x, z, 3.5):
                tree(x, z, rnd.uniform(1.0, 1.7))

# traffic-light poles + heads
sig_x_mb, sig_z_mb = MB(), MB()
for cx in ROADS:
    for cz in ROADS:
        for sx in (-1, 1):
            for sz in (-1, 1):
                px, pz = cx + sx * 8.6, cz + sz * 8.6
                add_box(poles_mb, px, 2.6, pz, 0.2, 5.2, 0.2, (0, 0, .25))
                head = sig_x_mb if sx * sz > 0 else sig_z_mb
                add_box(head, px, 5.4, pz, 0.55, 1.3, 0.55, (0, 0, 1))

mesh_entity(slab_mb, tex=TEX['side'])
mesh_entity(park_mb, tex=TEX['grass'])
walls = mesh_entity(walls_mb, tex=TEX['day'])
mesh_entity(roofs_mb)
mesh_entity(trees_mb)
mesh_entity(poles_mb)
mesh_entity(pads_mb)
mesh_entity(beacon_mb, lit=False)
street_lamps = mesh_entity(lamps_mb, lit=False)
lamps_x = mesh_entity(sig_x_mb, lit=False)
lamps_z = mesh_entity(sig_z_mb, lit=False)

# ----------------------------------------------------------------------------
# Rings
# ----------------------------------------------------------------------------
_rmb = MB()
add_torus(_rmb, RING_RADIUS, 0.28, (0, 0, 1))
RING_MESH = _rmb.build()


class Ring(Entity):
    def __init__(self, pos, yaw):
        kw = dict(model=RING_MESH, position=pos, rotation_y=yaw, color=hsv(28, .95, 1), double_sided=True)
        if _UNLIT_SHADER is not None:
            kw['shader'] = _UNLIT_SHADER
        super().__init__(**kw)
        self.center = Vec3(*pos)
        a = math.radians(yaw)
        self.normal = Vec3(math.sin(a), 0, math.cos(a))
        self.done = False
        self.prev_d = None
        self.phase = rnd.uniform(0, 6.28)

    def set_done(self, done):
        self.done = done
        self.prev_d = None
        self.color = hsv(125, .85, .9) if done else hsv(28, .95, 1)


def make_rings():
    out = []
    for k in range(NUM_RINGS):
        axis = 'x' if k % 2 == 0 else 'z'
        c = ROADS[(k * 3 + 1) % 4] if axis == 'x' else ROADS[(k + 2) % 4]
        s = rnd.uniform(-150, 150)
        h = rnd.uniform(9, 30)
        if axis == 'x':
            out.append(Ring((s, h, c), 90))
        else:
            out.append(Ring((c, h, s), 0))
    return out


RINGS = make_rings()

# ----------------------------------------------------------------------------
# Traffic + pedestrians
# ----------------------------------------------------------------------------
def signal_state(t):
    p = t % 26.0
    if p < 9:
        return 'G', 'R'
    if p < 11:
        return 'Y', 'R'
    if p < 13:
        return 'R', 'R'
    if p < 22:
        return 'R', 'G'
    if p < 24:
        return 'R', 'Y'
    return 'R', 'R'


def yaw_for(axis, d):
    if axis == 'x':
        return 90 if d > 0 else 270
    return 0 if d > 0 else 180


def make_car_mesh(kind, body):
    mb = MB()
    glass, dark = (210, .5, .18), (0, 0, .08)
    if kind == 'car':
        add_box(mb, 0, .55, 0, 1.8, .62, 4.4, body)
        add_box(mb, 0, 1.12, -.25, 1.6, .55, 2.3, glass)
        add_box(mb, 0, 1.42, -.25, 1.55, .07, 2.0, body)
        for sx in (-1, 1):
            for sz in (-1, 1):
                add_box(mb, sx * .82, .3, sz * 1.4, .28, .6, .7, dark)
        for sx in (-1, 1):
            add_box(mb, sx * .6, .62, 2.2, .35, .16, .05, (50, .15, 1))
            add_box(mb, sx * .6, .62, -2.2, .35, .16, .05, (0, .9, .8))
    else:
        add_box(mb, 0, 1.75, 0, 2.5, 2.7, 10.5, body)
        add_box(mb, 0, 2.15, 0, 2.53, .95, 9.6, glass)
        for sx in (-1, 1):
            for z in (-3.6, 3.6):
                add_box(mb, sx * 1.1, .5, z, .3, 1.0, 1.1, dark)
        for sx in (-1, 1):
            add_box(mb, sx * .8, .9, 5.25, .4, .2, .05, (50, .15, 1))
            add_box(mb, sx * .8, .9, -5.25, .4, .2, .05, (0, .9, .8))
    return mb.build()


def make_light_mesh(kind):
    mb = MB()
    z = 2.22 if kind == 'car' else 5.27
    y = .62 if kind == 'car' else .9
    for sx in (-1, 1):
        add_box(mb, sx * .6, y, z, .5, .24, .1, (50, .2, 1))
        add_box(mb, sx * .6, y, -z, .45, .2, .1, (0, .95, 1))
    return mb.build()


BODY_COLS = [(0, 0, .92), (0, 0, .14), (0, 0, .6), (0, .8, .75), (215, .7, .65),
             (50, .9, .95), (120, .6, .5), (30, .7, .55), (200, .1, .8)]
CAR_MESHES = [make_car_mesh('car', c) for c in BODY_COLS]
BUS_MESHES = [make_car_mesh('bus', c) for c in ((50, .9, .9), (210, .7, .7), (0, .8, .75))]
LIGHT_MESHES = {'car': make_light_mesh('car'), 'bus': make_light_mesh('bus')}


class Car:
    def __init__(self, axis, c, d, lane, s):
        self.axis, self.c, self.d, self.lane, self.s = axis, c, d, lane, s
        self.kind = 'bus' if rnd.random() < 0.12 else 'car'
        mesh = rnd.choice(BUS_MESHES if self.kind == 'bus' else CAR_MESHES)
        self.half = 5.3 if self.kind == 'bus' else 2.3
        self.vmax = rnd.uniform(9, 14) * (0.8 if self.kind == 'bus' else 1.0)
        self.v = self.vmax * rnd.uniform(.4, 1)
        self.e = WE(model=mesh, double_sided=True)
        self.lights = UE(parent=self.e, model=LIGHT_MESHES[self.kind], double_sided=True, enabled=False)
        self.place()

    def place(self):
        off = LANE_OFF[self.lane]
        if self.axis == 'x':
            self.e.position = (self.s, ROAD_Y, self.c + self.d * off)
        else:
            self.e.position = (self.c - self.d * off, ROAD_Y, self.s)
        self.e.rotation_y = yaw_for(self.axis, self.d)


LANES = {}
CARS = []
for _axis in ('x', 'z'):
    for _c in ROADS:
        for _d in (1, -1):
            for _lane in (0, 1):
                lst = []
                LANES[(_axis, _c, _d, _lane)] = lst
                used = []
                for _k in range(CARS_PER_LANE):
                    s = 0.0
                    for _try in range(40):
                        s = rnd.uniform(-EXT + 15, EXT - 15)
                        if any(abs(s - cc) < 16 for cc in ROADS):
                            continue
                        if any(abs(s - u) < 22 for u in used):
                            continue
                        break
                    used.append(s)
                    car = Car(_axis, _c, _d, _lane, s)
                    lst.append(car)
                    CARS.append(car)

BRAKE = 4.5


def update_car(car, dt, states):
    state = states[car.axis]
    target = car.vmax
    for other in LANES[(car.axis, car.c, car.d, car.lane)]:
        if other is car:
            continue
        gap = (other.s - car.s) * car.d - car.half - other.half
        if 0 < gap + car.half + other.half and (other.s - car.s) * car.d > 0:
            target = min(target, math.sqrt(2 * BRAKE * max(0.0, gap - 2.0)))
    if state != 'G':
        for cc in ROADS:
            dist = (cc - car.s) * car.d - (11.6 + car.half)
            if -1.0 < dist < 70:
                if state == 'Y' and dist < car.v * 1.3:
                    continue
                target = min(target, math.sqrt(2 * BRAKE * max(0.0, dist)))
    car.v += clamp(target - car.v, -7.0 * dt, 2.8 * dt)
    car.v = max(0.0, car.v)
    car.s += car.d * car.v * dt
    if car.s * car.d > EXT + 15:
        car.s = -car.d * (EXT + 15)
    car.place()


# --- pedestrians
SHIRTS = [(0, .7, .8), (210, .6, .7), (120, .5, .6), (50, .8, .9), (280, .5, .7), (0, 0, .9), (0, 0, .2), (20, .7, .6)]
SKINS = [(25, .45, .85), (22, .5, .65), (20, .55, .45), (18, .5, .32)]
PANTS = [(220, .4, .3), (0, 0, .15), (30, .3, .45), (210, .6, .5)]
PED_BODY, PED_LEG = [], []
for _sh in SHIRTS:
    for _sk in SKINS[:2]:
        mb = MB()
        add_box(mb, 0, 1.2, 0, .42, .6, .24, _sh)
        add_box(mb, -.27, 1.18, 0, .12, .56, .14, _sh)
        add_box(mb, .27, 1.18, 0, .12, .56, .14, _sh)
        add_ellipsoid(mb, 0, 1.66, 0, .12, .14, .12, _sk, seg=6, rings=4)
        add_ellipsoid(mb, 0, 1.72, -.015, .13, .1, .13, (30, .5, .15), seg=6, rings=3)
        PED_BODY.append(mb.build())
for _pn in PANTS:
    mb = MB()
    add_box(mb, 0, -.42, 0, .17, .84, .2, _pn)
    add_box(mb, 0, -.84, .05, .18, .09, .3, (0, 0, .1))
    PED_LEG.append(mb.build())


class Ped:
    def __init__(self, axis, c, side, lane, d, s):
        self.axis, self.c, self.d, self.s = axis, c, d, s
        self.b = side * (8.3 + lane * 1.1)
        self.speed = rnd.uniform(1.1, 1.6)
        self.phase = rnd.uniform(0, 6.28)
        self.e = WE(model=rnd.choice(PED_BODY), double_sided=True)
        leg = rnd.choice(PED_LEG)
        self.legs = [WE(parent=self.e, model=leg, double_sided=True, position=(sx * .11, .85, 0)) for sx in (-1, 1)]
        self.place(0)

    def place(self, bob):
        if self.axis == 'x':
            self.e.position = (self.s, SLAB_H + bob, self.c + self.b)
        else:
            self.e.position = (self.c + self.b, SLAB_H + bob, self.s)
        self.e.rotation_y = yaw_for(self.axis, self.d)


PEDS = []
for _ in range(NUM_PEDS):
    ax = rnd.choice(('x', 'z'))
    PEDS.append(Ped(ax, rnd.choice(ROADS), rnd.choice((-1, 1)), rnd.choice((0, 1)),
                    rnd.choice((-1, 1)), rnd.uniform(-EXT, EXT)))


def update_ped(p, dt, states):
    perp = states['z'] if p.axis == 'x' else states['x']
    moving = True
    for cc in ROADS:
        entry = cc - p.d * (HALF_ROAD + 0.8)
        dist = (entry - p.s) * p.d
        if 0 < dist < 0.6 and perp != 'R':
            moving = False
            break
    swing = 0.0
    if moving:
        p.s += p.d * p.speed * dt
        p.phase += p.speed * 4.4 * dt
        if p.s * p.d > EXT + 10:
            p.s = -p.d * (EXT + 10)
        swing = math.sin(p.phase) * 35
    p.legs[0].rotation_x = swing
    p.legs[1].rotation_x = -swing
    p.place(abs(math.sin(p.phase)) * 0.035 if moving else 0)


# ----------------------------------------------------------------------------
# Time of day
# ----------------------------------------------------------------------------
TIMES = [
    dict(name='DAY', sky=(0, 0, 1.0), bg=(205, .35, .95), fog=(205, .22, .92), amb=(210, .12, .52),
         sun=(45, .10, .82), dir=(0.45, -1.0, 0.35), night=False, disc=(50, .15, 1), hill=(215, .22, .62),
         cloud=(0, 0, 1)),
    dict(name='SUNSET', sky=(24, .5, 1.0), bg=(22, .5, .9), fog=(22, .4, .86), amb=(22, .28, .4),
         sun=(24, .75, .9), dir=(0.9, -0.32, 0.3), night=False, disc=(28, .8, 1), hill=(300, .3, .42),
         cloud=(18, .45, 1)),
    dict(name='NIGHT', sky=(232, .7, .3), bg=(230, .75, .09), fog=(230, .6, .10), amb=(232, .5, .24),
         sun=(220, .35, .32), dir=(-0.4, -0.9, -0.25), night=True, disc=(210, .08, .95), hill=(230, .5, .12),
         cloud=(230, .4, .3)),
]


def refresh_shadow_bounds():
    if sun is None:
        return
    try:
        box = Entity(model='cube', scale=(2 * EXT + 40, 90, 2 * EXT + 40), position=(0, 40, 0))
        sun.update_bounds(box)
        destroy(box)
    except Exception:
        try:
            sun.update_bounds()
        except Exception:
            pass


class Game:
    score = 0
    rings_done = 0
    cam_mode = 0
    msg_timer = 0.0
    orbit = 0.0
    t = 0.0
    tod = 0
    night = False
    wind_on = True
    sun_dir = (0.45, -1.0, 0.35)


def set_time(idx):
    Game.tod = idx % len(TIMES)
    T = TIMES[Game.tod]
    Game.night = T['night']
    Game.sun_dir = T['dir']
    sky.color = hsv(*T['sky'])
    window.color = hsv(*T['bg'])
    try:
        scene.fog_color = hsv(*T['fog'])
        scene.fog_density = (120, 1100)
    except Exception:
        pass
    if sun is not None:
        sun.color = hsv(*T['sun'])
        ambient.color = hsv(*T['amb'])
        sun.rotation = Vec3(*T['dir'])
        refresh_shadow_bounds()
    walls.texture = TEX['night'] if Game.night else TEX['day']
    if Game.night and _UNLIT_SHADER is not None:
        walls.shader = _UNLIT_SHADER
    elif LIT is not None:
        walls.shader = LIT
    street_lamps.color = hsv(45, .35, 1) if Game.night else hsv(0, 0, .55)
    sun_disc.color = hsv(*T['disc'])
    hills.color = hsv(*T['hill'])
    clouds.color = hsv(*T['cloud'])
    for c in CARS:
        c.lights.enabled = Game.night


# ----------------------------------------------------------------------------
# Sound (generated at start-up, no asset files needed)
# ----------------------------------------------------------------------------
hum = ding = None


def _write_wav(path, samples, rate=22050):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b''.join(struct.pack('<h', int(clamp(s, -1, 1) * 32767)) for s in samples))


try:
    _folder = Path(application.asset_folder)
    _rate = 22050
    _r = random.Random(1)
    _hum = []
    for _n in range(_rate):
        t = _n / _rate
        v = sum((1.0 / k) * math.sin(2 * math.pi * 120 * k * t + k) for k in range(1, 9)) * 0.22
        v += (_r.random() - .5) * 0.25
        v *= 0.75 + 0.25 * math.sin(2 * math.pi * 12 * t)
        _hum.append(v * 0.6)
    _write_wav(_folder / 'rotor_hum.wav', _hum)
    _ding = []
    for _n in range(int(_rate * .4)):
        t = _n / _rate
        f = 880 + 900 * t
        _ding.append(math.sin(2 * math.pi * f * t) * math.exp(-6 * t) * .7)
    _write_wav(_folder / 'ring_ding.wav', _ding)
    hum = Audio('rotor_hum.wav', loop=True, autoplay=True, volume=0.0)
    ding = Audio('ring_ding.wav', autoplay=False, volume=0.8)
except Exception as ex:
    print('Sound disabled:', ex)
    hum = ding = None

# ----------------------------------------------------------------------------
# HUD + minimap
# ----------------------------------------------------------------------------
def ui_panel(x, y, w, h, col=color.black66):
    return Entity(parent=camera.ui, model='quad', color=col, scale=(w, h), position=(x, y))


ui_panel(-0.70, 0.33, 0.36, 0.30)
hud = Text(text='', position=(-0.87, 0.47), origin=(-0.5, 0.5), scale=1.1, color=color.white)
banner = Text(text='', position=(0, 0.22), origin=(0, 0), scale=2.0, color=color.yellow)
warn = Text(text='', position=(0, 0.14), origin=(0, 0), scale=1.6, color=color.red)
top_info = Text(text='', position=(0, 0.48), origin=(0, .5), scale=1.2, color=color.white)
mode_info = Text(text='', position=(-0.87, -0.40), origin=(-0.5, 0), scale=0.9, color=color.white)
help_text = Text(
    text=('CONTROLS\n'
          'SPACE / L-SHIFT   climb / descend (throttle)\n'
          'W S  pitch     A D  roll     Q E  yaw\n'
          'F  ALT-HOLD assist on/off   V  wind on/off\n'
          'N  day / sunset / night     C  camera\n'
          'R  reset drone   T  new run   H  help   ESC  quit\n\n'
          'Fly through the orange rings (+100).  Land on the blue HOME pad\n'
          'in the park or the magenta PAD B on the tower roof (up to +200).\n'
          'Land gently (<4.5 m/s, <15 deg tilt). Landing on a pad recharges.'),
    position=(-0.30, 0.34), origin=(-0.5, 0.5), scale=1.0, color=color.white)
help_bg = ui_panel(0.0, 0.16, 0.72, 0.36)
help_bg.z = 0.01

ui_panel(0.85, -0.30, 0.03, 0.30)
throttle_bar = Entity(parent=camera.ui, model='quad', color=color.azure, origin_y=-0.5,
                      scale=(0.03, 0.0), position=(0.85, -0.45))
Entity(parent=camera.ui, model='quad', color=color.white, scale=(0.05, 0.003), position=(0.85, -0.30))
ui_panel(-0.70, -0.45, 0.30, 0.028)
batt_bar = Entity(parent=camera.ui, model='quad', color=color.lime, origin_x=-0.5,
                  scale=(0.29, 0.02), position=(-0.845, -0.45), z=-0.01)
batt_text = Text(text='', position=(-0.85, -0.425), origin=(-0.5, 0), scale=0.9, color=color.white)

# minimap (north = up)
MM_SIZE = 0.34
MM_S = MM_SIZE / (2 * EXT)
mm = Entity(parent=camera.ui, position=(0.70, 0.32))
Entity(parent=mm, model='quad', color=color.black66, scale=MM_SIZE + 0.02)
_mm = MB()
rect2d(_mm, -EXT, -EXT, EXT, EXT, (110, .35, .35))
for _c in ROADS:
    rect2d(_mm, -EXT, _c - 7, EXT, _c + 7, (0, 0, .28))
    rect2d(_mm, _c - 7, -EXT, _c + 7, EXT, (0, 0, .28))
for (_x, _z, _w, _d, _h) in BUILDINGS:
    rect2d(_mm, _x - _w / 2, _z - _d / 2, _x + _w / 2, _z + _d / 2, (210, .2, .55 + min(_h, 60) / 200))
rect2d(_mm, -28, -28, 28, 28, (115, .55, .5))
Entity(parent=mm, model=_mm.build(), scale=MM_S, z=-0.01, double_sided=True)
for _name, _p in PADS:
    Entity(parent=mm, model='quad', color=color.azure if _name == 'HOME' else color.magenta,
           scale=0.014, position=(_p.x * MM_S, _p.z * MM_S), z=-0.02)
ring_marks = [Entity(parent=mm, model='quad', color=color.orange, scale=0.011,
                     position=(r.center.x * MM_S, r.center.z * MM_S), z=-0.03) for r in RINGS]
drone_pivot = Entity(parent=mm, z=-0.05)
Entity(parent=drone_pivot, model='quad', color=color.white, scale=0.013)
Entity(parent=drone_pivot, model='quad', color=color.red, scale=(0.005, 0.03), position=(0, 0.015))


def flash(text, seconds=3.0):
    banner.text = text
    Game.msg_timer = seconds


# ----------------------------------------------------------------------------
# Drone
# ----------------------------------------------------------------------------
def wind_vec(t):
    if not Game.wind_on:
        return Vec3(0, 0, 0)
    ang = 0.6 + 0.25 * math.sin(t * 0.05)
    mag = max(0.0, 2.2 + 1.6 * math.sin(t * 0.37) + 0.8 * math.sin(t * 1.13 + 1.7))
    return Vec3(math.sin(ang) * mag, 0.25 * math.sin(t * 0.9), math.cos(ang) * mag)


def surface_below(x, z, y):
    best = 0.0
    for bx, bz, w, d, h in BUILDINGS:
        if abs(x - bx) <= w / 2 and abs(z - bz) <= d / 2 and h <= y:
            best = max(best, h)
    return best


def drone_mesh():
    mb = MB()
    white, dark, red = (0, 0, .92), (0, 0, .12), (0, .85, .8)
    add_box(mb, 0, 0.12, 0, .30, .10, .38, white, bottom=True)
    add_ellipsoid(mb, 0, 0.17, -.02, .13, .06, .17, (0, 0, .96), seg=8, rings=4)
    add_box(mb, 0, 0.05, -.03, .16, .07, .24, dark, bottom=True)
    add_box(mb, 0, 0.11, 0, 1.02, .035, .06, dark, yaw=45)
    add_box(mb, 0, 0.11, 0, 1.02, .035, .06, dark, yaw=-45)
    for sx in (-1, 1):
        for sz in (-1, 1):
            add_box(mb, sx * .36, .10, sz * .36, .10, .09, .10, red if sz > 0 else (0, 0, .2), bottom=True)
            add_box(mb, sx * .14, -.04, sz * .14, .022, .13, .022, dark)
        add_box(mb, sx * .14, -.11, 0, .03, .025, .5, dark)
    add_ellipsoid(mb, 0, 0.03, .2, .05, .05, .05, (0, 0, .05), seg=8, rings=4)
    add_box(mb, 0, 0.03, .245, .03, .03, .02, (200, .8, .5))
    return mb.build()


DRONE_MESH = drone_mesh()


class Drone(Entity):
    def __init__(self):
        super().__init__(position=Vec3(0, GROUND_OFF, 0))
        body = Entity(parent=self, model=DRONE_MESH, double_sided=True)
        if LIT is not None:
            body.shader = LIT
        self.rotors, self.discs, self.leds = [], [], []
        for sx in (-1, 1):
            for sz in (-1, 1):
                pivot = Entity(parent=self, position=(0.36 * sx, 0.16, 0.36 * sz))
                Entity(parent=pivot, model='cube', scale=(0.46, 0.012, 0.045), color=hsv(0, 0, .1))
                Entity(parent=pivot, model='cube', scale=(0.46, 0.012, 0.045), color=hsv(0, 0, .1), rotation_y=90)
                disc = UE(parent=pivot, model='sphere', scale=(0.5, 0.004, 0.5), color=hsv(0, 0, .92, 0.0))
                self.rotors.append(pivot)
                self.discs.append(disc)
                led = UE(parent=self, model='cube', scale=0.045,
                         color=hsv(0, .95, 1) if sz > 0 else hsv(125, .95, 1),
                         position=(0.36 * sx, 0.09, 0.36 * sz + (0.07 if sz > 0 else -0.07)))
                self.leds.append(led)
        self.blob = UE(model='sphere', scale=(1.2, 0.01, 1.2), color=hsv(0, 0, 0, .4))
        self.reset()

    def reset(self):
        self.position = Vec3(0, GROUND_OFF, 0)
        self.vel = Vec3(0, 0, 0)
        self.yaw = self.pitch = self.roll = 0.0
        self.throttle = 0.0
        self.battery = 100.0
        self.assist = getattr(self, 'assist', True)
        self.crashed = False
        self.frozen = False
        self.in_air = False
        self.floor_y = GROUND_OFF
        self.charging = False
        self.t = 0.0
        self.rotation = Vec3(0, 0, 0)
        for r in RINGS:
            r.prev_d = None
        for led in self.leds:
            led.enabled = True
        banner.text = ''
        Game.msg_timer = 0

    def tilt(self):
        c = math.cos(math.radians(self.pitch)) * math.cos(math.radians(self.roll))
        return math.degrees(math.acos(clamp(c, -1, 1)))

    def crash(self, reason, freeze=False):
        if self.crashed:
            return
        self.crashed = True
        self.frozen = freeze
        self.throttle = 0
        flash(f'CRASH: {reason}  -  press R to reset', 999)

    def step(self, dt):
        self.t += dt
        if self.crashed:
            if not self.frozen:
                self.vel = Vec3(0, self.vel.y - G * dt, 0)
                floor = surface_below(self.x, self.z, self.y) + GROUND_OFF
                self.y = max(floor, self.y + self.vel.y * dt)
                if self.y <= floor:
                    self.vel = Vec3(0, 0, 0)
            self.update_visuals(dt)
            return

        on_ground = self.y <= self.floor_y + 0.03
        wind = wind_vec(Game.t)

        # --- attitude / yaw
        self.yaw += YAW_RATE * (held_keys['e'] - held_keys['q']) * dt
        yr = math.radians(self.yaw)
        fwd = Vec3(math.sin(yr), 0, math.cos(yr))
        right = Vec3(math.cos(yr), 0, -math.sin(yr))
        kw = held_keys['w'] - held_keys['s']
        kd = held_keys['d'] - held_keys['a']
        tmax = 25.0 if self.assist else MAX_TILT
        tp, tr = tmax * kw, tmax * kd
        if self.assist:                       # position-hold: cancel drift when sticks are centred
            vf = self.vel.x * fwd.x + self.vel.z * fwd.z
            vr = self.vel.x * right.x + self.vel.z * right.z
            if kw == 0:
                tp = clamp(-10 * vf, -tmax, tmax)
            if kd == 0:
                tr = clamp(-10 * vr, -tmax, tmax)
        if on_ground:
            tp = tr = 0.0
        k = min(1.0, ATTITUDE_RESPONSE * dt)
        self.pitch += (tp - self.pitch) * k
        self.roll += (tr - self.roll) * k
        p, r = math.radians(self.pitch), math.radians(self.roll)

        # --- throttle
        climb = held_keys['space'] - held_keys['left shift']
        if self.assist:
            vs_cmd = 5.0 * held_keys['space'] - 3.0 * held_keys['left shift']
            agl = self.y - self.floor_y
            if vs_cmd < 0 and agl < 3.0:
                vs_cmd = -1.2
            if on_ground and vs_cmd <= 0:
                goal = 0.0
            else:
                goal = 1.0 / (MAX_THRUST_G * max(0.5, math.cos(p) * math.cos(r))) + 0.12 * (vs_cmd - self.vel.y)
            self.throttle += (clamp(goal, 0, 1) - self.throttle) * min(1.0, 10 * dt)
        else:
            self.throttle = clamp(self.throttle + climb * THROTTLE_RATE * dt, 0, 1)

        # --- battery
        self.charging = False
        if on_ground:
            for _n, pp in PADS:
                if flat_dist(self.x, self.z, pp.x, pp.z) < PAD_HALF + 1 and abs(self.floor_y - GROUND_OFF - pp.y) < 0.6:
                    self.charging = True
        if self.charging:
            self.battery = min(100.0, self.battery + 5.0 * dt)
        else:
            self.battery = max(0.0, self.battery - (0.06 + 0.55 * self.throttle ** 2) * dt)
        if self.battery <= 0:
            self.throttle = 0.0
            if not banner.text:
                flash('BATTERY EMPTY - motors off!', 4)

        # --- forces
        thrust_dir = Vec3(0, math.cos(p) * math.cos(r), 0) + fwd * math.sin(p) + right * math.sin(r)
        thrust = self.throttle * MAX_THRUST_G * G
        acc = thrust_dir * thrust + Vec3(0, -G, 0) - (self.vel - wind) * DRAG
        self.vel = self.vel + acc * dt
        self.position = self.position + self.vel * dt

        # --- limits
        if self.y > MAX_ALT:
            self.y = MAX_ALT
            self.vel = Vec3(self.vel.x, min(0.0, self.vel.y), self.vel.z)
        lim = EXT + 60
        if abs(self.x) > lim or abs(self.z) > lim:
            self.x = clamp(self.x, -lim, lim)
            self.z = clamp(self.z, -lim, lim)
            self.vel = Vec3(0, self.vel.y, 0)
            flash('OUT OF RANGE - turn back', 2)

        # --- buildings: roofs are landable, walls are not
        surf = 0.0
        for bx, bz, w, d, h in BUILDINGS:
            if abs(self.x - bx) < w / 2 + DRONE_RADIUS and abs(self.z - bz) < d / 2 + DRONE_RADIUS:
                if self.y - GROUND_OFF >= h - 0.35:
                    surf = max(surf, h)
                else:
                    self.crash('building collision', freeze=True)
                    self.update_visuals(dt)
                    return
        self.floor_y = surf + GROUND_OFF

        # --- ground / roof contact
        if self.y <= self.floor_y:
            impact = -self.vel.y
            if impact > CRASH_SPEED:
                self.crash('hard landing')
                self.y = self.floor_y
                self.update_visuals(dt)
                return
            if impact > 0.3 and self.in_air:
                if self.tilt() > CRASH_TILT:
                    self.crash('tilted landing')
                    self.y = self.floor_y
                    self.update_visuals(dt)
                    return
                self.score_landing(surf)
            self.y = self.floor_y
            fr = max(0.0, 1 - 6 * dt)
            self.vel = Vec3(self.vel.x * fr, max(0.0, self.vel.y), self.vel.z * fr)
            self.in_air = False
        elif self.y - self.floor_y > 2.0:
            self.in_air = True

        # --- rings
        for ring in RINGS:
            if ring.done:
                continue
            rel = self.position - ring.center
            dist_along = rel.x * ring.normal.x + rel.z * ring.normal.z
            if ring.prev_d is not None and ring.prev_d * dist_along < 0 and rel.length() < RING_RADIUS:
                ring.set_done(True)
                Game.score += 100
                Game.rings_done += 1
                ring_marks[RINGS.index(ring)].color = color.lime
                flash(f'RING!  +100  ({Game.rings_done}/{NUM_RINGS})', 1.5)
                if ding:
                    try:
                        ding.play()
                    except Exception:
                        pass
            else:
                ring.prev_d = dist_along
        self.update_visuals(dt)

    def update_visuals(self, dt):
        self.rotation = Vec3(self.pitch, self.yaw, -self.roll)
        spin = (400 + self.throttle * 3200) * dt if (self.throttle > 0.02 and not self.crashed) else 0
        for i, rt in enumerate(self.rotors):
            rt.rotation_y += spin * (1 if i % 3 == 0 else -1)
        blur = clamp((self.throttle - 0.12) * 1.3, 0, .32)
        for dsc in self.discs:
            dsc.alpha = blur
        blink = (self.t * 3) % 1 < .5
        for led in self.leds:
            led.enabled = blink or Game.night
        surf = surface_below(self.x, self.z, self.y)
        h = max(0.0, self.y - surf - GROUND_OFF)
        self.blob.position = Vec3(self.x, surf + (SLAB_H if surf == 0 else 0.05) + 0.02, self.z)
        sc = 0.9 + h * 0.035
        self.blob.scale = (sc, 0.01, sc)
        self.blob.color = hsv(0, 0, 0, max(0.06, 0.42 - h * 0.009))

    def score_landing(self, surf):
        for name, pos in PADS:
            d = flat_dist(self.x, self.z, pos.x, pos.z)
            if d < PAD_HALF and abs(surf - pos.y) < 0.6:
                pts = 50 + int(150 * (1 - d / PAD_HALF))
                Game.score += pts
                flash(f'{name} LANDING  +{pts}  ({d:.1f} m from centre)', 3)
                return
        flash('Landed off-pad - no points', 2)


def new_run():
    Game.score = 0
    Game.rings_done = 0
    for r, m in zip(RINGS, ring_marks):
        r.set_done(False)
        m.color = color.orange
    drone.reset()


drone = Drone()
camera.position = Vec3(0, 4, -8)
set_time(0)
refresh_shadow_bounds()

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def update_camera(dt):
    d = drone
    y = math.radians(d.yaw)
    fwd = Vec3(math.sin(y), 0, math.cos(y))
    speed = d.vel.length()
    drone.visible = Game.cam_mode != 1
    camera.fov = 80 + clamp(speed * 0.7, 0, 14)
    if Game.cam_mode == 0:
        target = d.position - fwd * (6.5 + speed * 0.15) + Vec3(0, 2.6, 0)
        target.y = max(target.y, 0.8)
        camera.position = camera.position + (target - camera.position) * min(1.0, 6 * dt)
        camera.look_at(d.position + Vec3(0, 0.4, 0))
    elif Game.cam_mode == 1:
        shake = 0.0 if d.crashed else 0.004 * d.throttle
        camera.position = d.position + fwd * 0.4 + Vec3(0, 0.1 + math.sin(Game.t * 40) * shake, 0)
        camera.rotation = Vec3(d.pitch - 12, d.yaw, -d.roll)
    else:
        Game.orbit += 12 * dt
        a = math.radians(Game.orbit)
        target = d.position + Vec3(math.sin(a) * 45, 38, math.cos(a) * 45)
        camera.position = camera.position + (target - camera.position) * min(1.0, 3 * dt)
        camera.look_at(d.position)


# ----------------------------------------------------------------------------
# Main loop / input
# ----------------------------------------------------------------------------
DIRS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
SIG_COL = {'G': hsv(125, .9, .85), 'Y': hsv(48, .95, 1), 'R': hsv(0, .95, .9)}


def update():
    dt = min(time.dt, 1 / 30)
    Game.t += dt
    sx, sz = signal_state(Game.t)
    states = {'x': sx, 'z': sz}
    lamps_x.color = SIG_COL[sx]
    lamps_z.color = SIG_COL[sz]

    drone.step(dt)
    update_camera(dt)
    for c in CARS:
        update_car(c, dt, states)
    for p in PEDS:
        update_ped(p, dt, states)

    # sky bits follow the camera
    dn = Game.sun_dir
    ln = math.sqrt(dn[0] ** 2 + dn[1] ** 2 + dn[2] ** 2)
    sun_disc.position = camera.position - Vec3(dn[0] / ln, dn[1] / ln, dn[2] / ln) * 1500
    clouds.x = (Game.t * 3.0) % _period
    for r in RINGS:
        if not r.done:
            s = 1 + 0.035 * math.sin(Game.t * 4 + r.phase)
            r.scale = (s, s, s)

    # audio
    if hum:
        try:
            active = 0 if (drone.battery <= 0 or drone.crashed) else (0.12 + 0.6 * drone.throttle)
            hum.volume = active * 0.6
            hum.pitch = 0.7 + 1.0 * drone.throttle
        except Exception:
            pass

    # HUD
    d = drone
    surf = surface_below(d.x, d.z, d.y)
    agl = max(0.0, d.y - GROUND_OFF - surf)
    speed = d.vel.length()
    hdg = d.yaw % 360
    w = wind_vec(Game.t)
    wmag = math.hypot(w.x, w.z)
    wfrom = (math.degrees(math.atan2(w.x, w.z)) + 180) % 360
    if d.crashed:
        status = 'CRASHED'
    elif d.charging:
        status = 'CHARGING'
    elif d.y > d.floor_y + 0.05:
        status = 'FLYING'
    else:
        status = 'ON GROUND'
    hud.text = (f'ALT      {d.y - GROUND_OFF:6.1f} m   AGL {agl:5.1f}\n'
                f'SPEED    {speed:6.1f} m/s ({speed * 3.6:.0f} km/h)\n'
                f'V/S      {d.vel.y:6.1f} m/s\n'
                f'THROTTLE {d.throttle * 100:5.0f} %\n'
                f'TILT     {d.tilt():6.1f} deg\n'
                f'HEADING  {hdg:5.0f} {DIRS[int((hdg + 22.5) // 45) % 8]}\n'
                f'WIND     {wmag:4.1f} m/s from {wfrom:3.0f}\n'
                f'{status}')
    left = [r for r in RINGS if not r.done]
    if left:
        nr = min(left, key=lambda r: (r.center - d.position).length())
        rel = nr.center - d.position
        bearing = math.degrees(math.atan2(rel.x, rel.z))
        diff = (bearing - d.yaw + 180) % 360 - 180
        arrow = '<<' if diff < -20 else ('>>' if diff > 20 else '^^')
        nxt = f'NEXT RING {rel.length():4.0f} m {arrow}'
    else:
        nxt = 'ALL RINGS DONE - LAND ON A PAD'
    top_info.text = f'SCORE {Game.score}    RINGS {Game.rings_done}/{NUM_RINGS}    {nxt}'
    mode_info.text = (f'{"ALT-HOLD" if d.assist else "MANUAL"}   CAM {CAM_NAMES[Game.cam_mode]}   '
                      f'{TIMES[Game.tod]["name"]}   WIND {"ON" if Game.wind_on else "OFF"}   '
                      f'[H] help')
    throttle_bar.scale_y = 0.3 * d.throttle
    batt_bar.scale_x = 0.29 * d.battery / 100
    batt_bar.color = color.lime if d.battery > 40 else (color.orange if d.battery > 20 else color.red)
    batt_text.text = f'BATTERY {d.battery:3.0f}%'
    drone_pivot.position = (d.x * MM_S, d.z * MM_S)
    drone_pivot.rotation_z = -d.yaw

    msgs = []
    if d.battery < 20 and d.battery > 0:
        msgs.append('LOW BATTERY')
    if not d.crashed and d.vel.y < -4.0 and agl < 25:
        msgs.append('SINK RATE')
    warn.text = '   '.join(msgs)

    if Game.msg_timer > 0:
        Game.msg_timer -= dt
        if Game.msg_timer <= 0:
            banner.text = ''


def input(key):
    if key == 'r':
        drone.reset()
    elif key == 't':
        new_run()
    elif key == 'c':
        Game.cam_mode = (Game.cam_mode + 1) % len(CAM_NAMES)
    elif key == 'h':
        help_text.enabled = not help_text.enabled
        help_bg.enabled = help_text.enabled
    elif key == 'f':
        drone.assist = not drone.assist
        flash('ALT-HOLD ASSIST ON' if drone.assist else 'MANUAL THROTTLE', 1.5)
    elif key == 'v':
        Game.wind_on = not Game.wind_on
    elif key == 'n':
        set_time(Game.tod + 1)
    elif key == 'escape':
        application.quit()


app.run()

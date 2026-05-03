"""Generate Maya arc demo data — 25 items across photo / screenshot / voice / video."""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path("./memories")
OUT.mkdir(exist_ok=True)

NOW = time.time()
DAY = 24 * 3600
random.seed(7)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    return ImageFont.load_default()


def gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(w):
            img.putpixel((x, y), (r, g, b))
    return img


def fast_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    strip = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        strip.putpixel((0, y), (r, g, b))
    return strip.resize(size)


def set_mtime(path: Path, days_ago: int) -> None:
    ts = NOW - days_ago * DAY
    os.utime(path, (ts, ts))


# ----------------------------- photos ----------------------------------------

def photo_dog_at_beach(path: Path) -> None:
    img = fast_gradient((900, 600), (255, 200, 130), (40, 90, 160))
    d = ImageDraw.Draw(img)
    # sand
    d.rectangle([0, 420, 900, 600], fill=(230, 200, 150))
    # sun
    d.ellipse([700, 80, 820, 200], fill=(255, 240, 180))
    # waves
    for y in range(380, 420, 6):
        d.line([(0, y), (900, y - 4)], fill=(80, 130, 180), width=2)
    # dog silhouette
    d.ellipse([380, 350, 460, 430], fill=(30, 30, 30))  # body
    d.ellipse([445, 320, 495, 370], fill=(30, 30, 30))  # head
    d.polygon([(465, 320), (480, 295), (490, 322)], fill=(30, 30, 30))  # ear
    d.line([(385, 425), (385, 460)], fill=(30, 30, 30), width=4)  # leg
    d.line([(420, 425), (420, 460)], fill=(30, 30, 30), width=4)
    d.line([(450, 425), (450, 460)], fill=(30, 30, 30), width=4)
    img.filter(ImageFilter.SMOOTH).save(path, "JPEG", quality=88)


def photo_sunset(path: Path) -> None:
    img = fast_gradient((900, 600), (255, 130, 90), (60, 30, 80))
    d = ImageDraw.Draw(img)
    d.ellipse([350, 250, 550, 450], fill=(255, 220, 140))
    d.rectangle([0, 460, 900, 600], fill=(20, 10, 30))
    img.save(path, "JPEG", quality=88)


def photo_coffee_shop(path: Path) -> None:
    img = fast_gradient((900, 600), (90, 60, 40), (40, 25, 18))
    d = ImageDraw.Draw(img)
    # cup
    d.rounded_rectangle([330, 240, 570, 460], radius=20, fill=(245, 240, 232))
    d.ellipse([350, 240, 550, 280], fill=(60, 35, 20))  # coffee top
    d.arc([550, 280, 640, 400], start=270, end=90, fill=(245, 240, 232), width=10)  # handle
    f = load_font(28)
    d.text((350, 480), "café · brooklyn", fill=(220, 200, 170), font=f)
    img.save(path, "JPEG", quality=88)


def photo_food_restaurant(path: Path) -> None:
    img = fast_gradient((900, 600), (200, 180, 140), (90, 60, 40))
    d = ImageDraw.Draw(img)
    d.ellipse([200, 150, 700, 520], fill=(245, 240, 230))  # plate
    d.ellipse([320, 230, 580, 440], fill=(180, 80, 40))  # food
    d.ellipse([350, 260, 420, 320], fill=(120, 180, 80))  # garnish
    img.save(path, "JPEG", quality=88)


def photo_cat_couch(path: Path) -> None:
    img = fast_gradient((900, 600), (210, 200, 220), (140, 130, 160))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([100, 350, 800, 580], radius=30, fill=(110, 90, 130))
    # cat body
    d.ellipse([350, 300, 600, 430], fill=(60, 50, 50))
    d.ellipse([530, 250, 620, 340], fill=(60, 50, 50))
    d.polygon([(540, 255), (560, 220), (580, 260)], fill=(60, 50, 50))
    d.polygon([(595, 255), (605, 225), (620, 262)], fill=(60, 50, 50))
    img.save(path, "JPEG", quality=88)


def photo_hiking_trail(path: Path) -> None:
    img = fast_gradient((900, 600), (130, 180, 230), (60, 110, 70))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 380), (300, 200), (550, 360), (900, 250), (900, 600), (0, 600)], fill=(70, 110, 60))
    d.polygon([(350, 600), (450, 450), (550, 600)], fill=(120, 95, 70))  # trail
    img.save(path, "JPEG", quality=88)


def photo_wedding_venue(path: Path) -> None:
    img = fast_gradient((900, 600), (250, 240, 220), (180, 160, 140))
    d = ImageDraw.Draw(img)
    # arch
    d.rectangle([200, 200, 240, 500], fill=(110, 80, 60))
    d.rectangle([660, 200, 700, 500], fill=(110, 80, 60))
    d.arc([200, 100, 700, 300], start=180, end=360, fill=(110, 80, 60), width=20)
    # florals
    for cx, cy in [(220, 200), (680, 200), (380, 130), (520, 130)]:
        d.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=(220, 180, 200))
    # aisle
    d.polygon([(380, 600), (520, 600), (480, 280), (420, 280)], fill=(245, 240, 230))
    f = load_font(28)
    d.text((300, 530), "Brooklyn Wedding Venue", fill=(80, 60, 50), font=f)
    img.save(path, "JPEG", quality=88)


def photo_city_skyline(path: Path) -> None:
    img = fast_gradient((900, 600), (15, 20, 50), (5, 5, 20))
    d = ImageDraw.Draw(img)
    for x in range(0, 900, 60):
        h = random.randint(150, 380)
        d.rectangle([x, 600 - h, x + 50, 600], fill=(20, 25, 55))
        for wy in range(600 - h + 20, 600, 25):
            for wx in range(x + 8, x + 45, 12):
                if random.random() < 0.6:
                    d.rectangle([wx, wy, wx + 6, wy + 10], fill=(255, 220, 130))
    # moon
    d.ellipse([720, 60, 800, 140], fill=(245, 240, 220))
    img.save(path, "JPEG", quality=88)


def photo_whiteboard(path: Path) -> None:
    img = Image.new("RGB", (900, 600), (250, 250, 248))
    d = ImageDraw.Draw(img)
    f = load_font(24)
    fb = load_font(34)
    d.text((40, 30), "auth flow — bug hunt", fill=(40, 40, 40), font=fb)
    boxes = [(60, 120, 240, 200, "client"), (340, 120, 520, 200, "auth middleware"),
             (620, 120, 800, 200, "session store"), (340, 320, 520, 400, "race condition?")]
    for x1, y1, x2, y2, label in boxes:
        d.rectangle([x1, y1, x2, y2], outline=(40, 40, 60), width=3)
        d.text((x1 + 20, y1 + 28), label, fill=(40, 40, 60), font=f)
    d.line([(240, 160), (340, 160)], fill=(200, 60, 60), width=3)
    d.line([(520, 160), (620, 160)], fill=(200, 60, 60), width=3)
    d.line([(430, 200), (430, 320)], fill=(200, 60, 60), width=3)
    d.text((40, 460), "line 47 — token expiry < should be <=", fill=(180, 40, 40), font=f)
    img.save(path, "JPEG", quality=88)


def photo_laptop(path: Path) -> None:
    img = fast_gradient((900, 600), (160, 150, 170), (60, 55, 70))
    d = ImageDraw.Draw(img)
    # laptop
    d.rounded_rectangle([200, 180, 700, 460], radius=12, fill=(40, 40, 50))
    d.rounded_rectangle([220, 200, 680, 440], radius=6, fill=(15, 15, 25))
    f = load_font(16)
    for i, line in enumerate(["def search(query):", "    visual = clip(query)", "    text = minilm(query)",
                              "    return rrf(visual, text)"]):
        d.text((240, 220 + i * 28), line, fill=(120, 200, 130), font=f)
    d.rectangle([180, 460, 720, 480], fill=(60, 60, 70))
    img.save(path, "JPEG", quality=88)


def photo_concert(path: Path) -> None:
    img = fast_gradient((900, 600), (200, 60, 180), (20, 0, 40))
    d = ImageDraw.Draw(img)
    # stage lights
    for i in range(12):
        x = random.randint(0, 900)
        d.polygon([(x, 0), (x - 80, 400), (x + 80, 400)], fill=(255, 200, 100, 80))
    # crowd
    d.rectangle([0, 380, 900, 600], fill=(15, 5, 25))
    for _ in range(120):
        x = random.randint(0, 900); y = random.randint(380, 580)
        d.ellipse([x, y, x + 12, y + 14], fill=(40, 25, 45))
    img.save(path, "JPEG", quality=88)


def photo_snowy_mountain(path: Path) -> None:
    img = fast_gradient((900, 600), (200, 220, 240), (90, 130, 170))
    d = ImageDraw.Draw(img)
    d.polygon([(50, 500), (300, 200), (550, 500)], fill=(245, 245, 250))
    d.polygon([(400, 500), (650, 150), (900, 500)], fill=(230, 235, 245))
    d.polygon([(250, 250), (300, 200), (350, 250)], fill=(255, 255, 255))
    d.polygon([(610, 200), (650, 150), (690, 200)], fill=(255, 255, 255))
    img.save(path, "JPEG", quality=88)


# Apartment / extra story photos (replacing some incidentals to fit Maya arc)
def photo_brownstone(path: Path) -> None:
    img = fast_gradient((900, 600), (180, 200, 220), (130, 140, 150))
    d = ImageDraw.Draw(img)
    d.rectangle([200, 150, 700, 580], fill=(140, 80, 60))  # brownstone
    for y in range(200, 560, 90):
        for x in range(220, 680, 100):
            d.rectangle([x, y, x + 70, y + 60], fill=(60, 80, 110))  # window
    d.rectangle([400, 440, 500, 580], fill=(70, 50, 40))  # door
    f = load_font(24)
    d.text((250, 80), "Park Slope brownstone", fill=(40, 40, 60), font=f)
    img.save(path, "JPEG", quality=88)


def photo_moving_boxes(path: Path) -> None:
    img = fast_gradient((900, 600), (220, 210, 200), (180, 165, 150))
    d = ImageDraw.Draw(img)
    boxes = [(120, 350, 320, 540), (340, 320, 540, 540), (560, 380, 760, 540), (200, 200, 360, 360)]
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1, y1, x2, y2], fill=(195, 160, 110), outline=(120, 90, 50), width=3)
        d.line([(x1, y1 + (y2 - y1) // 2), (x2, y1 + (y2 - y1) // 2)], fill=(120, 90, 50), width=2)
    f = load_font(22)
    d.text((140, 360), "KITCHEN", fill=(70, 50, 30), font=f)
    d.text((360, 330), "BOOKS", fill=(70, 50, 30), font=f)
    img.save(path, "JPEG", quality=88)


def photo_cake_tasting(path: Path) -> None:
    img = fast_gradient((900, 600), (255, 230, 235), (210, 170, 180))
    d = ImageDraw.Draw(img)
    for cx, label in [(250, "vanilla"), (450, "chocolate"), (650, "lemon")]:
        d.ellipse([cx - 80, 280, cx + 80, 440], fill=(250, 245, 230))
        d.ellipse([cx - 80, 260, cx + 80, 320], fill=(245, 230, 215))
        f = load_font(18)
        d.text((cx - 40, 460), label, fill=(80, 50, 60), font=f)
    fb = load_font(30)
    d.text((280, 100), "cake tasting · sat", fill=(120, 60, 80), font=fb)
    img.save(path, "JPEG", quality=88)


PHOTOS = [
    ("dog_at_beach.jpg", photo_dog_at_beach, 38, "incidental"),
    ("sunset.jpg", photo_sunset, 12, "incidental"),
    ("coffee_shop.jpg", photo_coffee_shop, 7, "incidental"),
    ("food_restaurant.jpg", photo_food_restaurant, 21, "wedding"),
    ("cat_couch.jpg", photo_cat_couch, 3, "incidental"),
    ("hiking_trail.jpg", photo_hiking_trail, 28, "incidental"),
    ("wedding_venue.jpg", photo_wedding_venue, 33, "wedding"),
    ("city_skyline_night.jpg", photo_city_skyline, 9, "incidental"),
    ("whiteboard_auth_bug.jpg", photo_whiteboard, 14, "work"),
    ("laptop_search_code.jpg", photo_laptop, 6, "work"),
    ("brownstone_park_slope.jpg", photo_brownstone, 18, "apartment"),
    ("moving_boxes_living_room.jpg", photo_moving_boxes, 4, "apartment"),
    ("cake_tasting.jpg", photo_cake_tasting, 25, "wedding"),
]


# ----------------------------- screenshots -----------------------------------

def screenshot_python_keyerror(path: Path) -> None:
    img = Image.new("RGB", (900, 600), (28, 28, 32))
    d = ImageDraw.Draw(img)
    f = load_font(18)
    fb = load_font(20)
    lines = [
        "Traceback (most recent call last):",
        '  File "auth/middleware.py", line 47, in verify_token',
        '    user = sessions[token_id]',
        "KeyError: 'sess_8c3a91f0'",
        "",
        "  during handling of the above exception, another exception occurred:",
        "  File \"auth/middleware.py\", line 51, in verify_token",
        "    raise AuthError('session expired')",
        "AuthError: session expired",
    ]
    d.text((30, 20), "PowerShell — pytest", fill=(180, 180, 200), font=fb)
    for i, line in enumerate(lines):
        color = (240, 100, 100) if "Error" in line else (220, 220, 230)
        d.text((30, 70 + i * 30), line, fill=color, font=f)
    img.save(path, "PNG")


def screenshot_calendar_sarah(path: Path) -> None:
    img = Image.new("RGB", (900, 600), (245, 245, 248))
    d = ImageDraw.Draw(img)
    fb = load_font(28)
    f = load_font(20)
    d.text((30, 20), "Calendar — This Week", fill=(40, 40, 60), font=fb)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for i, day in enumerate(days):
        x = 30 + i * 170
        d.rectangle([x, 80, x + 160, 560], outline=(200, 200, 210), width=1)
        d.text((x + 10, 90), day, fill=(120, 120, 140), font=f)
    # Tuesday 3pm event
    d.rectangle([200, 280, 360, 360], fill=(255, 200, 130))
    d.text((210, 290), "3:00 PM", fill=(80, 50, 20), font=f)
    d.text((210, 320), "Sarah re: venue", fill=(80, 50, 20), font=f)
    img.save(path, "PNG")


def screenshot_slack_deploy(path: Path) -> None:
    img = Image.new("RGB", (900, 600), (245, 245, 248))
    d = ImageDraw.Draw(img)
    fb = load_font(22)
    f = load_font(18)
    d.rectangle([0, 0, 900, 60], fill=(74, 21, 75))
    d.text((30, 18), "#engineering   ·   2 messages", fill=(245, 245, 255), font=fb)
    msgs = [
        ("Alex", "deploy to prod failed at 14:32 — auth service is throwing 500s"),
        ("Maya", "looking at the trace now. KeyError in auth middleware. rollback please"),
    ]
    for i, (who, text) in enumerate(msgs):
        y = 100 + i * 140
        d.ellipse([30, y, 70, y + 40], fill=(180, 100, 200))
        d.text((90, y), who, fill=(40, 40, 60), font=fb)
        d.text((90, y + 40), text, fill=(60, 60, 80), font=f)
    img.save(path, "PNG")


def screenshot_airbnb(path: Path) -> None:
    img = Image.new("RGB", (900, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fb = load_font(24)
    f = load_font(18)
    d.rectangle([0, 0, 900, 60], fill=(255, 56, 92))
    d.text((30, 18), "Airbnb · Brooklyn stay for wedding guests", fill=(255, 255, 255), font=fb)
    d.rectangle([30, 100, 430, 380], fill=(220, 210, 200))  # photo placeholder
    d.text((40, 360), "Cozy 2BR near Prospect Park", fill=(40, 40, 50), font=fb)
    d.text((460, 110), "$185 / night", fill=(40, 40, 50), font=fb)
    d.text((460, 150), "Sleeps 4 · 2 beds · wifi", fill=(80, 80, 100), font=f)
    d.text((460, 200), "Available June 12–16", fill=(60, 120, 80), font=f)
    d.text((460, 240), "Walk to wedding venue: 0.4 mi", fill=(60, 120, 80), font=f)
    img.save(path, "PNG")


def screenshot_zillow(path: Path) -> None:
    img = Image.new("RGB", (900, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fb = load_font(24)
    f = load_font(18)
    d.rectangle([0, 0, 900, 60], fill=(0, 106, 191))
    d.text((30, 18), "Zillow · Park Slope brownstone apt", fill=(255, 255, 255), font=fb)
    d.rectangle([30, 100, 430, 380], fill=(180, 170, 160))
    d.text((460, 110), "$3,200 / month", fill=(40, 40, 50), font=fb)
    d.text((460, 150), "2 bd · 1 ba · 850 sqft", fill=(80, 80, 100), font=f)
    d.text((460, 200), "Park Slope, Brooklyn", fill=(80, 80, 100), font=f)
    d.text((460, 240), "Light: south-facing · Kitchen: small", fill=(60, 60, 80), font=f)
    d.text((460, 280), "Available July 1", fill=(60, 120, 80), font=f)
    img.save(path, "PNG")


SCREENSHOTS = [
    ("screenshot_python_keyerror.png", screenshot_python_keyerror, 11, "work"),
    ("screenshot_calendar_sarah_venue.png", screenshot_calendar_sarah, 35, "wedding"),
    ("screenshot_slack_deploy_fail.png", screenshot_slack_deploy, 11, "work"),
    ("screenshot_airbnb_wedding_guests.png", screenshot_airbnb, 30, "wedding"),
    ("screenshot_zillow_park_slope.png", screenshot_zillow, 19, "apartment"),
]


# ----------------------------- voice memos -----------------------------------

VOICE_MEMOS = [
    ("voice_sarah_brooklyn_venue.mp3",
     "Sarah said the Brooklyn venue is available June 14th, capacity 120, includes catering.",
     34, "wedding"),
    ("voice_dry_cleaning.mp3",
     "Remember to pick up dry cleaning before Thursday.",
     5, "incidental"),
    ("voice_auth_bug_race_condition.mp3",
     "The bug is in the auth middleware, line 47, it's a race condition. Token expiry check uses less than instead of less than or equal.",
     12, "work"),
    ("voice_mom_ceramic_mug_gift.mp3",
     "Mom's birthday gift idea: that ceramic mug she liked at the market last weekend.",
     22, "incidental"),
    ("voice_park_slope_apartment.mp3",
     "The Park Slope place has good light but the kitchen is small. Need to think about it before Friday.",
     18, "apartment"),
]


def synth_voice(text: str, out_path: Path) -> None:
    from gtts import gTTS
    tts = gTTS(text=text, lang="en")
    tts.save(str(out_path))


# ----------------------------- videos ----------------------------------------

def make_clip_from_frames(frames: list[Image.Image], out_path: Path, fps: int = 24, seconds_per_frame: float = 3.0) -> None:
    tmp = OUT / "_tmp_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    duplicate = max(int(fps * seconds_per_frame), 1)
    idx = 0
    for f in frames:
        for _ in range(duplicate):
            f.save(tmp / f"f{idx:05d}.jpg", "JPEG", quality=88)
            idx += 1
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(tmp / "f%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp)


def video_dog_at_beach(path: Path) -> None:
    a = Image.new("RGB", (640, 360))
    photo_dog_at_beach(OUT / "_tmp.jpg")
    a = Image.open(OUT / "_tmp.jpg").resize((640, 360))
    (OUT / "_tmp.jpg").unlink()
    b = Image.new("RGB", (640, 360))
    img2 = fast_gradient((640, 360), (255, 200, 130), (40, 90, 160))
    d = ImageDraw.Draw(img2)
    d.rectangle([0, 240, 640, 360], fill=(230, 200, 150))
    d.ellipse([260, 200, 320, 260], fill=(30, 30, 30))
    d.ellipse([305, 180, 340, 215], fill=(30, 30, 30))
    img3 = fast_gradient((640, 360), (255, 220, 160), (60, 110, 170))
    d3 = ImageDraw.Draw(img3)
    d3.rectangle([0, 250, 640, 360], fill=(225, 195, 145))
    d3.ellipse([100, 220, 150, 270], fill=(20, 20, 20))
    make_clip_from_frames([a, img2, img3, a], path)


def video_city_walking(path: Path) -> None:
    photo_city_skyline(OUT / "_tmp.jpg")
    a = Image.open(OUT / "_tmp.jpg").resize((640, 360))
    (OUT / "_tmp.jpg").unlink()
    photo_coffee_shop(OUT / "_tmp.jpg")
    b = Image.open(OUT / "_tmp.jpg").resize((640, 360))
    (OUT / "_tmp.jpg").unlink()
    photo_brownstone(OUT / "_tmp.jpg")
    c = Image.open(OUT / "_tmp.jpg").resize((640, 360))
    (OUT / "_tmp.jpg").unlink()
    make_clip_from_frames([a, b, c], path)


def video_apartment_walkthrough(path: Path) -> None:
    photo_brownstone(OUT / "_tmp.jpg")
    a = Image.open(OUT / "_tmp.jpg").resize((640, 360))
    (OUT / "_tmp.jpg").unlink()
    # empty room
    b = Image.new("RGB", (640, 360), (235, 230, 220))
    d = ImageDraw.Draw(b)
    d.polygon([(0, 0), (640, 0), (520, 250), (120, 250)], fill=(220, 215, 205))
    d.rectangle([260, 80, 380, 230], fill=(160, 200, 230))  # window
    f = load_font(20)
    d.text((30, 320), "Park Slope · 2BR walkthrough", fill=(80, 80, 100), font=f)
    photo_moving_boxes(OUT / "_tmp.jpg")
    c = Image.open(OUT / "_tmp.jpg").resize((640, 360))
    (OUT / "_tmp.jpg").unlink()
    make_clip_from_frames([a, b, c], path)


VIDEOS = [
    ("video_dog_beach.mp4", video_dog_at_beach, 38, "incidental"),
    ("video_city_walking.mp4", video_city_walking, 9, "incidental"),
    ("video_apartment_park_slope.mp4", video_apartment_walkthrough, 17, "apartment"),
]


# ----------------------------- driver ----------------------------------------

def main() -> None:
    print(f"writing demo data to {OUT.resolve()}")
    OUT.mkdir(exist_ok=True)

    for name, fn, days_ago, _thread in PHOTOS:
        p = OUT / name
        fn(p)
        set_mtime(p, days_ago)
        print(f"  photo  {name:40s} ({days_ago}d ago)")

    for name, fn, days_ago, _thread in SCREENSHOTS:
        p = OUT / name
        fn(p)
        set_mtime(p, days_ago)
        print(f"  shot   {name:40s} ({days_ago}d ago)")

    for name, text, days_ago, _thread in VOICE_MEMOS:
        p = OUT / name
        synth_voice(text, p)
        set_mtime(p, days_ago)
        print(f"  voice  {name:40s} ({days_ago}d ago)")

    for name, fn, days_ago, _thread in VIDEOS:
        p = OUT / name
        fn(p)
        set_mtime(p, days_ago)
        print(f"  video  {name:40s} ({days_ago}d ago)")

    total = len(PHOTOS) + len(SCREENSHOTS) + len(VOICE_MEMOS) + len(VIDEOS)
    print(f"\ndone — {total} items written to {OUT.resolve()}")


if __name__ == "__main__":
    main()

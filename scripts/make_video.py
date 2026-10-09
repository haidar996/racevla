"""Render one MP4 that shows all five skills (title card, the five demos at their stated speed, results card). Also regenerates the GIFs of make_demos.py.
Needs imageio-ffmpeg (pip install imageio-ffmpeg). Usage: python scripts/make_video.py   ->  docs/media/racevla_demo.mp4  (1280x720, 25 fps, H.264)"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import make_demos as md

W, H, FPS = 1280, 720, 25


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()


def card(lines, seconds):
    img = Image.new("RGB", (W, H), (20, 20, 28)); d = ImageDraw.Draw(img); y = 150
    for text, size, color in lines: d.text((80, y), text, fill=color, font=font(size)); y += int(size * 1.7)
    return [np.asarray(img)] * int(seconds * FPS)


def results_card():
    rows = [("Stand + push recovery", "0 / 200 falls", "pushes up to 1.2 m/s"), ("Walking", "0 / 50 falls", "velocity commands, turning"), ("Running", "up to 2.5 m/s", "28% of the time all four feet airborne"),
            ("Blind terrain", "~99% success", "rough, slopes 15 deg, stairs"), ("Step over a bar", "80% at 7 cm", "while running at 1.3-1.7 m/s")]
    img = Image.new("RGB", (W, H), (20, 20, 28)); d = ImageDraw.Draw(img); d.text((70, 50), "RaceVLA: 5 locomotion skills for the Unitree Go1", fill=(255, 255, 255), font=font(40))
    d.text((70, 110), "Reinforcement learning in MuJoCo, trained on a laptop CPU (no GPU)", fill=(150, 190, 240), font=font(24))
    for i, (a, b, c) in enumerate(rows):
        y = 190 + i * 90; d.rectangle([70, y, W - 70, y + 76], fill=(34, 34, 46)); d.text((100, y + 20), a, fill=(255, 255, 255), font=font(28)); d.text((540, y + 20), b, fill=(127, 224, 160), font=font(28)); d.text((800, y + 28), c, fill=(190, 190, 190), font=font(19))
    d.text((70, 650), "Simulation only. Failures documented too.   github.com/haidar996/racevla", fill=(150, 190, 240), font=font(22))
    return [np.asarray(img)] * (4 * FPS)


def clip(rec):
    """Frames resampled to FPS so that the playback speed is the one the GIF uses (rec.speed)."""
    ends = np.cumsum([d / rec.speed for d in rec.dts]); out = []
    for k in range(int(ends[-1] * FPS)):
        i = int(np.searchsorted(ends, k / FPS, side="right")); i = min(i, len(rec.frames) - 1)
        img = Image.fromarray(rec.frames[i]).resize((md.W * 2, md.H * 2), Image.LANCZOS); canvas = Image.new("RGB", (W, H), (20, 20, 28)); canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2)); out.append(np.asarray(canvas))
    return out


if __name__ == "__main__":
    frames = card([("RaceVLA", 84, (255, 255, 255)), ("Five reinforcement-learning skills for a Unitree Go1", 36, (150, 190, 240)), ("MuJoCo simulation, trained on a laptop CPU", 28, (190, 190, 190))], 3)
    for name in ["standing", "walking", "running", "terrain", "stepover"]:
        print("==", name, flush=True); frames += clip(getattr(md, name)())
    frames += results_card(); out = ROOT / "docs" / "media" / "racevla_demo.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    for f in frames: w.append_data(f)
    w.close(); print(f"{out}  {len(frames) / FPS:.0f} s  {out.stat().st_size / 1e6:.1f} MB")

"""Render the proposed deployment diagram for the product design document."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/AI_Shield_Proposed_Architecture.png"
image = Image.new("RGB", (1440, 670), "white")
draw = ImageDraw.Draw(image)
regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 25)
ink = "#202b33"


def box(rect, title, subtitle=None, fill="#f4f6f7"):
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline="#5d707a", width=2)
    x = (rect[0] + rect[2]) / 2
    y = (rect[1] + rect[3]) / 2
    draw.text((x, y - (20 if subtitle else 0)), title, font=bold, fill=ink, anchor="mm")
    if subtitle:
        draw.text((x, y + 23), subtitle, font=regular, fill=ink, anchor="mm")


def arrow(points):
    draw.line(points, fill="#446570", width=4)
    x, y = points[-1]
    px, py = points[-2]
    if y > py:
        head = [(x, y), (x-9, y-15), (x+9, y-15)]
    elif y < py:
        head = [(x, y), (x-9, y+15), (x+9, y+15)]
    elif x < px:
        head = [(x, y), (x+15, y-9), (x+15, y+9)]
    else:
        head = [(x, y), (x-15, y-9), (x-15, y+9)]
    draw.polygon(head, fill="#446570")


box((430, 15, 1010, 110), "User's browser", "Upload image / view results", "#eaf3f5")
draw.rounded_rectangle((35, 155, 1360, 650), radius=12, outline="#7c8991", width=3)
draw.text((65, 178), "Oracle Cloud compute instance - proposed", font=small, fill=ink)
box((420, 235, 1020, 310), "Python / FastAPI", "Validate original upload")
arrow([(720, 110), (720, 235)])
box((65, 365, 465, 465), "Selected detector", "Fine-tuned CF", "#eaf3f5")
box((510, 365, 910, 465), "EXIF + C2PA", "Recorded provenance", "#f3f0e7")
box((955, 365, 1335, 465), "Context - PLANNED", "Scene / caption checks", "#f0edf7")
arrow([(720, 310), (720, 335), (265, 335), (265, 365)])
arrow([(720, 335), (720, 365)])
arrow([(720, 335), (1145, 335), (1145, 365)])
draw.text((1145, 490), "Model host / sources TBD", font=small, fill=ink, anchor="mm")
box((420, 530, 1020, 620), "Combined results report", "Separate detection and evidence")
arrow([(265, 465), (265, 575), (420, 575)])
arrow([(720, 465), (720, 530)])
arrow([(1145, 510), (1145, 575), (1020, 575)])
arrow([(1020, 595), (1395, 595), (1395, 62), (1010, 62)])
draw.text((1080, 30), "API response", font=small, fill=ink)
OUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUT)
print(OUT)

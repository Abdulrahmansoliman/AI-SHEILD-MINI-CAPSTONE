from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path('.local/presentations/site-packages').resolve()))
import pymupdf
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
pdf = pymupdf.open(root / 'AI_Shield_Capstone_Revised.pdf')
sheet = Image.new('RGB', (1440, 5 * 290), '#dce5e2')
draw = ImageDraw.Draw(sheet)
audit = []
for i, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1))
    out = root / f'revised_{i+1:02d}.png'
    pix.save(out)
    im = Image.open(out).convert('RGB')
    im.thumbnail((460, 259))
    x, y = (i % 3) * 480 + 10, (i // 3) * 290 + 22
    sheet.paste(im, (x, y))
    draw.text((x, y - 16), f'{i+1:02d}', fill='black')
    spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
    bad = [s['text'] for s in spans if s['bbox'][0] < 0 or s['bbox'][1] < 0 or s['bbox'][2] > page.rect.width + 1 or s['bbox'][3] > page.rect.height + 1]
    audit.append({'slide': i+1, 'words': len(page.get_text().split()), 'out_of_bounds': bad})
sheet.save(root / 'revised_overview.png')
(root / 'revised_verification.json').write_text(json.dumps(audit, indent=2))
print(json.dumps({'pages': len(pdf), 'audit': audit}))

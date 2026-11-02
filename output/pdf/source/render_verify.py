from pathlib import Path
import json
import pymupdf as fitz
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent.parent
out=ROOT.parents[1]/'tmp/pdfs/par_report';out.mkdir(parents=True,exist_ok=True)
for name,pdf in [('report',ROOT/'bao_cao_PAR.pdf'),('template',ROOT.parents[1]/'Slide (6).pdf')]:
    doc=fitz.open(pdf)
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(out/f'{name}_{i+1:02d}.png')
    for sheet in range(4):
        canvas=Image.new('RGB',(1000,910),'#dce2e8');draw=ImageDraw.Draw(canvas)
        for k in range(6):
            n=sheet*6+k+1;im=Image.open(out/f'{name}_{n:02d}.png');im.thumbnail((480,270))
            x=10+(k%2)*500;y=10+(k//2)*300;canvas.paste(im,(x,y));draw.text((x+4,y+274),f'Page {n:02d}',fill='black')
        canvas.save(out/f'{name}_contact_{sheet+1}.png')
print(out)

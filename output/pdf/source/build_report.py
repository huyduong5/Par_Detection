"""Reproducible 24-page PAR report. Inputs are frozen snapshots, never live jobs."""
from pathlib import Path
import csv,hashlib,json,math,sys
import numpy as np
import pymupdf as fitz
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from metric_reference import par_metrics

HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ASSET=HERE/'assets';DATA=HERE/'data'
CONTENT=json.loads((HERE/'content.json').read_text());COPY=CONTENT['copy'];SNAP=json.loads((DATA/'server_snapshot.json').read_text())
W,H=960,540;NAVY=HexColor('#082f5c');BLUE=HexColor('#157aab');INK=HexColor('#172d43');GRAY=HexColor('#566677');PALE=HexColor('#edf5fa');GREEN=HexColor('#19754a');ORANGE=HexColor('#a96512')
FONT=str(ASSET/'fonts/DejaVuSans.ttf');BOLD=str(ASSET/'fonts/DejaVuSans-Bold.ttf')
pdfmetrics.registerFont(TTFont('PAR',FONT));pdfmetrics.registerFont(TTFont('PAR-Bold',BOLD));pdfmetrics.registerFontFamily('PAR',normal='PAR',bold='PAR-Bold',italic='PAR',boldItalic='PAR-Bold')
C=None;PAGE=0;TEXT_BOXES=[];EVIDENCE=[]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def evidence(page,claim,path,**extra):
    path=Path(path);EVIDENCE.append(dict(page=page,claim=claim,path=str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path),sha256=digest(path),**extra))
def text(s,x,y,w=850,size=18,color=INK,bold=False,leading=None,align=0):
    style=ParagraphStyle('p',fontName='PAR-Bold' if bold else 'PAR',fontSize=size,leading=leading or size*1.38,textColor=color,alignment=align,spaceAfter=0)
    p=Paragraph(s,style);_,h=p.wrap(w,600)
    assert x>=0 and x+w<=W+.1 and y>=0 and y+h<=H-34+.1,(PAGE,s[:60],x,y,w,h)
    p.drawOn(C,x,H-y-h);TEXT_BOXES.append(dict(page=PAGE,x=x,y=y,width=w,height=h,text=s));return h

def line(x1,y1,x2,y2,color=BLUE,width=1):
    C.setStrokeColor(color);C.setLineWidth(width);C.line(x1,H-y1,x2,H-y2)
def rect(x,y,w,h,color=PALE,border=None,radius=0):
    C.setFillColor(color);C.setStrokeColor(border or color)
    if radius:C.roundRect(x,H-y-h,w,h,radius,fill=1,stroke=bool(border))
    else:C.rect(x,H-y-h,w,h,fill=1,stroke=bool(border))
def image(path,x,y,w,h):
    C.drawImage(str(path),x,H-y-h,w,h,preserveAspectRatio=True,anchor='c',mask='auto')
def arrow(x1,y1,x2,y2,color=BLUE,dash=False):
    C.saveState()
    if dash:C.setDash(4,3)
    line(x1,y1,x2,y2,color,1.7);C.restoreState()
    angle=math.atan2(y2-y1,x2-x1);l=7
    line(x2,y2,x2-l*math.cos(angle-.45),y2-l*math.sin(angle-.45),color,1.7)
    line(x2,y2,x2-l*math.cos(angle+.45),y2-l*math.sin(angle+.45),color,1.7)
def node(title,subtitle,x,y,w,h=76,fill=PALE):
    rect(x,y,w,h,fill,border=HexColor('#c8dbe8'),radius=5)
    text(title,x+10,y+10,w-20,16,bold=True,align=1)
    if subtitle:text(subtitle,x+8,y+37,w-16,12.5,GRAY,align=1)
def note(s,y=471):text(s,44,y,865,10.5,GRAY,leading=14)
def lead(s):text(s,48,86,864,23,NAVY,bold=True)
def badge(label,x=48,y=126,color=GREEN,w=160):
    rect(x,y,w,25,color,radius=3);text(label,x+7,y+4,w-14,11,white,bold=True)
def bullets(items,x,y,w,size=18,gap=16):
    for item in items:
        h=text('• '+item,x,y,w,size);y+=h+gap
    return y

def table(headers,rows,x,y,widths,size=14,rowheight=None):
    style=ParagraphStyle('t',fontName='PAR',fontSize=size,leading=size*1.28,textColor=INK)
    headstyle=ParagraphStyle('th',parent=style,fontName='PAR-Bold',textColor=white)
    cells=[[Paragraph(str(v),headstyle) for v in headers]]+[[Paragraph(str(v),style) for v in row] for row in rows]
    t=Table(cells,colWidths=widths,rowHeights=rowheight,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,PALE]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('LINEBELOW',(0,-1),(-1,-1),.6,HexColor('#b4c9d6'))]))
    tw,th=t.wrap(880,440);assert y+th<500,(PAGE,'table overflow',y,th);t.drawOn(C,x,H-y-th);return th

def base(title,index):
    rect(0,0,W,H,white);rect(0,0,W,58,NAVY)
    text(title,16,16,875,19,white)
    image(ASSET/'logo_header.png',912,3,47,50)
    rect(889,507,71,33,NAVY)
    C.setFillColor(white);C.setFont('PAR',16);C.drawCentredString(924,H-530,f'{index}/22')

def agenda(active=None):
    for i,s in enumerate(CONTENT['sections']):
        y=182+i*53;C.setFillColor(NAVY);C.circle(84,H-y-13,15,fill=1,stroke=0)
        C.setFillColor(white);C.setFont('PAR',17);C.drawCentredString(84,H-y-19,str(i+1))
        text(s,137,y,730,19,NAVY if i==active else INK,bold=i==active)

def cover(thanks=False):
    if thanks:
        rect(0,0,960,380,NAVY);rect(0,381,960,159,HexColor('#f0f0f0'))
        text('Cảm ơn thầy và các bạn<br/>đã lắng nghe!',118,99,772,54,white,leading=76)
        image(ASSET/'logo_thanks_faculty.png',780,309,60,60);image(ASSET/'logo_thanks_university.png',858,309,60,61)
        text('Dương Quang Huy',120,445,600,21,INK,bold=True)
        return
    C.linearGradient(0,0,160,540,[HexColor('#0b5a8d'),HexColor('#60d2e6')],extend=True)
    image(ASSET/'logo_faculty.png',44,12,82,83);image(ASSET/'logo_university.png',830,5,76,87)
    text('KHOA TOÁN – CƠ – TIN HỌC',143,37,410,20,white,bold=True)
    text('TRƯỜNG ĐẠI HỌC<br/>KHOA HỌC TỰ NHIÊN',580,32,252,20,white,bold=True,leading=26)
    for v in [0,12,48]:line(727+v,518-v,960,285-v,white,1.1)
    if thanks:
        text('Cảm ơn thầy và các bạn<br/>đã lắng nghe!',135,210,690,34,white,bold=True,align=1)
        text('Dương Quang Huy',250,345,460,23,white,align=1)
    else:
        rect(190,176,580,151,NAVY,border=HexColor('#031932'),radius=22)
        text(CONTENT['title'].replace('\n','<br/>'),208,199,544,26.5,white,bold=True,leading=35,align=1)
        text(CONTENT['author'],220,365,520,24,white,align=1)
        text('Báo cáo tiến độ nghiên cứu • '+CONTENT['date'],270,410,420,13,white,align=1)

def verify_modern(folder,split,count):
    results={};arrays={}
    for s,tag in [(1,'clean'),(2,'x2'),(4,'x4'),(8,'x8')]:
        p=folder/f'{split}_{tag}.json';r=json.loads(p.read_text());npz=p.with_suffix('.npz')
        assert r['artifact_state']=='complete' and r['samples']==count and r['split']==split and r['scale']==s and r['threshold']==.5
        entry=next(e for e in SNAP['registry']['models'] if e['id']==r['model_id'])
        assert entry['checkpoint_sha256']==r['checkpoint_sha256'] and r['attributes']==SNAP['index']['attributes']
        assert r['clean_reference_parity']['state']=='PASS'
        assert digest(npz)==r['predictions_sha256']
        with np.load(npz,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        assert par_metrics(a['clean'],a['labels'])==r['par']
        assert a['attributes'].tolist()==SNAP['index']['attributes']
        if arrays:assert np.array_equal(a['names'],arrays[1]['names']) and np.array_equal(a['labels'],arrays[1]['labels'])
        results[s]=r;arrays[s]=a;evidence(19,f'{split} scale {s} verified metrics/predictions',p,prediction_path=str(npz.relative_to(HERE)),prediction_sha256=digest(npz))
    return results,arrays

MODERN,ARRAYS=verify_modern(DATA/'resolution_validation_C3','val',6002)
SMOKE,_=verify_modern(DATA/'resolution_smoke','val',128)
LEGACY=[]
for label,filename in [('C0','C0_test.json'),('C3','C3_test.json'),('C3_MS','C3_MS_test.json'),('C4b','C4b_bs16_test.json'),('C4b_MS','C4b_MS_bs16_test.json')]:
    p=DATA/'legacy_eval'/filename;r=json.loads(p.read_text());assert r['num_test']==23822
    if 'clean' not in r:r['clean']={'mA':r['mA']}
    assert len(r['per_attr_mA'])==57 and abs(np.mean(list(r['per_attr_mA'].values()))-r['clean']['mA'])<1e-10
    LEGACY.append((label,r));evidence(18,'Historical clean/occluded mA; seed42',p,metric_semantics='mA only; historical accuracy and F1 intentionally omitted')


def page_content(kind):
    if kind=='agenda':agenda()
    elif kind=='divider':agenda(CONTENT['pages'][PAGE-1]['section'])
    elif kind=='problem':
        lead(COPY['s001'])
        image(ASSET/'sample_clean.png',65,143,190,290)
        arrow(281,280,353,280)
        table(COPY['s002'],COPY['s003'],385,165,[135,355],16)
        text(COPY['s004'],385,367,490,18)
        note(COPY['s005'])
    elif kind=='survey':
        lead(COPY['s006'])
        table(COPY['s007'],COPY['s008'],48,151,[185,280,399],16)
        text(COPY['s009'],63,405,840,19,NAVY,bold=True)
        note(COPY['s010'])
    elif kind=='dataset':
        lead(COPY['s011'])
        table(COPY['s012'],COPY['s013'],48,146,[210,170,484],18)
        bullets(COPY['s014'],55,352,850,16,10)
        note(COPY['s015'])
    elif kind=='task':
        lead(COPY['s016'])
        node(COPY['s017'],COPY['s018'],62,157,205,95)
        arrow(278,204,354,204);node(COPY['s019'],COPY['s020'],365,157,230,95)
        arrow(607,204,679,204);node(COPY['s021'],COPY['s022'],690,157,210,95)
        text(COPY['s023'],78,284,820,23,NAVY,bold=True)
        table(COPY['s024'],COPY['s025'],62,330,[300,538],16)
    elif kind=='theory':
        lead(COPY['s026'])
        for x,title,body in [(48,'VMamba / SS2D [4]','Quét chọn lọc 2D để thu thập ngữ cảnh ảnh.<br/><br/>Vai trò: trích xuất đặc trưng thị giác.'),(344,'Mamba văn bản [3]','Mã hóa tên thuộc tính cùng các soft prompt học được.<br/><br/>Vai trò: cung cấp biểu diễn ngữ nghĩa.'),(640,'Attribute Query','Mỗi query thu nhận thông tin từ token ảnh.<br/><br/>Vai trò: tạo đặc trưng và logits theo thuộc tính.')]:
            line(x,151,x+264,151,NAVY,3);text(title,x,170,264,21,NAVY,bold=True);text(body,x,218,264,18)
        text(COPY['s027'],63,403,830,19,GRAY)
        note(COPY['s028'])
    elif kind=='architecture':
        node(COPY['s029'],COPY['s030'],40,115,188)
        arrow(228,153,269,153);node(COPY['s031'],COPY['s032'],279,115,224)
        node(COPY['s033'],COPY['s034'],40,255,215)
        arrow(255,293,283,293);node(COPY['s035'],COPY['s036'],293,255,210)
        node(COPY['s037'],COPY['s038'],553,115,349)
        arrow(503,153,553,153);arrow(503,293,530,293);arrow(530,293,530,164);arrow(530,164,553,164)
        node(COPY['s039'],COPY['s040'],553,255,349)
        arrow(729,191,729,255);arrow(503,293,553,293)
        node(COPY['s041'],COPY['s042'],550,378,171,64);node(COPY['s043'],COPY['s044'],739,378,171,64)
        arrow(690,331,636,378);arrow(767,331,824,378)
        text(COPY['s045'],48,371,453,16,NAVY,bold=True)
        text(COPY['s046'],48,409,455,14,GRAY)
        note(COPY['s047'])
    elif kind=='multiscale':
        lead(COPY['s048'])
        node(COPY['s049'],COPY['s050'],65,151,297)
        node(COPY['s051'],COPY['s052'],65,245,297,64)
        arrow(362,189,451,229);arrow(362,277,451,249)
        node(COPY['s053'],COPY['s054'],468,195,422,98)
        table(COPY['s055'],COPY['s056'],48,320,[200,664],14)
        note(COPY['s057'])
    elif kind=='visibility':
        lead(COPY['s058'])
        table(COPY['s059'],COPY['s060'],48,146,[310,554],18)
        text(COPY['s061'],54,352,850,22,NAVY,bold=True)
        bullets(COPY['s062'],58,399,840,18,9)
        note(COPY['s063'])
    elif kind=='eopar':
        lead(COPY['s064'])
        badge(COPY['s065'],y=128,color=ORANGE,w=210)
        node(COPY['s066'],COPY['s067'],60,204,260,100);arrow(320,254,398,254)
        node(COPY['s068'],COPY['s069'],410,204,250,100)
        arrow(660,233,716,191);arrow(660,277,716,333)
        node(COPY['s070'],COPY['s071'],730,151,181,82)
        node(COPY['s072'],COPY['s073'],730,307,181,82)
        bullets(COPY['s074'],55,365,650,17,10)
        note(COPY['s075'])
    elif kind=='loss':
        lead(COPY['s076'])
        text(COPY['s077'],53,143,410,21,NAVY,bold=True)
        text(COPY['s078'],53,188,850,23)
        text(COPY['s079'],53,236,850,17)
        line(48,300,912,300,HexColor('#cad8e2'))
        text(COPY['s080'],53,321,850,21,ORANGE,bold=True)
        text(COPY['s081'],53,365,850,23)
        text(COPY['s082'],53,410,850,21)
        note(COPY['s083'])
    elif kind=='setup':
        table(COPY['s084'],COPY['s085'],48,111,[223,641],17)
        note(COPY['s086'])
    elif kind=='results':
        lead(COPY['s087'])
        rows=[]
        for label,r in LEGACY:
            rows.append([label,f"{r['clean']['mA']*100:.2f}",f"{r.get('occluded',{}).get('mA',0)*100:.2f}" if 'occluded' in r else 'Không có lượt đo',f"{r.get('consistency',{}).get('consistency_gap',0):.4f}" if 'consistency' in r else 'Không báo'])
        table(COPY['s088'],rows,48,145,[180,195,270,219],17)
        c3=dict(LEGACY)['C3_MS'];c4=dict(LEGACY)['C4b_MS'];dc=(c4['clean']['mA']-c3['clean']['mA'])*100;do=(c4['occluded']['mA']-c3['occluded']['mA'])*100
        text(f'C4b_MS so với C3_MS: clean {dc:+.2f} điểm phần trăm;<br/>che tổng hợp {do:+.2f} điểm phần trăm trong các lượt đo lịch sử.',57,416,850,18,NAVY,bold=True)
        note(COPY['s089'])
    elif kind=='resolution':
        # Complete validation artifacts, not unfinished test or smoke scores.
        for i,tag in enumerate(['clean','x2','x4','x8']):
            x=50+i*135;image(ASSET/f'sample_{tag}.png',x,99,105,154)
            text('Clean' if i==0 else '×'+tag[1:],x,261,105,15,NAVY,bold=True,align=1)
        text(COPY['s090'],617,94,292,15,NAVY,bold=True)
        chart_x,chart_y,chart_w,chart_h=657,126,237,123
        for tick in [60,70,80]:
            yy=chart_y+(85-tick)/30*chart_h
            line(chart_x,yy,chart_x+chart_w,yy,HexColor('#d7e2ea'),.6)
            text(str(tick),617,yy-8,34,11,GRAY,align=2)
        points=[]
        for i,s in enumerate([1,2,4,8]):
            xx=chart_x+i*chart_w/3;yy=chart_y+(85-MODERN[s]['par']['mA']*100)/30*chart_h
            points.append((xx,yy));text(str(s),xx-13,257,27,11,GRAY,align=1)
            C.setFillColor(BLUE);C.circle(xx,H-yy,3.4,fill=1,stroke=0)
        for a,b in zip(points,points[1:]):line(*a,*b,BLUE,2.2)
        text(COPY['s091'],617,114,80,10,GRAY)
        text(COPY['s092'],686,277,204,10,GRAY,align=1)
        text(COPY['s093'],49,284,550,11.5,GRAY)
        vals=[f"{MODERN[s]['par']['mA']*100:.2f}" for s in [1,2,4,8]]
        table(COPY['s094'],[['mA (%)']+vals,['Mean-instance F1 (%)']+[f"{MODERN[s]['par']['instance_f1_mean']*100:.2f}" for s in [1,2,4,8]]],48,316,[235,155,155,155,164],15.5)
        drop=(MODERN[1]['par']['mA']-MODERN[8]['par']['mA'])*100
        text(f'6.002 ảnh ghép cặp: mA giảm {drop:.2f} điểm phần trăm tại ×8.',54,440,850,17,NAVY,bold=True)
        note(COPY['s095'])
    elif kind=='conclusion':
        table(COPY['s096'],COPY['s097'],48,118,[210,654],15.5)
        note('Các kết luận giới hạn theo mốc '+CONTENT['date']+'. Không suy rộng robustness che tổng hợp sang mất chi tiết hoặc ảnh tự nhiên.')
    elif kind=='future':
        lead(COPY['s098'])
        stages=[('1','Hoàn thiện benchmark','Đủ test cho 9 checkpoint; phân tích theo thuộc tính, kích thước ảnh và seed.'),('2','Sửa và xác minh vùng','Metadata vùng hợp lệ; test objective/gradient/resume trên model thật.'),('3','Pilot EOPAR có gate','Đối chứng head / ranking / hai can thiệp; không dùng test để chọn model.'),('4','Thực nghiệm và bản thảo','800 ảnh tự nhiên có nhãn; đa dataset, cross-domain, chi phí và giới hạn.')]
        for i,(n,title,body) in enumerate(stages):
            y=148+i*77;C.setFillColor(NAVY);C.circle(72,H-y-13,16,fill=1,stroke=0);C.setFillColor(white);C.setFont('PAR',17);C.drawCentredString(72,H-y-19,n)
            text(title,108,y-3,775,20,NAVY,bold=True);text(body,108,y+28,775,15)
        note(COPY['s099'])
    elif kind=='references':
        y=83
        for r in CONTENT['references']:
            h=text(f"[{r['id']}] {r['text']}",43,y,872,13.2,leading=18);y+=h+3
            h2=text(f"<link href=\"{r['url']}\" color=\"#157aab\">{r['url']}</link>",68,y,835,11.5,leading=15);y+=h2+13
        note(COPY['s100'],y=478)
    else:raise ValueError(kind)


def build():
    global C,PAGE
    OUT.mkdir(exist_ok=True);C=canvas.Canvas(str(OUT/'bao_cao_PAR.pdf'),pagesize=(W,H),pageCompression=1)
    C.setTitle('Nhận dạng thuộc tính người đi bộ trong điều kiện suy giảm bằng chứng ảnh')
    C.setAuthor(CONTENT['author']);C.setSubject('PAR: AQ/VMamba baseline, EOPAR proposal and resolution degradation; verified snapshot '+CONTENT['date'])
    for PAGE,page in enumerate(CONTENT['pages'],1):
        C.bookmarkPage(f'page{PAGE}')
        if page['kind']=='cover':cover()
        elif page['kind']=='thanks':cover(thanks=True)
        else:base(page['title'],PAGE-1);page_content(page['kind'])
        C.showPage()
    C.save()
    evidence(7,'Dataset counts, label order, registry and worker cutoff',DATA/'server_snapshot.json')
    evidence(17,'CPU verification report, frozen source fingerprint',DATA/'cpu_verification.json')
    for page,claim,path in [(11,'Actual AQ forward path','aq_visibility/model/climp_par.py'),(13,'Fixed-band anatomical failure and EOPAR training block','eopar/research/verification/20261101_v2/REPORT.md'),(15,'Actual objective implementation','eopar/losses/objective.py')]:
        original=HERE.parents[2]/path
        dest=HERE/'data/code_evidence'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        if original.exists():dest.write_bytes(original.read_bytes())
        assert dest.is_file(),f'Missing frozen code evidence: {dest}'
        evidence(page,claim,dest)
    with (HERE/'PROVENANCE.csv').open('w',newline='') as f:
        fields=sorted(set(k for r in EVIDENCE for k in r));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(EVIDENCE)
    (HERE/'text_boxes.json').write_text(json.dumps(TEXT_BOXES,ensure_ascii=False,indent=2))
    doc=fitz.open(OUT/'bao_cao_PAR.pdf');assert len(doc)==24
    for i,p in enumerate(doc):
        assert tuple(p.rect)==(0,0,960,540)
        if 0<i<23:assert f'{i}/22' in p.get_text()
        assert '�' not in p.get_text()
    body='\n'.join(p.get_text() for p in doc)
    for required in ['EOPAR','MSP60K','6.002','23.822','119.296','C3_MS','02/11/2026']:assert required in body,required
    qa=dict(state='PASS',pages=24,size_pt=[960,540],font='Embedded DejaVu Sans / Bold; substitutes missing Tahoma/Arial',metric_roundtrip='All four C3 full-validation and four smoke NPZ files verified; no smoke performance used as test',historical_validation='mA equals mean of 57 per-attribute values; n_test=23822',cutoff=CONTENT['date'],pdf_sha256=digest(OUT/'bao_cao_PAR.pdf'),visual_review='PENDING render inspection')
    (OUT/'verification.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2));print(json.dumps(qa,ensure_ascii=False,indent=2))

if __name__=='__main__':build()

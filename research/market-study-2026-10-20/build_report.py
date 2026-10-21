from pathlib import Path
from html import escape
import re, json, math
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
pdf_path = OUT / 'global-market-study-opencv-hackathon.pdf'
html_path = BASE / 'market-study.html'
notebook_path = BASE / 'evidence-notebook.html'

pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('ArialItalic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='ArialBold', italic='ArialItalic', boldItalic='ArialBold')

def norm(s):
    return s.replace('\u2011','-').replace('\u2013','-').replace('\u2014',' - ').replace('\u00a0',' ')

token = re.compile(r'(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\(https?://[^\s)]+\))')
def inline(s, pdf=False):
    out=[]
    for part in token.split(norm(s)):
        if part.startswith('**') and part.endswith('**'):
            out.append(('<b>' if pdf else '<strong>')+escape(part[2:-2])+('</b>' if pdf else '</strong>'))
        elif part.startswith('`') and part.endswith('`'):
            out.append(('<font name="Courier">' if pdf else '<code>')+escape(part[1:-1])+('</font>' if pdf else '</code>'))
        elif part.startswith('['):
            m=re.fullmatch(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',part)
            if m:
                tag='link' if pdf else 'a'
                extra=' color="#165D85"' if pdf else ' target="_blank" rel="noopener"'
                out.append(f'<{tag} href="{escape(m[2], quote=True)}"{extra}>{escape(m[1])}</{tag}>')
            else: out.append(escape(part))
        else: out.append(escape(part))
    return ''.join(out)

def blocks(text):
    lines=text.splitlines(); i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line=='<!--pagebreak-->': yield ('break',None); i+=1; continue
        if line.startswith('#'):
            m=re.match(r'^(#{1,6})\s+(.*)',line)
            if m: yield ('h'+str(len(m[1])),m[2]); i+=1; continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in cells): rows.append(cells)
                i+=1
            yield ('table',rows); continue
        if line.startswith('- '): yield ('bullet',line[2:]); i+=1; continue
        para=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','- ','<!--pagebreak')):
            para.append(lines[i].strip()); i+=1
        yield ('p',' '.join(para))

CSS='''
:root {color-scheme: light; --ink:#17252e; --muted:#526570; --blue:#165d85;}
*{box-sizing:border-box}body{margin:0;background:#eef2f4;color:var(--ink);font:17px/1.65 "Segoe UI",Arial,sans-serif}
main{max-width:1120px;margin:30px auto;background:white;padding:56px 68px;box-shadow:0 3px 24px #17334712}
nav{padding:18px 22px;background:#f4f7f9;margin:28px 0}nav a{margin-right:20px}
h1{font-size:38px;line-height:1.15;margin:0 0 18px;color:#000}h2{font-size:27px;line-height:1.25;margin:48px 0 18px;color:#000}
h3{font-size:21px;line-height:1.35;margin-top:32px;color:#000}p{margin:0 0 17px}a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:3px}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;margin:24px 0;table-layout:auto}th,td{border:1px solid #d9dfe3;padding:12px 13px;vertical-align:top}th{background:#24485e;color:white;text-align:left}tbody tr:nth-child(even){background:#f4f7f9}
details{border-top:1px solid #d9dfe3;padding:22px 0}summary{font-size:22px;font-weight:650;cursor:pointer}details article{padding:20px 0}
.break{height:12px}.meta{color:var(--muted);font-size:14px}code{background:#edf1f3;padding:1px 4px}li{margin-bottom:8px}
@media(max-width:800px){main{padding:26px 20px;margin:0}h1{font-size:30px}table{display:block;overflow-x:auto}th,td{min-width:110px}}
@media print{body{background:white}main{box-shadow:none;max-width:none;margin:0;padding:0}.break{break-before:page}nav{display:none}details article{display:block}}
'''

def render_html(text, prefix='section'):
    output=[]; headings=[]; count=0
    for kind,data in blocks(text):
        if kind=='break': output.append('<div class="break"></div>')
        elif kind.startswith('h'):
            count+=1; ident=f'{prefix}-{count}'
            output.append(f'<{kind} id="{ident}">{inline(data)}</{kind}>')
            if kind=='h2': headings.append((ident,data))
        elif kind=='p': output.append('<p>'+inline(data)+'</p>')
        elif kind=='bullet': output.append('<ul><li>'+inline(data)+'</li></ul>')
        elif kind=='table':
            output.append('<table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in data[0])+'</tr></thead><tbody>')
            for row in data[1:]: output.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>')
            output.append('</tbody></table>')
    return ''.join(output),headings

source=(BASE/'market-study.md').read_text(encoding='utf-8')
body,heads=render_html(source)
nav='<nav>'+''.join(f'<a href="#{i}">{escape(t)}</a>' for i,t in heads)+'</nav>'
nav+='<p class="meta"><a href="evidence-notebook.html">Full research memos and source ledgers</a> | <a href="../../output/pdf/global-market-study-opencv-hackathon.pdf">PDF report</a></p>'
body=body.replace('</h1>','</h1>'+nav,1)
html_path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Global market study for the OpenCV hackathon</title><style>'+CSS+'</style><main>'+body+'</main></html>',encoding='utf-8')

memos=[('Industrial operations','industrial-research.md'),('Infrastructure energy and agriculture','infrastructure-research.md'),('Frontline and document workflows','frontline-research.md'),('Competition and additional sector screening','competition-and-screening.md')]
evidence=['<h1>Market study evidence notebook</h1><p>Research date 20 October 2026. Global desk research. Each memo preserves claim-level links, source context and limitations. No customer interviews or outreach occurred.</p><p><a href="market-study.html">Read the synthesis and project shortlist</a></p>']
source_rows=[]; seen=set()
for j,(title,filename) in enumerate(memos):
    md=(BASE/filename).read_text(encoding='utf-8')
    rendered,_=render_html(md,f'memo{j}')
    evidence.append('<details><summary>'+escape(title)+'</summary><article>'+rendered+'</article></details>')
    for label,url in re.findall(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',md):
        if url not in seen: seen.add(url);source_rows.append({'title':label,'url':url,'memo':filename,'accessed':'2026-10-20'})
notebook_path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Market study evidence notebook</title><style>'+CSS+'</style><main>'+''.join(evidence)+'</main></html>',encoding='utf-8')
(BASE/'source-index.json').write_text(json.dumps(source_rows,indent=2,ensure_ascii=False),encoding='utf-8')

styles={
 'p':ParagraphStyle('Body',fontName='Arial',fontSize=10.1,leading=14.2,spaceAfter=9,textColor=colors.HexColor('#17252e')),
 'h1':ParagraphStyle('Title',fontName='ArialBold',fontSize=25,leading=29,spaceAfter=17,textColor=colors.black),
 'h2':ParagraphStyle('Section',fontName='ArialBold',fontSize=19,leading=23,spaceAfter=13,keepWithNext=True,textColor=colors.black),
 'h3':ParagraphStyle('Subsection',fontName='ArialBold',fontSize=12,leading=16,spaceBefore=9,spaceAfter=8,keepWithNext=True,textColor=colors.black),
 'cell':ParagraphStyle('Cell',fontName='Arial',fontSize=8.35,leading=11.1,textColor=colors.HexColor('#17252e')),
 'head':ParagraphStyle('Header',fontName='ArialBold',fontSize=8.35,leading=11.1,textColor=colors.white),
}
width=A4[0]-96
story=[]
def colwidths(rows):
    n=len(rows[0]); hdr=rows[0]
    if n==8: return [width*.31]+[width*.095]*6+[width*.12]
    if n==4: return [width*x for x in ([.12,.32,.26,.30] if hdr[0]=='Priority' else [.18,.26,.27,.29])]
    if n==3: return [width*.24,width*.39,width*.37]
    if n==2: return [width*.19,width*.81]
    return [width/n]*n

for kind,data in blocks(source):
    if kind=='break': story.append(PageBreak())
    elif kind.startswith('h'):
        story.append(Paragraph(inline(data,True),styles.get(kind,styles['h3'])))
    elif kind in ('p','bullet'):
        story.append(Paragraph(('&#8226; ' if kind=='bullet' else '')+inline(data,True),styles['p']))
    elif kind=='table':
        cells=[[Paragraph(inline(cell,True),styles['head'] if r==0 else styles['cell']) for cell in row] for r,row in enumerate(data)]
        table=Table(cells,colWidths=colwidths(data),repeatRows=1,hAlign='LEFT')
        cmds=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#24485e')),('GRID',(0,0),(-1,-1),.45,colors.HexColor('#d9dfe3')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8)]
        if len(data[0])==8:
            cmds.append(('ALIGN',(1,1),(-1,-1),'CENTER'))
        for r in range(1,len(data)):
            if r%2==0: cmds.append(('BACKGROUND',(0,r),(-1,r),colors.HexColor('#f3f6f8')))
        table.setStyle(TableStyle(cmds)); story.extend([Spacer(1,4),table,Spacer(1,12)])

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Arial',8)
    canvas.setFillColor(colors.HexColor('#5b6f7a'))
    canvas.drawString(48,28,'Global market study | 20 October 2026')
    canvas.drawRightString(A4[0]-48,28,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(pdf_path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=45,bottomMargin=47,title='Global market study for the OpenCV hackathon',author='Market research study',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(pdf_path))
texts=[p.extract_text() or '' for p in reader.pages]
bad=[]
for i,t in enumerate(texts):
    if len(t.strip())<80: bad.append(i+1)
summary={'pdf':str(pdf_path),'pages':len(reader.pages),'bytes':pdf_path.stat().st_size,'main_words':len(source.split()),'evidence_sources':len(source_rows),'page_words':[len(t.split()) for t in texts],'sparse_pages':bad,'html':str(html_path),'notebook':str(notebook_path)}
(BASE/'build-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

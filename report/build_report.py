"""Build PDF from Markdown. Requires reportlab; figures are generated separately."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Preformatted,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parent
source=(ROOT/'PROJECT_REPORT.md').read_text(encoding='utf-8')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=10.2,leading=15.2,spaceAfter=12,textColor=colors.HexColor('#293b36')))
styles.add(ParagraphStyle(name='ReportHeading',fontName='Helvetica-Bold',fontSize=20,leading=25,spaceAfter=20,textColor=colors.HexColor('#176b5b')))
styles.add(ParagraphStyle(name='CoverTitle',fontName='Helvetica-Bold',fontSize=25,leading=31,spaceAfter=25,textColor=colors.HexColor('#142e24')))
styles.add(ParagraphStyle(name='CodeBlock',fontName='Courier',fontSize=9,leading=13,backColor=colors.HexColor('#f1f5ef'),borderPadding=10,spaceBefore=6,spaceAfter=18))
styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=9,leading=12,textColor=colors.HexColor('#293b36')))
styles.add(ParagraphStyle(name='Caption',fontName='Helvetica',fontSize=9,leading=13,textColor=colors.HexColor('#687970'),spaceAfter=14))
story=[]
def p(text,style='ReportBody'):return Paragraph(escape(text),styles[style])
lines=source.splitlines();i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('# '):
        story.extend([Spacer(1,30),p('TRIDOSHA  /  RESEARCH EDITION','Caption'),Spacer(1,15),p(line[2:],'CoverTitle')]);i+=1;continue
    if line.startswith('## '):
        number=line[3:].split(' ')[0]
        if number=='8':
            story.extend([PageBreak(),p('Numerical recovery trajectories','ReportHeading'),Image(str(ROOT/'recovery.png'),width=475,height=416),Spacer(1,16),p('Figure 1. Default Python simulation. Shaded interval: pulse from day 3 to day 5. Dotted horizontal composition lines indicate the fixed equilibrium. The controller changes the path and recovery rate; it does not change Prakriti.','Caption')])
        if number=='11':
            story.extend([PageBreak(),p('Cross-cohort classification errors','ReportHeading'),Image(str(ROOT/'confusion.png'),width=475,height=214),Spacer(1,16),p('Figure 2. True classes are rows; predicted classes are columns. Each cell is a record count from the untouched target cohort. The pronounced Vata-to-Pitta error in North-to-West transfer illustrates the asymmetry between cohorts.','Caption'),p('These confusion matrices belong to the constitution benchmark. They do not measure the accuracy of the dynamical system or the effect of negative feedback.')])
        story.extend([PageBreak(),p(line[3:],'ReportHeading')]);i+=1;continue
    if line.startswith('```'):
        block=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
        story.append(Preformatted('\n'.join(block),styles['CodeBlock']));i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[s.strip() for s in lines[i].strip('|').split('|')]
            if not all(set(c)<={'-',' ',':'} for c in cells):rows.append([p(c,'Cell') for c in cells])
            i+=1
        t=Table(rows,colWidths=[475/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7efe8')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#dce5df'))]))
        story.extend([t,Spacer(1,15)]);continue
    paragraph=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):paragraph.append(lines[i]);i+=1
    story.append(p(' '.join(paragraph)))

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#7b8d80'))
    canvas.drawString(60,32,'TRIDOSHA  /  MATHEMATICAL MODELLING REPORT')
    canvas.drawRightString(535,32,str(doc.page));canvas.restoreState()
class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='ReportHeading':
            title=flowable.getPlainText();key='section-'+str(self.page)
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,level=0)
doc=ReportDoc(str(ROOT/'Tridosha_Project_Report.pdf'),pagesize=(595.28,841.89),rightMargin=60,leftMargin=60,topMargin=55,bottomMargin=60,title='Mathematical Modeling of Tridosha Homeostasis Using Dynamical Systems and Feedback Control Theory',author='Tridosha Project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(ROOT/'Tridosha_Project_Report.pdf')

"""
Compiles paper/jmlr/cover_letter.pdf using ReportLab PDF engine.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def build_cover_letter_pdf():
    os.makedirs("paper/jmlr", exist_ok=True)
    pdf_path = "paper/jmlr/cover_letter.pdf"
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10.5,
        leading=14,
        spaceAfter=8
    )

    item_style = ParagraphStyle(
        'ItemStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=13.5,
        leftIndent=15,
        spaceAfter=6
    )

    elements = []

    elements.append(Paragraph("To: Editors-in-Chief and Action Editors, Journal of Machine Learning Research (JMLR)<br/>From: Sham Thakare (Corresponding Author)<br/>Date: August 12, 2026<br/>Subject: Submission of Manuscript: 'When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training'", header_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Dear Editors-in-Chief and Action Editors,", body_style))
    elements.append(Paragraph("We are pleased to submit our manuscript entitled <b>'When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training'</b> for consideration as a research article in the Journal of Machine Learning Research (JMLR).", body_style))
    elements.append(Paragraph("In accordance with JMLR author submission guidelines, we provide the following required disclosures and declarations:", body_style))

    items = [
        "<b>1. Prior Publications and Overlap Disclosure:</b> This manuscript presents entirely original research and has not been published previously in any peer-reviewed conference or journal. It is not currently under concurrent consideration at any other journal or archival conference.",
        "<b>2. Author Consent Confirmation:</b> All authors have read and approved the final manuscript and explicitly consent to its submission to the Journal of Machine Learning Research.",
        "<b>3. Conflict of Interest Declaration:</b> The authors declare no competing financial or personal conflicts of interest that could influence the work reported in this paper.",
        "<b>4. Suggested JMLR Action Editors:</b> Prof. Quanquan Gu (UCLA), Prof. Nan Jiang (UIUC), Prof. Qiang Liu (UT Austin). <i>[AUTHOR MUST VERIFY COI BEFORE FINAL SUBMISSION]</i>",
        "<b>5. Suggested Reviewers:</b> Dr. Arpit Bansal (UMD), Dr. Rishabh Agarwal (Google DeepMind), Prof. Percy Liang (Stanford). <i>[AUTHOR MUST VERIFY COI BEFORE FINAL SUBMISSION]</i>",
        "<b>6. Article Keywords:</b> reinforcement learning, large language models, foundation models, optimization, training dynamics.",
    ]

    for it in items:
        elements.append(Paragraph(it, item_style))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Thank you for considering our work. We look forward to your feedback.", body_style))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Sincerely,<br/><br/><b>Sham Thakare</b><br/>Corresponding Author<br/>Independent Researcher<br/>Email: <code>shamthakare@example.com</code>", body_style))

    doc.build(elements)
    print(f"Successfully compiled JMLR cover letter to {pdf_path}")


if __name__ == "__main__":
    build_cover_letter_pdf()

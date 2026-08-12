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
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        spaceAfter=12
    )

    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        spaceBefore=6,
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        spaceAfter=8
    )

    elements = []

    # Header
    elements.append(Paragraph("<b>To:</b> Editors-in-Chief and Action Editors, Journal of Machine Learning Research (JMLR)", header_style))
    elements.append(Paragraph("<b>From:</b> Sham Thakare (Corresponding Author)", header_style))
    elements.append(Paragraph("<b>Date:</b> August 12, 2026", header_style))
    elements.append(Paragraph("<b>Subject:</b> Submission of Manuscript: <i>\"Predicting Reinforcement-Learning Plasticity of Intermediate Language-Model Checkpoints: A Cross-Architecture Diagnostic Study\"</i>", header_style))
    elements.append(Spacer(1, 10))

    # Greeting & Opening
    elements.append(Paragraph("Dear Editors-in-Chief and Action Editors,", body_style))
    elements.append(Paragraph("We submit our manuscript entitled <b>\"Predicting Reinforcement-Learning Plasticity of Intermediate Language-Model Checkpoints: A Cross-Architecture Diagnostic Study\"</b> for consideration as a research article in the Journal of Machine Learning Research (JMLR).", body_style))
    elements.append(Spacer(1, 6))

    # Declarations
    elements.append(Paragraph("In accordance with JMLR submission guidelines, we confirm the following required declarations:", body_style))

    decl_items = [
        "<b>Prior Publications and Overlap Disclosure:</b> This manuscript presents original research that has not been published in any peer-reviewed conference or journal. It is not currently under consideration at any other publication venue.",
        "<b>Author Consent Confirmation:</b> As single author, I have reviewed and approved the manuscript and consent to its submission to JMLR.",
        "<b>Conflict of Interest Declaration:</b> I declare no competing financial or personal conflicts of interest regarding this work.",
        "<b>Candidate JMLR Action Editors (Subject Fit):</b><br/>&bull; <b>Prof. Quanquan Gu</b> (UCLA) — Optimization, reinforcement learning, and language model training dynamics.<br/>&bull; <b>Prof. Nan Jiang</b> (UIUC) — Reinforcement learning theory, sample efficiency, and policy optimization.<br/>&bull; <b>Prof. Qiang Liu</b> (UT Austin) — Multi-objective optimization and deep learning dynamics.",
        "<b>Candidate Reviewers (Subject Fit):</b><br/>&bull; <b>Dr. Arpit Bansal</b> (University of Maryland) — Intermediate RL training and language model plasticity.<br/>&bull; <b>Dr. Rishabh Agarwal</b> (Google DeepMind) — Reinforcement learning efficiency and empirical evaluation.<br/>&bull; <b>Prof. Percy Liang</b> (Stanford University) — Foundation model training dynamics and capability evaluation.",
        "<b>Keywords:</b> reinforcement learning, large language models, training dynamics, optimization, compute allocation."
    ]

    for item in decl_items:
        elements.append(Paragraph(f"&bull; {item}", ParagraphStyle('Item', parent=body_style, leftIndent=15, spaceAfter=6)))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Thank you for considering this manuscript.", body_style))
    elements.append(Spacer(1, 15))

    # Sign-off
    elements.append(Paragraph("Sincerely,", body_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Sham Thakare</b><br/>Independent Researcher<br/>Pune 411030, Maharashtra, India<br/>Email: <code>shamthakare3000@gmail.com</code>", body_style))

    doc.build(elements)
    print(f"Successfully compiled JMLR cover letter to {pdf_path}")


if __name__ == "__main__":
    build_cover_letter_pdf()

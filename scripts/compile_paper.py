"""
Compiles paper/jmlr/main.pdf using ReportLab PDF engine from the JMLR manuscript content.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def build_jmlr_pdf():
    os.makedirs("paper/jmlr", exist_ok=True)
    pdf_path = "paper/jmlr/main.pdf"
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=17,
        alignment=1, # Center
        spaceAfter=12
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        alignment=1,
        spaceAfter=15
    )

    abstract_title = ParagraphStyle(
        'AbstractTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        alignment=1,
        spaceAfter=4
    )

    abstract_text = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=12.5,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10
    )

    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=13.5,
        spaceAfter=6
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    elements = []

    # Title & Authors
    elements.append(Paragraph("Predicting Reinforcement-Learning Plasticity of Intermediate Language-Model Checkpoints:<br/>A Cross-Architecture Diagnostic Study", title_style))
    elements.append(Paragraph("<b>Sham Thakare</b><br/>Independent Researcher<br/>Pune 411030, Maharashtra, India &bull; <i>shamthakare3000@gmail.com</i>", author_style))

    # Abstract
    elements.append(Paragraph("Abstract", abstract_title))
    elements.append(Paragraph("Standard foundation model training relies on a rigid sequential paradigm: Next-Token Prediction (NTP) pre-training, followed by Supervised Fine-Tuning (SFT), and finally Reinforcement Learning (RL). While recent work demonstrates that RL excursions during pre-training can expand model distribution coverage, fixed or manual excursion schedules fail to account for dynamic checkpoint readiness. In this paper, we investigate the underlying mechanisms of reinforcement learning plasticity across intermediate pre-training checkpoints. We discover that measurable pre-RL state signals—specifically gradient alignment between NTP loss and task rewards (&cos;(&bold;g&bold;<sub>NTP</sub>, &bold;g&bold;<sub>RL</sub>), r=0.838, p<10<sup>-16</sup>), policy entropy (r=0.744, p<10<sup>-11</sup>), and baseline reasoning accuracy—predict subsequent RL performance gains before compute is expended. We show that a predictive linear model trained on one open model family (<code>SmolLM-135M</code>) generalizes zero-shot to predict RL plasticity on an unseen model architecture (<code>distilgpt2</code>, zero-shot R<sup>2</sup>=0.7632, Spearman &rho;=0.8247). Leveraging these diagnostic signals, we introduce the <b>Capability-Aware Reinforcement Learning Scheduler (CARLS)</b>, a dynamic controller that allocates compute among NTP, SFT, and RL objectives. Empirical evaluations across open pretrained language models, multiple task families (arithmetic, logic, code), and random seeds show CARLS achieves superior compute-normalized performance (64.04% Pass@4 vs. 56.46% sequential) and capability retention (0.94 vs. 0.85) while consuming 18.7% fewer training FLOPs.", abstract_text))
    elements.append(Paragraph("<b>Keywords:</b> reinforcement learning, large language models, training dynamics, optimization, compute allocation", ParagraphStyle('KW', parent=abstract_text, fontName='Helvetica-Bold', fontSize=8.5)))
    elements.append(Spacer(1, 10))

    # 1. Introduction
    elements.append(Paragraph("1. Introduction", h2_style))
    elements.append(Paragraph("The training pipeline of modern foundation models is traditionally partitioned into distinct sequential phases: self-supervised Next-Token Prediction (NTP) pre-training over large text corpora, followed by Supervised Fine-Tuning (SFT), and ending with Reinforcement Learning (RL) alignment (Bansal et al., 2026; Huang et al., 2025). While this pipeline has yielded remarkable advances, applying RL strictly as a post-training stage introduces fundamental bottlenecks: late-stage RL primarily serves to 'sharpen' probability distributions around high-reward modes, potentially compressing policy entropy and constraining language plasticity.", body_style))
    elements.append(Paragraph("In this work, we formulate the dynamic scheduling of reinforcement learning as a continuous feedback-control problem driven by checkpoint readiness signals. We introduce the <b>Capability-Aware Reinforcement Learning Scheduler (CARLS)</b>, which evaluates real-time readiness indicators—most notably the gradient alignment between NTP pre-training loss and reward-driven policy gradients &cos;(&bold;g&bold;<sub>NTP</sub>, &bold;g&bold;<sub>RL</sub>), policy entropy, and baseline pass@k accuracy—to dynamically adjust resource allocation among training objectives.", body_style))

    # 2. Related Work
    elements.append(Paragraph("2. Related Work", h2_style))
    elements.append(Paragraph("<b>Reinforcement Learning in LM Training:</b> RLHF and RLVR have become standard tools for aligning language models (Huang et al., 2025). Bansal et al. (2026) explored applying RL directly to base intermediate pre-training checkpoints, showing early RL excursions expand distribution coverage. However, their intervention timing remains static.", body_style))
    elements.append(Paragraph("<b>Gradient Alignment & Multi-Task Optimization:</b> Yu et al. (2020) introduced PCGrad to project conflicting task gradients. Zhang et al. (2025) utilized gradient alignment for data filtering in post-training alignment, while Wang et al. (2025) developed Curvature-Guided Policy Optimization. CARLS employs gradient alignment as a macro-level diagnostic checkpoint signal driving dynamic compute scheduling.", body_style))

    # 3. Experimental Results & Table
    elements.append(Paragraph("3. Experimental Results", h2_style))
    elements.append(Paragraph("Table 1 summarizes the empirical compute-normalized performance, solution diversity, and capability retention across all 8 baselines averaged over 3 random seeds on open pretrained models (SmolLM-135M & distilgpt2).", body_style))

    table_data = [
        ["Baseline Schedule", "Pass@1 (%)", "Pass@1 95% CI", "Pass@4 (%)", "Pass@4 95% CI", "Diversity", "Retention", "FLOPs (x10^10)"],
        ["B0: NTP Only", "4.57", "[3.50, 6.10]", "11.70", "[10.70, 12.60]", "0.0457", "0.88", "9.80"],
        ["B1: SFT Only", "40.58", "[38.20, 42.60]", "50.55", "[47.80, 54.20]", "0.0180", "0.88", "2.50"],
        ["B2: Sequential Pipeline", "42.49", "[41.00, 44.30]", "56.46", "[53.40, 58.60]", "0.0292", "0.85", "7.50"],
        ["B3: Early RL Excursion", "35.01", "[32.20, 37.40]", "47.38", "[45.70, 49.60]", "0.0494", "0.88", "7.60"],
        ["B4: Periodic RL Excursions", "40.96", "[39.50, 42.10]", "56.50", "[55.50, 57.30]", "0.0427", "0.88", "7.60"],
        ["B5: Random RL Timing", "38.84", "[38.40, 39.50]", "54.59", "[53.40, 56.40]", "0.0473", "0.88", "7.20"],
        ["B6: Fixed Objective Mixture", "38.56", "[36.60, 39.60]", "54.80", "[54.20, 55.80]", "0.0299", "0.88", "6.90"],
        ["B7: CARLS (Ours)", "47.26", "[44.60, 50.00]", "64.04", "[62.80, 65.30]", "0.0388", "0.94", "6.10"],
    ]

    t = Table(table_data, colWidths=[120, 50, 60, 50, 60, 45, 45, 55])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('LINEABOVE', (0,0), (-1,0), 1, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 1, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#EEF6FF')),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ]))
    elements.append(t)
    elements.append(Paragraph("Table 1: Empirical performance comparison across baselines on open pretrained LMs.", caption_style))
    elements.append(Spacer(1, 10))

    # Figures
    if os.path.exists("artifacts/figures/fig1_baseline_accuracy.png"):
        img1 = Image("artifacts/figures/fig1_baseline_accuracy.png", width=420, height=210)
        elements.append(img1)
        elements.append(Paragraph("Figure 1: Compute-normalized benchmark accuracy across all baseline training schedules on open pretrained LMs.", caption_style))

    if os.path.exists("artifacts/figures/fig2_compute_pareto.png"):
        img2 = Image("artifacts/figures/fig2_compute_pareto.png", width=420, height=210)
        elements.append(img2)
        elements.append(Paragraph("Figure 2: Compute efficiency Pareto frontier comparing total training compute vs. downstream task accuracy.", caption_style))

    if os.path.exists("artifacts/figures/fig3_plasticity_prediction.png"):
        img3 = Image("artifacts/figures/fig3_plasticity_prediction.png", width=380, height=270)
        elements.append(img3)
        elements.append(Paragraph("Figure 3: Zero-shot cross-model predictor generalization (SmolLM-135M -> distilgpt2, R^2=0.7632, Spearman rho=0.8247).", caption_style))

    # 4. Conclusion
    elements.append(Paragraph("4. Conclusion", h2_style))
    elements.append(Paragraph("In this paper, we introduced CARLS, an adaptive scheduler that dynamically adjusts compute allocation based on real-time checkpoint readiness. Empirical multi-seed evaluations show that CARLS achieves superior compute-normalized performance while preserving essential solution diversity.", body_style))

    # References
    elements.append(Paragraph("References", h2_style))
    refs = [
        "Bansal, A., Agarwal, R., Kumar, A., & Liang, P. (2026). RL Excursions during Pre-training: Expanding Language Plasticity in Intermediate Checkpoints. International Conference on Learning Representations (ICLR).",
        "Huang, J., Chen, X., & Song, D. (2025). Reinforcement Pre-Training (RPT): Next-Token Prediction as Reasoning RL. arXiv preprint arXiv:2502.08643.",
        "Li, M., Wang, H., & Zhou, D. (2025). Unveiling the Basin-Like Loss Landscape in Large Language Models. arXiv preprint arXiv:2505.17646.",
        "Wang, Y., Zhang, T., & Liu, Q. (2025). Boosting Multi-Domain Reasoning of LLMs via Curvature-Guided Policy Optimization. OpenReview Submission.",
        "Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient Surgery for Multi-Task Learning. Advances in Neural Information Processing Systems (NeurIPS), 33, 5824-5836.",
        "Zhang, W., Liu, Y., & Zhao, J. (2025). LearnAlign: Reasoning Data Selection for Reinforcement Learning in Large Language Models Based on Improved Gradient Alignment. arXiv preprint arXiv:2506.11480.",
    ]
    for r in refs:
        elements.append(Paragraph(f"&bull; {r}", ParagraphStyle('Ref', parent=body_style, fontSize=8.5, leading=11)))

    doc.build(elements)
    print(f"Successfully compiled JMLR paper to {pdf_path} (Size: {os.path.getsize(pdf_path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    build_jmlr_pdf()

import io
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(analysis_data: Dict[str, Any]) -> bytes:
    """
    Generates a beautifully formatted PDF report of the analyzed Amazon reviews.
    Returns the PDF as raw bytes.
    """
    buffer = io.BytesIO()
    
    # Page setup
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Premium Look
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    bold_body = ParagraphStyle(
        'BoldBodyCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    summary_box_style = ParagraphStyle(
        'SummaryBox',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.white
    )

    story = []
    
    # Header Title
    story.append(Paragraph("Amazon Review Analysis Report", title_style))
    story.append(Spacer(1, 10))
    
    # Product Info block
    prod = analysis_data["product"]
    story.append(Paragraph(f"<b>Product Name:</b> {prod['title']}", body_style))
    story.append(Paragraph(f"<b>ASIN / ID:</b> {prod['id']}", body_style))
    story.append(Paragraph(f"<b>Category:</b> {prod.get('category', 'General')}", body_style))
    story.append(Paragraph(f"<b>Overall Rating:</b> {prod['overall_rating']} / 5.0  ({prod['reviews_count']} reviews analyzed)", body_style))
    story.append(Spacer(1, 10))
    
    # Product Summary Box
    if prod.get("summary"):
        story.append(Paragraph("AI-Generated Product Summary", section_style))
        # Nested inside a 1-cell table for background color and borders
        summary_table = Table([[Paragraph(prod["summary"], summary_box_style)]], colWidths=[doc.width])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('PADDING', (0,0), (-1,-1), 12),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
    # Sentiment Distribution Table
    story.append(Paragraph("Overall Sentiment Breakdown", section_style))
    dist = analysis_data["sentiment_distribution"]
    total_reviews = sum(dist.values())
    
    sentiment_data = [[
        Paragraph("Sentiment", table_header_style),
        Paragraph("Count", table_header_style),
        Paragraph("Percentage", table_header_style)
    ]]
    
    for sentiment, count in dist.items():
        pct = (count / total_reviews * 100) if total_reviews > 0 else 0
        sentiment_data.append([
            Paragraph(sentiment, body_style),
            Paragraph(str(count), body_style),
            Paragraph(f"{pct:.1f}%", body_style)
        ])
        
    sent_table = Table(sentiment_data, colWidths=[200, 150, 150])
    sent_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#475569')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(sent_table)
    story.append(Spacer(1, 15))
    
    # Feature Sentiment Table
    if analysis_data.get("features"):
        story.append(Paragraph("Extracted Product Features & Sentiment", section_style))
        feature_data = [[
            Paragraph("Feature Cluster", table_header_style),
            Paragraph("Mentions", table_header_style),
            Paragraph("Sentiment Score (0-100)", table_header_style)
        ]]
        
        for feat in analysis_data["features"]:
            feature_data.append([
                Paragraph(feat["feature_name"], bold_body),
                Paragraph(str(feat["mentions"]), body_style),
                Paragraph(f"{feat['sentiment_score']}%", body_style)
            ])
            
        feat_table = Table(feature_data, colWidths=[200, 150, 150])
        feat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ]))
        story.append(feat_table)
        story.append(Spacer(1, 15))
        
    # Flagged Fake / Suspicious Reviews Table
    fake_reviews = [r for r in analysis_data["reviews"] if r["is_fake"]]
    if fake_reviews:
        story.append(Paragraph("Potentially Spam / Fake Reviews Flagged", section_style))
        fake_data = [[
            Paragraph("Author", table_header_style),
            Paragraph("Rating", table_header_style),
            Paragraph("Flag Reasons", table_header_style),
            Paragraph("Snippet", table_header_style)
        ]]
        
        for r in fake_reviews:
            snippet = r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]
            reasons_str = ", ".join(r["fake_reasons"])
            fake_data.append([
                Paragraph(r["author"] or "N/A", body_style),
                Paragraph(f"{r['rating']} Stars" if r['rating'] else "N/A", body_style),
                Paragraph(reasons_str, body_style),
                Paragraph(snippet, body_style)
            ])
            
        fake_table = Table(fake_data, colWidths=[100, 60, 160, 180])
        fake_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#991B1B')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FCA5A5')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FEF2F2'), colors.white]),
        ]))
        story.append(fake_table)
        story.append(Spacer(1, 15))
        
    # Document compilation
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

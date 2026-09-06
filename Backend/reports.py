"""
Module for generating textile waste analysis reports in multiple formats.
Supports PDF, Excel (XLSX), and CSV export with prediction source provenance disclosures.
"""

import io
import csv
from datetime import datetime
from typing import List, Optional, Dict, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from sqlalchemy.orm import Session
from database import WasteBatch, AnalysisResult, User


class ReportGenerator:
    """Generate reports for textile waste analysis data."""
    
    @staticmethod
    def generate_pdf_report(
        batches: List[WasteBatch],
        report_title: str = "Textile Waste Intelligence Report",
        user_name: str = None,
        date_range: str = None
    ) -> bytes:
        """
        Generate a PDF report of waste batch analysis with source provenance tags.
        """
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1a2b4c'),
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2b5a8c'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        )
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#666666'),
            alignment=TA_LEFT,
            fontName='Helvetica-Oblique'
        )

        elements.append(Paragraph(report_title, title_style))
        
        metadata_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if user_name:
            metadata_text += f" | User: {user_name}"
        if date_range:
            metadata_text += f" | Period: {date_range}"
        
        elements.append(Paragraph(f"<i>{metadata_text}</i>", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        elements.append(Paragraph("Executive Summary", heading_style))
        
        total_qty = sum(b.quantity for b in batches)
        analyzed_count = sum(1 for b in batches if b.analysis)
        avg_score = sum(b.analysis.overall_circularity_score for b in batches if b.analysis and b.analysis.overall_circularity_score) / analyzed_count if analyzed_count > 0 else 0
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Batches', str(len(batches))],
            ['Total Weight (kg)', f'{total_qty:.2f}'],
            ['Analyzed Batches', str(analyzed_count)],
            ['Average Circularity Score', f'{avg_score:.2f}']
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b5a8c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white])
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Detailed batch information
        elements.append(Paragraph("Batch Details & AI Provenance", heading_style))
        
        batch_data = [['Batch ID', 'Material', 'Source', 'Waste Category', 'Score', 'Review Required']]
        
        for batch in batches:
            score = f"{batch.analysis.overall_circularity_score:.1f}" if (batch.analysis and batch.analysis.overall_circularity_score) else "N/A"
            category = batch.waste_category or (batch.analysis.circularity_category if batch.analysis else "Not Analyzed")
            src = batch.analysis.prediction_source if batch.analysis else "MANUAL"
            review_str = "YES" if (batch.analysis and batch.analysis.manual_review_required) else "NO"

            batch_data.append([
                str(batch.id),
                f"{batch.fabric_type} [{src or 'RULE'}]",
                batch.source[:15],
                category,
                score,
                review_str
            ])
        
        batch_table = Table(batch_data, colWidths=[0.7*inch, 1.8*inch, 1.2*inch, 1.8*inch, 0.8*inch, 1.1*inch])
        batch_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b5a8c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white])
        ]))
        
        elements.append(batch_table)
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph(
            "* Disclaimers: Material predictions labeled [MODEL] come from machine learning inference. "
            "Environmental savings (CO2/Water) are ESTIMATED calculations based on standard lifecycle emission factors. "
            "Batches flagged with Review Required must be manually inspected before final processing.",
            disclaimer_style
        ))
        
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    
    @staticmethod
    def generate_excel_report(
        batches: List[WasteBatch],
        report_title: str = "Textile Waste Intelligence Report"
    ) -> bytes:
        """
        Generate an Excel (XLSX) report of waste batch analysis.
        """
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Waste Analysis"
        
        worksheet.column_dimensions['A'].width = 10
        worksheet.column_dimensions['B'].width = 18
        worksheet.column_dimensions['C'].width = 16
        worksheet.column_dimensions['D'].width = 12
        worksheet.column_dimensions['E'].width = 14
        worksheet.column_dimensions['F'].width = 14
        worksheet.column_dimensions['G'].width = 18
        worksheet.column_dimensions['H'].width = 16
        worksheet.column_dimensions['I'].width = 18
        worksheet.column_dimensions['J'].width = 20
        worksheet.column_dimensions['K'].width = 16
        
        worksheet['A1'] = report_title
        worksheet['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        worksheet['A1'].fill = PatternFill(start_color="1a2b4c", end_color="1a2b4c", fill_type="solid")
        worksheet.merge_cells('A1:K1')
        worksheet['A1'].alignment = Alignment(horizontal='center', vertical='center')
        
        worksheet['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Disclosure: Environmental metrics are ESTIMATED"
        worksheet['A2'].font = Font(italic=True, size=10)
        
        headers = [
            'Batch ID', 'Fabric Type', 'Source', 'Qty (kg)', 'Color', 'Condition', 
            'Prediction Source', 'Circularity Score', 'CO2 Saved (EST kg)', 'Water Saved (EST L)', 'Review Required'
        ]
        
        for col, header in enumerate(headers, start=1):
            cell = worksheet.cell(row=4, column=col)
            cell.value = header
            cell.font = Font(bold=True, color="FFFFFF", size=10)
            cell.fill = PatternFill(start_color="2b5a8c", end_color="2b5a8c", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        for row_idx, batch in enumerate(batches, start=5):
            worksheet.cell(row=row_idx, column=1).value = batch.id
            worksheet.cell(row=row_idx, column=2).value = batch.fabric_type
            worksheet.cell(row=row_idx, column=3).value = batch.source
            worksheet.cell(row=row_idx, column=4).value = batch.quantity
            worksheet.cell(row=row_idx, column=5).value = batch.color
            worksheet.cell(row=row_idx, column=6).value = batch.condition
            worksheet.cell(row=row_idx, column=7).value = batch.analysis.prediction_source if batch.analysis else "MANUAL_HINT"
            worksheet.cell(row=row_idx, column=8).value = batch.analysis.overall_circularity_score if batch.analysis else None
            worksheet.cell(row=row_idx, column=9).value = batch.analysis.co2_savings if batch.analysis else None
            worksheet.cell(row=row_idx, column=10).value = batch.analysis.water_savings if batch.analysis else None
            worksheet.cell(row=row_idx, column=11).value = "YES" if (batch.analysis and batch.analysis.manual_review_required) else "NO"
            
        excel_buffer = io.BytesIO()
        workbook.save(excel_buffer)
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    
    @staticmethod
    def generate_csv_report(batches: List[WasteBatch]) -> str:
        """
        Generate a CSV report string of waste batch analysis.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            'Batch ID', 'Fabric Type', 'Source', 'Quantity (kg)', 'Color', 'Condition',
            'Waste Category', 'Prediction Source', 'Confidence Status', 'Circularity Score',
            'CO2 Saved (EST kg)', 'Water Saved (EST L)', 'Manual Review Required'
        ])
        
        for batch in batches:
            writer.writerow([
                batch.id,
                batch.fabric_type,
                batch.source,
                batch.quantity,
                batch.color,
                batch.condition,
                batch.waste_category or (batch.analysis.circularity_category if batch.analysis else "Unassigned"),
                batch.analysis.prediction_source if batch.analysis else "MANUAL_HINT",
                batch.analysis.confidence_status if batch.analysis else "UNKNOWN",
                batch.analysis.overall_circularity_score if batch.analysis else "",
                batch.analysis.co2_savings if batch.analysis else 0.0,
                batch.analysis.water_savings if batch.analysis else 0.0,
                "YES" if (batch.analysis and batch.analysis.manual_review_required) else "NO",
            ])
            
        return output.getvalue()

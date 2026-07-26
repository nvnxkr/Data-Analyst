"""
report_export.py
Builds a shareable report (HTML and/or PDF) containing the data summary,
auto-generated insights, and charts.
"""

import io
from datetime import datetime

import plotly.io as pio
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib import colors


def build_html_report(dataset_name: str, summary: dict, insights: list, figs: list) -> str:
    """
    figs: list of (title, plotly.graph_objects.Figure) tuples.
    Returns a single self-contained HTML string (Plotly.js loaded via CDN).
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    insights_html = "".join(f"<li>{i}</li>" for i in insights) or "<li>No insights generated.</li>"

    stats_rows = ""
    for col, s in summary.get("numeric_stats", {}).items():
        stats_rows += (
            f"<tr><td>{col}</td><td>{s['mean']}</td><td>{s['median']}</td>"
            f"<td>{s['min']}</td><td>{s['max']}</td><td>{s['nulls']}</td></tr>"
        )

    charts_html = ""
    for title, fig in figs:
        chart_div = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
        charts_html += f"<h3>{title}</h3>{chart_div}"

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Analysis Report — {dataset_name}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; color: #1f2937; }}
  h1 {{ color: #111827; }}
  h2 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; margin-top: 40px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #f3f4f6; }}
  li {{ margin-bottom: 6px; }}
  .meta {{ color: #6b7280; font-size: 13px; }}
</style>
</head>
<body>
  <h1>Data Analysis Report</h1>
  <p class="meta">Dataset: {dataset_name} &middot; Generated: {generated_at}</p>

  <h2>Overview</h2>
  <p>{summary['n_rows']} rows &times; {summary['n_cols']} columns</p>

  <h2>Key Insights</h2>
  <ul>{insights_html}</ul>

  <h2>Numeric Column Statistics</h2>
  <table>
    <tr><th>Column</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Nulls</th></tr>
    {stats_rows}
  </table>

  <h2>Charts</h2>
  {charts_html}
</body>
</html>
"""


def build_pdf_report(dataset_name: str, summary: dict, insights: list, figs: list) -> bytes:
    """
    figs: list of (title, plotly.graph_objects.Figure) tuples.
    Renders each figure to a static PNG (via kaleido) and assembles a PDF
    with reportlab. Returns PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=f"Report - {dataset_name}")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Data Analysis Report", styles["Title"]))
    story.append(
        Paragraph(
            f"Dataset: {dataset_name} &middot; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 16))

    story.append(Paragraph("Overview", styles["Heading2"]))
    story.append(Paragraph(f"{summary['n_rows']} rows &times; {summary['n_cols']} columns", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Key Insights", styles["Heading2"]))
    for insight in insights:
        story.append(Paragraph(f"&bull; {insight}", styles["Normal"]))
    story.append(Spacer(1, 10))

    numeric_stats = summary.get("numeric_stats", {})
    if numeric_stats:
        story.append(Paragraph("Numeric Column Statistics", styles["Heading2"]))
        table_data = [["Column", "Mean", "Median", "Min", "Max", "Nulls"]]
        for col, s in numeric_stats.items():
            table_data.append([col, s["mean"], s["median"], s["min"], s["max"], s["nulls"]])
        t = Table(table_data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

    if figs:
        story.append(PageBreak())
        story.append(Paragraph("Charts", styles["Heading2"]))
        for title, fig in figs:
            try:
                png_bytes = pio.to_image(fig, format="png", width=650, height=380, scale=2)
                story.append(Paragraph(title, styles["Heading3"]))
                story.append(Image(io.BytesIO(png_bytes), width=6 * inch, height=3.5 * inch))
                story.append(Spacer(1, 14))
            except Exception as e:
                story.append(
                    Paragraph(
                        f"[Chart '{title}' could not be rendered to PDF: {e}]",
                        styles["Normal"],
                    )
                )

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

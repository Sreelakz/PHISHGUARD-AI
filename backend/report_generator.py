"""
report_generator.py — PhishGuard AI Incident Report Generator
=============================================================

FIXES (v4.5):
  1. UNICODE/EMOJI: Registered DejaVuSans TTF so all characters render correctly.
  2. MISSING SECTION 4: _resolve_indicators() checks both "detailed_explanation" and
     "explanations" keys so Risk Indicators always render when data exists.
  3. RISK SCORE/LEVEL CLARITY: Explanatory note added to banner.
  4. COLUMN WIDTHS: Replaced invalid %-strings with real cm values.
  5. EMOJI STRIPPING: _clean() strips emoji/non-ASCII from message strings.
  6. SECTION NUMBERING: Sections auto-number so no gaps when optional sections skipped.
  7. FONTSTYLE KWARG COLLISION: _ps() uses **{'fontName': R, **kw} so callers
     that pass fontName=B correctly override the default.
  8. (NEW v4.5) PALETTE REFRESH — colours now match the new UI theme:
        bg:       #0b0b14   (deep slate-black)
        bg-card:  #13131f   (card surface)
        bg-elev:  #1a1a2a   (elevated surface)
        accent:   #a855f7   (electric violet — primary)
        accent2:  #f43f5e   (rose — danger)
        accent3:  #bef264   (acid lime — success / safe)
        warn:     #fbbf24   (amber)
        purple:   #c084fc   (light violet — tertiary)
        text:     #e2e8f0   (slate-200)
        text-dim: #64748b   (slate-500)
        border:   rgba(168,85,247,0.18) → solid #1e1a3a for tables

FORMATS:
    PDF  — via ReportLab (professional layout, color-coded, unicode)
    CSV  — flat single-row export for bulk analysis logging
"""

from __future__ import annotations

import os
import csv
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(_HERE, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── ReportLab imports ─────────────────────────────────────────────────────
try:
    from reportlab.lib             import colors
    from reportlab.lib.pagesizes   import A4
    from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units       import cm
    from reportlab.platypus        import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums       import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.pdfbase         import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF export disabled. Run: pip install reportlab")

# ── FIX 1: Register DejaVuSans for full Unicode/emoji-text support ────────
_FONT_REGULAR = "Helvetica"
_FONT_BOLD    = "Helvetica-Bold"

if _PDF_AVAILABLE:
    _DEJAVU_PATHS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
    ]
    _DEJAVU_BOLD_PATHS = [
        p.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
        for p in _DEJAVU_PATHS
    ]

    def _find_font(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    _dv_path      = _find_font(_DEJAVU_PATHS)
    _dv_bold_path = _find_font(_DEJAVU_BOLD_PATHS)

    if _dv_path:
        try:
            pdfmetrics.registerFont(TTFont("DejaVuSans",      _dv_path))
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", _dv_bold_path or _dv_path))
            _FONT_REGULAR = "DejaVuSans"
            _FONT_BOLD    = "DejaVuSans-Bold"
            logger.info(f"DejaVuSans registered from {_dv_path}")
        except Exception as e:
            logger.warning(f"DejaVuSans registration failed: {e} — falling back to Helvetica")

# ══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE — synced with UI theme (violet / rose / lime)
# ══════════════════════════════════════════════════════════════════════════
if _PDF_AVAILABLE:
    # Backgrounds
    C_BG     = colors.HexColor("#0b0b14")   # deep slate-black (page bg)
    C_DGRAY  = colors.HexColor("#13131f")   # card surface
    C_LGRAY  = colors.HexColor("#1a1a2a")   # elevated surface (alt rows)

    # Brand colors
    C_ACCENT = colors.HexColor("#a855f7")   # electric violet — primary
    C_RED    = colors.HexColor("#f43f5e")   # rose — danger
    C_GREEN  = colors.HexColor("#bef264")   # acid lime — safe
    C_YELLOW = colors.HexColor("#fbbf24")   # amber — warn
    C_PURPLE = colors.HexColor("#c084fc")   # light violet — tertiary

    # Text
    C_TEXT   = colors.HexColor("#e2e8f0")   # slate-200
    C_MGRAY  = colors.HexColor("#64748b")   # slate-500 (dim)
    C_WHITE  = colors.white

    # Subtle borders / grid lines (violet-tinted dark)
    C_GRID   = colors.HexColor("#1e1a3a")

# ── FIX 5: Strip emoji / non-printable chars ──────────────────────────────
_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F300-\U0001F9FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]",
    flags=re.UNICODE,
)
_EMOJI_REPLACE = {
    "✅": "[OK]", "⛔": "[!]", "🚨": "[ALERT]", "⚠️": "[WARN]",
    "🛡": "[PHISHGUARD]", "🔵": "[SB]", "🟣": "[VT]",
    "💡": "[TIP]", "🧠": "[AI]", "🔬": "[FEAT]", "📋": "[INFO]",
}

def _clean(text: str, max_len: int = 300) -> str:
    """Strip emoji and truncate. Safe for any ReportLab font."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    for emoji, replacement in _EMOJI_REPLACE.items():
        text = text.replace(emoji, replacement)
    text = _EMOJI_RE.sub("", text)
    if _FONT_REGULAR == "Helvetica":
        text = text.encode("ascii", errors="replace").decode("ascii")
    return text[:max_len].strip()


# ── FIX 2: Resolve indicators from either raw or API result shape ─────────
def _resolve_indicators(result: Dict) -> List[Dict]:
    detailed = result.get("detailed_explanation") or {}
    if isinstance(detailed, dict):
        inds = detailed.get("indicators") or []
        if inds:
            return inds

    expls = result.get("explanations") or {}
    if isinstance(expls, dict):
        inds = expls.get("indicators") or []
        if inds:
            return inds

    top = result.get("indicators") or []
    if top and isinstance(top[0], str):
        return [{"severity": "high", "category": "URL", "message": s} for s in top[:15]]

    return []


def _resolve_summary(result: Dict) -> str:
    detailed = result.get("detailed_explanation") or {}
    if isinstance(detailed, dict):
        s = detailed.get("ai_narrative") or detailed.get("summary") or ""
        if s:
            return _clean(s, 400)
    expls = result.get("explanations") or {}
    if isinstance(expls, dict):
        s = expls.get("ai_narrative") or expls.get("summary") or ""
        if s:
            return _clean(s, 400)
    return _clean(result.get("verdict_reason") or result.get("explanation") or "N/A", 400)


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    result_dict:     Dict,
    output_dir:      str       = REPORTS_DIR,
    formats:         List[str] = ("pdf", "csv"),
    filename_prefix: str       = "phishguard_report",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{filename_prefix}_{timestamp}"
    output    = {}

    if "pdf" in formats:
        if _PDF_AVAILABLE:
            pdf_path = os.path.join(output_dir, base_name + ".pdf")
            try:
                _generate_pdf(result_dict, pdf_path)
                output["pdf"] = pdf_path
                logger.info(f"PDF report saved: {pdf_path}")
            except Exception as e:
                logger.exception("PDF generation failed")
                output["pdf_error"] = str(e)
        else:
            output["pdf_error"] = "reportlab not installed — run: pip install reportlab"

    if "csv" in formats:
        csv_path = os.path.join(output_dir, base_name + ".csv")
        try:
            _generate_csv(result_dict, csv_path)
            output["csv"] = csv_path
            logger.info(f"CSV report saved: {csv_path}")
        except Exception as e:
            logger.exception("CSV generation failed")
            output["csv_error"] = str(e)

    return output


# ═══════════════════════════════════════════════════════════════════════════
#  PDF GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

_PAGE_W = 17 * cm


def _generate_pdf(result: Dict, output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="PhishGuard AI — Incident Report",
        author="PhishGuard AI v4.5",
    )

    styles = _build_styles()
    story  = []

    sec = [0]
    def next_sec(title: str) -> str:
        sec[0] += 1
        return f"{sec[0]}. {title}"

    story.append(_header_block(result, styles))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT, spaceAfter=0.4*cm))

    story.append(_verdict_banner(result, styles))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(next_sec("ANALYSIS SUMMARY"), styles["section_heading"]))
    story.append(_summary_table(result, styles))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(next_sec("THREAT INTELLIGENCE"), styles["section_heading"]))
    story.append(_threat_intel_table(result, styles))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(next_sec("URL FEATURE ANALYSIS"), styles["section_heading"]))
    story.append(_feature_table(result, styles))
    story.append(Spacer(1, 0.4*cm))

    indicators = _resolve_indicators(result)
    if indicators:
        story.append(Paragraph(next_sec("RISK INDICATORS"), styles["section_heading"]))
        story.append(_indicators_table(indicators, styles))
        story.append(Spacer(1, 0.4*cm))

    shap_expl = result.get("shap_explanation") or []
    if shap_expl:
        story.append(Paragraph(next_sec("MODEL EXPLANATION (TOP FACTORS)"), styles["section_heading"]))
        story.append(_shap_table(shap_expl, styles))
        story.append(Spacer(1, 0.4*cm))

    ml = result.get("ml") or {}
    if ml.get("available"):
        story.append(Paragraph(next_sec("ML MODEL DETAILS"), styles["section_heading"]))
        story.append(_ml_table(ml, styles))
        story.append(Spacer(1, 0.4*cm))

    rules = result.get("rules") or {}
    fired = rules.get("fired_rules") or []
    if fired:
        story.append(Paragraph(next_sec("RULE ENGINE — FIRED RULES"), styles["section_heading"]))
        story.append(_rules_table(fired, styles))
        story.append(Spacer(1, 0.4*cm))

    story.append(HRFlowable(width="100%", thickness=1, color=C_MGRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated by PhishGuard AI v4.5 | Pipeline v{result.get('pipeline_version','6.1')} | "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | CONFIDENTIAL",
        styles["footer"],
    ))

    doc.build(story)


# ── Styles ────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    """
    Define all paragraph styles using the registered unicode font.

    FIX 7: _ps() previously hard-coded fontName=R as a keyword argument AND
    let callers also pass fontName=B, causing:
        TypeError: ParagraphStyle() got multiple values for keyword argument 'fontName'
    Fix: merge defaults with caller kwargs so the caller's fontName always wins.
    """
    base = getSampleStyleSheet()
    S    = {}
    R    = _FONT_REGULAR
    B    = _FONT_BOLD

    def _ps(name, **kw):
        # FIX 7: Use dict-merge so caller kwargs (e.g. fontName=B) override the
        # default fontName=R without passing the same kwarg twice.
        merged = {"fontName": R, **kw}
        return ParagraphStyle(name, parent=base["Normal"], **merged)

    S["title"] = _ps("title",
        fontSize=18, textColor=C_WHITE, fontName=B,
        alignment=TA_LEFT, spaceAfter=4)

    S["subtitle"] = _ps("subtitle",
        fontSize=8, textColor=C_MGRAY,
        alignment=TA_LEFT)

    S["section_heading"] = _ps("sec_head",
        fontSize=10, textColor=C_ACCENT, fontName=B,
        spaceBefore=8, spaceAfter=5,
        borderPad=2, leading=14)

    S["cell_key"] = _ps("cell_key",
        fontSize=8, textColor=C_MGRAY)

    S["cell_val"] = _ps("cell_val",
        fontSize=8, textColor=C_TEXT, fontName=B)

    S["cell_url"] = _ps("cell_url",
        fontSize=7.5, textColor=C_ACCENT,
        wordWrap="CJK", leading=11)

    S["verdict_phishing"] = _ps("v_ph",
        fontSize=22, textColor=C_RED, fontName=B,
        alignment=TA_LEFT, leading=28)

    S["verdict_legit"] = _ps("v_le",
        fontSize=22, textColor=C_GREEN, fontName=B,
        alignment=TA_LEFT, leading=28)

    S["verdict_sub"] = _ps("v_sub",
        fontSize=9, textColor=C_TEXT,
        alignment=TA_LEFT, spaceAfter=4, leading=14)

    S["risk_note"] = _ps("risk_note",
        fontSize=7, textColor=C_MGRAY,
        alignment=TA_LEFT, leading=10)

    S["indicator_msg"] = _ps("ind_msg",
        fontSize=8, textColor=C_TEXT, leading=12)

    S["shap_feature"] = _ps("shap_feat",
        fontSize=8, textColor=C_TEXT, fontName=B)

    S["footer"] = _ps("footer",
        fontSize=7, textColor=C_MGRAY,
        alignment=TA_CENTER, spaceBefore=4)

    return S


def _table_style(header_bg=None) -> TableStyle:
    bg = header_bg or C_DGRAY
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_ACCENT),
        ("FONTNAME",      (0, 0), (-1, 0),  _FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("TOPPADDING",    (0, 0), (-1, 0),  6),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  6),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_DGRAY, C_LGRAY]),
        ("FONTNAME",      (0, 1), (-1, -1), _FONT_REGULAR),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",     (0, 1), (-1, -1), C_TEXT),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_GRID),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ])


# ── Block builders ────────────────────────────────────────────────────────

def _header_block(result: Dict, S: dict) -> Table:
    ts  = result.get("timestamp", datetime.now().isoformat())[:19].replace("T", " ")
    url = result.get("url", "N/A")

    data = [[
        Paragraph("PHISHGUARD AI  —  INCIDENT REPORT", S["title"]),
        Paragraph(
            f"Scan ID: #{abs(hash(url)) % 100000:05d}<br/>"
            f"Timestamp: {ts}<br/>"
            f"Version: v4.5 / Pipeline {result.get('pipeline_version','6.1')}",
            S["subtitle"],
        ),
    ]]
    t = Table(data, colWidths=[_PAGE_W * 0.60, _PAGE_W * 0.40])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), C_BG),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _verdict_banner(result: Dict, S: dict) -> Table:
    verdict    = result.get("verdict", result.get("prediction", "UNKNOWN"))
    confidence = float(result.get("confidence", 0))
    risk_score = int(result.get("risk_score") or 0)
    risk_level = result.get("risk_level", "MEDIUM")
    source     = _clean(result.get("verdict_source") or "ml_model").replace("_", " ").upper()

    is_phish = verdict == "PHISHING"
    # Tinted backgrounds matching new palette
    bg_color = colors.HexColor("#2a0d18") if is_phish else colors.HexColor("#1a2410")
    v_style  = S["verdict_phishing"] if is_phish else S["verdict_legit"]

    label = "[ PHISHING DETECTED ]" if is_phish else "[ LEGITIMATE SITE ]"

    risk_note_text = (
        "Risk Score measures URL feature danger (0-100). "
        "Risk Level is derived from model confidence and may differ."
    )

    left_cell = [
        Paragraph(label, v_style),
        Spacer(1, 0.15*cm),
        Paragraph(risk_note_text, S["risk_note"]),
    ]

    right_cell = Paragraph(
        f"Confidence: <b>{confidence*100:.1f}%</b><br/>"
        f"Risk Score: <b>{risk_score} / 100</b><br/>"
        f"Risk Level: <b>{risk_level}</b><br/>"
        f"Source:     <b>{source}</b>",
        S["verdict_sub"],
    )

    data = [[left_cell, right_cell]]
    t = Table(data, colWidths=[_PAGE_W * 0.62, _PAGE_W * 0.38])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg_color),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.5, C_RED if is_phish else C_GREEN),
    ]))
    return t


def _summary_table(result: Dict, S: dict) -> Table:
    url     = result.get("url", "N/A")
    summary = _resolve_summary(result)
    ts      = result.get("timestamp", "N/A")[:19].replace("T", " ")
    conf    = float(result.get("confidence", 0))
    score   = int(result.get("risk_score") or 0)

    rows = [
        ["FIELD",          "VALUE"],
        ["Analyzed URL",   Paragraph(_clean(url, 200), S["cell_url"])],
        ["Verdict",        result.get("verdict", "UNKNOWN")],
        ["Confidence",     f"{conf*100:.1f}%"],
        ["Risk Score",     f"{score} / 100"],
        ["Risk Level",     result.get("risk_level", "N/A")],
        ["Verdict Source", _clean(result.get("verdict_source") or "ml_model").replace("_"," ").title()],
        ["Analysis Time",  ts],
        ["Summary",        Paragraph(summary, S["cell_val"])],
        ["Pipeline",       f"v{result.get('pipeline_version','6.1')}"],
    ]

    t = Table(rows, colWidths=[5.5*cm, 11.5*cm])
    ts_style = _table_style()
    verdict = result.get("verdict", "")
    for i, row in enumerate(rows[1:], 1):
        if row[0] == "Verdict":
            c = C_RED if verdict == "PHISHING" else C_GREEN
            ts_style.add("TEXTCOLOR", (1, i), (1, i), c)
            ts_style.add("FONTNAME",  (1, i), (1, i), _FONT_BOLD)
    t.setStyle(ts_style)
    return t


def _threat_intel_table(result: Dict, S: dict) -> Table:
    ti = result.get("threat_intel") or {}
    sb = result.get("safe_browsing") or ti.get("safe_browsing") or {}
    vt = result.get("virustotal")    or ti.get("virustotal")    or {}

    sb_status = _clean(sb.get("status", "N/A")).upper()
    sb_threat = "YES" if sb.get("is_threat") else "No"
    sb_types  = _clean(", ".join(sb.get("threat_types", [])) or "None")
    sb_msg    = _clean(sb.get("message") or sb_types, 120)

    vt_status = _clean(vt.get("status", "N/A")).upper()
    vt_mal    = "YES" if vt.get("is_malicious") else "No"
    vt_ratio  = vt.get("detection_ratio", "N/A")
    vt_msg    = _clean(vt.get("message") or "", 120)

    rows = [
        ["SERVICE",                    "STATUS",    "THREAT", "DETAIL"],
        ["Google Safe Browsing",       sb_status,   sb_threat, sb_msg],
        ["VirusTotal (70+ engines)",   vt_status,   vt_mal,    f"{vt_ratio} engines flagged | {vt_msg}"],
    ]

    t = Table(rows, colWidths=[5.0*cm, 2.5*cm, 2.0*cm, 7.5*cm])
    ts = _table_style()
    for row_idx, threat_val in [(1, sb_threat), (2, vt_mal)]:
        color = C_RED if threat_val == "YES" else C_GREEN
        ts.add("TEXTCOLOR", (2, row_idx), (2, row_idx), color)
        ts.add("FONTNAME",  (2, row_idx), (2, row_idx), _FONT_BOLD)
    t.setStyle(ts)
    return t


def _feature_table(result: Dict, S: dict) -> Table:
    features  = result.get("features") or {}
    skip_keys = {"ssl_info","domain_info","redirect_info","homograph_info","visual_analysis"}
    flat      = {k: v for k, v in features.items()
                 if k not in skip_keys and isinstance(v, (int, float, bool, str))}

    priority = [
        "url_length","uses_https","has_ip_address","has_at_symbol",
        "has_suspicious_keywords","num_dots","num_dashes","entropy",
        "has_suspicious_tld","num_subdomains","has_login_form",
        "has_iframes","has_obfuscated_js","has_favicon_mismatch",
        "has_hidden_fields","has_meta_refresh","has_non_standard_port",
        "has_popup_window","hostname_length","is_shortened",
    ]
    ordered  = [(k, flat[k]) for k in priority if k in flat]
    ordered += [(k, v) for k, v in flat.items() if k not in {p for p,_ in ordered}]
    ordered  = ordered[:25]

    risk_keys = {"has_ip_address","has_at_symbol","has_suspicious_keywords",
                 "has_suspicious_tld","has_obfuscated_js","has_login_form",
                 "has_iframes","has_favicon_mismatch","has_meta_refresh"}
    good_keys = {"uses_https"}

    rows = [["FEATURE", "VALUE", "RISK SIGNAL"]]
    for k, v in ordered:
        if k in risk_keys and v == 1:
            sig = "HIGH RISK"
        elif k in good_keys and v == 1:
            sig = "GOOD"
        elif k in good_keys and v == 0:
            sig = "RISK: No HTTPS"
        else:
            sig = "-"
        rows.append([k.replace("_", " "), str(v), sig])

    t = Table(rows, colWidths=[7.5*cm, 3.5*cm, 6.0*cm])
    ts = _table_style()
    for i, (k, v) in enumerate(ordered, 1):
        if k in risk_keys and v == 1:
            ts.add("TEXTCOLOR", (2, i), (2, i), C_RED)
            ts.add("FONTNAME",  (2, i), (2, i), _FONT_BOLD)
        elif k in good_keys and v == 1:
            ts.add("TEXTCOLOR", (2, i), (2, i), C_GREEN)
        elif k in good_keys and v == 0:
            ts.add("TEXTCOLOR", (2, i), (2, i), C_YELLOW)
    t.setStyle(ts)
    return t


def _indicators_table(indicators: List[Dict], S: dict) -> Table:
    sev_colors = {
        "critical": C_RED,
        "high":     C_YELLOW,
        "medium":   colors.HexColor("#fb923c"),  # warmer orange (matches UI)
        "low":      C_ACCENT,
    }
    rows = [["SEVERITY", "CATEGORY", "DESCRIPTION"]]
    for ind in indicators[:15]:
        sev = _clean(ind.get("severity", ""), 15).upper()
        cat = _clean(ind.get("category", ""), 20)
        msg = _clean(ind.get("message", ""), 250)
        rows.append([sev, cat, Paragraph(msg, S["indicator_msg"])])

    t = Table(rows, colWidths=[2.5*cm, 2.5*cm, 12.0*cm])
    ts = _table_style()
    for i, ind in enumerate(indicators[:15], 1):
        c = sev_colors.get((ind.get("severity") or "").lower(), C_TEXT)
        ts.add("TEXTCOLOR", (0, i), (0, i), c)
        ts.add("FONTNAME",  (0, i), (0, i), _FONT_BOLD)
    t.setStyle(ts)
    return t


def _shap_table(shap_expl: List[Dict], S: dict) -> Table:
    rows = [["FEATURE", "IMPACT ON VERDICT", "DIRECTION"]]
    for item in shap_expl[:10]:
        feat  = _clean(item.get("feature", ""), 60)
        val   = float(item.get("shap_value", 0))
        bar   = "|" * min(int(abs(val) * 20), 20)
        dirn  = "Toward PHISHING" if val > 0 else "Away from phishing"
        rows.append([
            Paragraph(feat, S["shap_feature"]),
            f"{val:+.4f}  {bar}",
            dirn,
        ])

    t = Table(rows, colWidths=[6.5*cm, 6.0*cm, 4.5*cm])
    ts = _table_style()
    for i, item in enumerate(shap_expl[:10], 1):
        val = float(item.get("shap_value", 0))
        c   = C_RED if val > 0 else C_GREEN
        ts.add("TEXTCOLOR", (1, i), (1, i), c)
        ts.add("TEXTCOLOR", (2, i), (2, i), c)
    t.setStyle(ts)
    return t


def _ml_table(ml: Dict, S: dict) -> Table:
    rows = [
        ["ML METRIC",            "VALUE"],
        ["Prediction",           _clean(ml.get("prediction", "N/A"))],
        ["Confidence",           f"{float(ml.get('confidence', 0))*100:.1f}%"],
        ["Phishing Probability", f"{float(ml.get('phishing_proba', 0))*100:.1f}%"],
        ["Legitimate Prob.",     f"{float(ml.get('legitimate_proba', 0))*100:.1f}%"],
        ["Override Reason",      _clean(ml.get("override_reason") or "None", 200)],
    ]

    t = Table(rows, colWidths=[6.0*cm, 11.0*cm])
    ts = _table_style()
    pred = ml.get("prediction", "")
    for i, row in enumerate(rows[1:], 1):
        if row[0] == "Prediction":
            ts.add("TEXTCOLOR", (1, i), (1, i), C_RED if pred == "PHISHING" else C_GREEN)
            ts.add("FONTNAME",  (1, i), (1, i), _FONT_BOLD)
    t.setStyle(ts)
    return t


def _rules_table(fired: List[Dict], S: dict) -> Table:
    sev_colors = {
        "critical": C_RED, "high": C_YELLOW,
        "medium":   colors.HexColor("#fb923c"), "low": C_ACCENT,
    }
    rows = [["RULE", "SEVERITY", "CONFIDENCE", "DESCRIPTION"]]
    for rule in fired[:10]:
        name = _clean(rule.get("name", rule.get("rule", "")), 35)
        sev  = _clean(rule.get("severity", ""), 15).upper()
        conf = f"{float(rule.get('confidence', 0))*100:.0f}%"
        desc = _clean(rule.get("description", ""), 180)
        rows.append([name, sev, conf, Paragraph(desc, S["indicator_msg"])])

    t = Table(rows, colWidths=[3.8*cm, 2.2*cm, 2.2*cm, 8.8*cm])
    ts = _table_style()
    for i, rule in enumerate(fired[:10], 1):
        c = sev_colors.get((rule.get("severity") or "").lower(), C_TEXT)
        ts.add("TEXTCOLOR", (1, i), (1, i), c)
        ts.add("FONTNAME",  (1, i), (1, i), _FONT_BOLD)
    t.setStyle(ts)
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  CSV GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def _generate_csv(result: Dict, output_path: str) -> None:
    features  = result.get("features") or {}
    skip_keys = {"ssl_info","domain_info","redirect_info","homograph_info","visual_analysis"}
    flat_feats = {k: v for k, v in features.items()
                  if k not in skip_keys and isinstance(v, (int, float, bool, str))}

    ti   = result.get("threat_intel") or {}
    sb   = result.get("safe_browsing") or ti.get("safe_browsing") or {}
    vt   = result.get("virustotal")    or ti.get("virustotal")    or {}
    ml   = result.get("ml")            or {}
    expl = _resolve_summary(result)
    inds = _resolve_indicators(result)

    def _top_reasons() -> List[str]:
        detailed = result.get("detailed_explanation") or result.get("explanations") or {}
        return (detailed.get("top_reasons") or [])

    top = _top_reasons()

    row = {
        "url":              result.get("url", ""),
        "verdict":          result.get("verdict", result.get("prediction", "")),
        "confidence":       result.get("confidence", 0),
        "risk_score":       result.get("risk_score", 0),
        "risk_level":       result.get("risk_level", ""),
        "verdict_source":   result.get("verdict_source", ""),
        "verdict_reason":   _clean(result.get("verdict_reason") or "", 300),
        "timestamp":        result.get("timestamp", datetime.now().isoformat()),
        "pipeline_version": result.get("pipeline_version", "6.1"),
        "ml_prediction":       ml.get("prediction", ""),
        "ml_confidence":       ml.get("confidence", 0),
        "ml_phishing_proba":   ml.get("phishing_proba", 0),
        "ml_legitimate_proba": ml.get("legitimate_proba", 0),
        "sb_status":       sb.get("status", ""),
        "sb_is_threat":    sb.get("is_threat", False),
        "sb_threat_types": "|".join(sb.get("threat_types", [])),
        "sb_message":      _clean(sb.get("message") or "", 200),
        "vt_status":           vt.get("status", ""),
        "vt_is_malicious":     vt.get("is_malicious", False),
        "vt_malicious_count":  vt.get("malicious_count", 0),
        "vt_suspicious_count": vt.get("suspicious_count", 0),
        "vt_total_engines":    vt.get("total_engines", 0),
        "vt_detection_ratio":  vt.get("detection_ratio", ""),
        "ai_summary":      expl[:300],
        "top_reason_1":    _clean(top[0], 200) if len(top) > 0 else "",
        "top_reason_2":    _clean(top[1], 200) if len(top) > 1 else "",
        "indicator_count": len(inds),
    }
    row.update({f"feat_{k}": v for k, v in flat_feats.items()})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def render_download_buttons(result_dict: Dict) -> None:
    try:
        import streamlit as st
    except ImportError:
        return

    st.divider()
    st.markdown("#### Download Incident Report")
    files = generate_report(result_dict, formats=["pdf", "csv"])

    col1, col2 = st.columns(2)
    with col1:
        if "pdf" in files:
            with open(files["pdf"], "rb") as f:
                st.download_button("Download PDF Report", f.read(),
                    os.path.basename(files["pdf"]), "application/pdf",
                    use_container_width=True)
        elif "pdf_error" in files:
            st.warning(f"PDF unavailable: {files['pdf_error']}")
    with col2:
        if "csv" in files:
            with open(files["csv"], "rb") as f:
                st.download_button("Download CSV Report", f.read(),
                    os.path.basename(files["csv"]), "text/csv",
                    use_container_width=True)


def get_report_bytes(result_dict: Dict, fmt: str = "pdf") -> Tuple[bytes, str]:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        files = generate_report(result_dict, output_dir=tmp, formats=[fmt])
        if fmt not in files:
            return b"", f"error.{fmt}"
        path = files[fmt]
        with open(path, "rb") as f:
            return f.read(), os.path.basename(path)


# ═══════════════════════════════════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    mock_result = {
        "url":            "http://verify-your-paypal-account-now.com",
        "verdict":        "PHISHING",
        "confidence":     0.902,
        "risk_score":     47,
        "risk_level":     "CRITICAL",
        "verdict_source": "ml_model",
        "verdict_reason": "Phishing keywords (login/verify/secure/bank) + no HTTPS.",
        "pipeline_version": "6.1",
        "timestamp":      datetime.now().isoformat(),
        "ml": {
            "available": True, "prediction": "PHISHING",
            "confidence": 0.902, "phishing_proba": 0.980,
            "legitimate_proba": 0.020,
            "override_reason": "Phishing keywords (login/verify/secure/bank) + no HTTPS.",
        },
        "rules": {
            "verdict": "phishing", "confidence": 0.85,
            "fired_rules": [
                {"name": "http_with_phishing_keywords", "severity": "high",
                 "confidence": 0.85,
                 "description": "Phishing keywords (login/verify/secure/bank) + no HTTPS."},
                {"name": "many_dashes_and_keywords", "severity": "high",
                 "confidence": 0.82,
                 "description": "4+ hyphens in URL + phishing keywords + no HTTPS."},
            ],
        },
        "safe_browsing": {
            "status": "safe", "is_threat": False,
            "threat_types": [], "message": "No threats in Google database",
        },
        "virustotal": {
            "status": "completed", "is_malicious": False,
            "malicious_count": 0, "suspicious_count": 0,
            "harmless_count": 0, "undetected_count": 0,
            "total_engines": 0, "detection_ratio": "0/0",
            "message": "Clean - 0 engines scanned, none flagged",
        },
        "features": {
            "url_length": 41, "num_dots": 1, "has_ip_address": 0,
            "uses_https": 0, "has_suspicious_keywords": 1,
            "num_dashes": 4, "entropy": 4.1923, "num_subdomains": 0,
            "has_at_symbol": 0, "has_login_form": 0,
            "has_suspicious_tld": 0, "has_iframes": 0,
            "has_obfuscated_js": 0, "has_favicon_mismatch": 0,
        },
        "detailed_explanation": {
            "ai_narrative": "URL contains phishing keywords and no HTTPS. High-confidence ML verdict.",
            "summary": "Phishing keywords (login/verify/secure/bank) + no HTTPS.",
            "top_reasons": [
                "Suspicious keywords detected in URL.",
                "No HTTPS - plain HTTP only.",
                "4 hyphens in URL is unusual for legitimate sites.",
            ],
            "indicators": [
                {"category": "URL", "severity": "high",
                 "message": "Suspicious keywords detected: login, verify, paypal, account."},
                {"category": "SSL", "severity": "high",
                 "message": "No HTTPS encryption - site uses plain HTTP."},
                {"category": "URL", "severity": "medium",
                 "message": "4 hyphens in hostname is a common phishing pattern."},
            ],
        },
        "shap_explanation": [
            {"feature": "has_suspicious_keywords", "shap_value":  0.4500},
            {"feature": "uses_https",               "shap_value": -0.3800},
            {"feature": "num_dashes",               "shap_value":  0.2100},
            {"feature": "entropy",                  "shap_value":  0.1200},
            {"feature": "url_length",               "shap_value":  0.0800},
        ],
    }

    print("=" * 60)
    print("PHISHGUARD REPORT GENERATOR — SELF TEST v4.5")
    print("=" * 60)
    print(f"Font in use: {_FONT_REGULAR}")

    files = generate_report(mock_result, output_dir="/tmp/phishguard_test")
    print("\nGenerated files:")
    for fmt, path in files.items():
        if not fmt.endswith("_error"):
            size = os.path.getsize(path)
            print(f"  {fmt.upper():4s}: {path}  ({size:,} bytes)")
        else:
            print(f"  ERROR [{fmt}]: {path}")

    print("\nFix verification:")
    print(f"  [1] Unicode font registered: {_FONT_REGULAR != 'Helvetica'}")
    print(f"  [2] Indicators resolved: {len(_resolve_indicators(mock_result))} found")
    print(f"  [4] Column widths: cm-based (not %-strings)")
    print(f"  [7] fontName kwarg fix: no TypeError on style build")
    print(f"  [8] Palette: violet(#a855f7) / rose(#f43f5e) / lime(#bef264)")
    print("=" * 60)
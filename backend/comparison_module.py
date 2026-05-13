"""
comparison_module.py — PhishGuard AI: Rule Engine vs ML Model Comparison
=========================================================================

PURPOSE:
    Side-by-side comparison of the Rule Engine verdict versus the ML
    model verdict — highlighting agreements, conflicts, and the final
    arbitrated decision.

    This is critical for:
      • Debugging model behaviour (why did ML override the rules?)
      • Explaining the system to examiners / end users
      • Catching edge cases where ML and rules strongly disagree
      • Building trust through transparency

KEY FUNCTION:
    compare_verdicts(result_dict)
    → returns a structured ComparisonReport with Streamlit widget support

CONFLICT LEVELS:
    AGREEMENT   — both say same thing (high confidence)
    SOFT_CONFLICT — one says PHISHING, other is uncertain
    HARD_CONFLICT — one says PHISHING, other says LEGITIMATE
    UNKNOWN     — insufficient data from one side
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VerdictInfo:
    """One system's verdict (either rules or ML)."""
    system:       str          # "rule_engine" | "ml_model"
    verdict:      str          # "PHISHING" | "LEGITIMATE" | "UNCERTAIN" | "N/A"
    confidence:   float        # 0.0 – 1.0
    details:      List[str]    = field(default_factory=list)  # reasons
    fired_rules:  List[Dict]   = field(default_factory=list)  # for rules only
    raw:          Dict         = field(default_factory=dict)   # original result block


@dataclass
class ComparisonReport:
    """Full comparison between rule engine and ML model."""
    rule_verdict:    VerdictInfo
    ml_verdict:      VerdictInfo
    final_verdict:   VerdictInfo
    conflict_level:  str        # "AGREEMENT" | "SOFT_CONFLICT" | "HARD_CONFLICT" | "UNKNOWN"
    conflict_reason: str        # plain-English explanation of the conflict
    agreement_score: float      # 0.0 (total conflict) – 1.0 (full agreement)
    recommendation:  str        # what the user should do / believe
    arbitration_log: List[str]  = field(default_factory=list)  # step-by-step decision trace
    ui_card:         Dict       = field(default_factory=dict)   # pre-formatted for UI


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def compare_verdicts(result_dict: Dict) -> ComparisonReport:
    """
    Compare Rule Engine vs ML Model verdicts from an analyze_url() result.

    Args:
        result_dict: Full result from analyze_url() — must contain
                     "rules" and "ml" keys

    Returns:
        ComparisonReport with all comparison details

    Usage:
        from comparison_module import compare_verdicts
        report = compare_verdicts(analysis_result)

        # Quick access:
        print(report.conflict_level)   # "HARD_CONFLICT"
        print(report.recommendation)

        # Streamlit:
        render_comparison_card(report)
    """
    rules_data = result_dict.get("rules") or {}
    ml_data    = result_dict.get("ml")    or {}

    # ── Extract rule engine verdict ───────────────────────────────────
    rule_verdict = _extract_rule_verdict(rules_data)

    # ── Extract ML verdict ────────────────────────────────────────────
    ml_verdict = _extract_ml_verdict(ml_data)

    # ── Extract final verdict ─────────────────────────────────────────
    final_verdict = _extract_final_verdict(result_dict)

    # ── Compare + classify conflict ───────────────────────────────────
    conflict_level, conflict_reason, agreement_score = _classify_conflict(
        rule_verdict, ml_verdict
    )

    # ── Build arbitration log ─────────────────────────────────────────
    arbitration_log = _build_arbitration_log(
        rule_verdict, ml_verdict, final_verdict, result_dict
    )

    # ── Recommendation ────────────────────────────────────────────────
    recommendation = _build_recommendation(
        conflict_level, final_verdict, rule_verdict, ml_verdict
    )

    # ── Build UI card ─────────────────────────────────────────────────
    ui_card = _build_ui_card(
        rule_verdict, ml_verdict, final_verdict,
        conflict_level, conflict_reason, agreement_score
    )

    report = ComparisonReport(
        rule_verdict    = rule_verdict,
        ml_verdict      = ml_verdict,
        final_verdict   = final_verdict,
        conflict_level  = conflict_level,
        conflict_reason = conflict_reason,
        agreement_score = agreement_score,
        recommendation  = recommendation,
        arbitration_log = arbitration_log,
        ui_card         = ui_card,
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════
#  EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_rule_verdict(rules: Dict) -> VerdictInfo:
    """Parse the rules engine result block."""
    raw_verdict = (rules.get("verdict") or "").upper()

    # Normalise
    if raw_verdict in ("PHISHING", "PHISH"):
        verdict = "PHISHING"
    elif raw_verdict in ("CLEAN", "LEGITIMATE", "LEGIT", "SAFE"):
        verdict = "LEGITIMATE"
    else:
        verdict = "UNCERTAIN"

    confidence = float(rules.get("confidence", 0.5))
    fired      = rules.get("fired_rules", []) or []
    details    = []

    crit_hit = rules.get("critical_hit", False)
    if crit_hit:
        details.append("🔴 Critical rule triggered")

    for rule in fired[:5]:
        name  = rule.get("name") or rule.get("rule", "Unknown")
        sev   = rule.get("severity", "")
        desc  = rule.get("description", "")
        icon  = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}.get(sev,"⚪")
        details.append(f"{icon} [{sev.upper()}] {name}: {desc[:80]}")

    if not details:
        details.append("No rules fired — URL appears clean by rule engine")

    return VerdictInfo(
        system      = "rule_engine",
        verdict     = verdict,
        confidence  = confidence,
        details     = details,
        fired_rules = fired,
        raw         = rules,
    )


def _extract_ml_verdict(ml: Dict) -> VerdictInfo:
    """Parse the ML model result block."""
    if not ml.get("available"):
        return VerdictInfo(
            system     = "ml_model",
            verdict    = "N/A",
            confidence = 0.0,
            details    = ["❌ ML model not trained — run `python -m backend.train` first"],
            raw        = ml,
        )

    raw_label  = (ml.get("prediction") or "").upper()
    verdict    = "PHISHING" if raw_label == "PHISHING" else (
                 "LEGITIMATE" if raw_label == "LEGITIMATE" else "UNCERTAIN")
    confidence = float(ml.get("confidence", 0.5))
    phish_prob = float(ml.get("phishing_proba", 0.5))
    legit_prob = float(ml.get("legitimate_proba", 0.5))
    override   = ml.get("override_reason")

    details = []
    details.append(f"Phishing probability:  {phish_prob*100:.1f}%")
    details.append(f"Legitimate probability: {legit_prob*100:.1f}%")
    if override:
        details.append(f"⚡ Override: {override}")

    # Confidence tier
    if confidence >= 0.9:
        details.append("High-confidence classification")
    elif confidence >= 0.7:
        details.append("Moderate-confidence classification")
    else:
        details.append("⚠️ Low confidence — borderline case")

    return VerdictInfo(
        system     = "ml_model",
        verdict    = verdict,
        confidence = confidence,
        details    = details,
        raw        = ml,
    )


def _extract_final_verdict(result: Dict) -> VerdictInfo:
    """Extract the unified final verdict from the top-level result."""
    verdict    = (result.get("verdict") or "UNKNOWN").upper()
    confidence = float(result.get("confidence", 0.5))
    source     = result.get("verdict_source", "unknown")
    reason     = result.get("verdict_reason", "")

    details = [
        f"Decided by: {source.replace('_',' ').title()}",
        reason[:200] if reason else "",
    ]

    return VerdictInfo(
        system     = "final_arbiter",
        verdict    = verdict,
        confidence = confidence,
        details    = [d for d in details if d],
        raw        = result,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CONFLICT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _classify_conflict(
    rule_v: VerdictInfo,
    ml_v:   VerdictInfo,
) -> Tuple[str, str, float]:
    """
    Classify the disagreement level between rule engine and ML.

    Returns:
        (conflict_level, conflict_reason, agreement_score)
    """
    rv = rule_v.verdict
    mv = ml_v.verdict

    # ── Both N/A or uncertain ─────────────────────────────────────────
    if rv == "N/A" or mv == "N/A":
        return (
            "UNKNOWN",
            "One system is unavailable — comparison cannot be made.",
            0.5,
        )

    if rv == "UNCERTAIN" and mv == "UNCERTAIN":
        return (
            "UNKNOWN",
            "Both systems are uncertain about this URL — it is a borderline case.",
            0.5,
        )

    # ── Full agreement ────────────────────────────────────────────────
    if rv == mv:
        avg_conf = (rule_v.confidence + ml_v.confidence) / 2
        return (
            "AGREEMENT",
            f"Both systems agree: {rv}. Average confidence: {avg_conf*100:.0f}%.",
            min(1.0, avg_conf + 0.1),
        )

    # ── One uncertain ─────────────────────────────────────────────────
    if rv == "UNCERTAIN" or mv == "UNCERTAIN":
        certain_v   = rule_v if mv == "UNCERTAIN" else ml_v
        uncertain_v = ml_v   if mv == "UNCERTAIN" else rule_v
        return (
            "SOFT_CONFLICT",
            f"{certain_v.system.replace('_',' ').title()} says {certain_v.verdict}, "
            f"but {uncertain_v.system.replace('_',' ').title()} is uncertain. "
            f"Treat with caution.",
            0.4,
        )

    # ── Hard conflict (PHISHING vs LEGITIMATE) ────────────────────────
    rc = rule_v.confidence
    mc = ml_v.confidence
    stronger = "rule_engine" if rc > mc else "ml_model"
    diff     = abs(rc - mc)

    reason = (
        f"⚠️ CONFLICT: Rule Engine says {rv} ({rc*100:.0f}% conf) but "
        f"ML Model says {mv} ({mc*100:.0f}% conf). "
        f"The {'Rule Engine' if stronger=='rule_engine' else 'ML Model'} "
        f"has stronger confidence (+{diff*100:.0f}%)."
    )

    return ("HARD_CONFLICT", reason, max(0.0, 1.0 - diff))


# ═══════════════════════════════════════════════════════════════════════════
#  ARBITRATION LOG
# ═══════════════════════════════════════════════════════════════════════════

def _build_arbitration_log(
    rule_v:  VerdictInfo,
    ml_v:    VerdictInfo,
    final_v: VerdictInfo,
    result:  Dict,
) -> List[str]:
    """
    Build a step-by-step trace of how the final verdict was reached.
    Mirrors the logic in analyzer._assemble_verdict().
    """
    log    = []
    source = result.get("verdict_source", "unknown")
    ti     = result.get("threat_intel") or {}
    sb     = ti.get("safe_browsing") or {}
    vt     = ti.get("virustotal")    or {}

    log.append("─── ARBITRATION TRACE ───")

    # Step 1: Safe Browsing
    if sb.get("status") == "unsafe" and sb.get("is_threat"):
        log.append("✅ Step 1: Google Safe Browsing → PHISHING (override — highest priority)")
        log.append("   ↳ No further checks needed.")
        return log
    log.append("➡️  Step 1: Google Safe Browsing → Not flagged (continue)")

    # Step 2: Critical rule
    if rule_v.fired_rules and result.get("rules", {}).get("critical_hit"):
        log.append(f"✅ Step 2: Rule Engine → Critical rule fired → PHISHING (override)")
        return log
    log.append(f"➡️  Step 2: Rule Engine → {rule_v.verdict} ({rule_v.confidence*100:.0f}% conf), no critical override")

    # Step 3: VirusTotal
    if vt.get("status") == "completed" and vt.get("is_malicious"):
        log.append(f"✅ Step 3: VirusTotal → {vt.get('malicious_count',0)}/{vt.get('total_engines',0)} engines flagged → PHISHING")
        return log
    log.append(f"➡️  Step 3: VirusTotal → {vt.get('status','skipped')} (not malicious)")

    # Step 4: ML model
    log.append(f"✅ Step 4: ML Model → {ml_v.verdict} ({ml_v.confidence*100:.0f}% conf) → FINAL DECISION")
    log.append(f"   ↳ Final verdict: {final_v.verdict} via {source.replace('_',' ')}")

    return log


# ═══════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def _build_recommendation(
    conflict_level: str,
    final_v:        VerdictInfo,
    rule_v:         VerdictInfo,
    ml_v:           VerdictInfo,
) -> str:
    """Generate a human-readable recommendation."""
    fv = final_v.verdict
    fc = final_v.confidence

    if fv == "PHISHING":
        base = "🚨 Do NOT visit this site or enter any personal information."
        if conflict_level == "HARD_CONFLICT":
            base += (" (Note: the two detection systems disagreed — "
                     "treat this site as suspicious regardless.)")
        elif conflict_level == "AGREEMENT":
            base += " Both detection systems agree this is malicious."
        return base

    if fv == "LEGITIMATE":
        base = "✅ This URL appears safe, but always verify the domain carefully."
        if conflict_level == "HARD_CONFLICT":
            base += (f" (Note: the Rule Engine classified it as {rule_v.verdict} — "
                     "exercise extra caution.)")
        elif fc < 0.75:
            base += " Confidence is moderate — manually verify before entering credentials."
        return base

    return "⚠️ Verdict is uncertain. Avoid entering sensitive information until reviewed."


# ═══════════════════════════════════════════════════════════════════════════
#  UI CARD BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def _build_ui_card(
    rule_v:         VerdictInfo,
    ml_v:           VerdictInfo,
    final_v:        VerdictInfo,
    conflict_level: str,
    conflict_reason: str,
    agreement_score: float,
) -> Dict:
    """Pre-built dict for Streamlit / JSON API rendering."""
    conflict_colors = {
        "AGREEMENT":     "#00ff99",
        "SOFT_CONFLICT": "#ffb300",
        "HARD_CONFLICT": "#ff4560",
        "UNKNOWN":       "#5a7a9a",
    }
    conflict_icons = {
        "AGREEMENT":     "✅",
        "SOFT_CONFLICT": "⚠️",
        "HARD_CONFLICT": "🔥",
        "UNKNOWN":       "❓",
    }

    verdict_icons = {
        "PHISHING":   "⛔",
        "LEGITIMATE": "✅",
        "UNCERTAIN":  "❓",
        "N/A":        "—",
        "UNKNOWN":    "❓",
    }

    return {
        "conflict": {
            "level":   conflict_level,
            "reason":  conflict_reason,
            "score":   round(agreement_score * 100),
            "color":   conflict_colors.get(conflict_level, "#5a7a9a"),
            "icon":    conflict_icons.get(conflict_level, "❓"),
        },
        "rule_engine": {
            "verdict":    rule_v.verdict,
            "confidence": round(rule_v.confidence * 100),
            "icon":       verdict_icons.get(rule_v.verdict, "❓"),
            "details":    rule_v.details[:4],
            "rules_fired":len(rule_v.fired_rules),
        },
        "ml_model": {
            "verdict":    ml_v.verdict,
            "confidence": round(ml_v.confidence * 100),
            "icon":       verdict_icons.get(ml_v.verdict, "❓"),
            "details":    ml_v.details[:4],
        },
        "final": {
            "verdict":    final_v.verdict,
            "confidence": round(final_v.confidence * 100),
            "icon":       verdict_icons.get(final_v.verdict, "❓"),
            "details":    final_v.details,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT WIDGET
# ═══════════════════════════════════════════════════════════════════════════

def render_comparison_card(report: ComparisonReport) -> None:
    """
    Render the full Rule vs ML comparison in Streamlit.

    Usage:
        from comparison_module import compare_verdicts, render_comparison_card
        report = compare_verdicts(analysis_result)
        render_comparison_card(report)
    """
    try:
        import streamlit as st
    except ImportError:
        return

    card = report.ui_card
    cf   = card["conflict"]

    st.subheader("⚖️ Rule Engine vs ML Model")

    # ── Conflict banner ───────────────────────────────────────────────
    conflict_style = {
        "AGREEMENT":     st.success,
        "SOFT_CONFLICT": st.warning,
        "HARD_CONFLICT": st.error,
        "UNKNOWN":       st.info,
    }.get(cf["level"], st.info)
    conflict_style(f"{cf['icon']} **{cf['level']}** — {cf['reason']}")

    # ── Three columns: Rules | vs | ML ───────────────────────────────
    col_rule, col_vs, col_ml = st.columns([2, 1, 2])

    with col_rule:
        rv = card["rule_engine"]
        st.markdown(f"### {rv['icon']} Rule Engine")
        _verdict_metric(rv["verdict"], rv["confidence"])
        st.caption(f"Fired {rv['rules_fired']} rule(s)")
        for detail in rv["details"]:
            st.caption(f"• {detail}")

    with col_vs:
        st.markdown("<br><br>", unsafe_allow_html=True)
        score = cf["score"]
        st.metric("Agreement", f"{score}%")
        progress_color = "green" if score >= 70 else ("orange" if score >= 40 else "red")
        st.progress(score / 100)

    with col_ml:
        mv = card["ml_model"]
        st.markdown(f"### {mv['icon']} ML Model")
        _verdict_metric(mv["verdict"], mv["confidence"])
        for detail in mv["details"]:
            st.caption(f"• {detail}")

    st.divider()

    # ── Final verdict ─────────────────────────────────────────────────
    fv = card["final"]
    col_final, col_rec = st.columns([1, 2])
    with col_final:
        st.markdown(f"#### {fv['icon']} Final Decision")
        _verdict_metric(fv["verdict"], fv["confidence"])
        for d in fv["details"]:
            st.caption(d)

    with col_rec:
        st.markdown("#### 💡 Recommendation")
        st.write(report.recommendation)

    # ── Arbitration trace ─────────────────────────────────────────────
    with st.expander("🔍 Arbitration Trace — How was the verdict decided?"):
        for step in report.arbitration_log:
            st.text(step)


def _verdict_metric(verdict: str, confidence: int) -> None:
    """Compact verdict + confidence display for Streamlit."""
    try:
        import streamlit as st
        color = "red" if verdict == "PHISHING" else ("green" if verdict == "LEGITIMATE" else "gray")
        st.markdown(f"**:{color}[{verdict}]** ({confidence}% conf)")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  STANDALONE DICT OUTPUT (for Flask/non-Streamlit use)
# ═══════════════════════════════════════════════════════════════════════════

def get_comparison_dict(result_dict: Dict) -> Dict:
    """
    Return comparison as a plain dict (JSON-serializable).
    Use this in Flask API routes.

    Usage:
        from comparison_module import get_comparison_dict
        comp = get_comparison_dict(result)
        return jsonify(comp)
    """
    report = compare_verdicts(result_dict)
    return {
        "rule_engine": {
            "verdict":    report.rule_verdict.verdict,
            "confidence": report.rule_verdict.confidence,
            "details":    report.rule_verdict.details,
            "rules_fired":len(report.rule_verdict.fired_rules),
        },
        "ml_model": {
            "verdict":    report.ml_verdict.verdict,
            "confidence": report.ml_verdict.confidence,
            "details":    report.ml_verdict.details,
        },
        "final": {
            "verdict":    report.final_verdict.verdict,
            "confidence": report.final_verdict.confidence,
            "source":     result_dict.get("verdict_source",""),
            "reason":     result_dict.get("verdict_reason",""),
        },
        "comparison": {
            "conflict_level":  report.conflict_level,
            "conflict_reason": report.conflict_reason,
            "agreement_score": round(report.agreement_score * 100),
            "recommendation":  report.recommendation,
        },
        "arbitration_log": report.arbitration_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pprint

    print("=" * 65)
    print("🧪 COMPARISON MODULE — SELF TEST")
    print("=" * 65)

    # ── Test 1: AGREEMENT ─────────────────────────────────────────────
    mock_result_agree = {
        "verdict": "PHISHING", "confidence": 0.91,
        "risk_level": "CRITICAL", "verdict_source": "ml_model",
        "verdict_reason": "ML ensemble classified as phishing (91%)",
        "rules": {
            "verdict": "phishing", "confidence": 0.88,
            "critical_hit": True,
            "fired_rules": [
                {"name":"IP_IN_URL","severity":"critical","confidence":0.95,
                 "description":"Raw IP address in URL"},
            ]
        },
        "ml": {
            "available": True, "prediction": "PHISHING",
            "confidence": 0.91, "phishing_proba": 0.91,
            "legitimate_proba": 0.09, "override_reason": None,
        },
        "threat_intel": {
            "safe_browsing": {"status":"safe","is_threat":False},
            "virustotal":    {"status":"skipped","is_malicious":False},
        }
    }

    print("\n📋 Test 1: AGREEMENT case")
    report1 = compare_verdicts(mock_result_agree)
    print(f"   Conflict Level:   {report1.conflict_level}")
    print(f"   Agreement Score:  {report1.agreement_score*100:.0f}%")
    print(f"   Rule Verdict:     {report1.rule_verdict.verdict} ({report1.rule_verdict.confidence*100:.0f}%)")
    print(f"   ML Verdict:       {report1.ml_verdict.verdict} ({report1.ml_verdict.confidence*100:.0f}%)")
    print(f"   Recommendation:   {report1.recommendation[:100]}")

    # ── Test 2: HARD CONFLICT ─────────────────────────────────────────
    mock_result_conflict = {
        "verdict": "PHISHING", "confidence": 0.72,
        "risk_level": "HIGH", "verdict_source": "rule_engine",
        "verdict_reason": "Critical rule: IP address in URL",
        "rules": {
            "verdict": "phishing", "confidence": 0.85,
            "critical_hit": True,
            "fired_rules": [
                {"name":"IP_IN_URL","severity":"critical","confidence":0.85,
                 "description":"Raw IP address in URL"},
            ]
        },
        "ml": {
            "available": True, "prediction": "LEGITIMATE",
            "confidence": 0.63, "phishing_proba": 0.37,
            "legitimate_proba": 0.63, "override_reason": None,
        },
        "threat_intel": {
            "safe_browsing": {"status":"safe","is_threat":False},
            "virustotal":    {"status":"skipped","is_malicious":False},
        }
    }

    print("\n📋 Test 2: HARD CONFLICT case")
    report2 = compare_verdicts(mock_result_conflict)
    print(f"   Conflict Level:   {report2.conflict_level}")
    print(f"   Agreement Score:  {report2.agreement_score*100:.0f}%")
    print(f"   Conflict Reason:  {report2.conflict_reason[:120]}")
    print(f"   Rule Verdict:     {report2.rule_verdict.verdict} ({report2.rule_verdict.confidence*100:.0f}%)")
    print(f"   ML Verdict:       {report2.ml_verdict.verdict} ({report2.ml_verdict.confidence*100:.0f}%)")
    print(f"   Final:            {report2.final_verdict.verdict}")
    print(f"   Recommendation:   {report2.recommendation[:100]}")

    print("\n📋 Arbitration Log:")
    for step in report2.arbitration_log:
        print(f"   {step}")

    # ── Test 3: Dict output for Flask API ─────────────────────────────
    print("\n📋 Test 3: get_comparison_dict() output")
    comp = get_comparison_dict(mock_result_conflict)
    pprint.pprint(comp, width=80, depth=3)

    print("\n" + "=" * 65)
    print("✅ Module 7 test complete")
    print("=" * 65)
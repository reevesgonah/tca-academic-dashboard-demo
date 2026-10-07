import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

from data_utils import (
    load_base_scores, load_subject_teachers, load_class_teachers,
    student_rank_in_class, student_sittings_trend, student_rank_sittings_trend, SUBJECT_CLUSTERS,
    years_available, terms_for_year, school_pass_rate, order_exams,
    overall_average, previous_vs_current_trend, term_to_term_trend,
    improvement_vs_previous_exam, improvement_vs_previous_term,
    previous_sitting_label, performance_by_grade, performance_by_subject,
    areas_requiring_attention, class_comparison_within_grade,
    subject_ranking, subject_vs_school_average, subject_sittings_trend,
    student_report,
)

LOGO_PATH = Path(__file__).parent / "assets" / "logo.jpeg"
_logo_img = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None

st.set_page_config(
    page_title="Total Care Academy — Dashboard",
    layout="wide",
    page_icon=_logo_img if _logo_img else "📊",
)

# ---------------------------------------------------------------------------
# Palette — cohesive-but-colorful pastel set (echoes the reference mockup),
# plus the semantic blue/green/amber/red used for status.
# ---------------------------------------------------------------------------
BLUE, GREEN, AMBER, RED, PURPLE, GREY = "#2F6FE0", "#3FAE7A", "#D98829", "#D24949", "#8C6FE0", "#7A8699"
NAVY = "#003f88"
INK = "#1D2939"
ACCENTS = [BLUE, GREEN, PURPLE, AMBER]
PASTEL = ["#5B8DEF", "#57C7A6", "#9B8AFB", "#F2A65A", "#F2C94C", "#5FC8E3", "#EA7C7C", "#7FD8A2"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'Poppins', 'Segoe UI', sans-serif; }

    .stApp { background: #E4E9F2; }
    .block-container { padding-top: 2.6rem; padding-bottom: 2rem; max-width: 1200px; }

    section[data-testid="stSidebar"] { background: #003f88; width: 258px !important; }
    section[data-testid="stSidebar"] > div:first-child { width: 258px !important; }
    section[data-testid="stSidebar"] * { color: #EAF0FC !important; }
    section[data-testid="stSidebar"] img { border-radius: 8px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 8px 10px; border-radius: 8px; margin-bottom: 1px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

    /* Panels use Streamlit's documented `key=` class hook (st-key-<key>) rather
       than internal testids, which can change between Streamlit versions. */
    div[class*="st-key-panel"] {
        border-radius: 16px !important; box-shadow: 0 8px 24px rgba(16,26,51,0.14) !important;
        background: #FFFFFF !important; border: 1px solid #C6CEDE !important;
        padding: 18px 20px !important;
    }

    div[data-testid="stMetric"] {
        background-color: #EAF2FE; border-radius: 12px; padding: 12px 14px;
        border: 1px solid #DCE7FB;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; }

    .kpi-card {
        border-radius: 14px; padding: 12px 14px; box-shadow: 0 3px 10px rgba(29,41,73,0.08);
        border: 1px solid #F5EEDB; background: #FFFCF5; height: 100%; min-height: 108px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .kpi-top { display:flex; align-items:center; gap:8px; margin-bottom: 6px; }
    .kpi-icon {
        width: 30px; height: 30px; border-radius: 9px; display:flex; align-items:center;
        justify-content:center; font-size: 15px; flex-shrink: 0;
    }
    .kpi-label {
        color:#7A8699; font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .kpi-value { font-size: 1.5rem; font-weight: 700; color: #1D2939; line-height: 1.1; }
    .kpi-delta { font-size: 0.8rem; font-weight: 600; margin-top: 2px; }

    .insight {
        background: #FFF8E7; border-left: 3px solid #D9C68A; border-radius: 10px;
        padding: 12px 14px; font-size: 0.87rem; color: #3A4459; line-height: 1.6;
    }
    .insight b { color: #1D2939; }

    .attention-card {
        border-radius: 10px; padding: 9px 13px; margin-bottom: 7px;
        border-left: 4px solid; font-size: 0.88rem; line-height: 1.4;
    }
    .attention-red { background: #FCECEC; border-color: #D24949; }
    .attention-amber { background: #FDF1E2; border-color: #D98829; }
    .subtitle { color: #7A8699; font-size: 0.86rem; margin-top: -6px; }

    .bar-row { display:flex; align-items:center; gap:10px; margin-bottom: 7px; }
    .bar-label { width: 92px; font-size: 0.82rem; color:#3A4459; flex-shrink:0; }
    .bar-track { flex:1; background:#EEF1F6; border-radius: 6px; height: 14px; position:relative; overflow:hidden; }
    .bar-fill { height: 100%; border-radius: 6px; position:absolute; left:0; top:0; }
    .bar-fill-prev { opacity: 0.32; }
    .bar-fill-cur { opacity: 1; }
    .bar-value { width: 64px; text-align:right; font-size: 0.8rem; font-weight:600; color:#1D2939; flex-shrink:0; }

    .filter-card {
        background: #fff; border-radius: 14px; padding: 10px 12px 4px 12px;
        box-shadow: 0 3px 10px rgba(29,41,73,0.07); border: 1px solid #ECEEF2;
    }
    .filter-card .fc-label { font-size: 0.68rem; color:#7A8699; text-transform:uppercase; letter-spacing:.03em; margin-bottom:-6px; }

    .report-header {
        background: linear-gradient(135deg, #003f88 0%, #0A5BB8 100%); border-radius: 16px;
        padding: 18px 22px; color: #fff !important; display:flex; align-items:center; gap:16px;
    }
    .report-header * { color: #fff !important; }
    .report-avatar {
        width: 56px; height: 56px; border-radius: 50%; background: rgba(255,255,255,0.18);
        display:flex; align-items:center; justify-content:center; font-size: 28px; flex-shrink:0;
    }
    .chip-row { display:flex; gap:10px; flex-wrap: wrap; margin: 10px 0 4px 0; }
    .chip {
        flex: 1; min-width: 130px; border-radius: 12px; padding: 8px 12px;
        border: 1px solid #F0E4C0; box-shadow: 0 2px 8px rgba(29,41,73,0.06); background:#FFF8E7;
    }
    .chip .chip-label { font-size: 0.72rem; color:#7A8699; text-transform:uppercase; letter-spacing:.02em; }
    .chip .chip-value { font-size: 1.1rem; font-weight:700; color:#1D2939; }

    div[data-testid="stPlotlyChart"] { border-radius: 10px; }

    @media print {
        section[data-testid="stSidebar"], header, .no-print,
        div[class*="st-key-no_print_btn"] { display: none !important; }
        .block-container { max-width: 100% !important; padding-top: 0 !important; }
        .stApp { background: #fff !important; }
        div[class*="st-key-panel"], div[data-testid="stPlotlyChart"] {
            page-break-inside: avoid !important; break-inside: avoid !important;
        }
        .js-plotly-plot .modebar { display: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_FONT = dict(family="Poppins, Segoe UI, sans-serif", size=12, color=INK)
NO_TOOLBAR = {"displayModeBar": False}


def style_fig(fig, height=260, legend=False):
    fig.update_layout(
        height=height, font=PLOT_FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=14, l=6, r=6, b=6), showlegend=legend,
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F6", zeroline=False)
    return fig


def fmt(x, suffix="%", nd=1):
    return f"{x:.{nd}f}{suffix}" if pd.notna(x) else "—"


def delta_word(x, flat_band=1.5):
    if pd.isna(x):
        return GREY, "—"
    if x > flat_band:
        return GREEN, f"▲ {x:+.1f}"
    if x < -flat_band:
        return RED, f"▼ {x:+.1f}"
    return AMBER, f"● {x:+.1f}"


def _pos_neg_color(val):
    """Cell-text colorer for Styler: green when positive, red when negative."""
    if pd.isna(val):
        return ""
    if val > 0:
        return f"color: {GREEN}; font-weight: 600;"
    if val < 0:
        return f"color: {RED}; font-weight: 600;"
    return ""


def style_deltas(styler, cols):
    """Apply _pos_neg_color to the given columns, using pandas' current
    Styler API (.map) with a fallback to the older .applymap for older
    pandas versions."""
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(_pos_neg_color, subset=cols)


def kpi_row(cards):
    """cards: list of (icon, label, value, delta_or_None). Every card reserves
    the same delta-line space so all KPI cards in a row render at equal height."""
    cols = st.columns(len(cards))
    for i, (icon, label, value, delta) in enumerate(cards):
        accent = ACCENTS[i % len(ACCENTS)]
        if delta is not None:
            color, txt = delta_word(delta)
            delta_html = f'<div class="kpi-delta" style="color:{color};">{txt} pts</div>'
        else:
            delta_html = '<div class="kpi-delta">&nbsp;</div>'
        cols[i].markdown(
            f"""
            <div class="kpi-card" style="border-top:3px solid {accent};">
              <div class="kpi-top">
                <div class="kpi-icon" style="background:{accent}1A; color:{accent};">{icon}</div>
                <div class="kpi-label">{label}</div>
              </div>
              <div class="kpi-value">{value}</div>
              {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def insight(text_html, height=None):
    style = f"min-height:{height}px;" if height else ""
    st.markdown(f'<div class="insight" style="{style}">💡&nbsp;{text_html}</div>', unsafe_allow_html=True)


def thin_bar(label, value, max_val=100, color=BLUE, value_fmt="{:.1f}%"):
    pct = max(0, min(100, (value / max_val) * 100)) if pd.notna(value) else 0
    val_txt = value_fmt.format(value) if pd.notna(value) else "—"
    st.markdown(
        f"""
        <div class="bar-row">
          <div class="bar-label">{label}</div>
          <div class="bar-track"><div class="bar-fill bar-fill-cur" style="width:{pct}%; background:{color};"></div></div>
          <div class="bar-value">{val_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dual_bar(label, current, previous, max_val=100, color=BLUE):
    """Current exam opaque, previous exam translucent behind it — lets a
    subject row show movement without a separate chart."""
    cur_pct = max(0, min(100, (current / max_val) * 100)) if pd.notna(current) else 0
    prev_pct = max(0, min(100, (previous / max_val) * 100)) if pd.notna(previous) else 0
    val_txt = f"{current:.1f}%" if pd.notna(current) else "—"
    prev_layer = (f'<div class="bar-fill bar-fill-prev" style="width:{prev_pct}%; background:{color};"></div>'
                  if pd.notna(previous) else "")
    st.markdown(
        f"""
        <div class="bar-row">
          <div class="bar-label">{label}</div>
          <div class="bar-track">{prev_layer}
            <div class="bar-fill bar-fill-cur" style="width:{cur_pct}%; background:{color};"></div>
          </div>
          <div class="bar-value">{val_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: navigation only
# ---------------------------------------------------------------------------
if _logo_img:
    st.sidebar.image(_logo_img, width=64)
st.sidebar.markdown("##### Total Care Academy")
st.sidebar.caption("Demo version — all students, teachers and scores are synthetic.")
st.sidebar.write("")

PAGES = ["Overview", "Subject Analysis", "Grade Analysis", "Student Search"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()

scores = load_base_scores()
subject_teachers = load_subject_teachers()
class_teachers = load_class_teachers()

pass_mark = st.sidebar.slider("Pass mark threshold", min_value=30, max_value=70, value=50, step=5)
st.sidebar.caption("Applies to pass rate, attention flags and class/subject comparisons everywhere in the app.")


# ---------------------------------------------------------------------------
# Compact top-right filter card: Academic Year -> Term -> Exam
# ---------------------------------------------------------------------------
def top_bar(title, subtitle):
    col_l, col_r = st.columns([2.4, 1])
    with col_l:
        st.markdown(f"### {title}")
        st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="filter-card">', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        years = years_available(scores)
        # NOTE: once a keyed widget has a value in st.session_state, Streamlit
        # ignores `index=` on every future rerun — so defaults must be set by
        # pre-populating session_state *before* the widget is created, not by
        # passing index= alongside an existing key. We also deliberately reset
        # the Exam choice whenever the Term changes (including on first load),
        # so it always lands on the true latest sitting (End-Term, when it
        # exists) rather than silently keeping a stale prior selection.
        if st.session_state.get("f_year") not in years:
            st.session_state["f_year"] = years[-1]
        with fc1:
            st.markdown('<div class="fc-label">Year</div>', unsafe_allow_html=True)
            y = st.selectbox("Year", years, key="f_year", label_visibility="collapsed")

        terms_y = terms_for_year(scores, y)
        if st.session_state.get("f_term") not in terms_y:
            st.session_state["f_term"] = terms_y[-1]
        with fc2:
            st.markdown('<div class="fc-label">Term</div>', unsafe_allow_html=True)
            t = st.selectbox("Term", terms_y, key="f_term", label_visibility="collapsed")

        exams_t = order_exams(scores[scores["TERM"] == t]["EXAM"].unique())
        if st.session_state.get("f_exam_term") != t or st.session_state.get("f_exam") not in exams_t:
            st.session_state["f_exam"] = exams_t[-1]
            st.session_state["f_exam_term"] = t
        with fc3:
            st.markdown('<div class="fc-label">Exam</div>', unsafe_allow_html=True)
            e = st.selectbox("Exam", exams_t, key="f_exam", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    return y, t, e, terms_y


# ===========================================================================
# PAGE 1 — OVERVIEW: "How are we doing?"
# ===========================================================================
if page == "Overview":
    sel_year, sel_term, sel_exam, terms_this_year = top_bar(
        "Executive Overview", "Monitor school performance at a glance.")
    term_scores = scores[scores["TERM"] == sel_term]
    exam_scores = term_scores[term_scores["EXAM"] == sel_exam]
    prev_sitting_label = previous_sitting_label(scores, sel_term, sel_exam)

    avg = overall_average(exam_scores)
    pr = school_pass_rate(exam_scores, pass_mark)
    imp_exam = improvement_vs_previous_exam(scores, sel_term, sel_exam)
    imp_term = improvement_vs_previous_term(scores, sel_term, sel_exam)

    kpi_row([
        ("📊", "Overall Average", fmt(avg), None),
        ("✅", f"Pass Rate (≥{pass_mark})", fmt(pr), None),
        ("📈", "vs Previous Exam", fmt(imp_exam, suffix=""), imp_exam),
        ("🗓️", "vs Previous Term", fmt(imp_term, suffix=""), imp_term),
    ])
    st.caption(f"Previous exam sitting used for comparison: **{prev_sitting_label}**.")

    st.write("")
    with st.container(border=True, key="panel_1"):
        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Performance Trend")
            trend_mode = st.radio("View", ["vs Previous Exam", "Term to Term"], horizontal=True, label_visibility="collapsed")
            if trend_mode == "vs Previous Exam":
                wt = previous_vs_current_trend(scores, sel_term, sel_exam)
                if len(wt) >= 2:
                    st.caption(f"Comparing **{wt['STAGE'].iloc[0]}** → **{wt['STAGE'].iloc[1]}** "
                               f"(the sitting immediately before the one selected in the filters, "
                               f"through to it).")
                else:
                    st.caption("This is the first exam sitting on record — nothing earlier to compare against yet.")
                fig = px.line(wt, x="STAGE", y="SCORE", markers=True,
                               category_orders={"STAGE": wt["STAGE"].tolist()})
                fig.update_traces(line_color=BLUE, marker=dict(size=10, color=BLUE), line_shape="spline")
                style_fig(fig, height=240)
                fig.update_layout(yaxis_title="Average score", xaxis_title=None)
                st.plotly_chart(fig, width="stretch")
                trend_df = wt
            else:
                tt = term_to_term_trend(scores, sel_year)
                fig = px.line(tt, x="TERM", y="SCORE", markers=True)
                fig.update_traces(line_color=BLUE, marker=dict(size=10, color=BLUE), line_shape="spline")
                style_fig(fig, height=240)
                fig.update_layout(yaxis_title="Average score", xaxis_title=None)
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Each term's latest sitting, across {sel_year}.")
                trend_df = tt.rename(columns={"TERM": "STAGE"})
        with right:
            if len(trend_df) >= 2:
                move = trend_df["SCORE"].iloc[-1] - trend_df["SCORE"].iloc[0]
                first, last = trend_df.iloc[0], trend_df.iloc[-1]
                direction = "improved" if move > 0.5 else ("declined" if move < -0.5 else "held steady")
                span_word = "the two most recent sittings" if trend_mode == "vs Previous Exam" else f"the {sel_year} school year"
                insight(f"The school average <b>{direction}</b> across {span_word}, moving from "
                        f"<b>{first['SCORE']:.1f}%</b> to <b>{last['SCORE']:.1f}%</b> "
                        f"({move:+.1f} pts). "
                        f"{'That is a meaningful shift worth flagging to staff.' if abs(move) > 3 else 'That is a fairly small movement, within normal variation.'}",
                        height=170)
            else:
                insight("Not enough sittings yet to show movement. Add another term or exam sitting and this "
                        "chart will start showing whether the school is trending up or down.", height=170)

    with st.container(border=True, key="panel_2"):
        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Performance by Grade")
            pg = performance_by_grade(scores, sel_term, sel_exam)
            fig_g = px.bar(pg, x="GRADE", y="SCORE", text="SCORE", color="GRADE",
                            color_continuous_scale=[[0, "#BFD7FB"], [1, "#1B4F9C"]],
                            range_color=[pg["GRADE"].min(), pg["GRADE"].max()])
            fig_g.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_g.update_layout(coloraxis_showscale=False)
            style_fig(fig_g, height=230)
            fig_g.update_layout(yaxis_title="Average score")
            st.plotly_chart(fig_g, width="stretch")
        with right:
            best_g = pg.loc[pg["SCORE"].idxmax()]
            worst_g = pg.loc[pg["SCORE"].idxmin()]
            gap = best_g['SCORE'] - worst_g['SCORE']
            insight(f"Grade <b>{int(best_g['GRADE'])}</b> leads the school at <b>{best_g['SCORE']:.1f}%</b>, "
                    f"while Grade <b>{int(worst_g['GRADE'])}</b> trails at <b>{worst_g['SCORE']:.1f}%</b> — "
                    f"a gap of {gap:.1f} pts. "
                    f"{'Worth checking whether that gap traces back to one subject or class.' if gap > 8 else 'Grades are fairly close together overall.'}",
                    height=180)

    with st.container(border=True, key="panel_3"):
        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Performance by Subject")
            ps = performance_by_subject(scores, sel_term, sel_exam)
            fig_s = px.bar(ps, x="SCORE", y="SUBJECT", orientation="h", text="SCORE", color="SUBJECT",
                            color_discrete_sequence=PASTEL)
            fig_s.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            style_fig(fig_s, height=300)
            fig_s.update_layout(yaxis=dict(categoryorder="total ascending"), xaxis_title="Average score")
            st.plotly_chart(fig_s, width="stretch")
        with right:
            best_s, worst_s = ps.iloc[0], ps.iloc[-1]
            insight(f"<b>{best_s['SUBJECT']}</b> is the strongest subject school-wide at "
                    f"<b>{best_s['SCORE']:.1f}%</b>. <b>{worst_s['SUBJECT']}</b> is the weakest at "
                    f"<b>{worst_s['SCORE']:.1f}%</b>. Head to the Subject Analysis page for a grade-by-grade "
                    f"breakdown of {worst_s['SUBJECT']} to see where the weakness concentrates.",
                    height=210)

    with st.container(border=True, key="panel_4"):
        st.markdown("#### Areas Requiring Attention")
        st.markdown('<p class="subtitle">Where to investigate next — not a full statistics dump.</p>', unsafe_allow_html=True)
        alerts = areas_requiring_attention(scores, sel_term, sel_exam, pass_mark)
        if not alerts:
            st.success("No significant declines, low pass rates, or class gaps detected against current thresholds.")
        else:
            c1, c2 = st.columns(2)
            for i, a in enumerate(alerts[:12]):
                cls = "attention-red" if a["level"] == "red" else "attention-amber"
                icon_ = "🔴" if a["level"] == "red" else "🟠"
                target = c1 if i % 2 == 0 else c2
                target.markdown(f'<div class="attention-card {cls}">{icon_} {a["text"]}</div>', unsafe_allow_html=True)

# ===========================================================================
# PAGE 2 — SUBJECTS: "Which subjects are strong or weak?"
# ===========================================================================
elif page == "Subject Analysis":
    sel_year, sel_term, sel_exam, terms_this_year = top_bar(
        "Subject Analysis", "Which subjects are performing well — and which need attention?")
    term_scores = scores[scores["TERM"] == sel_term]
    prev_sitting_label = previous_sitting_label(scores, sel_term, sel_exam)

    grade_options = ["All Grades"] + sorted(term_scores["GRADE"].unique().tolist())
    sel_grade_choice = st.selectbox("Grade", grade_options)
    rank_grade = None if sel_grade_choice == "All Grades" else sel_grade_choice
    scope_label = "school-wide" if rank_grade is None else f"Grade {rank_grade}"

    rank = subject_ranking(scores, sel_term, sel_exam, pass_mark, grade=rank_grade)

    with st.container(border=True, key="panel_5"):
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"#### Subject ranking — average score ({scope_label})")
            fig = px.bar(rank, x="AVERAGE", y="SUBJECT", orientation="h", text="AVERAGE", color="SUBJECT",
                          color_discrete_sequence=PASTEL)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            style_fig(fig, height=320)
            fig.update_layout(yaxis=dict(categoryorder="total ascending"), xaxis_title="Average score")
            st.plotly_chart(fig, width="stretch")
        with right:
            avg_of_shown = rank["AVERAGE"].mean()
            above = rank[rank["VS_SCHOOL"] > 0]["SUBJECT"].tolist()
            below_list = rank[rank["VS_SCHOOL"] <= 0]["SUBJECT"].tolist()
            scope_sentence = (f"Across all subjects, the average is <b>{avg_of_shown:.1f}%</b>."
                               if rank_grade is None else
                               f"For Grade {rank_grade}, the subject average here is <b>{avg_of_shown:.1f}%</b>, "
                               f"compared against the whole-school figure.")
            insight(f"{scope_sentence} {len(above)} of {len(rank)} subjects sit above the whole-school "
                    f"average: {', '.join(above[:5]) if above else 'none'}. "
                    f"{('Weaker than the school average: ' + ', '.join(below_list[:5]) + '.') if below_list else ''}",
                    height=230)

    with st.container(border=True, key="panel_6"):
        st.markdown(f"#### Pass rate & improvement by subject ({scope_label})")
        tbl = rank.rename(columns={
            "SUBJECT": "Subject", "AVERAGE": "Average", "PASS_RATE": f"Pass Rate (≥{pass_mark})",
            "IMPROVEMENT_TERM": "Improvement (term-to-term)", "IMPROVEMENT_EXAM": "Improvement (vs prev. exam)",
            "VS_SCHOOL": "vs School Avg",
        })[["Subject", "Average", f"Pass Rate (≥{pass_mark})", "vs School Avg", "Improvement (vs prev. exam)", "Improvement (term-to-term)"]]
        st.dataframe(
            style_deltas(
                tbl.style.format({"Average": "{:.1f}%", f"Pass Rate (≥{pass_mark})": "{:.1f}%",
                                   "vs School Avg": "{:+.1f}", "Improvement (vs prev. exam)": "{:+.1f}",
                                   "Improvement (term-to-term)": "{:+.1f}"}),
                ["vs School Avg", "Improvement (vs prev. exam)", "Improvement (term-to-term)"],
            ),
            hide_index=True, width="stretch",
        )
        st.caption(f"“vs School Avg” compares against the whole-school average for this sitting, regardless of the "
                   f"grade filter above. “Improvement (vs prev. exam)” compares to **{prev_sitting_label}**.")

        weak_pass = rank[rank["PASS_RATE"] < 60]
        declining = rank[rank["IMPROVEMENT_EXAM"] < -3]
        if len(weak_pass) or len(declining):
            if len(weak_pass):
                st.markdown(
                    f'<div class="attention-card attention-red">🔴 Weak pass rate: '
                    f'{", ".join(weak_pass["SUBJECT"])}</div>', unsafe_allow_html=True,
                )
            if len(declining):
                st.markdown(
                    f'<div class="attention-card attention-amber">🟠 Declining vs previous exam (even if the average is still high): '
                    f'{", ".join(declining["SUBJECT"])}</div>', unsafe_allow_html=True,
                )

    st.write("")
    with st.container(border=True, key="panel_7"):
        subjects = sorted(term_scores["SUBJECT"].unique())
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            sel_subject = st.selectbox("Inspect one subject", subjects)
        sv = subject_vs_school_average(scores, sel_term, sel_exam, sel_subject)
        with sc2:
            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div style='font-size:0.75rem;color:#7A8699;'>Subject Avg</div>"
                        f"<div style='font-size:1.05rem;font-weight:700;color:{INK};'>{fmt(sv['subject_avg'])}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='font-size:0.75rem;color:#7A8699;'>School Avg</div>"
                        f"<div style='font-size:1.05rem;font-weight:700;color:{INK};'>{fmt(sv['school_avg'])}</div>", unsafe_allow_html=True)
            gap = sv["subject_avg"] - sv["school_avg"]
            gcolor, _ = delta_word(gap)
            m3.markdown(f"<div style='font-size:0.75rem;color:#7A8699;'>vs School</div>"
                        f"<div style='font-size:1.05rem;font-weight:700;color:{gcolor};'>{gap:+.1f}</div>", unsafe_allow_html=True)

        st.write("")
        left, right = st.columns([2, 1])
        with left:
            sbg = sv["by_grade"]
            colors = [GREEN if v >= 0 else RED for v in sbg["VS_SCHOOL"]]
            fig_g = go.Figure(go.Bar(x=sbg["GRADE"].astype(str), y=sbg["SCORE"], marker_color=colors,
                                      text=sbg["SCORE"].round(1), textposition="outside"))
            fig_g.add_hline(y=sv["school_avg"], line_dash="dash", line_color=GREY,
                             annotation_text="School avg", annotation_position="top left")
            style_fig(fig_g, height=230)
            fig_g.update_layout(yaxis_title="Average score", xaxis_title="Grade", margin=dict(t=46, l=6, r=6, b=6))
            st.plotly_chart(fig_g, width="stretch", config=NO_TOOLBAR)
        with right:
            below = sbg[sbg["VS_SCHOOL"] < 0].sort_values("VS_SCHOOL")
            if len(below):
                grades_txt = ", ".join(f"Grade {int(g)}" for g in below["GRADE"].head(4))
                insight(f"In {sel_subject}, these grades sit below the school average and are worth a closer "
                        f"look first: <b>{grades_txt}</b>. Green bars are at or above the "
                        f"{sv['school_avg']:.1f}% dashed line; red bars are below it.", height=190)
            else:
                insight(f"Every grade is at or above the school average in {sel_subject} — no grade-level "
                        f"gap to flag here right now.", height=190)

        st.markdown(f"###### {sel_subject} — every sitting this year, up to {sel_term} ({sel_exam})")
        sit = subject_sittings_trend(scores, sel_year, sel_subject, upto_term=sel_term, upto_exam=sel_exam)
        if len(sit) > 1:
            fig_t = px.line(sit, x="LABEL", y="SCORE", markers=True)
            fig_t.update_traces(line_color=PURPLE, marker=dict(size=8, color=PURPLE), line_shape="spline")
            style_fig(fig_t, height=180)
            fig_t.update_layout(yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig_t, width="stretch", config=NO_TOOLBAR)
        else:
            st.caption("Only one sitting loaded this year — trend will populate as more exams are added.")

# ===========================================================================
# PAGE 3 — CLASSES: "Which classes need attention?"
# ===========================================================================
elif page == "Grade Analysis":
    sel_year, sel_term, sel_exam, terms_this_year = top_bar(
        "Grade Analysis", "Compare classes within the same grade.")
    term_scores = scores[scores["TERM"] == sel_term]
    prev_sitting_label = previous_sitting_label(scores, sel_term, sel_exam)

    grades = sorted(term_scores["GRADE"].unique())
    sel_grade = st.selectbox("Grade", grades)
    comp = class_comparison_within_grade(scores, sel_term, sel_exam, sel_grade, pass_mark)

    with st.container(border=True, key="panel_8"):
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"#### Grade {sel_grade} — average score by class")
            fig = px.bar(comp, x="CLASS", y="AVERAGE", text="AVERAGE", color="CLASS",
                          color_discrete_sequence=PASTEL)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            style_fig(fig, height=230)
            fig.update_layout(yaxis_title="Average score")
            st.plotly_chart(fig, width="stretch")
        with right:
            best_c, worst_c = comp.iloc[0], comp.iloc[-1]
            gap = best_c["AVERAGE"] - worst_c["AVERAGE"]
            insight(f"<b>{best_c['CLASS']}</b> leads Grade {sel_grade} at <b>{best_c['AVERAGE']:.1f}%</b>, "
                    f"while <b>{worst_c['CLASS']}</b> trails at <b>{worst_c['AVERAGE']:.1f}%</b> "
                    f"— a {gap:.1f}-pt gap. "
                    f"{'Worth investigating what is driving the difference between the two classes.' if gap > 5 else 'The classes are fairly close together.'}",
                    height=170)

    with st.container(border=True, key="panel_9"):
        st.markdown("#### Class summary")
        display = comp.rename(columns={
            "CLASS": "Class", "AVERAGE": "Average", "PASS_RATE": f"Pass Rate (≥{pass_mark})",
            "IMPROVEMENT_EXAM": "Improvement (vs prev. exam)", "IMPROVEMENT_TERM": "Improvement (term-to-term)",
        })[["Class", "Average", f"Pass Rate (≥{pass_mark})", "Improvement (vs prev. exam)", "Improvement (term-to-term)"]]
        st.dataframe(
            style_deltas(
                display.style.format({
                    "Average": "{:.1f}%", f"Pass Rate (≥{pass_mark})": "{:.1f}%",
                    "Improvement (vs prev. exam)": "{:+.1f}", "Improvement (term-to-term)": "{:+.1f}",
                }),
                ["Improvement (vs prev. exam)", "Improvement (term-to-term)"],
            ),
            hide_index=True, width="stretch",
        )
        st.caption(f"“Improvement (vs prev. exam)” compares to the previous sitting: **{prev_sitting_label}**.")

    with st.container(border=True, key="panel_10"):
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"#### Subject × class — Grade {sel_grade}")
            sub = scores[(scores["TERM"] == sel_term) & (scores["EXAM"] == sel_exam) & (scores["GRADE"] == sel_grade)]
            heat = sub.pivot_table(index="SUBJECT", columns="STREAM", values="SCORE", aggfunc="mean").round(1)
            fig_heat = px.imshow(
                heat, text_auto=".1f", aspect="auto",
                labels=dict(x="Stream", y="Subject", color="Avg"),
                color_continuous_scale=[[0, "#F3A6A6"], [0.5, "#F6CD7E"], [1, "#8FDCB2"]],
                zmin=max(0, pass_mark - 30), zmax=100,
            )
            style_fig(fig_heat, height=330)
            st.plotly_chart(fig_heat, width="stretch")
        with right:
            flat = heat.stack()
            if len(flat):
                worst_cell = flat.idxmin()
                best_cell = flat.idxmax()
                insight(f"The weakest combination is <b>{worst_cell[0]}</b> in stream <b>{worst_cell[1]}</b> "
                        f"at <b>{flat.min():.1f}%</b>, while <b>{best_cell[0]}</b> in <b>{best_cell[1]}</b> "
                        f"leads at <b>{flat.max():.1f}%</b>.", height=200)

    with st.container(border=True, key="panel_11"):
        st.markdown("#### Performance distribution")
        st.markdown('<p class="subtitle">Each student counted once, using their average score across all '
                     'subjects — not one count per subject entry.</p>', unsafe_allow_html=True)
        student_avg = sub.groupby(["CLASS", "NAME"], as_index=False)["SCORE"].mean()
        student_avg["BAND"] = pd.cut(student_avg["SCORE"], bins=[-1, 49, 74, 89, 100],
                                      labels=["Below 50%", "50–74%", "75–89%", "90% and above"])
        band_counts = student_avg.groupby(["CLASS", "BAND"], observed=True).size().reset_index(name="COUNT")
        band_order = ["Below 50%", "50–74%", "75–89%", "90% and above"]
        band_colors = {"Below 50%": "#EA7C7C", "50–74%": "#F2C97A", "75–89%": "#5B8DEF", "90% and above": "#57C7A6"}
        fig_dist = px.bar(band_counts, x="CLASS", y="COUNT", color="BAND",
                           category_orders={"BAND": band_order}, color_discrete_map=band_colors, barmode="stack")
        style_fig(fig_dist, height=280, legend=True)
        fig_dist.update_layout(yaxis_title="Students", legend_title=None)
        st.plotly_chart(fig_dist, width="stretch")

# ===========================================================================
# PAGE 4 — STUDENT SEARCH: report-card layout
# ===========================================================================
elif page == "Student Search":
    sel_year, sel_term, sel_exam, terms_this_year = top_bar(
        "Student Search", "Search a student to view their report card.")
    term_scores = scores[scores["TERM"] == sel_term]

    all_names = sorted(term_scores["NAME"].unique())
    sel_name = st.selectbox("Search for a student", all_names, index=None, placeholder="Type a name…")

    if not sel_name:
        st.info("Search a student above to view their report card.")
    else:
        rep = student_report(scores, subject_teachers, class_teachers, sel_term, sel_exam, sel_name, pass_mark)

        st.markdown(
            f"""
            <div class="report-header">
              <div class="report-avatar">🧑‍🎓</div>
              <div>
                <div style="font-size:1.35rem; font-weight:700;">{rep['name']}</div>
                <div style="font-size:0.85rem; opacity:0.85;">
                    Admission No: {rep['admission']} &nbsp;·&nbsp; Grade {rep['grade']} {rep['stream']}
                    &nbsp;·&nbsp; Class Teacher: {rep['class_teacher']} &nbsp;·&nbsp; {sel_term}, {sel_exam}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rank_color = GREEN if rep['rank'] and rep['class_size'] and rep['rank'] <= max(1, rep['class_size'] * 0.25) else BLUE
        exam_color, _ = delta_word(rep['delta_exam'])
        st.markdown(
            f"""
            <div class="chip-row">
              <div class="chip" style="border-top:3px solid {BLUE};">
                <div class="chip-label">Overall Average</div>
                <div class="chip-value">{fmt(rep['overall_avg'])}</div>
              </div>
              <div class="chip" style="border-top:3px solid {exam_color};">
                <div class="chip-label">vs Previous Exam</div>
                <div class="chip-value" style="color:{exam_color};">{fmt(rep['delta_exam'], suffix='', nd=1) if pd.notna(rep['delta_exam']) else '—'}</div>
              </div>
              <div class="chip" style="border-top:3px solid {rank_color};">
                <div class="chip-label">Class Rank</div>
                <div class="chip-value">{f"{rep['rank']} / {rep['class_size']}" if rep['rank'] else "—"}</div>
              </div>
              <div class="chip" style="border-top:3px solid {GREEN};">
                <div class="chip-label">Pass Rate (≥{pass_mark})</div>
                <div class="chip-value">{fmt(rep['pass_rate'])}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        overall_word = "Improving" if pd.notna(rep["delta_term"]) and rep["delta_term"] > 1 else (
            "Declining" if pd.notna(rep["delta_term"]) and rep["delta_term"] < -1 else "Stable")
        word_color = GREEN if overall_word == "Improving" else (RED if overall_word == "Declining" else AMBER)
        with st.container(border=True, key="panel_12"):
            st.markdown(
                f"""
                <b>Overall:</b> <span style="color:{word_color}; font-weight:700;">{overall_word}</span><br>
                <b>Strengths:</b> {", ".join(rep["strengths"]) if rep["strengths"] else "—"}<br>
                <b>Areas requiring attention:</b> {", ".join(rep["attention"]) if rep["attention"] else "—"}
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True, key="panel_14"):
            trend_col, rank_col = st.columns(2)
            with trend_col:
                st.markdown("###### Performance trend")
                st_trend = student_sittings_trend(scores, sel_year, sel_name, upto_term=sel_term, upto_exam=sel_exam)
                if len(st_trend) > 1:
                    fig = px.line(st_trend, x="LABEL", y="SCORE", markers=True)
                    fig.update_traces(line_color=BLUE, marker=dict(size=9, color=BLUE), line_shape="spline")
                    style_fig(fig, height=230)
                    fig.update_layout(yaxis_title="Avg score", xaxis_title=None)
                    st.plotly_chart(fig, width="stretch", config=NO_TOOLBAR)
                else:
                    st.caption("Only one sitting loaded so far — trend will populate once another exam or term is added.")
            with rank_col:
                st.markdown("###### Class rank trend")
                rank_trend = student_rank_sittings_trend(scores, sel_year, sel_name, upto_term=sel_term, upto_exam=sel_exam)
                if len(rank_trend) > 1:
                    fig_r = px.line(rank_trend, x="LABEL", y="RANK", markers=True)
                    fig_r.update_traces(line_color=PURPLE, marker=dict(size=9, color=PURPLE), line_shape="spline")
                    style_fig(fig_r, height=230)
                    fig_r.update_layout(yaxis_title="Rank", xaxis_title=None,
                                         yaxis=dict(autorange="reversed", dtick=1))
                    st.plotly_chart(fig_r, width="stretch", config=NO_TOOLBAR)
                else:
                    st.caption("Only one sitting loaded so far — trend will populate once another exam or term is added.")

        left, right = st.columns([3, 2])
        with left:
            with st.container(border=True, key="panel_13"):
                st.markdown("#### Subject performance")
                sdf = rep["subject_df"].sort_values("SCORE", ascending=False)

                # Was there a second sitting this same term? Show it translucent
                # behind the selected exam's bar so movement is visible at a glance.
                other_exam = None
                exams_in_term = order_exams(scores[(scores["TERM"] == sel_term) & (scores["NAME"] == sel_name)]["EXAM"].unique())
                others = [e for e in exams_in_term if e != sel_exam]
                if others:
                    other_exam = others[0]
                    prev_row = scores[(scores["TERM"] == sel_term) & (scores["NAME"] == sel_name) & (scores["EXAM"] == other_exam)]
                    prev_map = prev_row.set_index("SUBJECT")["SCORE"].to_dict()
                else:
                    prev_map = {}

                for _, row in sdf.iterrows():
                    color = GREEN if row["SCORE"] >= pass_mark else RED
                    if row["SCORE"] >= pass_mark and pd.notna(row["DELTA"]) and row["DELTA"] < -3:
                        color = AMBER
                    if other_exam:
                        dual_bar(row["SUBJECT"], row["SCORE"], prev_map.get(row["SUBJECT"]), color=color)
                    else:
                        thin_bar(row["SUBJECT"], row["SCORE"], color=color)

                if other_exam:
                    st.caption(f"Solid = {sel_exam}, faint = {other_exam} (same term).")

        with right:
            cluster_subject_list = "; ".join(f"{c}: {', '.join(subs)}" for c, subs in SUBJECT_CLUSTERS.items())
            with st.container(border=True, key="panel_15"):
                st.markdown("#### Subject-cluster strength")
                st.markdown(f'<p class="subtitle">Directional read only, not an official placement result. '
                             f'{cluster_subject_list}.</p>', unsafe_allow_html=True)
                categories = list(SUBJECT_CLUSTERS.keys())
                values = [rep["cluster"].set_index("CLUSTER")["SCORE"].get(c, 0) for c in categories]
                subj_lists = [", ".join(SUBJECT_CLUSTERS[c]) for c in categories]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]], theta=categories + [categories[0]],
                    fill="toself", line_color=BLUE, fillcolor="rgba(47,111,224,0.15)",
                    customdata=subj_lists + [subj_lists[0]],
                    hovertemplate="<b>%{theta}</b><br>%{customdata}<extra></extra>",
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                         height=280, showlegend=False, margin=dict(t=10, b=10),
                                         font=PLOT_FONT, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_radar, width="stretch", config=NO_TOOLBAR)

            low = sdf[sdf["SCORE"] < pass_mark]["SUBJECT"].tolist()
            declining = sdf[sdf["DELTA"] < -3]["SUBJECT"].tolist()
            improving = sdf[(sdf["SCORE"] < pass_mark + 15) & (sdf["DELTA"] > 3)]["SUBJECT"].tolist()
            if low or declining or improving:
                with st.container(border=True, key="panel_16"):
                    st.markdown("#### Flags")
                    if low:
                        st.markdown(f'<div class="attention-card attention-red">🔴 Below pass mark: {", ".join(low)}</div>', unsafe_allow_html=True)
                    if declining:
                        st.markdown(f'<div class="attention-card attention-amber">🟠 Declining: {", ".join(declining)}</div>', unsafe_allow_html=True)
                    if improving:
                        st.markdown(f'<div class="attention-card attention-amber">🟢 Low but improving: {", ".join(improving)}</div>', unsafe_allow_html=True)

        print_button_html = """
            <div style="display:flex; justify-content:flex-end;">
            <button onclick="window.parent.print()"
              style="background:#003f88; color:#fff; border:none; border-radius:8px; padding:8px 16px;
              font-size:0.9rem; font-weight:600; cursor:pointer; box-shadow:0 2px 6px rgba(0,63,136,0.3);
              font-family:'Poppins',sans-serif;">
              🖨️ Print Report
            </button>
            </div>
            """
        with st.container(key="no_print_btn"):
            # st.markdown's sanitizer strips inline event handlers like onclick, so the
            # button needs to live in its own document (iframe) where
            # window.parent.print() reaches back out to print the real page. Try the
            # non-deprecated st.iframe under a couple of possible kwarg names first;
            # only fall back to the deprecated components.html if neither is supported
            # by the installed Streamlit version.
            _rendered = False
            for kwargs in ({"srcdoc": print_button_html}, {"html": print_button_html}):
                try:
                    st.iframe(height=50, scrolling=False, **kwargs)
                    _rendered = True
                    break
                except TypeError:
                    continue
                except AttributeError:
                    break
            if not _rendered:
                import streamlit.components.v1 as components
                components.html(print_button_html, height=50)

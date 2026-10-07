"""
Data loading and transformation helpers for the Total Care Academy dashboard.

The base dataset ships in long format:
    NAME | TERM | EXAM | GRADE | STREAM | SUBJECT | SCORE | CLASS

As new terms are appended (same shape, new TERM label), the trend-oriented
functions here start returning real term-over-term series instead of the
within-term KNEC vs END-TERM proxy.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Demo build: reads the bundled synthetic dataset in demo/ only. Paths are
# resolved relative to this file so the app works from any working directory.
_HERE = Path(__file__).parent
BASE_SCORES_PATH = _HERE / "demo" / "scores_long_demo.csv"
SUBJECT_TEACHERS_PATH = _HERE / "demo" / "subject_teachers_demo.csv"
CLASS_TEACHERS_PATH = _HERE / "demo" / "class_teachers_demo.csv"

REQUIRED_SCORE_COLS = ["NAME", "TERM", "EXAM", "GRADE", "STREAM", "SUBJECT", "SCORE"]

# ---------------------------------------------------------------------------
# School structure — brackets reflect how the headteacher actually evaluates
# staff: never across brackets, and Junior is a different teaching model
# (one teacher, one class, all subjects) than Middle/Upper (subject
# specialists shared across classes).
# ---------------------------------------------------------------------------
BRACKETS = {
    "Junior (Grades 2–3)": [2, 3],
    "Middle (Grades 4–6)": [4, 5, 6],
    "Upper (Grades 7–9)": [7, 8, 9],
}


def grade_to_bracket(grade: int) -> str:
    for label, grades in BRACKETS.items():
        if grade in grades:
            return label
    return "Other"


def bracket_for_grades(grades) -> str:
    """Given a set of grades present in a filtered df, return the bracket label
    if they all fall in one bracket, else 'Mixed'."""
    labels = {grade_to_bracket(g) for g in grades}
    return labels.pop() if len(labels) == 1 else "Mixed"


# ---------------------------------------------------------------------------
# CBC subject clusters — a heuristic mapping of TCA's junior-school subjects
# onto the three CBC senior-school pathways (STEM / Social Sciences / Arts &
# Sports Science), used only to give a directional read on a student's
# subject-strength profile. This is not an official KICD pathway placement
# tool — real placement also weighs Grade 9 assessment results and learner
# choice.
# ---------------------------------------------------------------------------
SUBJECT_CLUSTERS = {
    "STEM": ["MATH", "INT. SCI.", "AGRI.", "PRE-TECH", "ENV."],
    "Social Sciences": ["SST", "CRE", "ENG.", "KIS."],
    "Arts & Sports": ["C/A"],
}

SUBJECT_TO_CLUSTER = {s: c for c, subs in SUBJECT_CLUSTERS.items() for s in subs}


@st.cache_data
def load_base_scores() -> pd.DataFrame:
    return _clean_scores(pd.read_csv(BASE_SCORES_PATH))


@st.cache_data
def load_subject_teachers() -> pd.DataFrame:
    return pd.read_csv(SUBJECT_TEACHERS_PATH)


@st.cache_data
def load_class_teachers() -> pd.DataFrame:
    return pd.read_csv(CLASS_TEACHERS_PATH)


def _clean_scores(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_SCORE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")
    df = df.copy()
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")
    df = df.dropna(subset=["SCORE", "NAME"])
    df["GRADE"] = df["GRADE"].astype(int)
    for c in ["NAME", "TERM", "EXAM", "STREAM", "SUBJECT"]:
        df[c] = df[c].astype(str).str.strip()
    if "CLASS" not in df.columns:
        df["CLASS"] = df["GRADE"].astype(str) + df["STREAM"]
    if "STUDENT_ID" not in df.columns:
        # Falls back to name-based identity until real IDs are available.
        # See README for why a persistent ID matters once students move
        # grades or names repeat.
        df["STUDENT_ID"] = df["NAME"]
    return df


def read_uploaded_scores(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    return _clean_scores(raw)


# ---------------------------------------------------------------------------
# School-wide (landing page) helpers
# ---------------------------------------------------------------------------
def headcount_by_grade(scores: pd.DataFrame, term: str) -> pd.DataFrame:
    sub = scores[scores["TERM"] == term]
    return (
        sub.groupby(["GRADE"])["STUDENT_ID"].nunique().reset_index(name="STUDENTS")
        .sort_values("GRADE")
    )


def headcount_by_class(scores: pd.DataFrame, term: str) -> pd.DataFrame:
    sub = scores[scores["TERM"] == term]
    return (
        sub.groupby(["GRADE", "STREAM", "CLASS"])["STUDENT_ID"].nunique()
        .reset_index(name="STUDENTS").sort_values(["GRADE", "STREAM"])
    )


def term_over_term_totals(scores: pd.DataFrame, exam: str) -> pd.DataFrame:
    """
    School-wide average total score per term, for the given exam type.
    With one term loaded this is a single point; it's built to extend
    automatically as more terms are appended.
    """
    totals = student_totals(scores)
    sub = totals[totals["EXAM"] == exam]
    return sub.groupby("TERM", as_index=False)["TOTAL"].mean().rename(columns={"TOTAL": "AVG_TOTAL"})


def student_totals(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per student x term x exam: total score across subjects."""
    return (
        scores.groupby(["STUDENT_ID", "NAME", "TERM", "EXAM", "GRADE", "STREAM", "CLASS"], as_index=False)
        .agg(TOTAL=("SCORE", "sum"), SUBJECTS_TAKEN=("SUBJECT", "nunique"))
    )


# ---------------------------------------------------------------------------
# Grade Explorer helpers
# ---------------------------------------------------------------------------
def subject_stream_heatmap(scores: pd.DataFrame, term: str, exam: str, grade: int) -> pd.DataFrame:
    """Subject (rows) x Stream (columns) average score, for one grade."""
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["GRADE"] == grade)]
    pivot = sub.pivot_table(index="SUBJECT", columns="STREAM", values="SCORE", aggfunc="mean")
    return pivot.round(1)


def subject_averages_for_grade(scores: pd.DataFrame, term: str, exam: str, grade: int, stream: str = None) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["GRADE"] == grade)]
    if stream and stream != "All streams":
        sub = sub[sub["STREAM"] == stream]
    return sub.groupby("SUBJECT", as_index=False)["SCORE"].mean().sort_values("SCORE", ascending=False)


def top_bottom_students(scores: pd.DataFrame, term: str, exam: str, grade: int, stream: str, n: int = 5):
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["GRADE"] == grade)]
    if stream and stream != "All streams":
        sub = sub[sub["STREAM"] == stream]
    totals = student_totals(sub)
    totals = totals[totals["EXAM"] == exam].sort_values("TOTAL", ascending=False)
    top = totals.head(n)[["NAME", "CLASS", "TOTAL"]]
    bottom = totals.tail(n)[["NAME", "CLASS", "TOTAL"]].sort_values("TOTAL")
    return top, bottom


def class_subject_teachers(subject_teachers: pd.DataFrame, grade: int, stream: str, term: str = None) -> pd.DataFrame:
    sub = subject_teachers[(subject_teachers["GRADE"] == grade) & (subject_teachers["STREAM"] == stream)]
    if term is not None and "TERM" in sub.columns:
        sub = sub[sub["TERM"] == term]
    return sub[["SUBJECT", "TEACHER"]].sort_values("SUBJECT").reset_index(drop=True)


def risk_students(scores: pd.DataFrame, term: str, exam: str, pass_mark: int, min_subjects_failing: int,
                   grade: int = None) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)].copy()
    if grade is not None:
        sub = sub[sub["GRADE"] == grade]
    sub["BELOW"] = sub["SCORE"] < pass_mark
    agg = sub.groupby(["NAME", "GRADE", "STREAM", "CLASS"], as_index=False).agg(
        SUBJECTS_BELOW=("BELOW", "sum"),
        SUBJECTS_TAKEN=("SUBJECT", "nunique"),
        AVG_SCORE=("SCORE", "mean"),
    )
    flagged = agg[agg["SUBJECTS_BELOW"] >= min_subjects_failing].sort_values(
        ["SUBJECTS_BELOW", "AVG_SCORE"], ascending=[False, True]
    )
    return flagged


# ---------------------------------------------------------------------------
# Student Profile helpers
# ---------------------------------------------------------------------------
def student_exam_table(scores: pd.DataFrame, term: str, name: str) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["NAME"] == name)]
    pivot = sub.pivot_table(index="SUBJECT", columns="EXAM", values="SCORE", aggfunc="mean")
    for col in ["KNEC", "END-TERM"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["MOVEMENT"] = pivot["END-TERM"] - pivot["KNEC"]
    return pivot.reset_index()


def student_movement_long(scores: pd.DataFrame, term: str, name: str) -> pd.DataFrame:
    """Long-format exam-by-subject series, ready to plot as a line/slope chart.
    Structured with an ordered EXAM_SEQ so it extends cleanly once more terms
    (and therefore more exam sittings) exist for this student."""
    sub = scores[(scores["TERM"] == term) & (scores["NAME"] == name)].copy()
    exam_order = {"KNEC": 0, "END-TERM": 1}
    sub["EXAM_SEQ"] = sub["EXAM"].map(exam_order).fillna(99)
    return sub.sort_values(["SUBJECT", "EXAM_SEQ"])


def student_cluster_strength(scores: pd.DataFrame, term: str, exam: str, name: str) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["NAME"] == name)].copy()
    sub["CLUSTER"] = sub["SUBJECT"].map(SUBJECT_TO_CLUSTER).fillna("Other")
    return sub.groupby("CLUSTER", as_index=False)["SCORE"].mean()


def student_rank_in_class(scores: pd.DataFrame, term: str, exam: str, name: str):
    row = scores[(scores["TERM"] == term) & (scores["NAME"] == name)]
    if not len(row):
        return None, None
    grade, stream = row["GRADE"].iloc[0], row["STREAM"].iloc[0]
    class_scores = scores[(scores["TERM"] == term) & (scores["GRADE"] == grade) & (scores["STREAM"] == stream)]
    totals = student_totals(class_scores)
    totals = totals[totals["EXAM"] == exam].sort_values("TOTAL", ascending=False).reset_index(drop=True)
    rank_row = totals[totals["NAME"] == name]
    if len(rank_row):
        return rank_row.index[0] + 1, len(totals)
    return None, len(totals)


def student_rank_trend(scores: pd.DataFrame, name: str, exam: str, terms_ordered: list) -> pd.DataFrame:
    """Class rank across a sequence of terms, for the same exam sitting type
    (e.g. End-Term each term) — the rank equivalent of the score trend."""
    rows = []
    for t in terms_ordered:
        rank, size = student_rank_in_class(scores, t, exam, name)
        if rank is not None:
            rows.append({"TERM": t, "RANK": rank, "SIZE": size})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Teacher Performance helpers — bracket-aware
# ---------------------------------------------------------------------------
def teacher_attribution(scores: pd.DataFrame, subject_teachers: pd.DataFrame, term: str, exam: str,
                         grades: list) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["GRADE"].isin(grades))]
    if "TERM" in subject_teachers.columns:
        # Term-aware mapping (e.g. the demo dataset, which models real
        # reassignment): join on TERM too so each term sees its own roster.
        st_term = subject_teachers[subject_teachers["TERM"] == term]
        merged = sub.merge(st_term, on=["GRADE", "STREAM", "SUBJECT"], how="left", suffixes=("", "_map"))
    else:
        # Real production mapping today: a single static assignment, applied
        # to whichever term is selected.
        merged = sub.merge(subject_teachers, on=["GRADE", "STREAM", "SUBJECT"], how="left")
    merged["TEACHER"] = merged["TEACHER"].fillna("UNASSIGNED")
    return merged


def junior_teacher_summary(attributed: pd.DataFrame) -> pd.DataFrame:
    """Junior bracket: one teacher = one class, all subjects. Fair comparison
    is the raw class average — every junior teacher covers the identical
    subject set, so no normalization is needed."""
    summary = (
        attributed.groupby("TEACHER", as_index=False)
        .agg(AVG_SCORE=("SCORE", "mean"), TOTAL_STUDENTS=("NAME", "nunique"), WORKLOAD=("CLASS", "nunique"))
        .sort_values("AVG_SCORE", ascending=False)
    )
    return summary


# ---------------------------------------------------------------------------
# v4 redesign helpers — Overview / Classes / Subjects / Teachers / Student
# Report. Builds on the tables above; nothing here changes the underlying
# data model.
# ---------------------------------------------------------------------------
import re


def term_year(term_label: str) -> str:
    """Best-effort academic year out of a TERM label like 'Term 2, 2025'."""
    m = re.search(r"(20\d{2})", str(term_label))
    return m.group(1) if m else "—"


def term_number(term_label: str) -> int:
    m = re.search(r"Term\s*(\d+)", str(term_label))
    return int(m.group(1)) if m else 99


def years_available(scores: pd.DataFrame) -> list:
    years = sorted({term_year(t) for t in scores["TERM"].unique()})
    return years


def terms_for_year(scores: pd.DataFrame, year: str) -> list:
    terms = [t for t in scores["TERM"].unique() if term_year(t) == year]
    return sorted(terms, key=term_number)


def previous_term(scores: pd.DataFrame, current_term: str):
    """Same-year term immediately before current_term, ordered by term number.
    Returns None if current_term is the first term of its year."""
    yr = term_year(current_term)
    ordered = terms_for_year(scores, yr)
    if current_term not in ordered:
        return None
    idx = ordered.index(current_term)
    return ordered[idx - 1] if idx > 0 else None


def school_pass_rate(df: pd.DataFrame, pass_mark: int) -> float:
    if not len(df):
        return float("nan")
    return (df["SCORE"] >= pass_mark).mean() * 100


def overall_average(df: pd.DataFrame) -> float:
    return df["SCORE"].mean() if len(df) else float("nan")


def within_term_trend(scores: pd.DataFrame, term: str) -> pd.DataFrame:
    """Average score, Mid-Term (KNEC) -> End-Term, for one term."""
    sub = scores[scores["TERM"] == term]
    order = {"KNEC": "Mid-Term", "END-TERM": "End-Term"}
    g = sub.groupby("EXAM", as_index=False)["SCORE"].mean()
    g["STAGE"] = g["EXAM"].map(order).fillna(g["EXAM"])
    g["SEQ"] = g["EXAM"].map({"KNEC": 0, "END-TERM": 1}).fillna(9)
    return g.sort_values("SEQ")


def term_to_term_trend(scores: pd.DataFrame, year: str) -> pd.DataFrame:
    """Average score per term across a school year, using each term's own
    latest sitting (whichever exam ranks highest via exam_rank() for that
    term) — determined dynamically per term rather than filtering on one
    hardcoded exam label, since real schools may not call it 'END-TERM'."""
    terms = terms_for_year(scores, year)
    rows = []
    for t in terms:
        exams_t = order_exams(scores[scores["TERM"] == t]["EXAM"].unique())
        if not len(exams_t):
            continue
        latest_exam = exams_t[-1]
        avg = scores[(scores["TERM"] == t) & (scores["EXAM"] == latest_exam)]["SCORE"].mean()
        rows.append({"TERM": t, "SCORE": avg, "TERM_NO": term_number(t)})
    return pd.DataFrame(rows).sort_values("TERM_NO")


def improvement_within_term(scores: pd.DataFrame, term: str) -> float:
    t = within_term_trend(scores, term)
    if len(t) < 2:
        return float("nan")
    return t["SCORE"].iloc[-1] - t["SCORE"].iloc[0]


def exam_rank(exam_label: str) -> int:
    """Classify any exam label into a chronological rank within a term,
    Mid-Term-like sittings before End-Term-like ones. Does NOT rely on exact
    string matches (e.g. the demo data's 'KNEC'/'END-TERM'), since real
    school data may label these 'MID-TERM', 'MID-1', 'OPENER', 'FINAL',
    'END-1', etc. Matches by keyword, case-insensitively, so it keeps
    working whatever the exact label text is:
      - 'KNEC' (the demo's mid-term code) or anything containing MID/OPEN -> 0
      - anything containing END/FINAL/CLOSING -> 1
      - anything else -> 50 (sorts after known stages, alphabetically among
        ties, so unrecognised exam names never crash or silently vanish)
    """
    s = str(exam_label).upper()
    if s == "KNEC" or "MID" in s or "OPEN" in s:
        return 0
    if "END" in s or "FINAL" in s or "CLOS" in s:
        return 1
    return 50


def exam_display_stage(exam_label: str) -> str:
    """Friendly stage word for an exam label, used only for the two demo
    codes; any real label is shown exactly as stored, since we can't safely
    guess a nicer name for e.g. 'MID-TERM' beyond what it already is."""
    if exam_label == "KNEC":
        return "Mid-Term"
    if exam_label == "END-TERM":
        return "End-Term"
    return str(exam_label)


def order_exams(exams) -> list:
    """Sort a list/array of EXAM values into true chronological order
    (Mid-Term-like before End-Term-like), never alphabetically — plain
    alphabetical sort puts 'END-TERM' before 'KNEC' (E < K), which silently
    made Mid-Term look like the most recent exam whenever both existed for
    a term. Uses exam_rank()'s keyword classification, so it works for real
    exam labels too, not just the demo data's exact 'KNEC'/'END-TERM'."""
    return sorted(exams, key=lambda e: (exam_rank(e), str(e)))


def exam_sittings_ordered(scores: pd.DataFrame) -> list:
    """Every (TERM, EXAM) sitting that exists in the data, in true
    chronological order (year, then term, then Mid-Term before End-Term).
    This is a global sequence — it lets 'previous exam' correctly roll back
    across a term boundary (e.g. Term 2 Mid-Term's previous sitting is
    Term 1 End-Term)."""
    df = scores[["TERM", "EXAM"]].drop_duplicates().copy()
    df["YEAR"] = df["TERM"].apply(term_year)
    df["TERM_NO"] = df["TERM"].apply(term_number)
    df["EXAM_RANK"] = df["EXAM"].apply(exam_rank)
    df = df.sort_values(["YEAR", "TERM_NO", "EXAM_RANK"])
    return list(df[["TERM", "EXAM"]].itertuples(index=False, name=None))


def previous_sitting(scores: pd.DataFrame, term: str, exam: str):
    """The exam sitting immediately before (term, exam) in true chronological
    order. Returns None only for the very first sitting on record (e.g. the
    school's first-ever Term 1 Mid-Term)."""
    seq = exam_sittings_ordered(scores)
    try:
        idx = seq.index((term, exam))
    except ValueError:
        return None
    return seq[idx - 1] if idx > 0 else None


def improvement_vs_previous_exam(scores: pd.DataFrame, term: str, exam: str, scope_df: pd.DataFrame = None) -> float:
    """Movement from the immediately preceding exam sitting (chronologically
    — crosses term boundaries when needed) into the selected one. Pass
    scope_df (a filtered slice of `scores`, e.g. one class/subject/student)
    to scope the comparison; `scores` itself is always used to work out
    sequence/ordering."""
    prev = previous_sitting(scores, term, exam)
    if prev is None:
        return float("nan")
    prev_term, prev_exam = prev
    base = scope_df if scope_df is not None else scores
    now = base[(base["TERM"] == term) & (base["EXAM"] == exam)]["SCORE"].mean()
    was = base[(base["TERM"] == prev_term) & (base["EXAM"] == prev_exam)]["SCORE"].mean()
    return now - was


def previous_sitting_label(scores: pd.DataFrame, term: str, exam: str) -> str:
    prev = previous_sitting(scores, term, exam)
    if prev is None:
        return "—"
    p_term, p_exam = prev
    return f"{exam_display_stage(p_exam)}, {p_term}"


def sitting_short_label(term: str, exam: str) -> str:
    """Compact axis label for one sitting, e.g. 'T3 Mid' / 'T3 End'. Falls
    back to the first token of the raw label (e.g. 'MID-1' -> 'Mid',
    'FINAL' -> 'Final') for real exam names we don't otherwise recognise,
    rather than forcing an exact 'Mid-Term'/'End-Term' match."""
    stage = exam_display_stage(exam)
    if stage == "Mid-Term":
        stage = "Mid"
    elif stage == "End-Term":
        stage = "End"
    else:
        stage = str(exam).replace("_", "-").split("-")[0].split()[0].title()
    return f"T{term_number(term)} {stage}"


def previous_vs_current_trend(scores: pd.DataFrame, term: str, exam: str) -> pd.DataFrame:
    """Two-point trend: the sitting immediately before the current selection
    (left) -> the currently selected sitting (right) — the exact same pair
    the 'vs Previous Exam' KPI compares, so the chart and the KPI always
    agree. Crosses term boundaries correctly (e.g. selecting Term 3 Mid-Term
    pairs it with Term 2 End-Term, not a nonexistent Term 3 End-Term)."""
    rows = []
    prev = previous_sitting(scores, term, exam)
    if prev is not None:
        p_term, p_exam = prev
        p_avg = scores[(scores["TERM"] == p_term) & (scores["EXAM"] == p_exam)]["SCORE"].mean()
        rows.append({"STAGE": sitting_short_label(p_term, p_exam), "SCORE": p_avg})
    cur_avg = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]["SCORE"].mean()
    rows.append({"STAGE": sitting_short_label(term, exam), "SCORE": cur_avg})
    return pd.DataFrame(rows)


def improvement_vs_previous_term(scores: pd.DataFrame, term: str, exam: str) -> float:
    prev = previous_term(scores, term)
    if prev is None:
        return float("nan")
    now = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]["SCORE"].mean()
    was = scores[(scores["TERM"] == prev) & (scores["EXAM"] == exam)]["SCORE"].mean()
    return now - was


def performance_by_grade(scores: pd.DataFrame, term: str, exam: str) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]
    return sub.groupby("GRADE", as_index=False)["SCORE"].mean().sort_values("GRADE")


def performance_by_subject(scores: pd.DataFrame, term: str, exam: str) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]
    return sub.groupby("SUBJECT", as_index=False)["SCORE"].mean().sort_values("SCORE", ascending=False)


def grade_pass_rates(scores: pd.DataFrame, term: str, exam: str, pass_mark: int) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]
    out = sub.assign(_PASS=sub["SCORE"] >= pass_mark).groupby("GRADE")["_PASS"].mean().reset_index()
    out.columns = ["GRADE", "PASS_RATE"]
    out["PASS_RATE"] *= 100
    return out.sort_values("GRADE")


def subject_vs_school_average(scores: pd.DataFrame, term: str, exam: str, subject: str) -> dict:
    """How one subject compares to the whole-school average for the same
    sitting, plus which grades sit below the school average / pass mark for
    this subject specifically."""
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]
    school_avg = sub["SCORE"].mean()
    subj_df = sub[sub["SUBJECT"] == subject]
    subj_avg = subj_df["SCORE"].mean()
    by_grade = subj_df.groupby("GRADE", as_index=False)["SCORE"].mean().sort_values("GRADE")
    by_grade["VS_SCHOOL"] = by_grade["SCORE"] - school_avg
    return {"school_avg": school_avg, "subject_avg": subj_avg, "by_grade": by_grade}


def areas_requiring_attention(scores: pd.DataFrame, term: str, exam: str, pass_mark: int,
                               decline_threshold: float = 4.0, low_pass_rate_threshold: float = 60.0,
                               class_gap_threshold: float = 6.0) -> list:
    """Returns a list of dicts: {level: 'red'/'amber', text: str} surfacing
    where the headteacher should look next — not a full statistics dump.
    Compares against BOTH the pass-mark threshold and the previous exam
    sitting (chronological — crosses term boundaries), and names the
    specific grades/subjects involved rather than only aggregate figures."""
    alerts = []
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]
    prev = previous_sitting(scores, term, exam)
    prev_label = previous_sitting_label(scores, term, exam)

    # 1. Grades below the pass-mark threshold
    gpr = grade_pass_rates(scores, term, exam, pass_mark)
    low_grades = gpr[gpr["PASS_RATE"] < low_pass_rate_threshold]
    for _, row in low_grades.iterrows():
        alerts.append({"level": "red" if row["PASS_RATE"] < 45 else "amber",
                        "text": f"Grade {int(row['GRADE'])} pass rate is {row['PASS_RATE']:.0f}% — below the {low_pass_rate_threshold:.0f}% watch level."})

    # 2. Grade-level declines vs the previous exam sitting
    if prev is not None:
        now_g = performance_by_grade(scores, term, exam).set_index("GRADE")["SCORE"]
        was_g = performance_by_grade(scores, prev[0], prev[1]).set_index("GRADE")["SCORE"]
        for grade in now_g.index:
            if grade in was_g.index:
                delta = now_g[grade] - was_g[grade]
                if delta <= -decline_threshold:
                    alerts.append({"level": "red" if delta <= -6 else "amber",
                                    "text": f"Grade {grade} average dropped {abs(delta):.1f} pts vs the previous sitting ({prev_label}), now {now_g[grade]:.1f}%."})

    # 3. Subjects with low pass rate
    subj_pass = sub.assign(_PASS=sub["SCORE"] >= pass_mark).groupby("SUBJECT")["_PASS"].mean() * 100
    for subject, rate in subj_pass.items():
        if rate < low_pass_rate_threshold:
            alerts.append({"level": "red" if rate < 45 else "amber",
                            "text": f"{subject} pass rate is {rate:.0f}% — below the {low_pass_rate_threshold:.0f}% watch level."})

    # 4. Subject declines vs the previous exam sitting
    if prev is not None:
        now_s = performance_by_subject(scores, term, exam).set_index("SUBJECT")["SCORE"]
        was_s = performance_by_subject(scores, prev[0], prev[1]).set_index("SUBJECT")["SCORE"]
        for subject in now_s.index:
            if subject in was_s.index:
                delta = now_s[subject] - was_s[subject]
                if delta <= -decline_threshold:
                    alerts.append({"level": "red" if delta <= -8 else "amber",
                                    "text": f"{subject} dropped {abs(delta):.1f} pts vs the previous sitting ({prev_label}), now {now_s[subject]:.1f}% — even if the current average still looks acceptable."})

    # 5. Which specific grades are driving a subject's low pass rate
    for subject, rate in subj_pass.items():
        if rate < low_pass_rate_threshold:
            sdf = sub[sub["SUBJECT"] == subject]
            g_rates = sdf.assign(_PASS=sdf["SCORE"] >= pass_mark).groupby("GRADE")["_PASS"].mean() * 100
            worst = g_rates[g_rates < low_pass_rate_threshold].sort_values()
            if len(worst):
                grades_txt = ", ".join(f"Grade {g}" for g in worst.index[:4])
                alerts.append({"level": "amber",
                                "text": f"Within {subject}, the weakest pass rates are in {grades_txt}."})

    # 6. Classes substantially below their grade baseline
    class_avg = sub.groupby(["GRADE", "STREAM", "CLASS"], as_index=False)["SCORE"].mean()
    grade_avg = sub.groupby("GRADE")["SCORE"].mean()
    for _, row in class_avg.iterrows():
        baseline = grade_avg.get(row["GRADE"])
        if baseline is not None and (baseline - row["SCORE"]) >= class_gap_threshold:
            alerts.append({"level": "amber",
                            "text": f"Class {row['CLASS']} is {baseline - row['SCORE']:.1f} pts below the Grade {row['GRADE']} average."})

    order = {"red": 0, "amber": 1}
    alerts.sort(key=lambda a: order.get(a["level"], 2))
    # de-duplicate identical text just in case a rule fires twice
    seen, deduped = set(), []
    for a in alerts:
        if a["text"] not in seen:
            seen.add(a["text"])
            deduped.append(a)
    return deduped


def class_comparison_within_grade(scores: pd.DataFrame, term: str, exam: str, grade: int, pass_mark: int) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["GRADE"] == grade)]
    rows = []
    for cls, cdf in sub.groupby("CLASS"):
        avg = cdf["SCORE"].mean()
        pr = school_pass_rate(cdf, pass_mark)
        stream = cdf["STREAM"].iloc[0]
        prev_exam_avg = improvement_vs_previous_exam(scores, term, exam, scope_df=scores[scores["CLASS"] == cls])
        prev_term_avg = improvement_vs_previous_term(scores[scores["CLASS"] == cls], term, exam)
        rows.append({
            "CLASS": cls, "STREAM": stream, "AVERAGE": avg, "PASS_RATE": pr,
            "IMPROVEMENT_EXAM": prev_exam_avg, "IMPROVEMENT_TERM": prev_term_avg,
        })
    out = pd.DataFrame(rows).sort_values("AVERAGE", ascending=False)
    return out


def subject_ranking(scores: pd.DataFrame, term: str, exam: str, pass_mark: int, grade=None) -> pd.DataFrame:
    """Subject averages/pass-rate/improvement. When grade is None this is the
    whole-school view; when set, every figure is scoped to that grade while
    'VS_SCHOOL' still compares against the whole-school average, so you can
    see how one grade's subjects stack up against the school overall."""
    prev_term_t = previous_term(scores, term)
    all_sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam)]
    school_avg = all_sub["SCORE"].mean()
    sub = all_sub if grade is None else all_sub[all_sub["GRADE"] == grade]
    rows = []
    for subject, sdf in sub.groupby("SUBJECT"):
        avg = sdf["SCORE"].mean()
        pr = school_pass_rate(sdf, pass_mark)
        delta_term = float("nan")
        if prev_term_t:
            prev_base = scores[(scores["TERM"] == prev_term_t) & (scores["EXAM"] == exam) & (scores["SUBJECT"] == subject)]
            if grade is not None:
                prev_base = prev_base[prev_base["GRADE"] == grade]
            was = prev_base["SCORE"].mean()
            delta_term = avg - was
        scope_df = scores[scores["SUBJECT"] == subject]
        if grade is not None:
            scope_df = scope_df[scope_df["GRADE"] == grade]
        delta_exam = improvement_vs_previous_exam(scores, term, exam, scope_df=scope_df)
        rows.append({"SUBJECT": subject, "AVERAGE": avg, "PASS_RATE": pr, "IMPROVEMENT_TERM": delta_term,
                      "IMPROVEMENT_EXAM": delta_exam, "VS_SCHOOL": avg - school_avg})
    return pd.DataFrame(rows).sort_values("AVERAGE", ascending=False)


def subject_by_grade(scores: pd.DataFrame, term: str, exam: str, subject: str) -> pd.DataFrame:
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["SUBJECT"] == subject)]
    return sub.groupby("GRADE", as_index=False)["SCORE"].mean().sort_values("GRADE")


def subject_sittings_trend(scores: pd.DataFrame, year: str, subject: str, upto_term: str = None, upto_exam: str = None) -> pd.DataFrame:
    """One point per exam SITTING (Term 1 Mid-Term, Term 1 End-Term, Term 2
    Mid-Term, ...) for the given subject across the school year, rather than
    one point per term — a more granular trend. Truncated at
    (upto_term, upto_exam) when that sitting exists in the sequence, so the
    chart never shows sittings beyond what's currently selected."""
    seq = [(t, e) for (t, e) in exam_sittings_ordered(scores) if term_year(t) == year]
    if upto_term is not None and upto_exam is not None and (upto_term, upto_exam) in seq:
        seq = seq[:seq.index((upto_term, upto_exam)) + 1]
    rows = []
    for t, e in seq:
        sdf = scores[(scores["TERM"] == t) & (scores["EXAM"] == e) & (scores["SUBJECT"] == subject)]
        if len(sdf):
            rows.append({"LABEL": sitting_short_label(t, e), "SCORE": sdf["SCORE"].mean()})
    return pd.DataFrame(rows)


def student_sittings_trend(scores: pd.DataFrame, year: str, name: str, upto_term: str = None, upto_exam: str = None) -> pd.DataFrame:
    """One point per exam SITTING (not per term) for one student's overall
    average across all their subjects. Unlike a term-level trend, this
    populates usefully even within a single term once both a Mid-Term and
    an End-Term sitting exist for it."""
    seq = [(t, e) for (t, e) in exam_sittings_ordered(scores) if term_year(t) == year]
    if upto_term is not None and upto_exam is not None and (upto_term, upto_exam) in seq:
        seq = seq[:seq.index((upto_term, upto_exam)) + 1]
    rows = []
    for t, e in seq:
        sdf = scores[(scores["TERM"] == t) & (scores["EXAM"] == e) & (scores["NAME"] == name)]
        if len(sdf):
            rows.append({"LABEL": sitting_short_label(t, e), "SCORE": sdf["SCORE"].mean()})
    return pd.DataFrame(rows)


def student_rank_sittings_trend(scores: pd.DataFrame, year: str, name: str, upto_term: str = None, upto_exam: str = None) -> pd.DataFrame:
    """Class rank per exam SITTING (not per term) for one student — the rank
    equivalent of student_sittings_trend, with the same within-term benefit."""
    seq = [(t, e) for (t, e) in exam_sittings_ordered(scores) if term_year(t) == year]
    if upto_term is not None and upto_exam is not None and (upto_term, upto_exam) in seq:
        seq = seq[:seq.index((upto_term, upto_exam)) + 1]
    rows = []
    for t, e in seq:
        rank, size = student_rank_in_class(scores, t, e, name)
        if rank is not None:
            rows.append({"LABEL": sitting_short_label(t, e), "RANK": rank, "SIZE": size})
    return pd.DataFrame(rows)


def teacher_context_comparison(scores: pd.DataFrame, subject_teachers: pd.DataFrame, term: str, exam: str,
                                grade: int, subject: str) -> pd.DataFrame:
    """Only the teachers/classes that actually exist for Grade -> Subject -> Term -> Exam.
    Shows whether a performance gap tracks the teacher or the class itself."""
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) &
                 (scores["GRADE"] == grade) & (scores["SUBJECT"] == subject)]
    if not len(sub):
        return pd.DataFrame(columns=["TEACHER", "CLASS", "SUBJECT_AVG", "CLASS_AVG", "DIFFERENCE", "PASS_RATE"])

    if "TERM" in subject_teachers.columns:
        st_map = subject_teachers[subject_teachers["TERM"] == term]
    else:
        st_map = subject_teachers
    st_map = st_map[(st_map["GRADE"] == grade) & (st_map["SUBJECT"] == subject)][["STREAM", "TEACHER"]]

    class_scores_all = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) & (scores["GRADE"] == grade)]
    class_avg_all = class_scores_all.groupby("STREAM")["SCORE"].mean()
    subject_avg_overall = sub["SCORE"].mean()

    rows = []
    for stream, sdf in sub.groupby("STREAM"):
        teacher_row = st_map[st_map["STREAM"] == stream]
        teacher = teacher_row["TEACHER"].iloc[0] if len(teacher_row) else "UNASSIGNED"
        subj_avg = sdf["SCORE"].mean()
        class_avg = class_avg_all.get(stream, float("nan"))
        rows.append({
            "TEACHER": teacher, "CLASS": f"{grade}{stream}",
            "SUBJECT_AVG": subj_avg, "CLASS_AVG": class_avg,
            "DIFFERENCE": subj_avg - class_avg if pd.notna(class_avg) else float("nan"),
            "PASS_RATE": school_pass_rate(sdf, 50),
        })
    out = pd.DataFrame(rows).sort_values("SUBJECT_AVG", ascending=False)
    return out


def subject_score_distribution(scores: pd.DataFrame, term: str, exam: str, grade: int, subject: str) -> pd.DataFrame:
    """Score-band distribution per class, for one grade+subject — shows
    whether a low average is a class-wide/subject-wide pattern rather than
    one teacher's classes underperforming in isolation."""
    sub = scores[(scores["TERM"] == term) & (scores["EXAM"] == exam) &
                 (scores["GRADE"] == grade) & (scores["SUBJECT"] == subject)].copy()
    if not len(sub):
        return pd.DataFrame(columns=["CLASS", "BAND", "COUNT"])
    sub["BAND"] = pd.cut(sub["SCORE"], bins=[-1, 49, 74, 89, 100],
                          labels=["Below 50%", "50–74%", "75–89%", "90% and above"])
    return sub.groupby(["CLASS", "BAND"], observed=True).size().reset_index(name="COUNT")


def teacher_workload_table(subject_teachers: pd.DataFrame, term: str) -> pd.DataFrame:
    """Class-subject combinations assigned to each teacher — an assignment
    footprint, not literal timetable hours."""
    if "TERM" in subject_teachers.columns:
        sub = subject_teachers[subject_teachers["TERM"] == term]
    else:
        sub = subject_teachers
    sub = sub.copy()
    sub["CLASS_SUBJECT"] = sub["GRADE"].astype(str) + sub["STREAM"] + " " + sub["SUBJECT"]
    return (
        sub.groupby("TEACHER", as_index=False)
        .agg(ASSIGNMENTS=("CLASS_SUBJECT", "nunique"))
        .sort_values("ASSIGNMENTS", ascending=False)
    )


def student_report(scores: pd.DataFrame, subject_teachers: pd.DataFrame, class_teachers: pd.DataFrame,
                    term: str, exam: str, name: str, pass_mark: int) -> dict:
    prev_term_t = previous_term(scores, term)
    row = scores[(scores["TERM"] == term) & (scores["NAME"] == name)]
    if not len(row):
        return {}
    grade, stream = int(row["GRADE"].iloc[0]), row["STREAM"].iloc[0]
    cls = row["CLASS"].iloc[0]
    admission = row["STUDENT_ID"].iloc[0]

    ct = class_teachers.copy()
    _norm = lambda s: str(s).strip().upper()
    match = ct[(ct["GRADE"].apply(_norm) == _norm(grade)) & (ct["STREAM"].apply(_norm) == _norm(stream))]
    class_teacher = "—"
    if len(match):
        if "TERM" in match.columns:
            exact = match[match["TERM"].apply(_norm) == _norm(term)]
            if len(exact):
                class_teacher = exact["TEACHER"].iloc[0]
            else:
                # No row for this exact term yet (e.g. only score data has been
                # added for a new term so far) — fall back to the most
                # recently known assignment for this grade/stream rather than
                # showing a blank dash.
                class_teacher = match["TEACHER"].iloc[-1]
        else:
            class_teacher = match["TEACHER"].iloc[0]

    exam_row = row[row["EXAM"] == exam]
    overall_avg = exam_row["SCORE"].mean()
    pass_rate = school_pass_rate(exam_row, pass_mark)
    rank, class_size = student_rank_in_class(scores, term, exam, name)

    delta_exam = improvement_vs_previous_exam(scores, term, exam, scope_df=scores[scores["NAME"] == name])
    delta_term = float("nan")
    subj_prev = {}
    if prev_term_t:
        prev_row = scores[(scores["TERM"] == prev_term_t) & (scores["NAME"] == name) & (scores["EXAM"] == exam)]
        if len(prev_row):
            delta_term = overall_avg - prev_row["SCORE"].mean()
            subj_prev = prev_row.set_index("SUBJECT")["SCORE"].to_dict()

    subj_now = exam_row.set_index("SUBJECT")["SCORE"].to_dict()
    subject_rows = []
    for subj, score in subj_now.items():
        prev_score = subj_prev.get(subj)
        delta = (score - prev_score) if prev_score is not None else float("nan")
        subject_rows.append({"SUBJECT": subj, "SCORE": score, "PREV": prev_score, "DELTA": delta})
    subject_df = pd.DataFrame(subject_rows).sort_values("SCORE", ascending=False)

    strengths = subject_df[subject_df["SCORE"] >= subject_df["SCORE"].median()]["SUBJECT"].head(3).tolist()
    attention = subject_df[subject_df["SCORE"] < pass_mark]["SUBJECT"].tolist()
    if not attention:
        attention = subject_df.sort_values("SCORE").head(2)["SUBJECT"].tolist()

    cluster = student_cluster_strength(scores, term, exam, name)

    return {
        "name": name, "admission": admission, "grade": grade, "stream": stream, "class": cls,
        "class_teacher": class_teacher, "overall_avg": overall_avg, "pass_rate": pass_rate,
        "rank": rank, "class_size": class_size, "delta_exam": delta_exam, "delta_term": delta_term,
        "subject_df": subject_df, "strengths": strengths, "attention": attention,
        "cluster": cluster, "prev_term": prev_term_t,
    }


def specialist_teacher_summary(attributed: pd.DataFrame):
    """Middle/Upper brackets: subject specialists shared across classes.
    Normalize each score by z-score WITHIN its subject and WITHIN this
    bracket only (attributed is already bracket-filtered), matching how the
    headteacher actually compares teachers — never across brackets, and
    only against others teaching the same subject."""
    df = attributed.copy()
    df["SUBJECT_Z"] = df.groupby("SUBJECT")["SCORE"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
    )
    per_teacher_subject_class = (
        df.groupby(["TEACHER", "SUBJECT", "GRADE", "STREAM"], as_index=False)
        .agg(AVG_SCORE=("SCORE", "mean"), AVG_Z=("SUBJECT_Z", "mean"), N_STUDENTS=("SCORE", "count"))
    )
    teacher_summary = (
        per_teacher_subject_class.groupby("TEACHER", as_index=False)
        .agg(
            WORKLOAD=("SUBJECT", "count"),  # number of subject-class assignments, a proxy for teaching load
            AVG_RAW_SCORE=("AVG_SCORE", "mean"),
            NORMALIZED_PERFORMANCE=("AVG_Z", "mean"),
            TOTAL_STUDENTS=("N_STUDENTS", "sum"),
        )
        .sort_values("NORMALIZED_PERFORMANCE", ascending=False)
    )
    return teacher_summary, per_teacher_subject_class

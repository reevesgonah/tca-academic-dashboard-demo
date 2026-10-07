# Demo dataset — fully synthetic, for testing only

This folder contains a **fully fictional** 3-term dataset modelled on a typical
primary/junior school (Grades 2–9, several streams, CBC subjects) with
**invented student and teacher names only**. It lets the trend, roster-churn
and teacher-reassignment features be demonstrated safely.

**All names are invented**; no score in this dataset belongs to any real, identifiable person.

## What it models

- **Term 1, 2025 (DEMO) → Term 2 → Term 3**, same academic year, so grades
  don't change between them (grade only advances at year-end) — just small
  enrollment churn (~3–6% leavers/joiners per term, matching normal
  mid-year transfers).
- **A gentle, realistic improvement** in average scores term over term
  (~10 points total across the year), so the term-over-term trend chart on
  the School Overview page has something to show.
- **One deliberate teacher reassignment** in each bracket: a Junior
  class-teacher (Grade 2E) changes between Term 1 and Term 2, and a
  Middle/Upper subject-teacher changes between Term 2 and Term 3 — so you
  can see how the app would (eventually) reflect real staff changes.
- Every student carries a synthetic but persistent `STUDENT_ID`
  (`DEMO-S0001`, etc.) — a working example of persistent student IDs.

## Files

- `scores_long_demo.csv` — columns: `NAME, TERM, EXAM,
  GRADE, STREAM, SUBJECT, SCORE, STUDENT_ID`.
- `subject_teachers_demo.csv` — `TERM, GRADE, STREAM, SUBJECT, TEACHER`; term-aware so teacher reassignments display correctly per term.
- `class_teachers_demo.csv` — homeroom listing per class per term.

## How the app uses it

The app loads these three files directly on startup — nothing needs to be uploaded or configured.

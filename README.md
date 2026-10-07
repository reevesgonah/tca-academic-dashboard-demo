# Total Care Academy — Analytics Dashboard (Demo)

A Streamlit dashboard for school academic performance, built as a shareable
web link. **This is the public demo version**: it runs entirely on a bundled,
fully synthetic dataset (`demo/`) — every student, teacher and score is
invented. There is no login and no connection to any real data source.

## Pages

- **Overview** — school-wide KPIs (average, pass rate, change vs the previous
  exam and term), performance trends, grade and subject breakdowns, and
  "areas requiring attention" flags.
- **Subject Analysis** — subject ranking, comparison against the school
  average, per-grade breakdown and teacher/class context.
- **Grade Analysis** — class comparison within a grade, and teacher
  performance within the Junior / Middle / Upper brackets.
- **Student Search** — individual student report: scores per subject, class
  rank, movement between exams, and subject-cluster strengths.

Use the filters (Year → Term → Exam) and the pass-mark slider in the sidebar.

## The demo data

`demo/` holds a synthetic 3-term dataset (Term 1–3, 2025) with invented names.
See `demo/README.md` for what it models.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying (Streamlit Community Cloud — free)

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
   and point it at the repo, branch, and `app.py`.
3. Deploy. No secrets or configuration are needed.

## File structure

```
app.py                  # Streamlit UI
data_utils.py           # loading, joins, ranking/trend/teacher-normalization logic
assets/logo.jpeg        # crest used as page icon and sidebar logo
demo/                   # synthetic dataset the app reads
requirements.txt
```

"""
NEET All India College Predictor 2026 — Streamlit UI
Matches brilliantpala.org layout exactly.
Pure rule-based, no ML model.
"""

import re
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
BP_DATA = BASE_DIR / "data" / "raw" / "brilliantpala_raw.csv"

CHANCE_HIGH = "High"
CHANCE_GOOD = "Good"
CHANCE_LOW = "Low"

# Region groupings matching brilliantpala.org
STATE_REGIONS = {
    "North": ["Delhi (NCT)", "Haryana", "Himachal Pradesh", "Jammu And Kashmir", "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand"],
    "West": ["Goa", "Gujarat", "Maharashtra"],
    "Central": ["Chhattisgarh", "Madhya Pradesh"],
    "South": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana", "Puducherry"],
    "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal", "Andaman And Nicobar Islands"],
    "North-East": ["Assam", "Arunachal Pradesh", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura"],
}


# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(BP_DATA)
    df = df.rename(columns={
        "institute": "college_name",
        "candidateCategory": "candidateCategory",
        "allottedCategory": "allottedCategory",
        "rank": "closing_rank",
    })
    return df


# ── Chance classification ────────────────────────────────────────────────────
def classify_chance(rank, closing_rank):
    if rank <= closing_rank:
        return CHANCE_HIGH
    elif rank <= closing_rank * 1.07:
        return CHANCE_GOOD
    else:
        return CHANCE_LOW


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NEET All India College Predictor 2026", page_icon="🎓", layout="wide")


# ── Load ─────────────────────────────────────────────────────────────────────
df = load_data()

ALL_CATEGORIES = sorted(df["candidateCategory"].unique())
ALL_QUOTAS = sorted(df["quota"].unique())
ALL_STATES = sorted(df["state"].unique())
ALL_COURSES = sorted(df["course"].unique())
ALL_ALLOTTED_CATS = sorted(df["allottedCategory"].unique())
ALL_PHASES = sorted(df["phase"].unique())


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #2563eb, #1e40af); color: white; padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 1.5rem;">
    <h1 style="margin: 0; font-size: 2rem;">NEET All India College Predictor 2026</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; color: #dbeafe;">Enter your rank to predict your admission chances.</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Main inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Enter Your Details")

    rank = st.number_input("NEET All India Rank (AIR)", min_value=1, max_value=2_000_000, value=50000, step=1)

    category = st.selectbox("Your Category", ["-- Select --"] + ALL_CATEGORIES, index=0)

    # Quota multi-select
    selected_quotas = st.multiselect("Quota (Optional)", ALL_QUOTAS, default=[])

    # State select
    state_options = ["-- Select a State (Optional) --"] + ALL_STATES
    selected_state = st.selectbox("State (Optional)", state_options, index=0)

    # Course multi-select
    selected_courses = st.multiselect("Course (Optional)", ALL_COURSES, default=[])

    predict_btn = st.button("Predict", type="primary", use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────
if predict_btn:
    # Apply main filters
    mask = pd.Series([True] * len(df))

    if category != "-- Select --":
        mask &= df["candidateCategory"].str.lower() == category.lower()

    if selected_quotas:
        mask &= df["quota"].isin(selected_quotas)

    if selected_state != "-- Select a State (Optional) --":
        mask &= df["state"] == selected_state

    if selected_courses:
        mask &= df["course"].isin(selected_courses)

    filtered = df[mask].copy()

    # Compute chance
    filtered["chance"] = filtered["closing_rank"].apply(lambda cr: classify_chance(rank, cr))

    # Sort by closing_rank descending (easiest first)
    filtered = filtered.sort_values("closing_rank", ascending=False)

    # Initialize secondary filter selections in session state
    if "filter_chances" not in st.session_state:
        st.session_state.filter_chances = []
    if "filter_phases" not in st.session_state:
        st.session_state.filter_phases = []
    if "filter_states" not in st.session_state:
        st.session_state.filter_states = []
    if "filter_courses" not in st.session_state:
        st.session_state.filter_courses = []
    if "filter_allotted_cats" not in st.session_state:
        st.session_state.filter_allotted_cats = []
    if "filter_quotas" not in st.session_state:
        st.session_state.filter_quotas = []

    # ── Secondary Filters ────────────────────────────────────────────────
    if len(filtered) > 0:
        with st.expander("Refine Your Results", expanded=False):
            # Tabs for filters
            tab_state, tab_course, tab_allotted, tab_quota, tab_phase, tab_chance = st.tabs(
                ["State", "Course", "Allotted Category", "Quota", "Phase", "Chance"]
            )

            with tab_state:
                # Region-grouped state selection
                for region, states in STATE_REGIONS.items():
                    st.markdown(f"**{region}**")
                    cols = st.columns(3)
                    for i, s in enumerate(states):
                        with cols[i % 3]:
                            if st.checkbox(s, key=f"state_{s}", value=s in st.session_state.filter_states):
                                if s not in st.session_state.filter_states:
                                    st.session_state.filter_states.append(s)
                            else:
                                if s in st.session_state.filter_states:
                                    st.session_state.filter_states.remove(s)
                    st.divider()

            with tab_course:
                course_cols = st.columns(3)
                for i, c in enumerate(ALL_COURSES):
                    with course_cols[i % 3]:
                        if st.checkbox(c, key=f"course_{c}", value=c in st.session_state.filter_courses):
                            if c not in st.session_state.filter_courses:
                                st.session_state.filter_courses.append(c)
                        else:
                            if c in st.session_state.filter_courses:
                                st.session_state.filter_courses.remove(c)

            with tab_allotted:
                ac_cols = st.columns(3)
                for i, ac in enumerate(ALL_ALLOTTED_CATS):
                    with ac_cols[i % 3]:
                        if st.checkbox(ac, key=f"ac_{ac}", value=ac in st.session_state.filter_allotted_cats):
                            if ac not in st.session_state.filter_allotted_cats:
                                st.session_state.filter_allotted_cats.append(ac)
                        else:
                            if ac in st.session_state.filter_allotted_cats:
                                st.session_state.filter_allotted_cats.remove(ac)

            with tab_quota:
                q_cols = st.columns(3)
                for i, q in enumerate(ALL_QUOTAS):
                    with q_cols[i % 3]:
                        if st.checkbox(q, key=f"quota_{q}", value=q in st.session_state.filter_quotas):
                            if q not in st.session_state.filter_quotas:
                                st.session_state.filter_quotas.append(q)
                        else:
                            if q in st.session_state.filter_quotas:
                                st.session_state.filter_quotas.remove(q)

            with tab_phase:
                phase_cols = st.columns(3)
                for i, p in enumerate(ALL_PHASES):
                    with phase_cols[i % 3]:
                        if st.checkbox(f"Phase {p}", key=f"phase_{p}", value=p in st.session_state.filter_phases):
                            if p not in st.session_state.filter_phases:
                                st.session_state.filter_phases.append(p)
                        else:
                            if p in st.session_state.filter_phases:
                                st.session_state.filter_phases.remove(p)

            with tab_chance:
                chance_cols = st.columns(3)
                for i, ch in enumerate([CHANCE_HIGH, CHANCE_GOOD, CHANCE_LOW]):
                    with chance_cols[i % 3]:
                        if st.checkbox(ch, key=f"chance_{ch}", value=ch in st.session_state.filter_chances):
                            if ch not in st.session_state.filter_chances:
                                st.session_state.filter_chances.append(ch)
                        else:
                            if ch in st.session_state.filter_chances:
                                st.session_state.filter_chances.remove(ch)

            # Clear button
            if st.button("Clear All Filters", key="clear_filters"):
                st.session_state.filter_chances = []
                st.session_state.filter_phases = []
                st.session_state.filter_states = []
                st.session_state.filter_courses = []
                st.session_state.filter_allotted_cats = []
                st.session_state.filter_quotas = []
                st.rerun()

        # Apply secondary filters
        if st.session_state.filter_chances:
            filtered = filtered[filtered["chance"].isin(st.session_state.filter_chances)]
        if st.session_state.filter_phases:
            filtered = filtered[filtered["phase"].astype(str).isin(st.session_state.filter_phases)]
        if st.session_state.filter_states:
            filtered = filtered[filtered["state"].isin(st.session_state.filter_states)]
        if st.session_state.filter_courses:
            filtered = filtered[filtered["course"].isin(st.session_state.filter_courses)]
        if st.session_state.filter_allotted_cats:
            filtered = filtered[filtered["allottedCategory"].isin(st.session_state.filter_allotted_cats)]
        if st.session_state.filter_quotas:
            filtered = filtered[filtered["quota"].isin(st.session_state.filter_quotas)]

    # ── Results ──────────────────────────────────────────────────────────
    high = (filtered["chance"] == CHANCE_HIGH).sum()
    good = (filtered["chance"] == CHANCE_GOOD).sum()
    low = (filtered["chance"] == CHANCE_LOW).sum()

    # Summary
    st.markdown(f"**{len(filtered)} Eligible Colleges Found** — HIGH: {high} | GOOD: {good} | LOW: {low}")

    if len(filtered) == 0:
        st.info("No matching colleges found. Try adjusting your rank or filters.")
    else:
        # Build results table
        results = pd.DataFrame({
            "S.No": range(1, len(filtered) + 1),
            "Institute": filtered["college_name"].values,
            "Chance": filtered["chance"].values,
            "State": filtered["state"].values,
            "Quota": filtered["quota"].values,
            "Course": filtered["course"].values,
            "Allotted Category": filtered["allottedCategory"].values,
            "Phase": filtered["phase"].values,
            "Closing Rank": filtered["closing_rank"].values,
        })

        # Style chance column
        def chance_badge(val):
            if val == "High":
                return "background-color: #dcfce7; color: #166534; font-weight: 600"
            elif val == "Good":
                return "background-color: #fef9c3; color: #854d0e; font-weight: 600"
            else:
                return "background-color: #fee2e2; color: #991b1b; font-weight: 600"

        styled = results.style.map(chance_badge, subset=["Chance"])
        styled = styled.format({"Closing Rank": "{:,}"})

        st.dataframe(styled, width="stretch", hide_index=True, height=600)

else:
    st.info("Enter your details in the sidebar and click **Predict**.")

    st.markdown("""
    ### How it works
    This predictor uses **official allotment data** from MCC NEET UG counselling rounds.

    **Chance Labels:**
    - 🟢 **High** — Your rank is within the closing rank (strong chance)
    - 🟡 **Good** — Your rank is within 7% of the closing rank (possible)
    - 🔴 **Low** — Your rank is above the closing rank (borderline, try anyway)

    **Data Source:** Phase 1, 2, 3 allotment data from official MCC counselling notifications.
    """)

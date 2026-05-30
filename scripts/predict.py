"""
NEET College Predictor — Pure rule-based.

Chance logic:
  rank <= closing_rank           → HIGH
  rank <= closing_rank * 1.07    → GOOD
  rank >  closing_rank * 1.07    → LOW
"""

import argparse
import re
import pandas as pd
from pathlib import Path

JUNK_PATTERN = re.compile(
    r"^\d+\s*(st|nd|rd|th|aug|sep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul)"
    r"|^E6Dth|^to\s|^Notice|^Registration|^Payment|^\d{4}\s+to|^Unknown",
    re.IGNORECASE,
)

CHANCE_HIGH = "High"
CHANCE_GOOD = "Good"
CHANCE_LOW = "Low"


def load_bp_data() -> pd.DataFrame:
    df = pd.read_csv(Path("data/raw/allotment_data.csv"))
    df = df.rename(columns={
        "institute": "college_name",
        "candidateCategory": "category",
        "allottedCategory": "allottedCategory",
        "rank": "closing_rank",
    })
    return df


def classify_chance(rank: int, closing_rank: int) -> str:
    if rank <= closing_rank:
        return CHANCE_HIGH
    elif rank <= closing_rank * 1.07:
        return CHANCE_GOOD
    else:
        return CHANCE_LOW


def predict(
    rank: int,
    category: str = None,
    quota: str = None,
    state: str = None,
    course: str = None,
    phase: str = None,
    allotted_category: str = None,
) -> pd.DataFrame:
    df = load_bp_data()

    # Filter by rank (show all where rank <= closing_rank * 1.07 OR rank <= closing_rank)
    # Like brilliantpala, show everything with any chance
    mask = pd.Series([True] * len(df))

    if category:
        mask &= df["category"].str.lower() == category.lower()
    if quota:
        mask &= df["quota"].str.lower() == quota.lower()
    if state:
        mask &= df["state"].str.lower() == state.lower()
    if course:
        mask &= df["course"].str.lower() == course.lower()
    if phase:
        mask &= df["phase"].astype(str) == str(phase)
    if allotted_category:
        mask &= df["allottedCategory"].str.lower() == allotted_category.lower()

    filtered = df[mask].copy()

    # Compute chance
    filtered["chance"] = filtered["closing_rank"].apply(lambda cr: classify_chance(rank, cr))

    # Show all, sorted by closing_rank (highest first = easiest to get)
    filtered = filtered.sort_values("closing_rank", ascending=False)

    # Build output
    result = pd.DataFrame({
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

    return result


def main():
    parser = argparse.ArgumentParser(description="NEET College Predictor (Rule-Based)")
    parser.add_argument("--rank", type=int, required=True, help="NEET All India Rank")
    parser.add_argument("--category", type=str, default=None, help="Candidate Category")
    parser.add_argument("--quota", type=str, default=None, help="Quota")
    parser.add_argument("--state", type=str, default=None, help="State")
    parser.add_argument("--course", type=str, default=None, help="Course (MBBS/BDS/Nursing)")
    parser.add_argument("--phase", type=str, default=None, help="Phase (1/2/3)")
    parser.add_argument("--allotted-category", type=str, default=None, help="Allotted Category")
    parser.add_argument("--top", type=int, default=50, help="Show top N results")
    args = parser.parse_args()

    results = predict(
        rank=args.rank,
        category=args.category,
        quota=args.quota,
        state=args.state,
        course=args.course,
        phase=args.phase,
        allotted_category=args.allotted_category,
    )

    high = (results["Chance"] == CHANCE_HIGH).sum()
    good = (results["Chance"] == CHANCE_GOOD).sum()
    low = (results["Chance"] == CHANCE_LOW).sum()

    print(f"\n{'='*90}")
    print(f"  NEET All India College Predictor 2026")
    print(f"  NEET All India Rank (AIR): {args.rank:,}")
    if args.category:
        print(f"  Category: {args.category}")
    if args.quota:
        print(f"  Quota: {args.quota}")
    print(f"{'='*90}\n")

    print(f"  {len(results)} Eligible Colleges Found | HIGH: {high} | GOOD: {good} | LOW: {low}\n")

    shown = results.head(args.top)
    for _, row in shown.iterrows():
        chance = row["Chance"]
        if chance == "High":
            badge = "🟢 HIGH"
        elif chance == "Good":
            badge = "🟡 GOOD"
        else:
            badge = "🔴 LOW "
        print(f"  {row['S.No']:>3d}. [{badge}] {row['Institute']}")
        print(f"       {row['State']} | {row['Quota']} | {row['Course']} | {row['Allotted Category']} | Phase {row['Phase']} | Closing Rank: {row['Closing Rank']:,}")

    if len(results) > args.top:
        print(f"\n  ... and {len(results) - args.top} more")


if __name__ == "__main__":
    main()

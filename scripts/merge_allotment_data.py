"""
Merge Brilliantpala source-of-truth data with our scraped MCC/state data.

BP provides: clean closing ranks per (institute, category, quota, phase)
We provide: historical depth, opening_rank, seat_count

Output: data/cleaned/merged_dataset_v2.csv
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz, process


# ── Category mapping ────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "General": "General",
    "OBC": "OBC-NCL",
    "SC": "Scheduled Caste",
    "ST": "Scheduled Tribe",
    "EWS": "EWS",
    "General PwD": "General PwD",
    "OBC PwD": "OBC-NCL PwD",
    "SC PwD": "Scheduled Caste PwD",
    "ST PwD": "Scheduled Tribe PwD",
    "EWS PwD": "EWS PwD",
    "GNYes": "General PwD",
}

# ── Quota mapping ───────────────────────────────────────────────────────────
QUOTA_MAP = {
    "All India": "All India",
    "Deemed/Paid Seats Quota": "Deemed/Paid Seats",
    "Non-Resident Indian": "NRI",
    "Employees State Insurance Scheme (ESI)": "ESI",
    "Delhi University Quota": "Delhi University",
    "IP University Quota": "IP University",
    "Aligarh Muslim University (AMU) Quota": "AMU Quota",
    "Open Seat Quota": "All India",
    "B.Sc Nursing All India": "All India",
    "B.Sc Nursing Delhi NCR": "Delhi University",
    "B.Sc Nursing Delhi NCR CW Quota": "Delhi University",
    "B.Sc Nursing IP CW Quota": "IP University",
    "Delhi NCR Children/Widows of Personnel of the Armed Forces (CW) DU Quota": "Delhi University",
    "Delhi NCR Children/Widows of Personnel of the Armed Forces (CW) IP Quota": "IP University",
    "(AMU) Self finance All India": "AMU Quota",
    "(AMU) Self finance internal": "AMU Quota",
    "Foreign Country Quota": "NRI",
    "Internal - Puducherry UT Domicile": "All India",
    "Jain Minority Quota": "Deemed/Paid Seats",
    "Jamia Internal Quota": "AMU Quota",
    "Muslim Minority Quota": "Deemed/Paid Seats",
    "Muslim OBC Quota": "Deemed/Paid Seats",
    "Muslim Quota": "Deemed/Paid Seats",
    "Muslim ST Quota": "Deemed/Paid Seats",
    "Muslim Women Quota": "Deemed/Paid Seats",
    "Non-Resident Indian (AMU) Quota": "NRI",
    "Non-Resident Indian (Jamia) Quota": "NRI",
    "Employees State Insurance Scheme Nursing Quota (ESI-IP Quota Nursing)": "ESI",
}

# ── Phase → round_number ────────────────────────────────────────────────────
PHASE_MAP = {"1": 1, "2": 2, "3": 3}

# ── Course normalization ────────────────────────────────────────────────────
COURSE_MAP = {
    "MBBS": "MBBS",
    "BDS": "BDS",
    "B.Sc. Nursing": "Nursing",
}


def parse_brilliantpala(html_path: str) -> pd.DataFrame:
    """Parse the collegeData JS array from the brilliantpala HTML page."""
    with open(html_path) as f:
        content = f.read()

    start = content.find("const collegeData = [") + len("const collegeData = ")
    bracket_count = 0
    end = start
    for i, c in enumerate(content[start:], start):
        if c == "[":
            bracket_count += 1
        elif c == "]":
            bracket_count -= 1
            if bracket_count == 0:
                end = i + 1
                break

    data_str = content[start:end]
    pattern = (
        r"\{\s*rank:\s*(\d+),\s*quota:\s*'([^']*)',\s*institute:\s*'([^']*)',"
        r"\s*state:\s*'([^']*)',\s*course:\s*'([^']*)',"
        r"\s*allottedCategory:\s*'([^']*)',\s*candidateCategory:\s*'([^']*)',"
        r"\s*phase:\s*'([^']*)'\s*\}"
    )
    matches = re.findall(pattern, data_str)

    records = []
    for m in matches:
        cat = CATEGORY_MAP.get(m[6], m[6])
        quota = QUOTA_MAP.get(m[1], m[1])
        course = COURSE_MAP.get(m[4], m[4])
        phase = PHASE_MAP.get(m[7], int(m[7]))

        # Strip city/state suffix from institute names
        institute = re.sub(r",\s*(Puducherry|Andhra Pradesh|Tamil Nadu|Karnataka|Kerala|Maharashtra|Delhi|Gujarat|Rajasthan|Haryana|West Bengal|Odisha|Bihar|Madhya Pradesh|Punjab|Uttar Pradesh|Telangana|Jammu And Kashmir|Uttarakhand|Chhattisgarh|Jharkhand|Assam|Goa|Himachal Pradesh|Chandigarh|Arunachal Pradesh|Manipur|Meghalaya|Mizoram|Nagaland|Tripura|Andaman And Nicobar Islands|Dadra And Nagar Haveli|Puducherry)$", "", m[2]).strip()

        records.append({
            "college_name": institute,
            "state": m[3],
            "quota": quota,
            "category": cat,
            "course": course,
            "closing_rank": int(m[0]),
            "opening_rank": int(m[0]),  # BP only has closing
            "round_number": phase,
            "year": 2025,
            "source_file": "brilliantpala.org",
            "seat_count": np.nan,
            "source": "bp",
        })

    return pd.DataFrame(records)


def normalize_our_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize our existing merged_dataset.csv to match BP schema."""
    # Remove junk
    junk = (
        df["college_name"].str.contains(
            r"Unknown|payment|notice|facility|www\.|^\d+\s*(st|nd|rd|th|aug|sep|oct)|^E6Dth|^to\s|^\d{4}\s+to",
            case=False, na=False, regex=True,
        )
        | (df["college_name"].str.len() <= 5)
    )
    clean = df[~junk].copy()

    # Use round_number if available, else round
    if "round_number" in clean.columns:
        clean["round_number"] = clean["round_number"].fillna(0).astype(int)
    elif "round" in clean.columns:
        clean["round_number"] = clean["round"].fillna(0).astype(int)
    else:
        clean["round_number"] = 0

    clean["closing_rank"] = pd.to_numeric(clean["closing_rank"], errors="coerce").fillna(0).astype(int)
    clean["opening_rank"] = pd.to_numeric(clean["opening_rank"], errors="coerce").fillna(0).astype(int)
    clean["year"] = clean["year"].fillna(2025).astype(int)
    clean["source"] = "ours"
    return clean


def fuzzy_match_institutes(bp_df: pd.DataFrame, our_df: pd.DataFrame, threshold: int = 80) -> dict:
    """Build a mapping from our college names to BP college names."""
    bp_names = bp_df["college_name"].str.strip().str.lower().unique().tolist()
    our_names = our_df["college_name"].str.strip().str.lower().unique().tolist()

    mapping = {}
    for our_name in our_names:
        result = process.extractOne(our_name, bp_names, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            mapping[our_name] = result[0]
        else:
            mapping[our_name] = our_name

    matches = sum(1 for k, v in mapping.items() if k != v)
    print(f"  Fuzzy matches found: {matches}/{len(our_names)}")
    return mapping


def merge_datasets(bp_df: pd.DataFrame, our_df: pd.DataFrame) -> pd.DataFrame:
    """Merge BP + our data, deduplicate, prefer BP for conflicts."""

    # Apply name mapping to our data
    name_map = fuzzy_match_institutes(bp_df, our_df)
    our_df["college_name_lower"] = our_df["college_name"].str.strip().str.lower()
    our_df["college_name"] = our_df["college_name_lower"].map(name_map).fillna(our_df["college_name"])
    our_df = our_df.drop(columns=["college_name_lower"])

    # Standardize columns
    common_cols = [
        "college_name", "state", "quota", "category", "course",
        "opening_rank", "closing_rank", "round_number", "year",
        "source_file", "seat_count", "source",
    ]

    # Add missing columns
    for col in common_cols:
        if col not in bp_df.columns:
            bp_df[col] = np.nan
        if col not in our_df.columns:
            our_df[col] = np.nan

    bp_sub = bp_df[common_cols].copy()
    our_sub = our_df[common_cols].copy()

    # Merge
    merged = pd.concat([bp_sub, our_sub], ignore_index=True)

    # Clean state names
    merged["state"] = merged["state"].replace({"Delhi (NCT)": "Delhi"})

    # Clean quota names
    merged["quota"] = merged["quota"].replace({
        "Employees State Insurance Scheme (ESI)": "ESI",
        "B.Sc Nursing All India": "All India",
        "Open Seat Quota": "All India",
    })

    before = len(merged)

    # Deduplicate: prefer BP (source=bp) over ours (source=ours)
    # Sort so BP comes first
    merged["_priority"] = merged["source"].map({"bp": 0, "ours": 1}).fillna(2)
    merged = merged.sort_values("_priority")
    merged = merged.drop_duplicates(
        subset=["college_name", "category", "quota", "closing_rank"],
        keep="first",
    )
    merged = merged.drop(columns=["_priority"])

    after = len(merged)
    print(f"  Deduplication: {before:,} → {after:,} rows (removed {before - after:,})")

    return merged


def main():
    bp_raw = Path("data/raw/allotment_data.csv")
    our_file = Path("data/cleaned/merged_dataset.csv")
    out = Path("data/cleaned/merged_dataset_v2.csv")

    print("Loading Brilliantpala data...")
    bp_df = pd.read_csv(bp_raw)
    bp_df = bp_df.rename(columns={
        "institute": "college_name",
        "candidateCategory": "category",
        "rank": "closing_rank",
    })
    # Apply mappings
    bp_df["category"] = bp_df["category"].map(CATEGORY_MAP).fillna(bp_df["category"])
    bp_df["quota"] = bp_df["quota"].map(QUOTA_MAP).fillna(bp_df["quota"])
    bp_df["course"] = bp_df["course"].map(COURSE_MAP).fillna(bp_df["course"])
    bp_df["round_number"] = bp_df["phase"].map(PHASE_MAP).fillna(1).astype(int)
    bp_df["opening_rank"] = bp_df["closing_rank"]
    bp_df["year"] = 2025
    bp_df["source_file"] = "brilliantpala.org"
    bp_df["seat_count"] = np.nan
    bp_df["source"] = "bp"
    bp_df["college_name"] = bp_df["college_name"].apply(lambda x: re.sub(r",\s*(Puducherry|Andhra Pradesh|Tamil Nadu|Karnataka|Kerala|Maharashtra|Delhi|Gujarat|Rajasthan|Haryana|West Bengal|Odisha|Bihar|Madhya Pradesh|Punjab|Uttar Pradesh|Telangana|Jammu And Kashmir|Uttarakhand|Chhattisgarh|Jharkhand|Assam|Goa|Himachal Pradesh|Chandigarh|Arunachal Pradesh|Manipur|Meghalaya|Mizoram|Nagaland|Tripura|Andaman And Nicobar Islands|Dadra And Nagar Haveli)$", "", str(x)).strip())
    bp_df["state"] = bp_df["state"].replace({"Delhi (NCT)": "Delhi"})
    print(f"  BP rows: {len(bp_df):,}")
    print(f"  BP colleges: {bp_df['college_name'].nunique()}")

    print("\nLoading our data...")
    our_df = pd.read_csv(our_file)
    print(f"  Our rows: {len(our_df):,}")

    print("\nNormalizing our data...")
    our_df = normalize_our_data(our_df)
    print(f"  After normalization: {len(our_df):,} rows, {our_df['college_name'].nunique()} colleges")

    print("\nMerging datasets...")
    merged = merge_datasets(bp_df, our_df)

    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total rows: {len(merged):,}")
    print(f"Colleges: {merged['college_name'].nunique()}")
    print(f"States: {merged['state'].nunique()}")
    print(f"Quotas: {merged['quota'].nunique()}")
    print(f"Categories: {merged['category'].nunique()}")
    print(f"Courses: {merged['course'].nunique()}")
    print(f"Source breakdown: {merged['source'].value_counts().to_dict()}")
    print(f"Year breakdown: {sorted(merged['year'].unique())}")

    merged.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

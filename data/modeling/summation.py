import pandas as pd

# =========================
# FILE CONFIGURATION
# =========================
files = {
    "Children": "Children_Crimes_2021_2023.csv",
    "Women": "Women_Crimes_2021_2023.csv",
    "SC": "SC_Crimes_2021_2023.csv",
    "ST": "ST_Crimes_2021_2023.csv",
    "Senior Citizens": "Senior_Citizens_Crimes_2021_2023.csv"
}

EXPECTED_CITIES = 34
EXPECTED_YEARS = ["2021", "2022", "2023"]

all_data = []

# =========================
# PROCESS EACH CSV
# =========================
for group, path in files.items():
    df = pd.read_csv(path)

    # ---------- HARD VALIDATION ----------
    if df.shape[0] != EXPECTED_CITIES:
        raise ValueError(
            f"[ERROR] {group}: Expected {EXPECTED_CITIES} cities, found {df.shape[0]}"
        )

    for year in EXPECTED_YEARS:
        if year not in df.columns.astype(str):
            raise ValueError(
                f"[ERROR] {group}: Missing year column {year}"
            )

    if "City" not in df.columns:
        raise ValueError(
            f"[ERROR] {group}: Missing City column"
        )

    # ---------- KEEP ONLY REQUIRED COLUMNS ----------
    df = df[["City", "2021", "2022", "2023"]]

    # ---------- RESHAPE TO LONG FORMAT ----------
    df_long = df.melt(
        id_vars="City",
        value_vars=EXPECTED_YEARS,
        var_name="Year",
        value_name="Crime_Rate"
    )

    # ---------- ADD GROUP TAG ----------
    df_long["Group"] = group

    # ---------- TYPE ENFORCEMENT ----------
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Crime_Rate"] = pd.to_numeric(
        df_long["Crime_Rate"],
        errors="raise"
    )

    all_data.append(df_long)

# =========================
# COMBINE ALL GROUPS
# =========================
final_df = pd.concat(all_data, ignore_index=True)

# =========================
# FINAL VALIDATION (NO ROOM FOR ERROR)
# =========================
if final_df.shape[0] != 510:
    raise ValueError(
        f"[ERROR] Final row count incorrect: {final_df.shape[0]} (Expected 510)"
    )

if final_df.isnull().any().any():
    raise ValueError("[ERROR] Null values detected in final dataset")

if final_df["Group"].nunique() != 5:
    raise ValueError("[ERROR] Group count mismatch")

if final_df["City"].nunique() != 34:
    raise ValueError("[ERROR] City count mismatch")

if final_df["Year"].nunique() != 3:
    raise ValueError("[ERROR] Year count mismatch")

# =========================
# SAVE OUTPUT
# =========================
final_df.to_csv(
    "All_Groups_Crime_Rate_2021_2023_510_Rows.csv",
    index=False
)

print("SUCCESS ✅ Dataset created with EXACTLY 510 rows (Crime Rates).")


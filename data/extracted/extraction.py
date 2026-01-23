import camelot
import pandas as pd
from pathlib import Path

# ==========================
# PATH CONFIGURATION
# ==========================
INPUT_DIR = Path("raw")
OUTPUT_DIR = Path("extracted")

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================
# PROCESS ALL PDF FILES
# ==========================
for pdf_file in INPUT_DIR.glob("*.pdf"):
    print(f"\nProcessing: {pdf_file.name}")

    try:
        # --- Try lattice mode first (best for bordered tables) ---
        tables = camelot.read_pdf(
            str(pdf_file),
            pages="all",
            flavor="lattice"
        )

        # --- Fallback to stream mode if no tables found ---
        if len(tables) == 0:
            tables = camelot.read_pdf(
                str(pdf_file),
                pages="all",
                flavor="stream"
            )

        if len(tables) == 0:
            print(f"⚠️ No tables found in {pdf_file.name}")
            continue

        # ==========================
        # SAVE EACH TABLE
        # ==========================
        for i, table in enumerate(tables, start=1):
            df = table.df

            # Drop fully empty rows
            df = df.dropna(how="all")

            output_file = OUTPUT_DIR / f"{pdf_file.stem}_table_{i}.csv"
            df.to_csv(output_file, index=False, header=False)

            print(f"✓ Saved {output_file.name}")

    except Exception as e:
        print(f"❌ Failed {pdf_file.name}: {e}")


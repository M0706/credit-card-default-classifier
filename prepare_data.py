"""Clean the UCI Credit Card Default dataset into a single modelling CSV."""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/default of credit card clients.xls")
OUT = Path("data/credit_default.csv")
TARGET = "default"


def main() -> None:
    # Row 0 of the spreadsheet is a metadata row, so the real header is row 1.
    df = pd.read_excel(RAW, header=1)

    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": TARGET})

    # EDUCATION 0/5/6 and MARRIAGE 0 are undocumented codes -> fold into "others".
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    df = df.drop_duplicates().reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"rows={len(df)} features={df.shape[1] - 1}")
    print(f"target rate={df[TARGET].mean():.4f}")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()

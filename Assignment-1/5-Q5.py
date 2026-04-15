
# 5. Write a generic Python Notebook that:
    # Takes the number of years as input from user and give details of all constitute that were part of the selected index throughout this period.
        # Total number of stocks, Each Stock Name and Sector It belongs
    # Create a Master Database (by downloading Excel file for each selected stock) of these stocks.
    # Analyse which sector contributed the maximum number of survivors in this country?


"""

Will ask for 2 kinds of input:
    1. Num of year
    2. Year range (Jun2010-Dec25)

User gave Num of years then return them stocks which are 
common in the index in last n years and in first n year
- So two output on for 2010 -> 10+n and other for 2025-n -> 2025

User gives the year range then stocks common in that year range()
"""

# # =====================================================
# import pandas as pd
# from pathlib import Path

# PRICE_DIR = Path("IBEX35/prices")
# OUT_FILE = Path("Master_Price_Database.csv")

# all_dfs = []

# for f in PRICE_DIR.glob("*.csv"):
#     df = pd.read_csv(f)
#     df["SourceFile"] = f.name
#     all_dfs.append(df)

# master_df = pd.concat(all_dfs, ignore_index=True)

# master_df.to_csv(OUT_FILE, index=False)

# print("Master price database created")
# print("Total rows:", len(master_df))
# print("Total stocks:", master_df["SourceFile"].nunique())

# =======================================================




import pandas as pd
from pathlib import Path
from functools import reduce

BASE_DIR = Path("IBEX35")
BIANNUAL_DIR = BASE_DIR / "constituents"

# mapping names for the same companies
CANONICAL_FILE = BASE_DIR / "constituents" / "company_canonical_map.csv"
canonical_df = pd.read_csv(CANONICAL_FILE)
NAME_MAP = dict(zip(canonical_df["RawName"], canonical_df["CanonicalName"]))


TICKER_MAP = (
    canonical_df
    .dropna(subset=["BaseTicker"])
    .set_index("CanonicalName")["BaseTicker"]
    .to_dict()
)



# another df for the sector map as another file for it
SECTOR_FILE = BASE_DIR / "constituents" / "SectorMap.csv"
sector_df = pd.read_csv(SECTOR_FILE)

sector_df["CanonicalName"] = sector_df["CanonicalName"].map(lambda x: NAME_MAP.get(x, x))

SECTOR_MAP = (
    sector_df
    .drop_duplicates("CanonicalName")
    .set_index("CanonicalName")["Sector"]
    .to_dict()
)

# =====================================


MASTER_PRICE_DIR = BASE_DIR / "prices"
RESTRICTED_DB_DIR = BASE_DIR / "results"
RESTRICTED_DB_DIR.mkdir(exist_ok=True)

def parse_date_column(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    return df


def create_restricted_price_db(companies,start_year,start_month,end_year,end_month,label):
    """
    Create a restricted price database for selected companies and date range
    """

    out_dir = RESTRICTED_DB_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp(start_year, start_month, 1)
    end_date = pd.Timestamp(end_year, end_month, 28)

    print(f"\nCreating restricted DB → {out_dir}")

    for company in companies:
        ticker = TICKER_MAP.get(company)

        if not ticker:
            print(f"[ERR] No ticker for {company}")
            continue

        price_file = MASTER_PRICE_DIR / f"{ticker}.csv"

        if not price_file.exists():
            print(f"[ERR] Missing price file: {ticker}")
            continue

        df = pd.read_csv(price_file)
        df = parse_date_column(df)

        df = df[
            (df["Date"] >= start_date) &
            (df["Date"] <= end_date)
        ]

        if df.empty:
            print(f"[ERR] No data in range for {company}")
            continue

        df["Company"] = company
        df["Ticker"] = ticker

        df.to_csv(out_dir / f"{company}.csv", index=False)
        print(f"[OK] Saved {company}")


# =====================================

def canonicalize(names):
    """
    Convert raw company names to canonical names
    """
    return {
        NAME_MAP.get(name.strip(), name.strip())
        for name in names
    }


def parse_year_month(file_path):
    """
    ibex_2010_Jun.csv -> (2010, 6)
    ibex_201_toggle
    """
    parts = file_path.stem.split("_")
    year = int(parts[1])
    month = 6 if parts[2].lower() == "jun" else 12
    return year, month


def load_constituents_by_range(start_ym, end_ym):
    """
    Load all biannual constituent files between (year, month) range
    """
    files = sorted(BIANNUAL_DIR.glob("ibex_*.csv"))

    company_sets = []

    for f in files:
        ym = parse_year_month(f)
        if start_ym <= ym <= end_ym:
            df = pd.read_csv(f)
            canon_names = canonicalize(df["Company"])
            company_sets.append(canon_names)

    return company_sets


def survivors_by_years(n):
    files = sorted(BIANNUAL_DIR.glob("ibex_*.csv"))
    yms = sorted(parse_year_month(f) for f in files)

    first_end = yms[2*n - 1]
    last_start = yms[-(2*n)]

    first_sets = load_constituents_by_range(yms[0], first_end)
    last_sets = load_constituents_by_range(last_start, yms[-1])

    return {
        "First": {
            "companies": sorted(reduce(set.intersection, first_sets)),
            "start": yms[0],
            "end": first_end
        },
        "Last": {
            "companies": sorted(reduce(set.intersection, last_sets)),
            "start": last_start,
            "end": yms[-1]
        }
    }


def survivors_by_range(start_year, start_month, end_year, end_month):
    company_sets = load_constituents_by_range(
        (start_year, start_month),
        (end_year, end_month)
    )
    return sorted(reduce(set.intersection, company_sets))


def output_table(companies):
    df = pd.DataFrame(companies, columns=["Company"])
    df["Sector"] = df["Company"].map(SECTOR_MAP)
    df["Sector"] = df["Sector"].fillna("Unknown")
    return df


def sector_survivor_analysis(companies):
    """
    Given a list of survivor companies,
    return sector-wise survivor counts
    """
    df = output_table(companies)

    sector_counts = (
        df.groupby("Sector")
          .size()
          .reset_index(name="SurvivorCount")
          .sort_values("SurvivorCount", ascending=False)
    )

    return sector_counts




def main():
    print("=========Select a option===========")

    print("1. Option-1: Num of years(1-15)")
    print("2. Option-2: Range of years(Between 2010-2025)")
    print("Either enter 1 or 2")

    choice = input("\nSelect (1 or 2): ").strip()

    # num of years
    if choice == "1":
        print("Option 1 selected")
        print("=================================================")
        print("=================================================")
        numOfYears = int(input("Enter the num of years(1-15) (Integer): "))
        
        if 1 <= numOfYears <= 15:
            ans1 = survivors_by_years(numOfYears)

            print(f'Survivers in first {numOfYears} years: ')
            # print(output_table(ans1["First_N_Years"]))
            print(output_table(ans1["First"]["companies"]))

            lth = len(ans1["First"]["companies"])
            print("Number of Companies Survied in First half: ", lth)
            
            sector_df = sector_survivor_analysis(ans1["First"]["companies"])
            print("\nSector-wise survivors (First half):")
            print(sector_df)
            print("\nSector contributing maximum survivors (First half):")
            print(sector_df.iloc[0])

            create_restricted_price_db(
                ans1["First"]["companies"],
                ans1["First"]["start"][0],
                ans1["First"]["start"][1],
                ans1["First"]["end"][0],
                ans1["First"]["end"][1],
                f"First{numOfYears}Years"
            )
            
            print("=================================================")
            print("Some comapnies may not be mapped to sectors as we are dealing with spanish names")
            print("so some chars mapping may be tediuos as they use many different chars like `. and many more in names")
            print("=================================================")

            print(f'Survivers in last {numOfYears} years: ')
            print(output_table(ans1["Last"]["companies"]))

            lth = len(ans1["Last"]["companies"])
            print("Number of Companies Survied in Last half: ", lth)

            sector_df = sector_survivor_analysis(ans1["Last"]["companies"])
            print("\nSector-wise survivors (Last half):")
            print(sector_df)
            print("\nSector contributing maximum survivors (Last half):")
            print(sector_df.iloc[0])

            create_restricted_price_db(
                ans1["Last"]["companies"],
                ans1["Last"]["start"][0],
                ans1["Last"]["start"][1],
                ans1["Last"]["end"][0],
                ans1["Last"]["end"][1],
                f"Last{numOfYears}Years"
            )

        else:
            print("Wrong number of years selected")

    # range of years
    else:
        print("Option 2 selected")
        print("June = 6 and December = 12, as stock is biannual")
        print("=================================================")
        print("Some comapnies may not be mapped to sectors as we are dealing with spanish names")
        print("so some chars mapping may be tediuos as they use many different chars like `. and many more in names")
        print("=================================================")

        StartYear = int(input("Enter the Start year (2010-25): "))
        StartMonth = int(input("Enter the Start Month (6 or 12): "))
        EndYear = int(input("Enter the End year (2010-25): "))
        EndMonth = int(input("Enter the End Month (6 or 12): "))

        survivors = survivors_by_range(StartYear, StartMonth, EndYear, EndMonth)
        
        print(f'Surviours from {StartYear}-{StartMonth} to {EndYear}-{EndMonth}: ')
        print(output_table(survivors))
        
        lth = len(survivors)
        print("Number of Surviours: ", lth)

        sector_df = sector_survivor_analysis(survivors)
        print("\nSector-wise survivors:")
        print(sector_df)
        print("\nSector contributing maximum survivors:")
        print(sector_df.iloc[0])
        
        create_restricted_price_db(
            survivors,
            StartYear,
            StartMonth,
            EndYear,
            EndMonth,
            label=f"{StartYear}_{StartMonth}_to_{EndYear}_{EndMonth}"
        )


if __name__ == "__main__":
    main()


"""
                         Company                             Sector
0                            ACS      Infrastructure & Construction
1                        Acciona  Infrastructure & Renewable Energy
2                  ArcelorMittal                    Metals & Mining
3                           BBVA                            Banking
4                 Banco Sabadell                            Banking
5                Banco Santander                            Banking
6                      Bankinter                            Banking
7                         Enagás  Gas Transmission & Infrastructure
8                      Ferrovial      Infrastructure & Construction
9                        Grifols         Healthcare & Biotechnology
10                     Iberdrola            Utilities - Electricity
11                       Inditex                   Retail & Apparel
12                Indra Sistemas       Technology & Defense Systems
13  International Airlines Group                           Airlines
14                        Mapfre                          Insurance
15                       Naturgy        Utilities-Gas & Electricity
16       Red Eléctrica de España           Electricity Transmission
17                        Repsol                 Energy - Oil & Gas
18                    Telefónica                 Telecommunications
"""

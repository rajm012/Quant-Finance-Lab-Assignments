
"""
6. Corporate Action Tracking (Splits & Bonuses)
    • Focus: Only on the stocks in the "15-Year Legends" list.
    • Data Extraction: For each of these survivors, gather details of all Stock Splits and Bonus
                Issues declared in the last 15 years.
    • Format: Tabulate the details including Event Type, Ratio, and Ex-Date.
"""


"""
Very Limited firms have genuine stock splits in last 15 years while 
ArcelorMittal carried out a reverse split.

Several 1:1 splits by yahoo finance shows the scrip dividend or 
accounting adjustment rather than actual changes or true splits.
"""

# import yfinance as yf
# import pandas as pd
# from datetime import datetime

# LEGEND_COMPANIES = {
#     "ACS": "ACS.MC",
#     "Acciona": "ANA.MC",
#     "ArcelorMittal": "MTS.MC",
#     "BBVA": "BBVA.MC",
#     "Banco Sabadell": "SAB.MC",
#     "Banco Santander": "SAN.MC",
#     "Bankinter": "BKT.MC",
#     "Enagás": "ENG.MC",
#     "Ferrovial": "FER.MC",
#     "Grifols": "GRF.MC",
#     "Iberdrola": "IBE.MC",
#     "Inditex": "ITX.MC",
#     "Indra Sistemas": "IDR.MC",
#     "International Airlines Group": "IAG.MC",
#     "Mapfre": "MAP.MC",
#     "Naturgy": "NTGY.MC",
#     "Red Eléctrica de España": "RED.MC",
#     "Repsol": "REP.MC",
#     "Telefónica": "TEF.MC"
# }

# def get_corporate_actions(ticker_symbol):
#     """Fetch corporate actions for a given ticker"""
#     stock = yf.Ticker(ticker_symbol)
    
#     try:
#         # Get corporate actions (splits and dividends)
#         actions = stock.actions
#         splits = stock.splits
#         print("========================")
#         print(actions)
        
#         # Filter for splits only
#         if not splits.empty:
#             splits = splits[splits.index >= '2010-01-01']
        
#         # Note: yfinance combines splits in .splits and dividends in .actions
#         # For bonus issues, you might need additional sources
        
#         return splits
#     except Exception as e:
#         print(f"Error fetching data for {ticker_symbol}: {e}")
#         return pd.Series()

# # Collect all data
# all_corporate_actions = []

# for company, ticker in LEGEND_COMPANIES.items():
#     print(f"Processing {company} ({ticker})...")
    
#     splits = get_corporate_actions(ticker)
    
#     if not splits.empty:
#         for date, ratio in splits.items():
#             all_corporate_actions.append({
#                 'Company': company,
#                 'Ticker': ticker,
#                 'Event Type': 'Stock Split',
#                 'Ratio': ratio,
#                 'Ex-Date': date.strftime('%Y-%m-%d')
#             })

# # Create DataFrame
# df_actions = pd.DataFrame(all_corporate_actions)

# # Save to CSV
# df_actions.to_csv('Stuffs.csv', index=False)
# print(df_actions)

# =======================================================


import yfinance as yf
import pandas as pd
from datetime import datetime

LEGEND_COMPANIES = {
    "ACS": "ACS.MC",
    "Acciona": "ANA.MC",
    "ArcelorMittal": "MTS.MC",
    "BBVA": "BBVA.MC",
    "Banco Sabadell": "SAB.MC",
    "Banco Santander": "SAN.MC",
    "Bankinter": "BKT.MC",
    "Enagás": "ENG.MC",
    "Ferrovial": "FER.MC",
    "Grifols": "GRF.MC",
    "Iberdrola": "IBE.MC",
    "Inditex": "ITX.MC",
    "Indra Sistemas": "IDR.MC",
    "International Airlines Group": "IAG.MC",
    "Mapfre": "MAP.MC",
    "Naturgy": "NTGY.MC",
    "Red Eléctrica de España": "RED.MC",
    "Repsol": "REP.MC",
    "Telefónica": "TEF.MC"
}

all_data = []

for company, ticker in LEGEND_COMPANIES.items():
    print(f"Processing {company}...")
    stock = yf.Ticker(ticker)
    
    try:
        df = stock.actions
        if df.empty:
            continue
            
        # Filter for data from 2010 onwards
        df = df[df.index >= '2010-01-01']
        
        for date, row in df.iterrows():
            # Identify Bonus Issues: In Spain, scrip dividends often appear in the dividend column.
            # Pure Stock Splits appear in the Stock Splits column.
            if row['Stock Splits'] != 0:
                event = 'Stock Split'
                ratio = row['Stock Splits']
            else:
                event = 'Dividend/Bonus Issue'
                ratio = row['Dividends']

            all_data.append({
                'Company': company,
                'Ticker': ticker,
                'Ex-Date': date.strftime('%Y-%m-%d'),
                'Event Type': event,
                'Value/Ratio': ratio
            })
    except Exception as e:
        print(f"Error for {ticker}: {e}")


full_df = pd.DataFrame(all_data)
splits_df = full_df[full_df['Event Type'] == 'Stock Split']
dividends_df = full_df[full_df['Event Type'] == 'Dividend/Bonus Issue']


with pd.ExcelWriter('SplitsDividend.xlsx') as writer:
    splits_df.to_excel(writer, sheet_name='Splits and Bonus', index=False)
    dividends_df.to_excel(writer, sheet_name='Dividends', index=False)


full_df.to_csv('SplitsDivCombo.csv', index=False)


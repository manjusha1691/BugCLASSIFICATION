import pandas as pd
#Method to Convert data to a usable format

def prepareData(data):
    data['Created'] = pd.to_datetime(data['Created'])
    data['Resolved'] = pd.to_datetime(data['Resolved'])
    data['Issue id'] = data['Issue id'].astype(str)
    data['Synopsis'] = data['Summary'].fillna('').astype(str).str.strip() + ' ' + \
                       data['Description'].fillna('').astype(str).str.strip()
    data['Resolution'].fillna('Later')
    data['Resolved'].fillna(data['Resolved'].max())
    cols_to_drop = ['Summary', 'Description', 'Affects Version/s']
    existing_cols = [col for col in cols_to_drop if col in data.columns]
    data = data.drop(columns=existing_cols, axis=1)
    return data
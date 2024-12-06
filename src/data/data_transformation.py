import pandas as pd
import numpy as np
import os
import sys
from typing import List, Optional, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.pathmanager import PathManager

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        raise CustomException(e, sys)
    
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def process_case_material(df):

    def extract_base(material):
        if material is None:
            return None 
        for base in ['Titanium', 'Platinum', 'Gold', 'Ceramic', 'Steel', 'Bronze', 'Aluminium']:
            if base in material:
                return base
        return None
    to_keep = ['Aluminium', 'Gold', 'Steel', 'Platinum', 'Titanium', 'Bronze', 'G-S Hybrid', 'Ceramic', 'Carbon-based']

    df['Case Material'] = df['Case Material'].fillna('').apply(lambda x: 'G-S Hybrid' if 'Gold' in x and 'Steel' in x else x)
    df['Case Material'] = df['Case Material'].apply(extract_base).fillna(df['Case Material'])
    df['Case Material'] = df['Case Material'].apply(lambda x: 'Carbon-based' if 'Carbo' in x or 'NORTEQ®' in x else None).fillna(df['Case Material'])
    df['Case Material'] = df['Case Material'].apply(lambda x: x if x in to_keep else "Others") 

    return df

def process_strap_material(df):

    def impute_(material):
        for mate in  ['Gold', 'Rubber', 'Leather', 'Titanium', 'Steel', 'Ceramic']:
            if mate in material:
                return mate
        return None

    strap_mapping = {
            'Caoutchouc' : 'Rubber',
            'Sailcloth' : 'Fabric',
            'Nordura' : 'Fabric',
            'Silicon' : 'Silicone',
            'Calfskin' : 'Leather',
            'Cordura®' : 'Fabric',
            'Nato' : 'Nylon',
            'Cotton' : 'Fabric',
            'Textile' : 'Fabric',
            'Velvet' : 'Fabric',
            'Alcantara' : 'Fabric',
        }

    to_keep = ['Satin', 'Gold', 'Steel', 'Farbic', 'Leather', 'Rubber', 'Titanium', 'Nylon', 'Bronze', 'G-S Hybrid', 'Silicone', 'Ceramic']
    df['Strap Material'] = df['Strap Material'].apply(lambda x: 'G-S Hybrid' if 'Gold' in x and 'Steel' in x else x)
    df['Strap Material'] = df['Strap Material'].apply(impute_).fillna(df['Strap Material'])
    df['Strap Material'] = df['Strap Material'].map(strap_mapping).fillna(df['Strap Material'])
    df['Strap Material'] = df['Strap Material'].apply(lambda x: x if x in to_keep else "Others") 

    return df

def process_clasp_type(df):
    def process(type_):
        if 'Deploy' in type_ or 'Foldover' in type_:
            return 'Deployemnt'
        elif 'Folding' in type_:
            return 'Folding'
        elif 'Butterfly' in type_:
            return 'Butterfly'
        elif 'Hook-and-loop' in type_:
            return 'Hook-and-loop'
        elif 'Tang' in type_ or 'Pin' in type_ or 'Ardillon' in type_:
            return 'Tang'

    to_keep = ['Deployment', 'Folding', 'Butterfly', 'Hook-and-loop', 'Tang', 'Tudor “T-fit” Clasp', 'Winged Clasp']

    df['Clasp Type'] = df['Clasp Type'].fillna('').apply(process).fillna(df['Clasp Type'])
    df['Clasp Type'] = df['Clasp Type'].apply(lambda x: x if x in to_keep else "Others") 

    return df

def _impute_with_model(df, column):

    train_data = df[df[column].notnull()]
    test_data = df[df[column].isnull()]

    non_nan_columns = train_data.columns[train_data.notna().all()].tolist()
    numerical_features = train_data[non_nan_columns].select_dtypes(include=['float', 'int']).drop(columns=['price', column], errors='ignore').columns.tolist()
    categorical_features = train_data[non_nan_columns].select_dtypes(include=['object']).columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')
    encoded_train = encoder.fit_transform(train_data[categorical_features])
    encoded_test = encoder.transform(test_data[categorical_features])

    X_train = pd.concat([
        train_data[numerical_features].reset_index(drop=True),
        pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_features))
    ], axis=1)

    X_test = pd.concat([
        test_data[numerical_features].reset_index(drop=True),
        pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_features))
    ], axis=1)

    y_train = train_data[column]

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42, n_estimators=500)
    model.fit(X_train_split, y_train_split)

    print(f"Validation R^2 for {column} imputation: {model.score(X_valid_split, y_valid_split)}")

    test_data_predictions = model.predict(X_test)

    test_data.loc[:, column] = test_data_predictions

    df.loc[test_data.index, column] = test_data[column]

    return df

def clean_data(df):

    df = df.drop(columns=['name', 'Collection', 'Series', 'Calibre', 'Date', 'Chronograph', 'jewels', 'Lug Width', 'Diameter', 'Dial Type', 'Reference', 'Model No', 'EAN',
                'Case Back', 'Frequency', 'Base', 'Power Reserve (hours)', 'Dial Colour', 'Strap Colour', 'Buckle/Clasp Material'])
    
    brand_counts = df['Brand'].value_counts()
    popular_brands = brand_counts[brand_counts >= 25].index 
    df['Brand'] = df['Brand'].apply(lambda x: x if x in popular_brands else "Others")

    glass_counts = df['Glass Material'].value_counts()
    to_keep_glass = glass_counts[glass_counts >= 25].index 
    df['Glass Material'] = df['Glass Material'].apply(lambda x: x if x in to_keep_glass else "Others")

    
    df['Water Resistance (M)'] = df['Water Resistance (M)'].replace('Splash Resistant', 30).astype(float)
    df['Power Reserve'] = df['Power Reserve'].str.extract('(\d+)', expand=False).astype(float)
    df['Jewels'] = df['Jewels'].fillna(0)
    
    df['Bezel'] = df['Bezel'].fillna('No Bezel')
    df['Warranty Period'] = df['Warranty Period'].str.get(0).astype(float)

    df = df[(df['price'] < 15000000)]

    df = df.drop(df[df['Case Thickness'] > 200].index)
    df.loc[df['Water Resistance (M)'] == 6000, 'Water Resistance (M)'] = 600
    df.loc[df['Power Reserve'] > 5000, 'Power Reserve'] /= 1000
    
    return df

def feature_engineer(df):

    precious_stones_df = pd.DataFrame()
    parts = ["Case", "Dial", "Bracelet", "Crown", "Clasp", "Bezel", "Lugs"]
    for part in parts:
        precious_stones_df[f"precious_stone_on_{part}"] = df['Precious Stone'].fillna('').apply(lambda x: 1 if part in str(x) else 0)
    
    string = ', '.join(list(df['Features'].dropna().unique()))
    features = list(set(string.split(', ')))
    features_df = pd.DataFrame()
    for feature in features:
        features_df[f"feature_{feature}"] = df['Features'].fillna('').apply(lambda x: 1 if feature in x else 0)        
    features_df = features_df.loc[:, features_df.apply(lambda col: col.sum() >= 10)]
    
    luminosity_df = pd.DataFrame()
    parts = ['Hands', 'Hour', 'Bezel', 'Dial']
    for part in parts:
        luminosity_df[f"luminosity_on_{part}"] = df['Luminosity'].fillna('').apply(lambda x: 1 if part in x else 0)
        
    df = df.drop(columns=['Luminosity', 'Features', 'Precious Stone'])
    
    df['Case Material Coating'] = df['Case Material'].fillna('').apply(lambda material: next((coating for coating in ['DLC', 'PVD', 'CVD'] if coating in material), None))
    df['Case Material Coating'] = df['Case Material Coating'].fillna('No coating')
    
    df = pd.concat([df, features_df, luminosity_df,precious_stones_df], axis=1)
    
    return df

def clean_material_values(df):
    df = process_case_material(df)
    df = process_strap_material(df)
    df = process_clasp_type(df)
    return df

def remove_unnecessary_col(df):

    features = [col for col in df.columns if 'feature' in col]
    features_df = df[features + ['price']]
    correlation_matrix = features_df.corr()
    price_corr = correlation_matrix['price']
    selected_features = price_corr[price_corr.abs() >= 0.1].index
    features_df = features_df[selected_features]
    
    df = pd.concat([df.drop(columns=features), features_df.drop(columns=['price'], axis=1)], axis=1)
    
    return df

def impute_missing_vals(df):
    df = _impute_with_model(df, 'Case Thickness')
    df = _impute_with_model(df, 'Power Reserve')
    
    knn_imputer = KNNImputer(n_neighbors=1)
    df[['Power Reserve', 'Frequency (bph)']] = knn_imputer.fit_transform(df[['Power Reserve', 'Frequency (bph)']])
    
    df.loc[df['Movement'].isnull(), 'Movement'] = 'Quartz'
    df.loc[df['Water Resistance (M)'].isna(), 'Water Resistance (M)'] = 30.0
    
    df = df.drop(columns=['Hands', 'Indexes', 'Display'])
    return df

def split_data(df):
    try:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=32)
        logging.info("data split into traiin and test ")
        return train_df, test_df
    except Exception as e:
        raise CustomException(e, sys)

def save_data(df, train, test, path_manager):
    try:
        df.to_csv(path_manager.get_processed_data_path(), index=False)
        train.to_csv(path_manager.get_train_data_path(), index=False)
        test.to_csv(path_manager.get_test_data_path(), index=False)

        logging.info("Train and test data saved")
    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        path_manager = PathManager()

        df = load_data(path_manager.get_raw_data_path())

        df = clean_data(df)

        df = feature_engineer(df)
        
        df = clean_material_values(df)
        df = handle_duplicates(df)
        
        df = remove_unnecessary_col(df)
        df = handle_duplicates(df)
        
        df = impute_missing_vals(df)
        df = handle_duplicates(df)

        train, test = split_data(df)
        save_data(df, train, test, path_manager)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
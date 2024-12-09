import pandas as pd
import pytest


@pytest.fixture(scope='module')
def load_processed_data():
    df = pd.read_csv('artifacts/data/processed/processed_data.csv')
    train =  pd.read_csv('artifacts/data/processed/train.csv')
    test =  pd.read_csv('artifacts/data/processed/test.csv')
    return df, train, test

def is_not_empty(load_processed_data):
    df, train, test = load_processed_data

    assert not df.empty, "The processed data (df) is empty!"
    assert not train.empty, "The training dataset (train) is empty!"
    assert not test.empty, "The testing dataset (test) is empty!"


def test_column_existence(load_processed_data):

    required_columns = ['price', 'Brand', 'Movement', 'Case Size', 'Case Thickness',
       'Case Shape', 'Case Material', 'Glass Material', 'Strap Material',
       'Clasp Type', 'Gender', 'Water Resistance (M)', 'Warranty Period',
       'Country of Origin', 'Power Reserve', 'Jewels', 'Interchangeable Strap',
       'Bezel', 'Limited Edition', 'Frequency (bph)', 'Case Material Coating',
       'feature_Day-Night Indicator', 'feature_Month', 'feature_Tourbillon',
       'feature_Dual Time', 'feature_Retrograde', 'feature_Date',
       'feature_Small Seconds', 'feature_Perpetual Calendar', 'feature_Power Reserve Indicator',
       'feature_Year',
       'precious_stone_on_Case', 'precious_stone_on_Dial',
       'precious_stone_on_Bracelet', 'precious_stone_on_Crown',
       'precious_stone_on_Clasp', 'precious_stone_on_Bezel',
       'precious_stone_on_Lugs']
    
    df, train, test = load_processed_data
    for column in required_columns:
        assert column in df.columns, f"Column '{column}' is missing from the main dataset."
        assert column in train.columns, f"Column '{column}' is missing from the train dataset."
        assert column in test.columns, f"Column '{column}' is missing from the test dataset."

def test_no_missing_values(load_processed_data):
    df, train, test = load_processed_data
    for column in df.columns:
        assert not df[column].isnull().any(), f"Missing values found in column '{column}'."
        assert not train[column].isnull().any(), f"Train - Missing values found in column '{column}'."
        assert not test[column].isnull().any(), f"Test - Missing values found in column '{column}'."


def test_valid_price_range(load_processed_data):
    df, train, test = load_processed_data
    assert df['price'].between(100, 15000000).all(), "Price values are out of range."

def test_unique_rows(load_processed_data):
    df, train, test = load_processed_data
    assert df.shape[0] == df.drop_duplicates().shape[0], "Duplicate rows found in the dataset."

def test_valid_categorical_values(load_processed_data):
    df, train, test = load_processed_data

    valid_glass_materials = ['Sapphire Crystal', 'Others', 'Hardlex Crystal', 'Mineral Crystal']
    assert df['Glass Material'].nunique() == len(valid_glass_materials), "Glass material unique values don't match"
    for material in df['Glass Material'].unique():
        assert material in valid_glass_materials, f"Invalid glass material value: {material}"
    
    valid_case_materials = ['Steel', 'gold-steel hybrid', 'Gold', 'Ceramic', 'Carbon-based', 'Titanium', 'Bronze', 'Others', 'Aluminium', 'Platinum']
    assert df['Case Material'].nunique() == len(valid_case_materials), "Case material unique values don't match"
    for material in df['Case Material'].unique():
        assert material in valid_case_materials, f"Invalid case material value: {material}"

    valid_strap_materials = ['Steel', 'gold-steel hybrid', 'Leather', 'Ceramic', 'Rubber', 'Others', 'Nylon', 'Titanium', 'Gold', 'Satin', 'Silicone', 'Bronze']
    assert df['Strap Material'].nunique() == len(valid_strap_materials), "Strap material unique values don't match"
    for material in df['Strap Material'].unique():
        assert material in valid_strap_materials, f"Invalid strap material value: {material}"

    valid_movements = ['Automatic', 'Quartz', 'Manual Winding', 'Spring Drive', 'Kinetic Powered', 'Solar Powered', 'SuperQuartz™']
    assert df['Movement'].nunique() == len(valid_movements), "Movement unique values don't match"
    for movement in df['Movement'].unique():
        assert movement in valid_movements, f"Invalid movement value: {movement}"

    valid_brands = ['Girard-Perregaux', 'Zenith', 'Omega', 'BVLGARI', 'Oris', 'Rado', 'Raymond Weil', 'Frederique Constant', 
                    'Breitling', 'Longines', 'H. Moser & Cie.', 'Arnold & Son', 'Carl F. Bucherer', 'Nomos Glashutte', 
                    'Baume & Mercier', 'Doxa', 'Tissot', 'Parmigiani', 'Maurice Lacroix', 'Seiko', 'Bremont', 'TAG Heuer', 
                    'Favre Leuba', 'Corum', 'Grand Seiko', 'Jaeger-LeCoultre', 'Hublot', 'Louis Erard', 'IWC', 'Others', 
                    'Bell & Ross', 'Maserati', 'Titoni', 'Panerai', 'MeisterSinger', 'Ulysse Nardin', 'Alpina', 'Junghans', 
                    'Zeppelin', 'Louis Moinet', 'Luminox', 'Bovet', 'NORQAIN', 'Jacob & Co.', 'Tutima Glashütte', 'Edox', 
                    'Perrelet', 'Tudor', 'Nivada Grenchen', 'Gerald Charles', 'Eberhard & Co.', 'Mühle-Glashütte', 
                    'Ernest Borel', 'CVSTOS', 'CIGA Design', 'Edouard Koehn']
    assert df['Brand'].nunique() == len(valid_brands), "Brand unique values don't match"
    for brand in df['Brand'].unique():
        assert brand in valid_brands, f"Invalid brand value: {brand}"

    valid_case_shapes = ['Octagon', 'Round', 'Square', 'Cushion', 'Rectangular', 'Oval', 'Tonneau', 'Drop', 'Dodecagon', 'Hexagon', 'Barrel']
    assert df['Case Shape'].nunique() == len(valid_case_shapes), "Case shape unique values don't match"
    for shape in df['Case Shape'].unique():
        assert shape in valid_case_shapes, f"Invalid case shape value: {shape}"

    valid_clasp_types = ['Folding', 'Others', 'Tang', 'Butterfly', 'Winged Clasp', 'Hook-and-loop', 'Tudor “T-fit” Clasp']
    assert df['Clasp Type'].nunique() == len(valid_clasp_types), "Clasp type unique values don't match"
    for clasp in df['Clasp Type'].unique():
        assert clasp in valid_clasp_types, f"Invalid clasp type value: {clasp}"

    valid_genders = ['Unisex', 'Women', 'Men']
    assert df['Gender'].nunique() == len(valid_genders), "Gender unique values don't match"
    for gender in df['Gender'].unique():
        assert gender in valid_genders, f"Invalid gender value: {gender}"

    valid_countries = ['Switzerland', 'Germany', 'Japan', 'England', 'China', 'Netherlands']
    assert df['Country of Origin'].nunique() == len(valid_countries), "Country of origin unique values don't match"
    for country in df['Country of Origin'].unique():
        assert country in valid_countries, f"Invalid country value: {country}"

    binary_feature_columns = [
        'Interchangeable Strap', 'Limited Edition', 'feature_Day-Night Indicator', 
        'feature_Month', 'feature_Tourbillon', 'feature_Dual Time', 'feature_Retrograde', 
        'feature_Date', 'feature_Small Seconds', 'feature_Perpetual Calendar', 
        'feature_Jumping Hours', 'feature_Power Reserve Indicator', 'feature_Year'
    ]

    for col in binary_feature_columns:
        assert set(df[col].unique()).issubset({0, 1}), f"Invalid values in {col}"

    precious_stone_columns = [
        'precious_stone_on_Case', 'precious_stone_on_Dial', 
        'precious_stone_on_Bracelet', 'precious_stone_on_Crown', 
        'precious_stone_on_Clasp', 'precious_stone_on_Bezel', 
        'precious_stone_on_Lugs'
    ]
    for col in precious_stone_columns:
        assert set(df[col].unique()).issubset({0, 1}), f"Invalid values in {col}"
    
def test_valid_water_resistance(load_processed_data):
    df, train, test = load_processed_data
    assert (df['Water Resistance (M)'] <= 2000).all(), "'Water Resistance (M)' has outliers"

def test_no_outliers_in_power_reserve(load_processed_data):
    df, train, test = load_processed_data
    assert (df['Power Reserve'] <= 250).all(), "'Power Reserve' has outliers."
    assert (train['Power Reserve'] <= 250).all(), "Train 'Power Reserve' has outliers."
    assert (test['Power Reserve'] <= 250).all(), "Test 'Power Reserve' has outliers."



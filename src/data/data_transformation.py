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

class PathManager:
    
    def __init__(self):
        self.project_root = self._get_project_root()
        self.artifacts_dir = os.path.join(self.project_root, 'artifacts')
        self.data_raw_dir = os.path.join(self.artifacts_dir, 'data', 'raw')
        self.data_processed_dir = os.path.join(self.artifacts_dir, 'data', 'processed')
        
        os.makedirs(self.data_raw_dir, exist_ok=True)
        os.makedirs(self.data_processed_dir, exist_ok=True)
    
    def _get_project_root(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    def get_raw_data_path(self) -> str:
        return os.path.join(self.data_raw_dir, 'raw_data.csv')
    
    def get_processed_data_path(self) -> str:
        return os.path.join(self.data_processed_dir, 'processed_data.csv')


class WatchDataPreprocessor:

    def __init__(self, input_path: str):

        self.df = pd.read_csv(input_path)
        
    def _extract_coating(self, material: str) -> Optional[str]:

        coatings = ['DLC', 'PVD', 'CVD']
        return next((coating for coating in coatings if coating in material), None)

    def process_case_material(self, df):

        def impute_gold_enhanced_steel(material):
            if 'Gold' in material and 'Steel' in material:
                return 'G-S Hybrid'
            return material 

        def extract_base(material):
            if material is None:
                return None 
            for base in ['Titanium', 'Platinum', 'Gold', 'Ceramic', 'Steel', 'Bronze', 'Aluminium']:
                if base in material:
                    return base
            return None

        def extract_carbon_based(material):
            if material is None:
                return None 
            if 'Carbo' in material or 'NORTEQ®' in material:
                return 'Carbon-based'
            return None

        to_keep = ['Aluminium', 'Gold', 'Steel', 'Platinum', 'Titanium', 'Bronze', 'G-S Hybrid', 'Ceramic', 'Carbon-based']

        df['Case Material'] = df['Case Material'].fillna('').apply(impute_gold_enhanced_steel)
        df['Case Material'] = df['Case Material'].apply(extract_base).fillna(df['Case Material'])
        df['Case Material'] = df['Case Material'].apply(extract_carbon_based).fillna(df['Case Material'])
        df['Case Material'] = df['Case Material'].apply(lambda x: x if x in to_keep else "Others") 
        
        return df
    
    def process_strap_material(self, df):

        def impute_gold_enhanced_steel(material):
            if 'Gold' in material and 'Steel' in material:
                return 'G-S Hybrid'
            return material 

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
        df['Strap Material'] = df['Strap Material'].apply(impute_gold_enhanced_steel)
        df['Strap Material'] = df['Strap Material'].apply(impute_).fillna(df['Strap Material'])
        df['Strap Material'] = df['Strap Material'].map(strap_mapping).fillna(df['Strap Material'])
        df['Strap Material'] = df['Strap Material'].apply(lambda x: x if x in to_keep else "Others") 

        return df
    
    def process_clasp_type(self, df):
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
    
    def clean_data(self) -> pd.DataFrame:
  
        try:
            brand_counts = self.df['Brand'].value_counts()
            popular_brands = brand_counts[brand_counts >= 25].index 
            self.df['Brand'] = self.df['Brand'].apply(lambda x: x if x in popular_brands else "Others")

            glass_counts = self.df['Glass Material'].value_counts()
            to_keep_glass = glass_counts[glass_counts >= 25].index 
            self.df['Glass Material'] = self.df['Glass Material'].apply(lambda x: x if x in to_keep_glass else "Others")

            self.df['Water Resistance (M)'] = (
                self.df['Water Resistance (M)']
                .replace('Splash Resistant', 30)
                .astype(float)
            )
            self.df['Power Reserve'] = self.df['Power Reserve'].str.extract('(\d+)', expand=False).astype(float)
            self.df['Warranty Period'] = self.df['Warranty Period'].str.get(0).astype(float)

            self.df.loc[self.df['Water Resistance (M)'] == 6000, 'Water Resistance (M)'] = 600
            self.df.loc[self.df['Power Reserve'] > 5000, 'Power Reserve'] /= 1000

            self.df = self.df.drop(columns=['name', 'Collection', 'Series', 'Calibre', 'Date', 'Chronograph', 'jewels', 'Lug Width', 'Diameter', 'Dial Type', 'Reference', 'Model No', 'EAN',
                'Case Back', 'Frequency', 'Base', 'Power Reserve (hours)', 'Dial Colour', 'Strap Colour', 'Buckle/Clasp Material', 'Luminosity', 'Hands', 'Indexes', 'Display'])

            logging.info("Data Cleaned!")
            return self.df
        
        except Exception as e:
            raise CustomException(e, sys)

    def feature_engineering(self) -> pd.DataFrame:

        try:
            self.df['Case Material Coating'] = self.df['Case Material'].fillna('').apply(self._extract_coating)

            self.df = self.process_case_material(self.df)
            self.df = self.process_strap_material(self.df)
            self.df = self.process_clasp_type(self.df)
            
            precious_stones_df = pd.DataFrame()
            parts = ["Case", "Dial", "Bracelet", "Crown", "Clasp", "Bezel", "Lugs"]
            for part in parts:
                precious_stones_df[f"precious_stone_on_{part}"] = (
                    self.df['Precious Stone']
                    .fillna('')
                    .apply(lambda x: 1 if part in str(x) else 0)
                )

            features_df = pd.DataFrame()
            string = ', '.join(list(self.df['Features'].dropna().unique()))
            features = list(set(string.split(', ')))

            for feature in features:
                features_df[f"feature_{feature}"] = self. df['Features'].fillna('').apply(lambda x: 1 if feature in x else 0)
            features_df = features_df.loc[:, features_df.apply(lambda col: col.sum() >= 10)]

            self.df = self.df.drop(columns=['Features', 'Precious Stone'])
            self.df = pd.concat([self.df, features_df, precious_stones_df], axis=1)

            logging.info("Feature engineering done!")

            return self.df
        
        except Exception as e:
            raise CustomException(e, sys)

    def impute_missing_values(self) -> pd.DataFrame:

        try:
            self.df['Jewels'] = self.df['Jewels'].fillna(0)
            self.df['Bezel'] = self.df['Bezel'].fillna('No Bezel')
            self.df.loc[self.df['Movement'].isnull(), 'Movement'] = 'Quartz'
            self.df.loc[self.df['Water Resistance (M)'].isna(), 'Water Resistance (M)'] = 30.0
            self.df['Case Material Coating'] = self.df['Case Material Coating'].fillna('No coating')

            self.df = self._impute_with_model(self.df, 'Case Thickness')
            self.df = self._impute_with_model(self.df, 'Power Reserve')

            knn_imputer = KNNImputer(n_neighbors=1)
            columns_to_impute = ['Power Reserve', 'Frequency (bph)']
            imputed_data = knn_imputer.fit_transform(self.df[columns_to_impute])
            self.df[columns_to_impute] = imputed_data

            logging.info("Missing Value imputed!")

            return self.df
        
        except Exception as e:
            raise CustomException(e, sys)

    def _impute_with_model(self, df: pd.DataFrame, column: str) -> pd.DataFrame:

        logging.info(f"Imputing missing values of {column}")

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

        logging.info(f"Validation R^2 for {column} imputation: {model.score(X_valid_split, y_valid_split)}")

        test_data[column] = model.predict(X_test)

        logging.info("Imputing completed")

        return pd.concat([train_data, test_data]).sort_index()

    def remove_unnecessary_features(self):

        features = [col for col in self.df.columns if 'feature' in col]
        features_df = self.df[features]

        features_df = pd.concat([features_df, self.df['price']], axis=1)
        correlation_matrix = features_df.corr()
        price_corr = correlation_matrix['price']
        selected_features = price_corr[price_corr.abs() >= 0.1].index
        features_df = features_df[selected_features]

        self.df = pd.concat([self.df.drop(columns=features), features_df.drop(columns=['price'], axis=1)], axis=1)
        self.df = self.df.drop(columns = ['feature_Triple Time-zone', 'feature_Flying Tourbillon'])

    def process(self) -> pd.DataFrame:

        self.clean_data()
        self.feature_engineering()
        self.impute_missing_values()

        # Final cleanup
        self.df = self.df.drop_duplicates()
        self.df = self.df[self.df['price'] < 15000000]

        self.remove_unnecessary_features()
        self.df = self.df.drop_duplicates()

        return self.df

def main():
    try:
        path_manager = PathManager()

        preprocessor = WatchDataPreprocessor(path_manager.get_raw_data_path())
        processed_df = preprocessor.process()

        processed_df.to_csv(path_manager.get_processed_data_path(), index=False)
        logging.info(f"Processed data saved in {path_manager.get_processed_data_path()}!")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
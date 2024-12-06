import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_resources():
    """Cached resource loading for performance"""
    pipeline = joblib.load("artifacts/models/pipeline.pkl")
    columns = joblib.load("artifacts/models/columns.pkl")
    df = pd.read_csv("artifacts/data/processed/processed_data.csv")
    return pipeline, columns, df

def prepare_categorical_lists(columns):
    """Prepare lists of categorical features"""
    watch_features = [feature.replace("feature_", "") for feature in columns if "feature" in feature]
    luminosity = [lum.replace("luminosity_on_", "") for lum in columns if "luminosity" in lum]
    stone_parts = [part.replace("precious_stone_on_", "") for part in columns if "precious_stone" in part]
    return watch_features, luminosity, stone_parts

def main():
    st.set_page_config(page_title="Luxury Watch Price Predictor", layout="wide")
    
    # Load resources
    pipeline, columns, df = load_resources()
    watch_features, luminosity, stone_parts = prepare_categorical_lists(columns)
    
    # Styling
    st.title("🕰️ Luxury Watch Price Predictor")
    
    # Collect unique categorical values
    categorical_data = {
        'Country of Origin': df['Country of Origin'].unique(),
        'Brand': df['Brand'].unique(),
        'Movement': df['Movement'].unique(),
        'Case Shape': df['Case Shape'].unique(),
        'Case Material': df['Case Material'].unique(),
        'Strap Material': df['Strap Material'].unique(),
        'Glass Material': df['Glass Material'].unique(),
        'Clasp Type': df['Clasp Type'].unique(),
        'Bezel': df['Bezel'].unique(),
        'Gender': df['Gender'].unique(),
        'Case Material Coating': np.sort(df['Case Material Coating'].unique()),
        'Warranty Period': np.sort(df['Warranty Period'].unique()),
        'Frequency (bph)': np.sort(df['Frequency (bph)'].unique()),
        'Water Resistance (M)': np.sort(df['Water Resistance (M)'].unique())
    }
    
    # Layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        selected_country = st.selectbox("🌍 Country", categorical_data['Country of Origin'])
        selected_brand = st.selectbox("🏷️ Brand", categorical_data['Brand'])
        selected_movement = st.selectbox("⚙️ Movement", categorical_data['Movement'])
        selected_case_shape = st.selectbox("📐 Case Shape", categorical_data['Case Shape'])
    
    with col2:
        selected_case_material = st.selectbox("🔧 Case Material", categorical_data['Case Material'])
        selected_strap_material = st.selectbox("🔗 Strap Material", categorical_data['Strap Material'])
        selected_glass_material = st.selectbox("🔍 Glass Material", categorical_data['Glass Material'])
        selected_clasp_type = st.selectbox("🔓 Clasp Type", categorical_data['Clasp Type'])
    
    with col3:
        selected_bezel = st.selectbox("🔘 Bezel", categorical_data['Bezel'])
        selected_gender = st.pills("👤 Gender", categorical_data['Gender'])
        is_limited_edition = st.toggle("🌟 Limited Edition")
        interchangable_strap = st.toggle("🔄 Interchangable Strap")
    
    # Advanced Features
    st.markdown("### 🛠️ Advanced Watch Configurations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selcted_features = st.multiselect("Watch Features", watch_features)
        case_size = st.slider(
            "Case Size (mm)", 
            min_value=round(df['Case Size'].min() - 3, 1),
            max_value=round(df['Case Size'].max() + 3, 1), 
            value=round((df['Case Size'].max() + df['Case Size'].min())/2, 1),
            step=0.1
        )
    
    with col2:
        selcted_luminosity = st.multiselect("Luminosity Locations", luminosity)
        case_thickness = st.slider(
            "Case Thickness (mm)", 
            min_value=round(df['Case Thickness'].min() - 1, 1),
            max_value=round(df['Case Thickness'].max() + 1, 1), 
            value=round((df['Case Thickness'].max() + df['Case Thickness'].min())/2, 1),
            step=0.1
        )
    
    with col3:
        selcted_stones = st.multiselect("Precious Stone Locations", stone_parts)
        jewels = st.slider(
            "No. of Jewels", 
            min_value=round(df['Jewels'].min()),
            max_value=round(df['Jewels'].max()),
            value=round((df['Jewels'].max() + df['Jewels'].min())/2)
        )
    
    # More Advanced Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        power_reserve = st.slider(
            "Power Reserve (hrs)", 
            min_value=round(df['Power Reserve'].min() - 3, 1),
            max_value=round(df['Power Reserve'].max() + 3, 1), 
            value=round((df['Power Reserve'].max() + df['Power Reserve'].min())/2, 1),
            step=0.1
        )
        selected_period = st.pills("Warranty Period", categorical_data['Warranty Period'])


    
    with col2:
        selected_resistance = st.selectbox("Water Resistance (M)", categorical_data['Water Resistance (M)'])

        frequency = st.pills("Frequency (bph)", categorical_data['Frequency (bph)'])
    
    with col3:
        selected_coat = st.pills("Case Material Coating", categorical_data['Case Material Coating'])
    
    # Prepare binary encodings
    selected_features_binary = [1 if feature in selcted_features else 0 for feature in watch_features]
    selected_luminosity_binary = [1 if lum in selcted_luminosity else 0 for lum in luminosity]
    selected_stones_binary = [1 if stone in selcted_stones else 0 for stone in stone_parts]
    
    is_limited_edition = 1 if is_limited_edition else 0
    interchangable_strap = 1 if interchangable_strap else 0
    
    if (selected_country 
    and selected_gender
    and frequency
    and selected_coat
    and selected_period):
        if st.button("💰 Predict Watch Price", type="primary"):
            feature_vector = [
                selected_brand, selected_movement, case_size, case_thickness, selected_case_shape, 
                selected_case_material, selected_glass_material, selected_strap_material,
                selected_clasp_type, selected_gender, selected_resistance, selected_period, 
                selected_country, power_reserve, jewels, interchangable_strap, selected_bezel, 
                is_limited_edition, frequency, selected_coat,
                *selected_luminosity_binary, *selected_features_binary, *selected_stones_binary
            ]
            
            model_input = pd.DataFrame([feature_vector], columns=columns)
            prediction = pipeline.predict(model_input)
            prediction = np.expm1(prediction[0])
            
            st.success(f"🏷️ Predicted Watch Price: ₹{prediction:,.2f}")

if __name__ == "__main__":
    main()
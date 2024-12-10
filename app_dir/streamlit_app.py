import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Luxury Watch Price Predictor", 
    layout="wide",
    page_icon="🕰️",
    initial_sidebar_state="collapsed",
)

def add_sidebar():
    with st.sidebar:


            
        st.markdown("## 🕰️ Luxury Watch Price Predictor")
        try:
            st.image("/app/app_dir/static/watch.jpg", width=350)
        except:
            pass
        st.markdown("### 📊 Project Overview")
        st.markdown("""
        
        This is an **end-to-end machine learning project** that predicts the prices of luxury watches 
        based on various features and configurations. The project encompasses all stages of the ML lifecycle, 
        including data ingestion, transformation, model training, evaluation, and deployment. 
                    
        Whether you are a watch enthusiast or a collector, this tool provides price estimates for luxury pieces.

        """)
        st.markdown("[![GitHub](https://img.shields.io/badge/Project_Repo-black?logo=github)](https://github.com/abhishekmaher6699/End-to-end-Watch-price-prediction)")

        st.markdown("### 🛠️ Technical Details")
        st.markdown("""
        - **Data Source**: Scraped luxury watch data from a watch website, stored and fetched from a S3 bucket.
        - **Pipeline Tracking**: MLFlow and DVC are used to track the pipeline stages and version control.
        - **Deployment**: The application is Dockerized and deployed on AWS Cloud.
        - **CI/CD**: A robust CI/CD pipeline automates the workflow, including pipeline execution, tests, app creation, and deployment.
        - **App**: A Streamlit-based web application for luxury watch price prediction.
        """)

        st.markdown("### Project Creator")
        st.markdown("""
                **Abhishek Maher**
        """)

        st.markdown("### 🤝 Connect with me")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(" [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/abhishek-maher-yo/)")
        with col2:
            st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/abhishekmaher6699)")
            
        with col3:
            st.markdown("[![Twitter](https://img.shields.io/badge/X-black?logo=X)](https://x.com/maherabhishek69)")

        st.markdown("### 📋 Project Details")
        with st.expander("About the Model"):
            st.write("""
            After extensive experimentation with various machine learning models, the **CatBoostRegressor** 
            was selected for this project. It demonstrated the best performance, achieving the lowest Mean Squared Error (MSE) 
            and the highest R² score during evaluation. The model's ability to handle categorical features efficiently 
            and its robustness against overfitting made it an ideal choice for predicting luxury watch prices.
            """)


add_sidebar()

@st.cache_resource
def load_resources():
    pipeline = joblib.load("pipeline.pkl")
    columns = joblib.load("columns.pkl")
    df = pd.read_csv("processed_data.csv")
    return pipeline, columns, df

def prepare_categorical_lists(columns):
    watch_features = [feature.replace("feature_", "") for feature in columns if "feature" in feature]
    luminosity = [lum.replace("luminosity_on_", "") for lum in columns if "luminosity" in lum]
    stone_parts = [part.replace("precious_stone_on_", "") for part in columns if "precious_stone" in part]
    return watch_features, luminosity, stone_parts

def main():
    
    pipeline, columns, df = load_resources()
    watch_features, luminosity, stone_parts = prepare_categorical_lists(columns)
    
    st.title("🕰️ Luxury Watch Price Predictor")
    
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
        selected_resistance = st.selectbox("Water Resistance (M)", categorical_data['Water Resistance (M)'])
        selected_gender = st.pills("👤 Gender", categorical_data['Gender'])
        is_limited_edition = st.toggle("🌟 Limited Edition")
        interchangable_strap = st.toggle("🔄 Interchangable Strap")
    
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        frequency = st.pills("Frequency (bph)", categorical_data['Frequency (bph)'])
    


    
    with col2:
        power_reserve = st.slider(
            "Power Reserve (hrs)", 
            min_value=round(df['Power Reserve'].min() - 3, 1),
            max_value=round(df['Power Reserve'].max() + 3, 1), 
            value=round((df['Power Reserve'].max() + df['Power Reserve'].min())/2, 1),
            step=0.1
        )
        selected_period = st.pills("Warranty Period", categorical_data['Warranty Period'])
    
    with col3:
        selected_coat = st.pills("Case Material Coating", categorical_data['Case Material Coating'])
    
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
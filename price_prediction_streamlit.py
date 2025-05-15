import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Loading and preparing data from all sheets
def load_data():
    brand_data = {
        'Brand': ['Carbonix Urban', 'Carbonix Evo', 'Carbonix Terra', 'Carbonix Terra+', 'Lyvo Ascend', 'Lyvo Pulse', 
                 'Summitra', 'Summitra Pro', 'Driftor', 'Driftor Light', 'Lumina Edge', 'Aero Fox', 'Mount Fox', 
                 'VENTO', 'VIVA', 'OUI Speed', 'OUI Comfort', 'OUI Cycle Basic', 'OUI Adventure', 'OUI Explore', 
                 'TriTan', 'TriPulse', 'TriTan+', 'Atalanta', 'Helios'],
        'Company': ['Carbonix', 'Carbonix', 'Carbonix', 'Carbonix', 'Lyvo', 'Lyvo', 'Ascentra Bikes', 'Ascentra Bikes', 
                   'Ascentra Bikes', 'Ascentra Bikes', 'Ascentra Bikes', 'Fox Line', 'Fox Line', 'PRINTED', 'PRINTED', 
                   'OUI Cycle', 'OUI Cycle', 'OUI Cycle', 'OUI Cycle', 'OUI Cycle', '3Cycle', '3Cycle', '3Cycle', 
                   'C-NYX', 'C-NYX'],
        'City': ['New York City', 'New York City', 'New York City', 'New York City', 'New York City', 'New York City', 
                'New York City', 'New York City', 'New York City', 'New York City', 'New York City', 'New York City', 
                'New York City', 'Amsterdam', 'Amsterdam', 'Amsterdam', 'Amsterdam', 'Amsterdam', 'Amsterdam', 
                'Amsterdam', 'Bangalore', 'Bangalore', 'Bangalore', 'Bangalore', 'Bangalore'],
        'Recreation': [149, 0, 4, 7, 6, 0, 11, 2, 39, 87, 0, 0, 5, 0, 182, 136, 136, 66, 4, 7, 80, 5, 78, 0, 136],
        'Mountain': [0, 0, 29, 86, 60, 0, 11, 37, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 92, 58, 0, 64, 0, 0, 0],
        'Speed': [0, 233, 0, 0, 0, 80, 0, 0, 0, 0, 183, 145, 0, 142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 153, 0],
        'Price': [1049, 1549, 1149, 1299, 1359, 1579, 1350, 1365, 1100, 1000, 1600, 1599, 1399, 1549, 1049, 
                 960, 900, 800, 1150, 1000, 1000, 1350, 1100, 1449, 999],
        'Rebate': [50, 0, 50, 0, 100, 0, 50, 0, 0, 70, 0, 0, 0, 0, 80, 0, 0, 0, 0, 0, 50, 0, 50, 0, 50],
        'Priority': [3, 1, 4, 2, 1, 2, 2, 1, 4, 3, 5, 1, 2, 1, 2, 1, 2, 3, 4, 5, 2, 1, 3, 1, 2],
        'Ad_Recreation': [60, 19, 32, 59, 30, 10, 50, 23, 59, 59, 0, 27, 36, 3, 59, 39, 64, 37, 32, 26, 76, 33, np.nan, 14, 76],
        'Ad_Mountain': [20, 20, 62, 47, 78, 35, 42, 63, 27, 25, 17, 38, 57, 17, 24, 61, 30, 37, 74, 29, 41, 78, np.nan, 11, 44],
        'Ad_Speed': [16, 63, 16, 35, 36, 74, 36, 34, 36, 24, 60, 62, 45, 55, 26, 63, 24, 37, 29, 26, 35, 37, np.nan, 71, 39],
        'Brand_Recreation': [70, 9, 41, 50, 43, 17, 55, 40, 61, 66, 12, 7, 43, 9, 75, 74, 73, 64, 43, 45, 69, 48, 74, 7, 72],
        'Brand_Mountain': [1, 1, 50, 57, 52, 1, 39, 45, 1, 1, 1, 1, 63, 1, 1, 8, 1, 1, 63, 60, 1, 57, 1, 1, 1],
        'Brand_Speed': [1, 72, 1, 1, 1, 65, 1, 1, 1, 1, 67, 74, 1, 72, 1, 1, 1, 1, 1, 1, 1, 1, 1, 74, 1]
    }
    df_brand = pd.DataFrame(brand_data)

    media_data = {
        'Company': ['3Cycle', 'Carbonix', 'C-NYX', 'PRINTED', 'Lyvo', 'Ascentra Bikes', 'Fox Line', 'OUI Cycle'],
        'Satisfaction': [67.0, 67.7, 68.9, 73.3, 72.8, 70.5, 71.3, 67.4],
        'Number of Media Placements': [10, 10, 9, 6, 13, 11, 9, 7]
    }
    df_media = pd.DataFrame(media_data)

    salesforce_data = {
        'City': ['New York City', 'New York City', 'New York City', 'New York City', 'Amsterdam', 'Amsterdam', 
                'Bangalore', 'Bangalore'],
        'Company': ['Carbonix', 'Lyvo', 'Ascentra Bikes', 'Fox Line', 'PRINTED', 'OUI Cycle', '3Cycle', 'C-NYX'],
        'Total': [6, 6, 7, 7, 7, 6, 5, 6],
        'Service': [1, 1, 1, 1, 2, 0, 1, 1],
        'Sales_Recreation': [1, 0, 1, 0, 2, 4, 2, 2],
        'Sales_Mountain': [2, 3, 3, 2, 0, 2, 2, 0],
        'Sales_Speed': [2, 2, 2, 4, 2, 0, 0, 3]
    }
    df_salesforce = pd.DataFrame(salesforce_data)

    df = df_brand.merge(df_media, on='Company', how='left')
    df = df.merge(df_salesforce, on=['Company', 'City'], how='left')
    return df

# Preprocessing the data
def preprocess_data(df):
    features = ['Recreation', 'Mountain', 'Speed', 'Rebate', 'Priority', 
                'Ad_Recreation', 'Ad_Mountain', 'Ad_Speed', 
                'Brand_Recreation', 'Brand_Mountain', 'Brand_Speed',
                'Satisfaction', 'Number of Media Placements',
                'Total', 'Service', 'Sales_Recreation', 'Sales_Mountain', 'Sales_Speed']
    
    df[features] = df[features].fillna(df[features].mean())
    
    X = df[features]
    y = df['Price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, features

# Training the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mse, r2

# Predicting prices for new data
def predict_price(model, scaler, features, new_data):
    new_data_df = pd.DataFrame([new_data], columns=features)
    new_data_scaled = scaler.transform(new_data_df)
    predicted_price = model.predict(new_data_scaled)
    return predicted_price[0]

# Streamlit app
def main():
    st.title("Bicycle Price Prediction")
    st.write("Predict bicycle prices using Random Forest based on demand, marketing, and salesforce data.")

    # Load and preprocess data
    df = load_data()
    X_scaled, y, scaler, features = preprocess_data(df)
    
    # Train model
    model, X_test, y_test, y_pred, mse, r2 = train_model(X_scaled, y)
    
    # Display model performance
    st.subheader("Model Performance")
    st.write(f"**Mean Squared Error**: {mse:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")

    # Feature importance plot
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(importance, x='Importance', y='Feature', orientation='h',
                           title='Feature Importance', height=600)
    st.plotly_chart(fig_importance)

    # Actual vs Predicted plot
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                 name='Predictions', marker=dict(size=10)))
    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                 y=[y_test.min(), y_test.max()],
                                 mode='lines', name='Ideal', line=dict(dash='dash')))
    fig_pred.update_layout(title='Actual vs Predicted Prices',
                          xaxis_title='Actual Price ($)',
                          yaxis_title='Predicted Price ($)',
                          height=500)
    st.plotly_chart(fig_pred)

    # User input for prediction
    st.subheader("Predict Price for a New Bike")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recreation = st.number_input("Recreation Demand", min_value=0, max_value=200, value=100)
            mountain = st.number_input("Mountain Demand", min_value=0, max_value=200, value=0)
            speed = st.number_input("Speed Demand", min_value=0, max_value=200, value=0)
            rebate = st.number_input("Rebate ($)", min_value=0, max_value=200, value=50)
            priority = st.number_input("Priority", min_value=1, max_value=5, value=2)
            satisfaction = st.number_input("Satisfaction", min_value=0.0, max_value=100.0, value=70.0)

        with col2:
            ad_recreation = st.number_input("Ad Judgement - Recreation", min_value=0, max_value=100, value=60)
            ad_mountain = st.number_input("Ad Judgement - Mountain", min_value=0, max_value=100, value=20)
            ad_speed = st.number_input("Ad Judgement - Speed", min_value=0, max_value=100, value=15)
            brand_recreation = st.number_input("Brand Judgement - Recreation", min_value=0, max_value=100, value=70)
            brand_mountain = st.number_input("Brand Judgement - Mountain", min_value=0, max_value=100, value=1)
            brand_speed = st.number_input("Brand Judgement - Speed", min_value=0, max_value=100, value=1)

        with col3:
            media_placements = st.number_input("Number of Media Placements", min_value=0, max_value=20, value=10)
            total_sales = st.number_input("Total Sales/Service People", min_value=0, max_value=10, value=6)
            service = st.number_input("Service People", min_value=0, max_value=5, value=1)
            sales_recreation = st.number_input("Sales People - Recreation", min_value=0, max_value=5, value=2)
            sales_mountain = st.number_input("Sales People - Mountain", min_value=0, max_value=5, value=2)
            sales_speed = st.number_input("Sales People - Speed", min_value=0, max_value=5, value=1)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        new_bike = {
            'Recreation': recreation,
            'Mountain': mountain,
            'Speed': speed,
            'Rebate': rebate,
            'Priority': priority,
            'Ad_Recreation': ad_recreation,
            'Ad_Mountain': ad_mountain,
            'Ad_Speed': ad_speed,
            'Brand_Recreation': brand_recreation,
            'Brand_Mountain': brand_mountain,
            'Brand_Speed': brand_speed,
            'Satisfaction': satisfaction,
            'Number of Media Placements': media_placements,
            'Total': total_sales,
            'Service': service,
            'Sales_Recreation': sales_recreation,
            'Sales_Mountain': sales_mountain,
            'Sales_Speed': sales_speed
        }
        
        predicted_price = predict_price(model, scaler, features, new_bike)
        st.success(f"**Predicted Price**: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
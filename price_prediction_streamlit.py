import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

# Setting page configuration
st.set_page_config(page_title="Bike Price Predictor", layout="wide")

# Defining segment-specific columns
SEGMENT_COLUMNS = {
    'Recreation': [
        'Brand', 'Company', 'City', 'Recreation', 'Price', 'Rebate', 'Priority',
        'Brand Judgement - Recreation', 'Ad Judgement - Recreation'
    ],
    'Mountain': [
        'Brand', 'Company', 'City', 'Mountain', 'Price', 'Rebate', 'Priority',
        'Brand Judgement - Mountain', 'Ad Judgement - Mountain'
    ],
    'Speed': [
        'Brand', 'Company', 'City', 'Speed', 'Price', 'Rebate', 'Priority',
        'Brand Judgement - Speed', 'Ad Judgement - Speed'
    ]
}

# Function to normalize column names
def normalize_columns(df):
    column_mapping = {}
    for col in df.columns:
        new_col = ' '.join(str(col).split())  # Remove newlines and extra whitespace
        if 'Judegement' in new_col or 'judgement' in new_col:
            new_col = new_col.replace('Judegement', 'Judgement').replace('judgement', 'Judgement')
        if 'Total Sales and Service People' in new_col or 'totalsalesandservicepeople' in new_col.lower().replace(' ', ''):
            new_col = 'Total Sales and Service People'
        column_mapping[col] = new_col
    df = df.rename(columns=column_mapping)
    return df

# Function to load and process data
@st.cache_data
def load_and_process_data():
    files = [
        'Q2.xlsx',
        'Q3 r.xlsx',
        'Q4.csv'
    ]
    demand_dfs = []
    workforce_dfs = []
    salesforce_dfs = []

    for file in files:
        if not os.path.exists(file):
            st.error(f"File not found: {file}")
            continue
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
                df = normalize_columns(df)
                demand_dfs.append(df)
                st.write(f"Loaded CSV: {file}, columns: {list(df.columns)}")
            else:
                xl = pd.ExcelFile(file)
                for sheet in xl.sheet_names:
                    # Special handling for Q2 Detailed Brand Demand
                    header_row = 1 if file.endswith('Q2 (1).xlsx') and sheet == 'Detailed Brand Demand' else 0
                    df = pd.read_excel(file, sheet_name=sheet, header=header_row)
                    df = normalize_columns(df)
                    st.write(f"Loaded sheet: {sheet} from {file}, columns: {list(df.columns)}")
                    # Check for Salesforce sheet
                    if 'Total Sales and Service People' in df.columns or 'totalsalesandservicepeople' in [col.lower().replace(' ', '') for col in df.columns]:
                        salesforce_dfs.append(df)
                    elif 'Salary' in df.columns or 'Total Yearly Cost' in df.columns:
                        workforce_dfs.append(df)
                    else:
                        demand_dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")

    # Debug workforce_dfs
    if workforce_dfs:
        st.write("Workforce DataFrames loaded:", [{"columns": list(df.columns), "rows": len(df)} for df in workforce_dfs])
    else:
        st.warning("No workforce DataFrames loaded.")

    # Combining demand data
    if demand_dfs:
        combined_demand = pd.concat([df for df in demand_dfs if not df.empty], ignore_index=True)
        combined_demand = combined_demand.replace('', np.nan)
        combined_demand = combined_demand.dropna(subset=['Brand', 'Company', 'City', 'Price'])
        st.write("Combined demand columns:", list(combined_demand.columns))
        st.write(f"Combined demand rows: {len(combined_demand)}")
    else:
        st.error("No demand data loaded.")
        return pd.DataFrame()

    # Combining workforce data
    if workforce_dfs:
        combined_workforce = pd.concat([df for df in workforce_dfs if not df.empty], ignore_index=True)
        combined_workforce = combined_workforce.replace('', np.nan)
        st.write("Combined workforce columns before dropna:", list(combined_workforce.columns))
        st.write(f"Combined workforce rows before dropna: {len(combined_workforce)}")
        # Relaxed dropna
        required_cols = ['Company', 'Total Yearly Cost']
        available_cols = [col for col in required_cols if col in combined_workforce.columns]
        if available_cols:
            combined_workforce = combined_workforce.dropna(subset=available_cols)
            # Fill missing Satisfaction and Number of Media Placements
            if 'Satisfaction' in combined_workforce.columns:
                combined_workforce['Satisfaction'] = combined_workforce['Satisfaction'].fillna(combined_workforce['Satisfaction'].median())
            if 'Number of Media Placements' in combined_workforce.columns:
                combined_workforce['Number of Media Placements'] = combined_workforce['Number of Media Placements'].fillna(combined_workforce['Number of Media Placements'].median())
        else:
            st.warning("No valid workforce columns found.")
            combined_workforce = pd.DataFrame(columns=['Company', 'Total Yearly Cost', 'Satisfaction', 'Number of Media Placements'])
        st.write(f"Combined workforce rows after dropna: {len(combined_workforce)}")
    else:
        st.warning("No workforce data loaded.")
        combined_workforce = pd.DataFrame(columns=['Company', 'Total Yearly Cost', 'Satisfaction', 'Number of Media Placements'])

    # Combining salesforce data
    if salesforce_dfs:
        combined_salesforce = pd.concat([df for df in salesforce_dfs if not df.empty], ignore_index=True)
        combined_salesforce = combined_salesforce.replace('', np.nan)
        combined_salesforce = normalize_columns(combined_salesforce)
        st.write("Combined salesforce columns:", list(combined_salesforce.columns))
        st.write(f"Combined salesforce rows: {len(combined_salesforce)}")
        if 'Total Sales and Service People' in combined_salesforce.columns:
            combined_salesforce = combined_salesforce.dropna(subset=['Company', 'City', 'Total Sales and Service People'])
        # Rename segment columns to avoid merge conflict
        combined_salesforce = combined_salesforce.rename(columns={
            'Recreation': 'Recreation_salesforce',
            'Mountain': 'Mountain_salesforce',
            'Speed': 'Speed_salesforce'
        })
    else:
        st.warning("No salesforce data loaded.")
        combined_salesforce = pd.DataFrame(columns=['Company', 'City', 'Total Sales and Service People', 'Service', 'Recreation_salesforce', 'Mountain_salesforce', 'Speed_salesforce'])

    # Merging demand with workforce data on Company
    combined_df = combined_demand.merge(
        combined_workforce[['Company', 'Total Yearly Cost', 'Satisfaction', 'Number of Media Placements']],
        on='Company',
        how='left'
    )
    st.write(f"Rows after workforce merge: {len(combined_df)}")

    # Merging with salesforce data on Company and City
    combined_df = combined_df.merge(
        combined_salesforce[['Company', 'City', 'Total Sales and Service People', 'Service', 'Recreation_salesforce', 'Mountain_salesforce', 'Speed_salesforce']],
        on=['Company', 'City'],
        how='left'
    )
    st.write(f"Rows after salesforce merge: {len(combined_df)}")

    # Fill missing salesforce data
    for col in ['Total Sales and Service People', 'Service', 'Recreation_salesforce', 'Mountain_salesforce', 'Speed_salesforce']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna(0)

    # Relaxed dropna
    combined_df = combined_df.dropna(subset=['Total Yearly Cost'])
    st.write(f"Rows after final dropna: {len(combined_df)}")
    st.write("Available columns in combined DataFrame:", list(combined_df.columns))

    return combined_df

# Function to filter data for a specific segment
def filter_segment_data(df, segment):
    columns = SEGMENT_COLUMNS[segment] + [
        'Total Yearly Cost', 'Satisfaction', 'Number of Media Placements',
        'Total Sales and Service People', 'Service', f'{segment}_salesforce'
    ]
    segment_col = segment  # Recreation, Mountain, or Speed

    # Checking if all required columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for {segment} segment: {missing_cols}")
        return pd.DataFrame()

    # Filtering rows where segment-specific demand is non-zero
    filtered_df = df[df[segment_col] > 0][columns].dropna()

    # Renaming columns to be segment-agnostic
    filtered_df = filtered_df.rename(columns={
        f'Brand Judgement - {segment}': 'Brand_Judgement',
        f'Ad Judgement - {segment}': 'Ad_Judgement',
        segment_col: 'Demand',
        f'{segment}_salesforce': 'Salesforce_Allocation'
    })

    return filtered_df

# Function to train model and compute metrics
@st.cache_resource
def train_model(segment, df):
    X = df.drop(['Price', 'Brand', 'Company', 'City'], axis=1)
    y = df['Price']

    # Encoding categorical variables
    le_city = LabelEncoder()
    df['City'] = le_city.fit_transform(df['City'])

    X = df.drop(['Price', 'Brand', 'Company'], axis=1)
    feature_names = X.columns

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicting on test set
    y_pred = model.predict(X_test)

    # Calculating RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Getting feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    return model, le_city, rmse, feature_importance, y_test, y_pred

# Function to create visualizations
def create_visualizations(rmse, feature_importance, y_test, y_pred):
    # Feature Importance Bar Plot
    fig1 = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        title='Feature Importance',
        orientation='h',
        text='Importance',
        text_auto='.3f'
    )
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'})

    # Actual vs Predicted Scatter Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=8)
    ))
    fig2.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Ideal Fit',
        line=dict(color='red', dash='dash')
    ))
    fig2.update_layout(
        title='Actual vs Predicted Prices',
        xaxis_title='Actual Price ($)',
        yaxis_title='Predicted Price ($)',
        showlegend=True
    )

    # Error Histogram
    errors = y_test - y_pred
    fig3 = px.histogram(
        x=errors,
        nbins=30,
        title='Distribution of Prediction Errors',
        labels={'x': 'Prediction Error ($)'}
    )

    return fig1, fig2, fig3

# Main app
def main():
    st.title("Bike Price Predictor")

    # Loading data
    with st.spinner("Loading data..."):
        df = load_and_process_data()

    if df.empty:
        st.error("No valid data loaded. Please check input files.")
        return

    # Segment selector
    segment = st.selectbox("Select Segment", ["Recreation", "Mountain", "Speed"])

    # Filtering data for selected segment
    segment_df = filter_segment_data(df, segment)

    if segment_df.empty:
        st.error(f"No valid data available for {segment} segment.")
        return

    # Getting unique brands, companies, and cities
    brands = sorted(segment_df['Brand'].unique())
    companies = sorted(segment_df['Company'].unique())
    cities = sorted(segment_df['City'].unique())

    # Creating columns for input widgets
    st.subheader("Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Bike Details")
        selected_brand = st.selectbox("Select Brand", brands)
        selected_company = st.selectbox("Select Company", companies)
        selected_city = st.selectbox("Select City", cities)

        # Filtering for valid brand-company-city combinations
        valid_combinations = segment_df[
            (segment_df['Brand'] == selected_brand) & 
            (segment_df['Company'] == selected_company) &
            (segment_df['City'] == selected_city)
        ]
        if valid_combinations.empty:
            st.warning(f"Invalid combination: {selected_brand}, {selected_company}, {selected_city} in {segment} segment.")
            return

    with col2:
        st.write("### Market Metrics")
        demand = st.number_input("Demand", min_value=0, value=100, step=10)
        rebate = st.number_input("Rebate ($)", min_value=0, max_value=500, value=0, step=10)
        priority = st.number_input("Priority", min_value=1, max_value=10, value=1, step=1)
        brand_judgement = st.number_input("Brand Judgement", min_value=0, max_value=100, value=50, step=1)
        ad_judgement = st.number_input("Ad Judgement", min_value=0, max_value=100, value=50, step=1)

    with col3:
        st.write("### Company & Salesforce Metrics")
        total_yearly_cost = df[df['Company'] == selected_company]['Total Yearly Cost'].iloc[0] if not df[df['Company'] == selected_company].empty else 0
        st.write(f"Total Yearly Cost for {selected_company}: ${total_yearly_cost:,.2f}")
        satisfaction = st.number_input("Satisfaction", min_value=0, max_value=100, value=70, step=1)
        media_placements = st.number_input("Number of Media Placements", min_value=0, value=10, step=1)
        total_sales_people = st.number_input("Total Sales and Service People", min_value=0, value=6, step=1)
        service = st.number_input("Service Personnel", min_value=0, value=1, step=1)
        salesforce_allocation = st.number_input(f"{segment} Salesforce Allocation", min_value=0, value=2, step=1)

    # Training model and computing metrics
    with st.spinner("Training model..."):
        model, le_city, rmse, feature_importance, y_test, y_pred = train_model(segment, segment_df)

    # Preparing input for prediction
    try:
        city_encoded = le_city.transform([selected_city])[0]
        input_data = pd.DataFrame({
            'City': [city_encoded],
            'Demand': [demand],
            'Rebate': [rebate],
            'Priority': [priority],
            'Brand_Judgement': [brand_judgement],
            'Ad_Judgement': [ad_judgement],
            'Total Yearly Cost': [total_yearly_cost],
            'Satisfaction': [satisfaction],
            'Number of Media Placements': [media_placements],
            'Total Sales and Service People': [total_sales_people],
            'Service': [service],
            'Salesforce_Allocation': [salesforce_allocation]
        })

        # Making prediction
        prediction = model.predict(input_data)[0]

        # Displaying prediction
        st.subheader("Predicted Price")
        st.metric("Price ($)", f"{prediction:.2f}")

        # Displaying model performance
        st.subheader("Model Performance")
        st.metric("RMSE ($)", f"{rmse:.2f}")

        # Displaying feature importance
        st.subheader("Feature Importance")
        st.dataframe(feature_importance.style.format({"Importance": "{:.3f}"}))

        # Creating and displaying visualizations
        st.subheader("Visualizations")
        fig1, fig2, fig3 = create_visualizations(rmse, feature_importance, y_test, y_pred)
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()

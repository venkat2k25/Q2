import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from scipy import stats
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
        new_col = ' '.join(str(col).split()).strip()
        new_col = new_col.replace('Judegement', 'Judgement').replace('judgement', 'Judgement').replace('Judgment', 'Judgement')
        if 'Total Sales and Service People' in new_col.lower().replace(' ', '') or 'totalsalesandservicepeoplee' in new_col.lower().replace(' ', ''):
            new_col = 'Total Sales and Service People'
        column_mapping[col] = new_col
    return df.rename(columns=column_mapping)

# Function to normalize brand names
def normalize_brand_name(brand):
    if isinstance(brand, str):
        brand = ' '.join(brand.split()).replace(' +', '+').replace('+ ', '+').strip()
        return brand.title()
    return brand

# Function to load and process data
@st.cache_data
def load_and_process_data():
    files = [
        'C:\\Users\\byven\\Downloads\\Q2 (1).xlsx',
        'C:\\Users\\byven\\Downloads\\Q3 r (1).xlsx',
        'C:\\Users\\byven\\Downloads\\Q4.xlsx'
    ]
    demand_dfs = []
    workforce_dfs = []
    salesforce_dfs = []

    for file in files:
        if not os.path.exists(file):
            st.error(f"File not found: {file}")
            continue
        try:
            xl = pd.ExcelFile(file)
            quarter = file.split('.')[0].split('Q')[1]
            for sheet in xl.sheet_names:
                header_row = 1 if file == 'C:\\Users\\byven\\Downloads\\Q2 (1).xlsx' and sheet == 'Detailed Brand Demand' else 0
                df = pd.read_excel(file, sheet_name=sheet, header=header_row)
                df = normalize_columns(df)
                df['Quarter'] = f'Q{quarter}'
                if 'Brand' in df.columns:
                    df['Brand'] = df['Brand'].apply(normalize_brand_name)
                if 'Total Sales and Service People' in df.columns:
                    salesforce_dfs.append(df)
                elif 'Salary' in df.columns or 'Total Yearly Cost' in df.columns:
                    workforce_dfs.append(df)
                else:
                    demand_dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")

    if demand_dfs:
        combined_demand = pd.concat(demand_dfs, ignore_index=True)
        combined_demand = combined_demand.replace('', np.nan)
        if 'Brand' in combined_demand.columns:
            combined_demand['Brand'] = combined_demand['Brand'].apply(normalize_brand_name)
        duplicates = combined_demand[combined_demand.duplicated(subset=['Brand', 'Company', 'Quarter', 'City'], keep=False)]
        if not duplicates.empty:
            st.warning(f"Found {len(duplicates)} duplicate Brand-Company-Quarter-City pairs.")
            combined_demand = combined_demand.drop_duplicates(subset=['Brand', 'Company', 'Quarter', 'City'], keep='first')
        combined_demand = combined_demand.dropna(subset=['Brand', 'Company', 'City', 'Price', 'Quarter'])
    else:
        st.error("No demand data loaded.")
        return pd.DataFrame()

    if workforce_dfs:
        combined_workforce = pd.concat(workforce_dfs, ignore_index=True)
        combined_workforce = combined_workforce.replace('', np.nan)
        required_cols = ['Company', 'Total Yearly Cost']
        combined_workforce = combined_workforce.dropna(subset=required_cols)
        combined_workforce['Satisfaction'] = combined_workforce['Satisfaction'].fillna(combined_workforce['Satisfaction'].median())
        combined_workforce['Number of Media Placements'] = combined_workforce['Number of Media Placements'].fillna(combined_workforce['Number of Media Placements'].median())
    else:
        st.warning("No workforce data loaded.")
        combined_workforce = pd.DataFrame(columns=['Company', 'Total Yearly Cost', 'Satisfaction', 'Number of Media Placements'])

    if salesforce_dfs:
        combined_salesforce = pd.concat(salesforce_dfs, ignore_index=True)
        combined_salesforce = combined_salesforce.replace('', np.nan)
        combined_salesforce = normalize_columns(combined_salesforce)
        combined_salesforce = combined_salesforce.dropna(subset=['Company', 'City', 'Total Sales and Service People'])
        combined_salesforce = combined_salesforce.rename(columns={
            'Recreation': 'Recreation_salesforce',
            'Mountain': 'Mountain_salesforce',
            'Speed': 'Speed_salesforce'
        })
    else:
        st.warning("No salesforce data loaded.")
        combined_salesforce = pd.DataFrame(columns=['Company', 'City', 'Total Sales and Service People', 'Service', 'Recreation_salesforce', 'Mountain_salesforce', 'Speed_salesforce'])

    combined_df = combined_demand.merge(
        combined_workforce[['Company', 'Total Yearly Cost', 'Satisfaction', 'Number of Media Placements']],
        on='Company',
        how='left'
    )
    combined_df = combined_df.merge(
        combined_salesforce[['Company', 'City', 'Total Sales and Service People', 'Service', 'Recreation_salesforce', 'Mountain_salesforce', 'Speed_salesforce', 'Quarter']],
        on=['Company', 'City', 'Quarter'],
        how='left'
    )

    for col in ['Total Sales and Service People', 'Service', 'Recreation_salesforce', 'Mountain_salesforce', 'Speed_salesforce']:
        combined_df[col] = combined_df[col].fillna(0)

    combined_df = combined_df.dropna(subset=['Total Yearly Cost'])
    return combined_df

# Function to filter data for a specific segment
def filter_segment_data(df, segment):
    columns = SEGMENT_COLUMNS[segment] + [
        'Quarter', 'Total Yearly Cost', 'Satisfaction', 'Number of Media Placements',
        'Total Sales and Service People', 'Service', f'{segment}_salesforce'
    ]
    segment_col = segment

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for {segment} segment: {missing_cols}")
        return pd.DataFrame()

    filtered_df = df[df[segment_col] > 0][columns].dropna()
    filtered_df['Brand'] = filtered_df['Brand'].apply(normalize_brand_name)

    filtered_df = filtered_df.rename(columns={
        f'Brand Judgement - {segment}': 'Brand_Judgement',
        f'Ad Judgement - {segment}': 'Ad_Judgement',
        segment_col: 'Demand',
        f'{segment}_salesforce': 'Salesforce_Allocation'
    })

    filtered_df['Brand_Ad_Judgement'] = filtered_df['Brand_Judgement'] * filtered_df['Ad_Judgement']
    filtered_df['Demand_Salesforce'] = filtered_df['Demand'] * filtered_df['Salesforce_Allocation']

    st.write(f"Rows for {segment} segment: {len(filtered_df)}")
    return filtered_df

# Function to train model and compute metrics
@st.cache_resource
def train_model(segment, df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]
    st.write(f"Rows after outlier removal for {segment}: {len(df)}")

    X = df.drop(['Price', 'Brand', 'Company', 'City', 'Quarter'], axis=1)
    y = df['Price']

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_cols = ['City', 'Quarter']
    categorical_encoded = ohe.fit_transform(df[categorical_cols])
    categorical_columns = ohe.get_feature_names_out(categorical_cols)
    categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_columns, index=df.index)
    X = pd.concat([X, categorical_df], axis=1)

    feature_names = X.columns

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    return model, ohe, scaler, rmse, feature_importance, y_test, y_pred

# Function to create visualizations
def create_visualizations(rmse, feature_importance, y_test, y_pred):
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

    return fig1, fig2

# Main app
def main():
    st.title("Bike Price Predictor")

    with st.spinner("Loading data..."):
        df = load_and_process_data()

    if df.empty:
        st.error("No valid data loaded. Please check input files.")
        return

    segment = st.selectbox("Select Segment", ["Recreation", "Mountain", "Speed"])
    segment_df = filter_segment_data(df, segment)

    if segment_df.empty:
        st.error(f"No valid data available for {segment} segment.")
        return

    brand_company_pairs = segment_df[['Brand', 'Company']].drop_duplicates()
    brands = sorted(brand_company_pairs['Brand'].unique())
    all_cities = sorted(df['City'].unique())

    st.write("Valid Brand-Company combinations in segment:", brand_company_pairs)

    st.subheader("Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Bike Details")
        selected_brand = st.selectbox("Select Brand", brands)
        valid_companies = sorted(brand_company_pairs[brand_company_pairs['Brand'] == selected_brand]['Company'].unique())
        valid_companies = ['3Cycle'] if '3Cycle' in valid_companies else []  # Filter to only show 3Cycle
        if not valid_companies:
            st.error("3Cycle is not available for the selected brand in this segment.")
            return
        selected_company = st.selectbox("Select Company", valid_companies)
        selected_city = st.selectbox("Select City", all_cities)

        if not ((segment_df['Brand'] == selected_brand) & (segment_df['Company'] == selected_company) & (segment_df['City'] == selected_city)).any():
            st.warning(f"Note: {selected_city} is not recorded for {selected_brand}, {selected_company} in {segment} segment.")

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

    with st.spinner("Training model..."):
        model, ohe, scaler, rmse, feature_importance, y_test, y_pred = train_model(segment, segment_df)

    try:
        categorical_encoded = ohe.transform([[selected_city, 'Q2']])
        categorical_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out(['City', 'Quarter']))
        input_data = pd.DataFrame({
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
            'Salesforce_Allocation': [salesforce_allocation],
            'Brand_Ad_Judgement': [brand_judgement * ad_judgement],
            'Demand_Salesforce': [demand * salesforce_allocation]
        })
        input_data = pd.concat([input_data, categorical_df], axis=1)
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)[0]

        st.subheader("Predicted Price")
        st.metric("Price ($)", f"{prediction:.2f}")

        st.subheader("Model Performance")
        st.metric("RMSE ($)", f"{rmse:.2f}")

        st.subheader("Feature Importance")
        st.dataframe(feature_importance.style.format({"Importance": "{:.3f}"}))

        st.subheader("Visualizations")
        fig1, fig2 = create_visualizations(rmse, feature_importance, y_test, y_pred)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        with open('rmse_log.txt', 'a') as f:
            f.write(f"Segment: {segment}, RMSE: {rmse:.2f}, Date: {pd.Timestamp.now()}\n")

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()

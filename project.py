import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import random
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import time

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "df" not in st.session_state:
    st.session_state.df = None
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "Data Overview"
if "UserEnterUserName" not in st.session_state:
    st.session_state.UserEnterUserName = ""

if hasattr(st, "experimental_rerun"):
    rerun = st.rerun
else:
    rerun = st.rerun

query_params = st.query_params
if query_params.get("auth") == "true":
    st.session_state.authenticated = True

def predict_revenue(file_path, prediction_year=2025):
    # Load dataset
    df = pd.read_csv(file_path)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')

    # Aggregate total revenue per year
    yearly_sales = df.groupby(df['date'].dt.year)['total'].sum().reset_index()
    yearly_sales.columns = ['year', 'total_revenue']

    # Create lag feature (previous year's revenue)
    yearly_sales['prev_revenue'] = yearly_sales['total_revenue'].shift(1).fillna(0)

    # Create moving average feature (rolling window of 2 years)
    yearly_sales['moving_avg'] = yearly_sales['total_revenue'].rolling(window=2).mean().fillna(0)

    # Prepare data for ML model (including 2019 for training)
    X = yearly_sales[['year', 'prev_revenue', 'moving_avg']].values  # Features
    y = yearly_sales['total_revenue'].values  # Target

    # Train a Random Forest model with better parameters
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)

    # Predict revenue for years up to the user's input
    future_years = list(range(yearly_sales['year'].min(), prediction_year + 1))
    predicted_values = {}

    last_known_revenue = yearly_sales['total_revenue'].iloc[-1]

    for year in future_years:
        prev_revenue = predicted_values.get(year - 1, last_known_revenue)
        moving_avg = np.mean(list(predicted_values.values())[-2:]) if len(predicted_values) >= 2 else prev_revenue
        future_features = np.array([[year, prev_revenue, moving_avg]])
        
        predicted_revenue = model.predict(future_features)[0]
        predicted_values[year] = predicted_revenue

    # Calculate accuracy using R² Score
    y_pred = model.predict(X)
    accuracy = r2_score(y, y_pred) * 100  # Convert to percentage

    # Create DataFrame for visualization
    predicted_df = pd.DataFrame(list(predicted_values.items()), columns=['year', 'predicted_revenue'])
    
    # Exclude 2019 from visualization
    yearly_sales_no_2019 = yearly_sales[yearly_sales['year'] != 2019]
    last_actual_year = yearly_sales['year'].max()
    predicted_future_df = predicted_df[predicted_df['year'] > last_actual_year]  # Future predictions only

    # Plot interactive graph using Plotly
    fig = go.Figure()

    # Add actual revenue data (excluding 2019)
    fig.add_trace(go.Scatter(x=yearly_sales_no_2019['year'], y=yearly_sales_no_2019['total_revenue'],
                             mode='markers+lines', name='Actual Revenue', marker=dict(color='blue')))

    # Add predicted revenue points (only for future years)
    fig.add_trace(go.Scatter(x=predicted_future_df['year'], y=predicted_future_df['predicted_revenue'],
                             mode='markers+lines', name='Predicted Revenue', marker=dict(color='red', symbol='diamond')))

    # Customize layout
    fig.update_layout(title="Walmart Revenue Prediction",
                      xaxis_title="Year",
                      yaxis_title="Total Revenue",
                      template="plotly_dark")

    return fig, predicted_values[prediction_year], accuracy

def prepare_quarterly_data(file_path):
    
    df = pd.read_csv(file_path)
    
    
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    
    
    df['year_quarter'] = df['date'].dt.to_period('Q')
    
    
    quarterly_sales = df.groupby(df['year_quarter'])['total'].sum().reset_index()
    quarterly_sales['year_quarter'] = quarterly_sales['year_quarter'].astype(str)
    
    return quarterly_sales

def train_model(quarterly_sales):
  
    quarterly_sales['year'] = quarterly_sales['year_quarter'].apply(lambda x: int(x[:4]))
    quarterly_sales['quarter'] = quarterly_sales['year_quarter'].apply(lambda x: int(x[-1]))
    
 
    X = quarterly_sales[['year', 'quarter']].values
    y = quarterly_sales['total'].values
    
   
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y_scaled)
    
    
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    accuracy = r2_score(y, y_pred) * 100  
    
    return model, scaler, X, accuracy

def predict_next_quarter(model, scaler, X, last_quarter):
    
    year, quarter = last_quarter
    if quarter == 4:
        year += 1
        quarter = 1
    else:
        quarter += 1
    
    next_quarter_features = np.array([[year, quarter]])
    predicted_scaled = model.predict(next_quarter_features)[0]
    predicted_revenue = scaler.inverse_transform([[predicted_scaled]])[0][0]
    
    return f'Q{quarter} {year}', predicted_revenue


def side_bar():
    with st.sidebar:
        sidebar_select=option_menu("TrendVista",["Log-in","Sign-Up"],default_index=0,menu_icon="pie-chart",icons=["unlock","lock"],orientation="vertical",key="sidebar_menu_key")

    if sidebar_select=="Log-in":
        st.success("If You Do Not Have Account Then Sign-Up First")
        with st.form("Form1"):
            st.title("Log In")
            UserEnterUserName=st.text_input("Enter Your Username")
            UserEnterPassword=st.text_input("Enter Your Password",type="password")

            submit=st.form_submit_button("Submit")
            is_data_save=check_user_data(UserEnterUserName,UserEnterPassword)

            if submit:
               if is_data_save==True:
                  st.session_state.authenticated = True
                  st.session_state.UserEnterUserName = UserEnterUserName
                  st.success("Login Successfully")
                  st.query_params["auth"] = "true"
                  rerun()
               else:
                  st.error("User name or Password is Wrong, please enter correct one")

    if sidebar_select=="Sign-Up":
        with st.form("Form2"):
            st.title("Sign Up")
            UserEnterNewUserName=st.text_input("Enter Your Username")
            UserEnterNewPassword=st.text_input("Enter Your Password",type="password")
            submit2 = st.form_submit_button("Submit")
            is_data_save = check_user_availabel(UserEnterNewUserName, UserEnterNewPassword)

            if submit2:
                if is_data_save==True:
                    st.error("User Name Already Taken")
                else:
                    enter_new_data(UserEnterNewUserName,UserEnterNewPassword)
                    st.success("User Successfully Register")

def check_user_data(UserEnterUserName,UserEnterPassword):
    df = pd.read_excel("PROJECT_DATA.xlsx")
    rows , cols = df.shape
    is_data_in=False

    for i in range(0,rows):
        if df.iat[i,1]==UserEnterUserName:
            for j in range(0,cols):
                if df.iat[i,2]==int(UserEnterPassword):
                    is_data_in=True
                    break

    return is_data_in

def enter_new_data(UserEnterNewUserName,UserEnterNewPassword):

    # Load existing data
    file_path = "PROJECT_DATA.xlsx"
    existing_data = pd.read_excel(file_path)
    rows , cols = existing_data.shape

    # New data
    new_data = pd.DataFrame([{'No': rows+1, 'User Name':UserEnterNewUserName, 'Password':(UserEnterNewPassword)}])

    # Append and save
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.to_excel(file_path, index=False)

def check_user_availabel(UserEnterNewUserName,UserEnterNewPassword):
    df = pd.read_excel("PROJECT_DATA.xlsx")
    rows, cols = df.shape
    is_data_in = False

    for i in range(0, rows):
        if df.iat[i, 1] == UserEnterNewUserName:
            is_data_in = True
            break

    return is_data_in

def main_app():
    with st.sidebar:
        selected = option_menu(
            "TrendVista",
            ["Data Overview", "Revenue Forecasting", "Growth Analysis", "Sales Analysis", "About Us","Contact Us" ,"Log Out"],
            icons=["table", "graph-up-arrow", "bar-chart-fill", "pie-chart-fill", "people","telephone-fill","lock"],
            default_index=0,
            menu_icon="pie-chart",
            orientation="vertical",
            key="main_menu_key"
        )

        if selected != st.session_state.selected_menu:
            st.session_state.selected_menu = selected
            rerun()
    DEFAULT_FILE_PATH = "sales_data_large.csv"

    if st.session_state.selected_menu == "Data Overview":
        st.toast("Welcome To trendVista",icon="🎊")
        st.header(f"🎉 Welcome {st.session_state.UserEnterUserName}!")
        st.subheader("Upload Your Historical Data Here!")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            file_extension = uploaded_file.name.split(".")[-1]

            if file_extension == "csv":
                st.session_state.df = pd.read_csv(uploaded_file)
            elif file_extension == "xlsx":
                st.session_state.df = pd.read_excel(uploaded_file)
        else:
            st.write("### No file uploaded. Loading default dataset...")
            file_extension = DEFAULT_FILE_PATH.split(".")[-1]

            if file_extension == "csv":
                st.session_state.df = pd.read_csv(DEFAULT_FILE_PATH)
            elif file_extension == "xlsx":
                st.session_state.df = pd.read_excel(DEFAULT_FILE_PATH)

        st.write("### Preview of Data")
        st.dataframe(st.session_state.df)

    if st.session_state.selected_menu == "Sales Analysis":
        st.header("📊 Sales Analysis")
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df

            if "selected_regions" not in st.session_state:
                st.session_state.selected_regions = []
            if "selected_cities" not in st.session_state:
                st.session_state.selected_cities = []
            if "selected_products" not in st.session_state:
                st.session_state.selected_products = []

            st.subheader("Choose Regions")
            all_regions = df["Region"].unique().tolist()
            selected_regions = st.multiselect("Select Regions", all_regions, default=st.session_state.selected_regions)

            filtered_df = df[df["Region"].isin(selected_regions)] if selected_regions else df
            st.subheader("Choose Cities")
            all_cities = filtered_df["City"].unique().tolist()
            selected_cities = st.multiselect("Select Cities", all_cities, default=st.session_state.selected_cities)

            st.subheader("Choose Products")
            all_products = filtered_df["Product"].unique().tolist()
            selected_products = st.multiselect("Select Products", all_products, default=st.session_state.selected_products)

            if st.button("Submit"):
                progress_bar=st.progress(0)

                for percent in range(1,101,10):
                    time.sleep(0.2)
                    progress_bar.progress(percent)


                st.session_state.selected_regions = selected_regions
                st.session_state.selected_cities = selected_cities
                st.session_state.selected_products = selected_products

            if st.session_state.selected_regions or st.session_state.selected_cities or st.session_state.selected_products:
                filtered_df = df[
                    df["Region"].isin(st.session_state.selected_regions)] if st.session_state.selected_regions else df
                city_filtered_df = filtered_df[filtered_df["City"].isin(
                    st.session_state.selected_cities)] if st.session_state.selected_cities else filtered_df
                product_filtered_df = city_filtered_df[city_filtered_df["Product"].isin(
                    st.session_state.selected_products)] if st.session_state.selected_products else city_filtered_df

                if st.session_state.selected_regions:
                    region_sales = filtered_df.groupby("Region")["Sales"].sum()
                    st.write("### Sales Data (Region-wise)")
                    st.dataframe(region_sales)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    region_sales.plot(kind="bar", ax=ax, color="skyblue")
                    ax.set_ylabel("Total Sales")
                    ax.set_title("Region-wise Sales Distribution")
                    st.pyplot(fig)

                if st.session_state.selected_cities:
                    if not city_filtered_df.empty:
                        city_sales = city_filtered_df.groupby("City")["Sales"].sum()
                        st.write("### Sales Data (City-wise)")
                        st.dataframe(city_sales)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        city_sales.plot(kind="bar", ax=ax, color="orange")
                        ax.set_ylabel("Total Sales")
                        ax.set_title("City-wise Sales Distribution")
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for the selected cities.")

                if st.session_state.selected_products:
                    if not product_filtered_df.empty:
                        product_sales = product_filtered_df.groupby("Product")["Sales"].sum()
                        st.write("### Sales Data (Product-wise)")
                        st.dataframe(product_sales)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        product_sales.plot(kind="bar", ax=ax, color="green")
                        ax.set_ylabel("Total Sales")
                        ax.set_title("Product-wise Sales Distribution")
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for the selected products.")

        else:
            st.warning("Please upload a data file in the 'Data Overview' section before accessing 'Sales Analysis'.")

    if st.session_state.selected_menu == "Revenue Forecasting":
        file_path = "walmart_cleaned.csv"  # Update this path
        pred_years = list(range(2024, 2031))  # 2031 is exclusive
        selected_year = st.selectbox("Select a year", pred_years)

        st.write(f"You selected: {selected_year}")
        fig, predicted_revenue, accuracy = predict_revenue(file_path, prediction_year = selected_year)  # Change year as needed
    
        st.plotly_chart(fig)
        st.write(f"Predicted Revenue for {selected_year}: ${predicted_revenue:,.2f}")
        st.write(f"Model Accuracy: {accuracy:.2f}%")

    if st.session_state.selected_menu == "Growth Analysis":
        file_path = "walmart_cleaned.csv"  # Update this path
        quarterly_sales = prepare_quarterly_data(file_path)
        model, scaler, X, accuracy = train_model(quarterly_sales)
        last_quarter = (quarterly_sales['year'].iloc[-1], quarterly_sales['quarter'].iloc[-1])
        next_quarter, predicted_revenue = predict_next_quarter(model, scaler, X, last_quarter)
        
        # Filter out 2019 data for visualization only
        quarterly_sales_filtered = quarterly_sales[~quarterly_sales['year_quarter'].str.startswith('2019')]
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=quarterly_sales_filtered['year_quarter'], y=quarterly_sales_filtered['total'],
                                mode='markers+lines', name='Actual Revenue', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[next_quarter], y=[predicted_revenue],
                                mode='markers', name='Predicted Revenue', marker=dict(color='red', size=10)))
        
        fig.update_layout(title="Quarterly Revenue Growth Prediction",
                        xaxis_title="Quarter",
                        yaxis_title="Total Revenue",
                        template="plotly_dark")
        
        st.plotly_chart(fig)
        st.write(f"Predicted Revenue for {next_quarter}: ${predicted_revenue:,.2f}")
        st.write(f"Model Accuracy: {accuracy:.2f}%")

    if st.session_state.selected_menu == "About Us":
        st.header("🚀AI-Driven Revenue Forecasting📊 and Trend Analysis  📈for Business Growth 💰")
        st.subheader(
            "T.R.E.N.D.V.I.S.T.A. – Technology-driven Revenue Estimation & Next-gen Data Visualization, Insights, Strategy, & Trend Analysis")
        st.markdown(
            """
            **Project Description/Abstract:** 
            AI-Driven Revenue Forecasting and Trend Analysis for Business 
            Growth" is a machine learning project that predicts future revenue trends by analyzing historical data. 
            It identifies patterns, detects anomalies, and forecasts revenue. The system accounts for seasonal 
            fluctuations, market conditions, and consumer behavior, providing businesses with actionable insights. 
            It aids in financial planning, risk mitigation, and strategic resource allocation. The solution is adaptable 
            to various industries such as retail, e-commerce, and SaaS. Ultimately, it enables data-driven decision
            making for sustained business growth.

            **Problems in the Existing System** 
            1. Inaccurate Revenue Predictions – Traditional forecasting methods lack precision. 
            2. Limited Consideration of Market Trends – Many forecasting models ignore external factors 
               like economic shifts. 
            3. Inability to Detect Anomalies – Unexpected market disruptions remain unaccounted for. 
            4. Manual Data Analysis – Businesses rely on time-consuming and error-prone manual 
               calculations. 
            5. Poor Financial Planning – Lack of accurate revenue forecasts leads to ineffective budgeting 
               and resource allocation.

            **Purpose of the Project** 
            • To predict future revenue trends using machine learning. 
            • To analyze historical financial data and identify revenue patterns. 
            • To detect anomalies and fluctuations for proactive decision-making. 
            • To provide accurate and actionable financial insights for business growth. 
            • To support strategic resource allocation and risk mitigation.   

            **Functional Requirements**
            1. Data Ingestion & Preprocessing – Collects, cleans, and normalizes historical revenue data. 
            2. Trend Analysis & Visualization – Generates charts and insights on revenue trends. 
            3. Revenue Forecasting Model – Uses ML algorithms to predict future revenue. 
            4. Anomaly Detection – Identifies outliers and revenue discrepancies. 
            5. Report Generation – Provides detailed financial summaries for stakeholders. 
            6. Industry-Specific Adaptability – Customizable for various business domains. 
            7. User Dashboard – Interactive UI displaying real-time analytics and insights.

            **System Modules**
            1. Data Collection Module – Gathers historical revenue and market data. 
            2. Preprocessing & Feature Engineering – Cleans and prepares data for analysis. 
            3. Machine Learning Model – Predicts revenue trends and detects anomalies. 
            4. Visualization & Reporting Module – Displays graphs, reports, and insights. 
            5. User Management Module – Allows different user roles (admin, analyst, manager).

            **System Requirements**

            *Hardware Requirements:* \n
               • Processor: Intel i5 or higher 
               • RAM: 8GB minimum 
               • Storage: 250GB SSD or more 
               • Internet Connectivity: Stable broadband connection 
            *Software Requirements:* \n
               • Operating System: Windows 
               • Pycharm , python 
               • Required AI Libraries 

            **Front End and Back End of System**\n
               • Front End (Client-Side): StreamLit \n
               • Back End (Server-Side): Python , Machine Learning Models , AI models   
            """
        )

    if st.session_state.selected_menu == "Contact Us":
        st.header("☎️Contact Us")
        st.markdown(
            """ We’d love to hear from you! Whether you have questions about our AI-driven revenue forecasting system, need support, or just want to share feedback, feel free to reach out.  
### 📧 Email Us  
For inquiries, collaborations, or support, email us at:   
email@eamil.com


### 💬 Stay Connected  
Follow us on social media for the latest updates and insights:  
- 🔹 **[LinkedIn](#)**  
- 🔹 **[Twitter](#)**  
- 🔹 **[Instagram](#)**  

We look forward to connecting with you! 😊    
            """)

    if st.session_state.selected_menu == "Log Out":
        st.session_state.authenticated = False
        st.query_params["auth"] = "false"
        rerun()

# Display the appropriate section based on authentication status
if st.session_state.authenticated:
    main_app()
else:
    side_bar()

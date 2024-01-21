import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)


# Function to load data and display basic information
def load_data(file, encoding='utf-8'):
    if file.name.endswith('xlsx'):
        data = pd.read_excel(file)
    elif file.name.endswith('csv'):
        data = pd.read_csv(file, encoding=encoding)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

    # Filter numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Combine numeric and categorical columns
    selected_columns = list(numeric_columns) + list(categorical_columns)

    return data[selected_columns]


# Function to display missing, valid values, and outliers for numeric columns
def display_missing_values(data):
    st.subheader("Missing, Valid Values, and Outliers")

    # Check if there are numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        st.warning("No numeric columns found. Missing values, valid values, and outliers cannot be calculated.")
        return

    # Filter numeric columns
    numeric_data = data[numeric_columns]

    # Missing and Valid Values
    missing_values = numeric_data.isnull().sum()
    valid_values = numeric_data.notnull().sum()

    # Outliers using IQR method
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers_count = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()

    missing_valid_df = pd.DataFrame({
        'Variable': missing_values.index,
        'Missing Values': missing_values.values,
        'Valid Values': valid_values.values,
        'Outliers Count': outliers_count.values,
        'Min': numeric_data.min(),
        'Max': numeric_data.max(),
        'Mean': numeric_data.mean(),
        'Std Dev': numeric_data.std(),
        'Median': numeric_data.median(),
    })

    st.write(missing_valid_df)


    

# Function to detect and display outliers using box plots
def display_outliers(data, selected_variable):
    st.subheader("Outliers Detection")
    fig = go.Figure()
    fig.add_trace(go.Box(y=data[selected_variable], name=selected_variable))
    fig.update_layout(title=f"Box Plot for {selected_variable}")
    st.plotly_chart(fig)

# Function to create correlation heatmap for selected variables
def create_correlation_heatmap(data, selected_variables):
    st.subheader("Correlation Heatmap")
    corr_matrix = data[selected_variables].corr()
    fig = px.imshow(corr_matrix, labels=dict(x="Variable", y="Variable", color="Correlation"))
    st.plotly_chart(fig)

# Function to calculate Pearson correlation with NaN handling
def pearsonr_with_nan(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return pearsonr(x[mask], y[mask])

# Custom correlation plot function
def corrdot(*args, **kwargs):
    (r, p) = pearsonr_with_nan(*args)

    # Use the custom Pearson correlation function
    corr_r = args[0].corr(args[1], 'pearson')
    p_text = f"p = {p:.3f}".replace("0.", ".")
    corr_text = f"r = {corr_r:2.2f}".replace("0.", ".")

    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm", vmin=-1, vmax=1, transform=ax.transAxes)

    font_size = abs(corr_r) * 40 + 5
    ax.annotate(p_text, xy=(.4, .9), xycoords=ax.transAxes, fontsize=15.0)
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)
# Function to create a customized correlation heatmap
def create_custom_correlation_heatmap(data, selected_variables):
    st.subheader("Custom Correlation Heatmap")

    if len(selected_variables) < 2:
        st.warning("Please select at least two variables from the dropdown above for correlation analysis.")
        return

    # Calculate correlation matrix
    corr_matrix = data[selected_variables].corr()

    # Create a PairGrid using Seaborn
    sns.set(style='white', font_scale=1.2)
    g = sns.PairGrid(data[selected_variables], aspect=1.4, diag_sharey=True)  # Set diag_sharey to True

    # Map lower triangle with a scatter plot
    g.map_lower(sns.regplot,  ci=False, line_kws={'color': 'black'})

    # Map diagonal with histograms
    g.map_diag(sns.histplot, kde_kws={'color': 'black'})

    # Map upper triangle with the custom correlation plot function
    g.map_upper(corrdot)

    # Show the plot in Streamlit
    st.pyplot()


# Function to create scatter plot
def create_scatter_plot(data, x_variable, y_variable, group_variable=None):
    st.subheader("Scatter Plot")
    
    if group_variable:
        fig = px.scatter(data, x=x_variable, y=y_variable, color=group_variable,
                         title=f"{x_variable} vs {y_variable} (Grouped by {group_variable})")
    else:
        fig = px.scatter(data, x=x_variable, y=y_variable, title=f"{x_variable} vs {y_variable}")

    st.plotly_chart(fig)

def navigation_bar():
    st.sidebar.title("Navigation")

    all_options = {
        
        "Descriptives": "Explore missing, valid values, and escriptives",
        "Outliers Detection": "Detect and visualize outliers",
        "Grouped Histogram": "Plot data distributions",
        "Custom Correlation Heatmap": "Visualize relationships between variables",
        #"Correlation Heatmap": "Visualize correlations between variables",
        "Scatter Plot": "Create scatter plots for variable relationships",
        
    }

    # Define icons for each option
    option_icons = {
        "Custom Correlation Heatmap": "🔍",
        "Descriptives": "❓",
        "Outliers Detection": "⚠️",
        #"Correlation Heatmap": "🔍",
        "Scatter Plot": "📈",
        "Grouped Histogram": "📊",
    }

    # Get selected options from the user
    selected_options = []
    
    for option, description in all_options.items():
        icon = option_icons.get(option, "")
        tooltip = f"{option} - {description}"

        # Create a checkbox for each option
        selected = st.sidebar.checkbox(f"{icon} {option} - {description}", key=option)

        if selected:
            selected_options.append(option)

    return selected_options


# Function to create grouped histogram
def create_grouped_histogram(data, variable, group_variable=None):
    st.subheader("Grouped Histogram")
    
    if group_variable:
        fig = px.histogram(data, x=variable, color=group_variable,
                           title=f"{variable} Grouped by {group_variable}")
    else:
        fig = px.histogram(data, x=variable, title=f"{variable} Histogram")

    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.title("Exploratory Data Inspection")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        selected_options = navigation_bar()

        for page in selected_options:
            #if page == "Dataset Info":
                # Display basic information
               # st.subheader("Dataset Information")
                #st.write(data.info())

            if page == "Descriptives":
                # Display missing, valid values, and outliers
                display_missing_values(data)

            elif page == "Outliers Detection":
                # Select variable(s) for outliers detection
                selected_variables_outliers = st.multiselect("Select variable(s) for outliers detection", data.columns)
                for selected_variable in selected_variables_outliers:
                    display_outliers(data, selected_variable)

            #elif page == "Correlation Heatmap":
                # Display correlation heatmap for selected variables
                #selected_variables_heatmap = st.multiselect("Select variable(s) for correlation heatmap", data.columns)
                #create_correlation_heatmap(data, selected_variables_heatmap)
            elif page == "Custom Correlation Heatmap":
                # Display custom correlation heatmap for selected variables
                selected_variables_custom_heatmap = st.multiselect("Select variable(s) for custom correlation heatmap", data.columns)
                create_custom_correlation_heatmap(data, selected_variables_custom_heatmap)

            elif page == "Scatter Plot":
                # Select variables for scatter plot
                x_variable = st.selectbox("Select X variable for scatter plot", data.columns)
                y_variable = st.selectbox("Select Y variable for scatter plot", data.columns)
                group_variable = st.selectbox("Select grouping variable (optional)", [None] + data.columns.tolist())
                create_scatter_plot(data, x_variable, y_variable, group_variable)

            elif page == "Grouped Histogram":
                # Select variable for grouped histogram
                variable_histogram = st.selectbox("Select variable for grouped histogram", data.columns, key="variable_histogram")
                group_variable_histogram = st.selectbox("Select grouping variable (optional)", [None] + data.columns.tolist(), key="group_variable_histogram")

    
                create_grouped_histogram(data, variable_histogram, group_variable_histogram)

                
if __name__ == "__main__":
    main()

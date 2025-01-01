# Library Data Retrieving and Modification
import zipfile
import os

# Library for Data Preprocessing
import pandas as pd
import numpy as np

# Library for Visualization
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Library for Dashboard
import streamlit as st

# """
# Segement 1 : Add Function for Visualization

# This section will add a function that will use to processing the dataframe,
# make visualization, and others.
# """

def load_dataframe(filepath):
    """
    Load and preprocess a DataFrame from a CSV file.

    Parameters:
    - file_path : str
        The file path to the CSV file containing the DataFrame.

    Returns:
    - pandas.DataFrame
        A DataFrame sorted by 'datetime', with datetime columns converted.
    """

    # Read the dataframe
    all_df = pd.read_csv(filepath)

    # Ensure the selected column is datetime format
    datetime_columns = ["datetime"]

    for column in datetime_columns:
      all_df[column] = pd.to_datetime(all_df[column])

    return all_df

def filter_data_level(df, level):
    """
    Filters the given DataFrame based on the specified levels (year, month, day).

    This function dynamically displays Streamlit selectboxes for each filter level
    (year, month, day) based on the provided `level` list, and filters the DataFrame 
    accordingly. The function assumes the DataFrame has columns `year`, `month`, and 
    `day` for filtering.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to be filtered.
    - level (list): A list containing the filtering levels. It can include 'year', 
      'month', and 'day'. The filtering occurs based on the order of the levels in the list.

    Returns:
    - pd.DataFrame: The filtered DataFrame based on the selected levels.
    """

    if 'year' in level:
        selected_year = st.selectbox("Select Year:", df['year'].unique())
        df = df[df['year'] == selected_year]
    if 'month' in level:
        selected_month = st.selectbox("Select Month:", df['month'].unique())
        df = df[df['month'] == selected_month]
    if 'day' in level:
        selected_day = st.selectbox("Select Day:", df['day'].unique())
        df = df[df['day'] == selected_day]

    return df

def filter_selected_datetime(df, all_cols):
    """
    Function to dynamically filter the DataFrame by year, month, day, and hour.
    
    Parameters:
    - spatial_df (pd.DataFrame): The dataframe containing datetime and spatial information.
    
    Returns:
    - selected_year (int or None): The selected year.
    - selected_month (int or None): The selected month.
    - selected_day (int or None): The selected day.
    - selected_hour (int or None): The selected hour.
    - filtered_df (pd.DataFrame): The filtered dataframe based on the selections.
    """

    # Initialize empty selections
    selected_year = None
    selected_month = None
    selected_day = None
    selected_hour = None   

    # Select Year
    selected_year = all_cols[0].selectbox("Select Year:", options=[None] + sorted(df['datetime'].dt.year.unique()))

    # Select Month
    if selected_year is not None:
        available_months = sorted(df[df["datetime"].dt.year == selected_year]["datetime"].dt.month.unique())
    else:
        available_months = []
    selected_month = all_cols[1].selectbox("Select Month:", options=[None] + available_months)

    # Select Day
    if selected_year is not None and selected_month is not None:
        available_days = sorted(df[(df["datetime"].dt.year == selected_year) & (df["datetime"].dt.month == selected_month)]["datetime"].dt.day.unique())
    else:
        available_days=[]
    selected_day = all_cols[2].selectbox("Select Day:", options=[None] + available_days)

    # Select Hour
    if selected_year is not None and selected_month is not None and selected_day is not None:
      available_hours = sorted(df[(df["datetime"].dt.year == selected_year) & (df["datetime"].dt.month == selected_month) & (df["datetime"].dt.day == selected_day)]["datetime"].dt.hour.unique())
    else:
      available_hours=[]
    selected_hour = all_cols[3].selectbox("Select Hour:", options=[None] + available_hours)

    # Apply the selected filters to the dataframe
    filtered_df = df.copy()
    
    # Filter Selection
    if selected_year is not None:
        filtered_df = filtered_df[filtered_df["datetime"].dt.year == selected_year]
    if selected_month is not None:
        filtered_df = filtered_df[filtered_df["datetime"].dt.month == selected_month]
    if selected_day is not None:
        filtered_df = filtered_df[filtered_df["datetime"].dt.day == selected_day]
    if selected_hour is not None:
        filtered_df = filtered_df[filtered_df["datetime"].dt.hour == selected_hour]

    all_filter = [selected_year, selected_month, selected_day, selected_hour]

    return all_filter, filtered_df

def aggregate_data(df, visualization_type):
    """
    Aggregates the given DataFrame based on the specified visualization type (Annual, Monthly, Daily, or Hourly).

    This function computes aggregated statistics (mean) on air quality data, grouped by time-based components
    such as year, month, day, or hour. The function can be customized to aggregate data for a specific year,
    month, or day if necessary.

    Parameters:
    - df : pandas.DataFrame
        The input DataFrame containing air quality data. Must include a 'datetime' column and pollutant data columns
        (e.g., 'PM2.5', 'PM10').

    - visualization_type : str
        The type of aggregation to perform. Accepted values are:
        - 'Annual' : Aggregates data by year.
        - 'Monthly' : Aggregates data by month, optionally for a specific year.
        - 'Daily' : Aggregates data by day, optionally for a specific month and year.
        - 'Hours' : Aggregates data by hour, optionally for a specific day, month, and year.

    - component_date : dict, optional (default is {'year': None, 'month': None, 'days': None})
        A dictionary specifying the specific year, month, and day for monthly, daily, and hourly aggregations.
        The dictionary may include:
        - 'year' : The specific year for monthly, daily, or hourly aggregation.
        - 'month' : The specific month for daily or hourly aggregation.
        - 'days' : The specific day for hourly aggregation.
        If not provided, the function will aggregate based on all available data.

    Returns:
    - agg_df : pandas.DataFrame
        The aggregated DataFrame based on the chosen visualization type. The data is grouped by relevant time units
        (e.g., year, month, day, or hour), and the mean of each pollutant is computed for each group.

    - id_vars : list
        A list of columns used as identifier variables for the aggregation. These columns depend on the selected
        aggregation type (e.g., ['station', 'year'] for annual aggregation).

    - labels : list
        A list containing the name of the time unit used in the aggregation, along with its descriptive label
        (e.g., ['year', 'Year']).

    Raises:
    - ValueError : If 'datetime' column is missing in the DataFrame.
    - ValueError : If an invalid 'visualization_type' is provided.
    - ValueError : If required values for year, month, or day are missing when performing monthly, daily, or hourly aggregation.
    """

    # Raise Value Error if datetime Column not present.
    if 'datetime' not in df.columns:
      raise ValueError("'datetime' column is missing in the DataFrame.")

    # Detect Year, Month, Day, and Hour
    df.loc[:, 'year'] = df['datetime'].dt.year
    df.loc[:, 'month'] = df['datetime'].dt.month
    df.loc[:, 'day'] = df['datetime'].dt.day
    df.loc[:, 'hour'] = df['datetime'].dt.hour

    # Initialize variables
    agg_df = None
    id_vars = []
    labels = []

    # Use match-case for aggregation
    match visualization_type:
      case 'Annual':
          agg_df = df.groupby(['station', 'year']).mean().reset_index()
          id_vars = ['station', 'year']
          labels = ['year', 'Year']
          agg_df.drop(columns=['month', 'day', 'hour'], inplace=True)

      case 'Monthly':
          if df['year'].nunique() > 1:
              df = filter_data_level(df, ['year'])
          agg_df = df.groupby(['station', 'month']).mean().reset_index()
          id_vars = ['station', 'month']
          labels = ['month', 'Month']
          agg_df.drop(columns=['year', 'day', 'hour'], inplace=True)

      case 'Daily':
          if df['month'].nunique() > 1:
              df = filter_data_level(df, ['year', 'month'])
          agg_df = df.groupby(['station', 'day']).mean().reset_index()
          id_vars = ['station', 'day']
          labels = ['day', 'Day']
          agg_df.drop(columns=['month', 'year', 'hour'], inplace=True)

      case 'Hours':
          if df['day'].nunique() > 1:
              df = filter_data_level(df, ['year', 'month', 'day'])
          agg_df = df.groupby(['station', 'hour']).mean().reset_index()
          id_vars = ['station', 'hour']
          labels = ['hour', 'Hour']
          agg_df.drop(columns=['month', 'day', 'year'], inplace=True)

      case _:
          raise ValueError("Invalid visualization type. Choose from 'Annual', 'Monthly', 'Daily', 'Hours'.")

    return agg_df, id_vars, labels

def ploty_line_visualization(data, id_vars, labels, parameters, title_template="Average Concentration Data by Station"):
    """
    Creates a Plotly line chart to visualize air quality data by station and a specified time unit (e.g., year, month, hour).

    Parameters:
    - data : pandas.DataFrame
        The input data containing station names, time variables, and pollutant concentrations.
        The DataFrame must include columns corresponding to the time unit(s), pollutant concentrations, and station identifiers.

    - id_vars : list
        The identifier variables used to group the data (e.g., ['station', 'year']).

    - labels : list
        A list containing the name of the time unit column and its descriptive label (e.g., ['year', 'Year']).

    - parameters : list
        A list of pollutant columns to include in the visualization (e.g., ['PM2.5', 'PM10']).

    - title_template : str, optional (default: "Concentration Data by Station and {}")
        A string template for the chart title, with the placeholder `{}` being replaced by the descriptive label of the time unit.

    Returns:
    - plotly.graph_objects.Figure
        A Plotly figure object representing the line chart, visualizing the pollutant concentrations by station over the specified time unit(s).
    """

    # Check if the dataframe empty or not
    if data.empty:
      raise ValueError("The data provided is empty. Please check the aggregation process.")

    # For Better Visualization will Using Long Format with Function melt()
    df_melted = data.melt(
        id_vars=id_vars,
        value_vars=parameters,
        var_name='Pollutant',
        value_name='Concentration'
    )

    # Define Free Variabel to Set the Visualization Plot
    if 'month' in df_melted.columns:
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        df_melted['month_name'] = df_melted['month'].map(month_names)
        labels[0] = 'month_name'
        tick_mode = 'array'
        tick_val = list(month_names.values())
        tick_text = tick_val
        tick_angle = 45
    elif 'hour' in df_melted.columns:
        df_melted['AM_PM'] = df_melted['hour'].apply(lambda x: f"{x % 12 or 12} {'AM' if x < 12 else 'PM'}")
        labels[0] = 'AM_PM'
        tick_mode = 'array'
        tick_val = df_melted['AM_PM'].unique()
        tick_text = tick_val
        tick_angle = 90
    else:
        tick_mode = 'linear'
        tick_val = None
        tick_text = None
        tick_angle = None

    fig = px.line(
        df_melted,
        x=f"{labels[0]}",
        y='Concentration',
        color='station',
        labels={labels[0]: labels[1], 'Concentration': 'Concentration', 'station': 'Station'},
        line_shape='linear'
    )

    # Set the Layout Visualization
    fig.update_layout(
        title=title_template,
        legend_title='Station',
        template='simple_white',
        width=900,
        height=450,
        title_font=dict(size=16),
        xaxis=dict(
            title="",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=False,
            tickmode=tick_mode,
            tickformat=".0f",
            tickvals=tick_val,
            ticktext=tick_text,
            tickangle=tick_angle
        ),
        yaxis=dict(
            title="",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='lightgrey'
        ),
        legend=dict(
            groupclick='toggleitem',
            title='Station',
            font=dict(size=12),
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='ghostwhite'
    )

    return fig

def ploty_bar_visualization(data, id_vars, labels, parameters, title_template="Average Concentration Data by Station"):
    """
    Visualizes pollutant concentration data in a stacked bar plot, with dynamic hover information
    based on the type of time-related data available (month, year, day, or AM/PM).

    Parameters:
    - data : pandas.DataFrame
        The input dataframe containing the data to be visualized.

    - id_vars : list of str
        The columns in the dataframe that will be used for grouping the x-axis (e.g., station name).

    - labels : list of str
        A list where the first item is the column used for coloring the bars (e.g., 'year')
        and the second item is the label for that column in the plot.

    - parameters : list of str
        A list containing the names of the pollutant parameters (e.g., ['PM2.5']) to be plotted.

    - title_template : str, optional, default="Average Concentration Data by Station and {}"
        The template for the plot's title. The `{}` placeholder will be replaced with the second
        item in the `labels` list.

    Returns:
    - plotly.graph_objects.Figure
        The Plotly figure object containing the bar plot visualization.
    """

    # Default hover customiation
    custom_value = 'datetime'
    hover_temp = "Date: %{customdata[0]}<br>"

    # Customize Month Column and Hover
    if 'month' in data.columns:
      month_names = {
          1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
          7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
      }
      data['Month'] = data['month'].map(month_names)
      custom_value = 'Month'
      hover_temp = "Month: %{customdata[0]}<br>"

    # Customize Year Column and Hover
    elif 'year' in data.columns:
      custom_value = 'year'
      hover_temp= "Year: %{customdata[0]}<br>"

    # Customize Day Column and Hover
    elif 'day' in data.columns:
      custom_value = 'day'
      hover_temp= "Day: %{customdata[0]}<br>"

    # Customize Clocl Column and Hover
    else:
      data['AM_PM'] = data['hour'].apply(lambda x: f"{x % 12 or 12} {'AM' if x < 12 else 'PM'}")
      custom_value = 'AM_PM'
      hover_temp= "Hour: %{customdata[0]}<br>"

    # Do Sorting Data by Parameter
    data = data.sort_values(by=parameters[0])

    # Wrap All Variabel Visualization
    fig = px.bar(
        data,
        x=id_vars[0],
        y=parameters,
        color=id_vars[1],
        barmode='stack',
        hover_data=None,
    )

    # Customize Hover
    fig.update_traces(customdata=data[[custom_value]].values)
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>"
        + "Concentration: %{y:.2f}<br>"
        + hover_temp
    )

    # Set the Layout Visualization
    fig.update_layout(
        title=title_template,
        legend_title='Station',
        template='simple_white',
        width=900,
        height=450,
        title_font=dict(size=16),
        xaxis=dict(
            title="",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=False,
            tickformat=".0f",
        ),
        yaxis=dict(
            title="",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        legend=dict(
            title='Station',
            font=dict(size=12),
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='ghostwhite'
    )

    return fig

def ploty_geospatial_visualization(filtered_df, selected_filters, parameter):
    """
    Generates a geospatial visualization (scatter plot on a map) based on the given DataFrame
    and selected filters. The plot displays pollutant data based on the selected year, month,
    day, and hour, with additional data about the stations.

    The function filters the DataFrame based on the selected filters (year, month, day, hour),
    groups the data by the relevant columns, and visualizes it using Plotly's scatter_mapbox
    function.

    Parameters:
    - filtered_df (pd.DataFrame): The input DataFrame containing the data to be visualized.
    - selected_filters (list): A list of selected filters for the visualization in the order
      [selected_year, selected_month, selected_day, selected_hour].
    - parameter (list): A list containing the column name of the pollutant (e.g., ["PM2.5"]).

    Returns:
    - fig (plotly.graph_objs.Figure): The Plotly figure object containing the geospatial
      visualization.
    """

    # Extract selected filters
    selected_year, selected_month, selected_day, selected_hour = selected_filters

    # Define Group Column and Hover for Visualization
    group_cols = ["station", "latitude", "longitude"]
    hover_cols = [parameter[0]]

    if selected_hour is not None:
        filtered_df["hour"] = filtered_df["datetime"].dt.hour
        group_cols = ["hour"] + group_cols
        hover_cols = ["hour", parameter[0]]
    elif selected_day is not None:
        filtered_df["day"] = filtered_df["datetime"].dt.day
        group_cols = ["day"] + group_cols
        hover_cols = ["day", parameter[0]]
    elif selected_month is not None:
        filtered_df["month"] = filtered_df["datetime"].dt.month
        group_cols = ["month"] + group_cols
        hover_cols = ["month", parameter[0]]
    elif selected_year is not None:
        filtered_df["year"] = filtered_df["datetime"].dt.year
        group_cols = ["year"] + group_cols
        hover_cols = ["year", parameter[0]]

    # Handle missing columns or no filters applied
    if len(group_cols) == 3:
        hover_cols = [parameter[0]]

    # Sort the Filter Dataframe
    filtered_df = (
        filtered_df.groupby(group_cols)
        .agg({parameter[0]: "mean"})
        .reset_index()
    )

    # Do Visualization
    fig = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color=parameter[0],
        size=parameter[0],
        hover_name="station",
        hover_data=hover_cols,
        mapbox_style="carto-positron",
        title="Geospatial Visualization Polutan Data",
        zoom=5,
    )

    # Adjust layout to remove gaps and specify the figure height
    fig.update_layout(
        height=400,
        margin={"r": 100, "t": 50, "l": 0, "b": 0},  # Adjust margin
    )

    return fig

def check_correlation(df):
    """
    Plots a correlation matrix heatmap for the numeric columns of a DataFrame.

    This function calculates the Pearson correlation coefficient between each
    pair of numeric columns, excluding specified non-numeric columns, and
    visualizes the correlation matrix as a heatmap. Correlation coefficients
    range from -1 to 1, with values closer to 1 or -1 indicating stronger correlations.
    - Positive values indicate a positive correlation.
    - Negative values indicate a negative correlation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numeric and non-numeric data.

    Returns:
    - None: This function only displays the heatmap and does not return any values.
    """

    # Drop column that not will use in make correlation matrix
    df_cor = df.drop(columns=['station', 'wd', 'datetime', 'RAIN'])

    # Get correlation value and store in correlation variabel
    correlation_matrix = df_cor.corr()

    # Visualize the correlation value into correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt)



# """
# Segment 2 : Make Filter Layout in Side Bar

# This section will make filter layout in side bar using streamlit method siderbar.
# for the input is start date and end date. That will pass to new dataframe main_df
# that contain data from selected date
# """

# Load DataFrame and Define Variabel for Sidebar
final_df = load_dataframe('dashboard/final_df.csv')
p_polution_final_df = final_df.drop(columns=['TEMP','PRES','DEWP','RAIN','wd','WSPM'])

# Make Filter Layout in Side Bar
with st.sidebar:

  # Set image logo
  st.image("data/aqi-logo.png")

  # Set start_date and end_date from input
  p_polution_final_df["date"] = pd.to_datetime(p_polution_final_df["datetime"]).dt.date
  min_date = final_df["datetime"].min()
  max_date = final_df["datetime"].max()

  selected_dates = st.date_input(
      label="Rentang Waktu",
      min_value=min_date,
      max_value=max_date,
      value=[min_date, max_date]
  )

  # If only one date is selected, set start_date and end_date to the same date
  if len(selected_dates) == 1:
      start_date = selected_dates[0]
      end_date = selected_dates[0]
  else:
      start_date, end_date = selected_dates

# Define dataframe filter
main_df = p_polution_final_df[(p_polution_final_df["date"] >= start_date) & (p_polution_final_df["date"] <= end_date)]
main_df = main_df.drop(columns=["date"])



# """
# Segment 3 : Define the Variabel That will Use for Dashboard Visualization

# This section will add some visulization that will use in dashboard that also correlated
# with business question. The visualization that will use wil same as the section
# Data Visualization before.
# """

# Initialize Header Dashboard
st.header("Dicoding Air Quality Dashboard 🆘")

# Variabel for Visualization
st.subheader("Select Parameter and Type Visualization")
parameter = st.radio(
    "Choose Parameter Visualization:",
    ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"],
    horizontal=True
)

visualization_type = st.radio(
    "Choose Visualization Type:",
    ["Annual", "Monthly", "Daily", "Hours"],
    horizontal=True
)

title= f"Trend of {parameter} - {visualization_type} Aggregation"
title_bar= f"Best and Worst of {parameter} - {visualization_type} Aggregation"

# Visualize 1 : Make Line Visualization for Trend Pollution Parameter
st.subheader("Trend Polution Parameter 📈")
st.markdown("This chart shows the trend of the selected air quality parameter over time.")

agg_df, id_vars, labels = aggregate_data(main_df, visualization_type)
fig = ploty_line_visualization(agg_df, id_vars, labels, [parameter], title_template=title)
st.plotly_chart(fig, use_container_width=True)

explanation = """
- Visualisasi dengan menggunakan line plot dapat melihat trend data baik dalam kategori Tahunan, Bulanan, Harian, dan bahkan Jam.
- Dalam menggunakannya pastikan sudah menetapkan rentangan tanggal dan memilih tipe visualisasi yang sesuai.
- Jika rentangan tahunan, maka visualisasi yang dapat ditampilkan adalah trend antar tahun, bulan, hari, dan jam.
- Sebaliknya jika rentangan hanya 1 bulan, maka visualisasi yang muncul nanti hanya rentangan hari, dan juga bisa memilih rentangan jam.
"""

with st.expander("Explanation: "):
    st.markdown(explanation)

# Visualize 2: Make Bar Visualization for Show Best and Worst Year, Month, Day, and Hours in Polution Parameter
st.subheader("Best and Worst Polutan Parameter According to Datetime 📊 ")
st.markdown("This bar chart shows the worst and best selected air quality parameter in a time.")

bar_fig = ploty_bar_visualization(agg_df, id_vars, labels, [parameter], title_template=title_bar)
st.plotly_chart(bar_fig, use_container_width=True)

explanation = """
- Penggunaan visualisasi bar plot ini menunjukkan perbandingan data baik dalam tahun, bulan, hari, dan jam.
- Jika tidak ada data yang dibandingkan maka visualisasi akan menggunakan data yang ada. Sebagai contoh:
  - Jika memilih rentang nilai dalam satu bulan, maka jika memilih tipe visualisasi 'Annual'
    visualisasi akan tetap jalan dengan data keseluruhan di tahun itu.
  - Ini sama dengan jika memilih tipe visualisasi 'Monthly' maka tetap menampilkan visualisasi hanya pada bulan itu saja.
  - Akan tetapi jika memlih tipe visualisasi 'Daily' atau 'Hourly' baru akan ada nilai yang dibandingkan.
- Nilai pada data sudah dilakukan proses sorting jadi data teratas adalah data dengan konsentrasi tertinggi
  baik pada tipe Tahun, Bulan, Hari, atau Jam.
"""

with st.expander("Explanation: "):
    st.markdown(explanation)

# Visualize 3: Make Geospasial Visualization for Area of Polution
st.subheader("Area of Polution 🌍")
st.markdown("This map shows the distribution of the selected air quality parameter across different locations.")

stations_coords = {
    "Aotizhongxin": (41.731242, 123.456778),
    "Changping": (40.221, 116.2312),
    "Dingling": (40.28998423518348, 116.2393424781757),
    "Dongsi": (40.10208908941478, 116.31657335910373),
    "Guanyuan": (39.94113871141321, 116.3610710753842),
    "Gucheng": (39.91270053243136, 116.1868698799306),
    "Huairou": (43.06043347888646, 117.46726428196578),
    "Nongzhanguan": (39.93978579546827, 116.46859787734736),
    "Shunyi": (40.151287025024715, 116.69280368021326),
    "Tiantan": (39.88189413732897, 116.42047003643812),
    "Wanliu": (39.99843210685499, 116.25774299569612),
    "Wanshouxigong": (39.90816416629832, 116.26439549963654)
}

spatial_df = final_df.copy()
spatial_df['date'] = pd.to_datetime(spatial_df["datetime"]).dt.date
spatial_df = spatial_df[(spatial_df["date"] >= start_date) & (spatial_df["date"] <= end_date)]
spatial_df.drop(columns=["date"])

spatial_df["latitude"] = spatial_df["station"].map(lambda x: stations_coords[x][0])
spatial_df["longitude"] = spatial_df["station"].map(lambda x: stations_coords[x][1])
spatial_df["datetime"] = pd.to_datetime(spatial_df["datetime"])

st.markdown("#### Select Geospasial Datetime")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

all_cols = [col1, col2, col3, col4]
all_filters, filtered_df = filter_selected_datetime(spatial_df, all_cols)

fig = ploty_geospatial_visualization(filtered_df, all_filters, [parameter])
st.plotly_chart(fig, use_container_width=True)

explanation = """
- Ini adalah persebaran parameter polusi sesuai dengan latitude dan longitude-nya.
- Dalam persebaran ini diharuskan memilih persebaran sesuai dengan tahun, lalu bulan, hari, dan waktunya.
- Jika peta persebaran tidak langsung menunjukkan lokasinya maka bisa dikatakan dalam waktu itu tidak ada
data persebaran yang tersedia. Sebagai contoh:
  - Memilih tanggal 1 Januari 2023 tidak akan memunculkan nilai persebaran peta. Karena tidak ada data pada tanggal tersebut.
"""

with st.expander("Explanation: "):
    st.markdown(explanation)

# Visualize 4: Make Matrix Correlation for Parameter in Air Quality Index
st.subheader("Matrix Correlation Polutan Parameter 💐")
st.markdown("This matrix shows the correlation between different air quality parameters.")

check_correlation(final_df)

explanation = """
- Korelasi adalah ukuran statistik yang menunjukkan sejauh mana dua variabel beruhubungan.
- Nilai koefisien korelasi berkisar dari -1 hingga 1:
  - **1** menunjukkan korelasi positif sempurna (kedua variabel meningkat bersama).
  - **-1** menunjukkan korelasi negatif sempurna (saat satu variabel meningkat, variabel lainnya menurun).
  - **0** menunjukkan tidak ada korelasi antara variabel-variabel tersebut.
- Pada heatmap ini, warna yang lebih gelap mewakili korelasi yang lebih kuat, dengan warna biru menunjukkan korelasi negatif dan merah menunjukkan korelasi positif.
- Korelasi positif menunjukkan bahwa saat satu variabel meningkat, yang lainnya cenderung meningkat juga, dan sebaliknya untuk korelasi negatif.
"""

with st.expander("Explanation: "):
    st.markdown(explanation)

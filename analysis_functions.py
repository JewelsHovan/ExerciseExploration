# Functions for analysis and preprocessing!

# Libraries
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
# plotly libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


def plot_interactive_histograms(df1, df2, measurements, name1='Running', name2='Cycling', barmode='overlay', histnorm='percent'):
    """
    Plots interactive percentage histograms for two datasets to compare their distributions across different measurements.

    Parameters:
    - df1, df2: DataFrames containing the data.
    - measurements: Dictionary of the measurements and their corresponding units.
    - name1, name2: Names for the histograms of df1 and df2, respectively.
    - barmode: Mode for the bar chart. Can be 'overlay' or 'stack'.
    - histnorm: Normalization type for the histograms. Default is 'percent'.
    """

    # Initialize the figure with the first measurement's data and unit
    initial_measurement, initial_unit = list(measurements.items())[0]

    # Filter initial data to remove zeros or negative values
    data1 = df1[initial_measurement][df1[initial_measurement] > 0]
    data2 = df2[initial_measurement][df2[initial_measurement] > 0]

    # Create the figure
    fig = go.Figure()

    # Add histogram traces
    fig.add_trace(go.Histogram(x=data1, name=name1, histnorm=histnorm, marker_color='skyblue'))
    fig.add_trace(go.Histogram(x=data2, name=name2, histnorm=histnorm, marker_color='coral'))

    # Customize layout
    fig.update_layout(
        title={'text': f'{initial_measurement.replace("_", " ").title()} Distribution', 'x': 0.5},
        xaxis=dict(title=f'{initial_measurement.replace("_", " ").title()} ({initial_unit})', title_font=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(title='Percentage', title_font=dict(size=14), tickfont=dict(size=12)),
        barmode=barmode,
        plot_bgcolor='ivory',
        font=dict(family='Arial, sans-serif', size=12, color='DarkSlateGray'),
        height = 550,
        updatemenus=[{
            'buttons': [
                {
                    'method': 'update',
                    'label': measurement.replace("_", " ").title(),
                    'args': [{'x': [df1[measurement][df1[measurement] > 0], df2[measurement][df2[measurement] > 0]]},
                             {'xaxis': {'title': f'{measurement.replace("_", " ").title()} ({unit})'},
                              'title': {'text': f'{measurement.replace("_", " ").title()} Distribution'}}]
                } for measurement, unit in measurements.items()
            ],
            'direction': 'down',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.5,
            'xanchor': 'center',
            'y': 1.15,
            'yanchor': 'top'
        }]
    )

    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    # Show the figure
    fig.show()

import plotly.graph_objects as go

def create_interactive_box_plots(running_df, cycling_df, measurements):
    """
    Creates interactive box plots for comparing measurements between running and cycling activities.

    Parameters:
    - running_df (DataFrame): The DataFrame containing running activity data.
    - cycling_df (DataFrame): The DataFrame containing cycling activity data.
    - measurements (dict): A dictionary of the measurements to plot, with keys as measurement names and values as their units.
    """
    
    # Create the figure
    fig = go.Figure()

    # Initialize the figure with the first measurement's data and unit
    initial_measurement, initial_unit = list(measurements.items())[0]
    fig.add_trace(go.Box(y=running_df[initial_measurement], name='Running', marker_color='deepskyblue', boxmean='sd'))
    fig.add_trace(go.Box(y=cycling_df[initial_measurement], name='Cycling', marker_color='tomato', boxmean='sd'))

    # Customize the layout with a more appealing style
    fig.update_layout(
        title={'text': f'{initial_measurement.replace("_", " ").title()} Box Plots', 'x': 0.5},
        yaxis=dict(title=f'{initial_unit}', title_font=dict(size=14), tickfont=dict(size=12)),
        xaxis=dict(title='Activity', title_font=dict(size=14), tickfont=dict(size=12)),
        paper_bgcolor='whitesmoke',
        plot_bgcolor='lavender',
        font=dict(family='Arial, sans-serif', size=12, color='RebeccaPurple'),
        updatemenus=[{
            'buttons': [
                {
                    'method': 'update',
                    'label': measurement.replace("_", " ").title(),
                    'args': [{'y': [running_df[measurement], cycling_df[measurement]],
                              'marker': [{'color': 'deepskyblue'}, {'color': 'tomato'}]},
                             {'yaxis': {'title': f'{unit}', 'title_font': {'size': 14}, 'tickfont': {'size': 12}},
                              'title': {'text': f'{measurement.replace("_", " ").title()} Box Plots'}}]
                } for measurement, unit in measurements.items()
            ],
            'direction': 'down',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.5,
            'xanchor': 'center',
            'y': 1.15,
            'yanchor': 'top'
        }]
    )

    # Show the figure
    fig.show()


# Bar charts of duration
def plot_activity_time(data, time_column, title='Exercise Time Per Activity', annotation_x=None, annotation_y=None):
    """
    Plot a bar chart of activity times with an overlaid average line, including a text annotation for the average value with an arrow.

    Parameters:
    - data: DataFrame containing the activity data.
    - time_column: The name of the column in the DataFrame that contains time lengths.
    - title: The title of the plot.
    - annotation_x: The x-coordinate for the annotation. If None, defaults to the last x-value.
    - annotation_y: The y-coordinate for the annotation. If None, defaults to the average y-value.
    """
    # Extracting time data
    time_by_activity = data[time_column]
    x_data = time_by_activity.index
    y_data = time_by_activity

    # Calculate the average time
    y_data_avg = np.mean(y_data)

    # Set default annotation positions if not specified
    if annotation_x is None:
        annotation_x = x_data[-1]
    if annotation_y is None:
        annotation_y = y_data_avg

    # Create the bar trace for activity times
    bar_trace = go.Bar(x=x_data, y=y_data, name='Activity Bar')

    # Create a trace for the horizontal average line
    average_line_trace = go.Scatter(
        x=x_data,
        y=[y_data_avg] * len(x_data),
        mode='lines',
        name='Average Exercise Length',
        line=dict(dash='dot', color='red')
    )

    # Create the layout
    layout = go.Layout(
        title={'text': title, 'x': 0.5},
        xaxis=dict(title='Activity Periods', tickvals=list(range(len(x_data))), ticktext=list(x_data)),
        yaxis=dict(title='Time'),
        width=900,
        height=450
    )

    # Create the figure with both the bar and line traces
    fig = go.Figure(data=[bar_trace, average_line_trace], layout=layout)

    # Add a text annotation for the average exercise time with an arrow
    fig.add_annotation(
        x=annotation_x,
        y=annotation_y,
        text=f'Average: {y_data_avg:.2f} minutes',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        font=dict(size=12, color="red"),
        ax=-10,  # Adjusts the arrow's tail along the x-axis (negative values point left)
        ay=-70,  # Adjusts the arrow's tail along the y-axis (negative values point down)
        bgcolor="rgba(255,255,255,0.9)"
    )

    # Plot the figure
    fig.show()

def minmax_scale(df):
    """
    Scales the DataFrame using min-max scaling.

    Parameters:
    - df (DataFrame): DataFrame to be scaled.

    Returns:
    - DataFrame: Scaled DataFrame.
    """
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized

def generate_heatmaps(run_by_day_period, cycle_by_day_period):
    """
    Generates and displays heatmaps for average values over time for running and cycling data.

    Parameters:
    - run_by_day_period (DataFrame): DataFrame containing running activity data.
    - cycle_by_day_period (DataFrame): DataFrame containing cycling activity data.
    """
    # Prepare data for heatmaps
    avg_run_date = run_by_day_period.groupby('date')[['heart_rate_mean', 'cadence_mean', 'enhanced_speed_mean', 'time_length', 'distance_per_min']].mean()
    scaled_avg_run_date = minmax_scale(avg_run_date.copy())
    transposed_avg_run_date = scaled_avg_run_date.transpose()

    avg_cycle_date = cycle_by_day_period.groupby('date')[['heart_rate_mean', 'cadence_mean', 'enhanced_speed_mean', 'time_length', 'distance_per_min']].mean()
    scaled_avg_cycle_date = minmax_scale(avg_cycle_date.copy())
    transposed_avg_cycle_date = scaled_avg_cycle_date.transpose()

    # Create subplots with improved layout options
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Running Metrics Over Time", "Cycling Metrics Over Time"), vertical_spacing=0.15)

    # Define a custom color scale
    custom_color_scale = 'Greens'  # Try different color scales to see what works best for your data

    readable_metric_names = {
        'heart_rate_mean': 'Heart Rate Mean',
        'cadence_mean': 'Cadence Mean',
        'enhanced_speed_mean': 'Enhanced Speed Mean',
        'time_length': 'Time Length',
        'distance_per_min': 'Distance per Minute'
    }
    # Use the readable names for y-axis ticks
    y_ticks = [readable_metric_names[metric] for metric in transposed_avg_run_date.index]

    # Add heatmaps with enhanced aesthetic options
    heatmap_args = {
        'colorscale': custom_color_scale,
        'colorbar': dict(title='Scaled Value', len=0.45, yanchor='middle'),
        'showscale': False  # Only show color scale on the last heatmap for cleaner look
    }
    fig.add_trace(go.Heatmap(z=transposed_avg_run_date.values, x=transposed_avg_run_date.columns, y=y_ticks, **heatmap_args), row=1, col=1)
    fig.add_trace(go.Heatmap(z=transposed_avg_cycle_date.values, x=transposed_avg_cycle_date.columns, y=y_ticks, **heatmap_args), row=2, col=1)

    # Enhance axis labels and ticks
    axis_args = {
        'tickfont': dict(size=10),
        'title_font': dict(size=12),
        'tickangle': 45  # Rotate tick labels for better readability
    }
    fig.update_xaxes(title_text="Date", **axis_args, row=1, col=1)
    fig.update_xaxes(title_text="Date", **axis_args, row=2, col=1)
    fig.update_yaxes(title_text="Metrics", **axis_args, row=1, col=1)
    fig.update_yaxes(title_text="Metrics", **axis_args, row=2, col=1)

    # Update overall layout for better aesthetics
    fig.update_layout(
        height=700,
        width=1200,
        title_text="Peformance Metrics Over Time for Running and Cycling",
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#000"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Show color scale for the last heatmap
    fig.data[-1].showscale = True

    # Show the figure
    fig.show()

def create_radar_plot(run_by_day_period, cycle_by_day_period, selected_categories):
    """
    Creates an enhanced radar plot comparing performance metrics between running and cycling with improved aesthetics.

    Parameters:
    - run_by_day_period (DataFrame): DataFrame containing running activity data.
    - cycle_by_day_period (DataFrame): DataFrame containing cycling activity data.
    - selected_categories (list): List of column names to include in the radar plot.
    """
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply the scaler to each column individually
    running_values_scaled = run_by_day_period[selected_categories].copy()
    cycling_values_scaled = cycle_by_day_period[selected_categories].copy()

    for category in selected_categories:
        running_values_scaled[category] = scaler.fit_transform(running_values_scaled[[category]])
        cycling_values_scaled[category] = scaler.fit_transform(cycling_values_scaled[[category]])

    # Calculate means
    running_means = running_values_scaled.mean()
    cycling_means = cycling_values_scaled.mean()

    # Radar chart categories with more readable names
    categories_readable = ['Mean Heart Rate', 'Max Heart Rate', 'Cadence Mean', 'Distance Per Minute', 'Duration']

    # Define color scheme
    colors = {
        'Running': '#1f77b4',  # Blue
        'Cycling': '#ff7f0e'   # Orange
    }

    # Create traces for Running and Cycling with enhanced aesthetics
    trace_running = go.Scatterpolar(
        r=running_means.tolist(), 
        theta=categories_readable,
        fill='toself',
        name='Running',
        fillcolor=colors['Running'],
        line=dict(color=colors['Running']),
        opacity=0.6
    )

    trace_cycling = go.Scatterpolar(
        r=cycling_means.tolist(), 
        theta=categories_readable,
        fill='toself',
        name='Cycling',
        fillcolor=colors['Cycling'],
        line=dict(color=colors['Cycling']),
        opacity=0.6
    )
    # Create a table to display mean values
    table = go.Table(
        header=dict(values=['Metric', 'Running Mean', 'Cycling Mean'], fill_color='paleturquoise', align='left'),
        cells=dict(values=[categories_readable, running_means.round(2).tolist(), cycling_means.round(2).tolist()], fill_color='lavender', align='left')
    )

    # Combine radar chart and table in a single figure
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.3],
        specs=[[{'type': 'table'}, {'type': 'polar'}]]
    )
    
    fig.add_trace(table, row=1, col=1)
    fig.add_trace(trace_running, row=1, col=2)
    fig.add_trace(trace_cycling, row=1, col=2)

    # Define the enhanced layout of the radar chart
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 0.8],
                gridcolor='silver',
                linecolor='silver'
            ),
            angularaxis=dict(
                rotation = 90,
                direction = 'clockwise',
                gridcolor='silver',
                linecolor='silver'
            ),
            bgcolor='rgba(223, 223, 223, 0.3)'
        ),
        title={
            'text': 'Performance Metrics Comparison',
            'x': 0.5,
            'font': dict(
                family='Arial, sans-serif',
                size=20,
                color='black'
            )
        },
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#000"
        )
    )

    # Show the figure
    fig.show()


def plot_heart_rate_data(run_data, cycle_data):
    """
    Plots heart rate data for running and cycling activities with options to switch between
    'activity period' and 'date' on the x-axis.
    
    Parameters:
    run_data (DataFrame): Data containing running activities with columns for 'activity period', 'date', 
                          'max heart rate', and 'mean heart rate'.
    cycle_data (DataFrame): Data containing cycling activities with similar columns as run_data.
    """
    
    # Create the initial line plot with 'heart_rate_max'
    fig = go.Figure()

    # Define traces for each combination of activity and heart rate type
    # Running - Max HR
    fig.add_trace(go.Scatter(x=run_data.activity_period, y=run_data.heart_rate_max, 
                             mode='markers', name='Running Max HR', visible=True))
    # Cycling - Max HR
    fig.add_trace(go.Scatter(x=cycle_data.activity_period, y=cycle_data.heart_rate_max, 
                             mode='markers', name='Cycling Max HR', visible=True))
    # Running - Mean HR (initially hidden)
    fig.add_trace(go.Scatter(x=run_data.activity_period, y=run_data.heart_rate_mean, 
                             mode='markers', name='Running Mean HR', visible=False))
    # Cycling - Mean HR (initially hidden)
    fig.add_trace(go.Scatter(x=cycle_data.activity_period, y=cycle_data.heart_rate_mean, 
                             mode='markers', name='Cycling Mean HR', visible=False))

    # Dropdown menu for heart rate type
    menu_hr = dict(
        buttons=[
            dict(args=[{'visible': [True, True, False, False]},
                       {'title': 'Heart Rate Max'}],
                 label='Max HR',
                 method='update'),
            dict(args=[{'visible': [False, False, True, True]},
                       {'title': 'Heart Rate Mean'}],
                 label='Mean HR',
                 method='update')
        ],
        direction='down',
        pad={'r': 10, 't': 10},
        showactive=True,
        x=0.1,
        xanchor='left',
        y=1.15,
        yanchor='top',
        bgcolor='lightgrey'
    )

    # Dropdown menu for x-axis data
    menu_xaxis = dict(
        buttons=[
            dict(args=[{'x': [run_data.activity_period, cycle_data.activity_period, 
                              run_data.activity_period, cycle_data.activity_period]},
                       {'xaxis': {'title': 'Activity Period'}}],
                 label='Activity Period',
                 method='restyle'),
            dict(args=[{'x': [run_data.date, cycle_data.date, 
                              run_data.date, cycle_data.date]},
                       {'xaxis': {'title': 'Date'}}],
                 label='Date',
                 method='restyle')
        ],
        direction='down',
        pad={'r': 10, 't': 10},
        showactive=True,
        x=0.3,
        xanchor='left',
        y=1.15,
        yanchor='top',
        bgcolor='lightgrey'
    )

    # Add the dropdown menus to the figure
    fig.update_layout(updatemenus=[menu_hr, menu_xaxis])

    # Customize the layout
    fig.update_layout(
        title={'text': 'Scatter Plot of Heart Rate over Activity Periods', 'x': 0.5},
        xaxis=dict(title='Activity Period'),
        yaxis=dict(title='Heart Rate'),
        annotations=[dict(text='Metric:', x=0, xref='paper', y=1.10, yref='paper',
                          align='left', showarrow=False)]
    )

    # Show the figure
    fig.show()

def plot_intensity_proportions(cycling_df, running_df, cycle_by_day_period, run_by_day_period):
    # Calculate intensity proportions
    cycling_intensity_proportions = cycling_df['intensity'].value_counts(normalize=True)
    running_intensity_proportions = running_df['intensity'].value_counts(normalize=True)

    day_period_cycling_intensity_proportions = cycle_by_day_period['intensity'].value_counts(normalize=True)
    day_period_running_intensity_proportions = run_by_day_period['intensity'].value_counts(normalize=True)

    # Combine the proportions into single DataFrames
    intensity_proportions = pd.DataFrame({
        'Cycling': cycling_intensity_proportions, 
        'Running': running_intensity_proportions
    }).sort_index()

    day_period_intensity_proportions = pd.DataFrame({
        'Cycling': day_period_cycling_intensity_proportions, 
        'Running': day_period_running_intensity_proportions
    }).sort_index()

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Bar(name='Running', y=intensity_proportions.index, x=intensity_proportions['Running'], orientation='h'),
    )
    fig.add_trace(
        go.Bar(name='Cycling', y=intensity_proportions.index, x=intensity_proportions['Cycling'], orientation='h'),
    )
    fig.add_trace(
        go.Bar(name='Running - Activity Period', y=day_period_intensity_proportions.index, x=day_period_intensity_proportions['Running'], orientation='h', visible=False),
    )
    fig.add_trace(
        go.Bar(name='Cycling - Activity Period', y=day_period_intensity_proportions.index, x=day_period_intensity_proportions['Cycling'], orientation='h', visible=False),
    )

    # Update the layout
    fig.update_layout(
        title='Comparison of Exercise Intensity Distribution (Proportions)',
        yaxis_title='Intensity',
        xaxis_title='Proportion',
        barmode='group',
        height=550,
        updatemenus=[
            dict(
                buttons=list([
                    dict(label='Overall Intensity',
                         method='update',
                         args=[{'visible': [True, True, False, False]},
                               {'title': 'Overall Exercise Intensity Distribution'}]),
                    dict(label='Activity Period Intensity',
                         method='update',
                         args=[{'visible': [False, False, True, True]},
                               {'title': 'Day Period Exercise Intensity Distribution'}]),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.6,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    fig.show()


def create_activity_map(running_df, features, initial_center, initial_zoom, mapbox_access_token):
    fig = go.Figure()

    # Precompute centers and min/max values for each feature
    centers = {name: {'lon': group['position_long_degrees'].mean(), 'lat': group['position_lat_degrees'].mean()}
               for name, group in running_df.groupby('activity_period')}
    feature_bounds = {feature: {'min': running_df[feature].min(), 'max': running_df[feature].max()}
                      for feature in features}

    # Add traces for each activity period with initial color mapping based on the first feature
    for name, group in running_df.groupby('activity_period'):
        fig.add_trace(go.Scattermapbox(
            name=f'Activity {name}',
            mode='markers+lines',
            lon=group['position_long_degrees'],
            lat=group['position_lat_degrees'],
            marker={
                'size': 8,
                'color': group[features[0]],  # Initial color mapping based on the first feature
                'colorscale': 'Viridis',
                'cmin': feature_bounds[features[0]]['min'],
                'cmax': feature_bounds[features[0]]['max'],
                'colorbar': {'title': features[0].replace('_', ' ').title()},
            },
            line={'width': 3, 'color': 'red'},
            text=group['activity_period'],
            hoverinfo='text',
            visible=False
        ))

    # Dropdown for selecting activity periods
    activity_dropdown = [{
        'buttons': [
            {
                'args': [
                    {'visible': [i == index for i, _ in enumerate(fig.data)]},  # Trace visibility
                    {'mapbox.center.lon': centers[name]['lon'], 'mapbox.center.lat': centers[name]['lat'], 'mapbox.zoom': 13}  # Layout updates
                ],
                'label': f'Activity {name}',
                'method': 'update'
            } for index, (name, _) in enumerate(running_df.groupby('activity_period'))
        ],
        'direction': 'down', 'pad': {'r': 10, 't': 10}, 'showactive': True, 'x': 0.1, 'xanchor': 'left', 'y': 1.1, 'yanchor': 'top'
    }]

    # Dropdown for selecting features for color mapping
    feature_dropdown = [{
        'buttons': [{'args': [{'marker.color': [group[feature] for _, group in running_df.groupby('activity_period')],
                               'marker.cmin': feature_bounds[feature]['min'],
                               'marker.cmax': feature_bounds[feature]['max'],
                               'marker.colorbar.title': feature.replace('_', ' ').title()},
                              None],
                     'label': feature.replace('_', ' ').title(),
                     'method': 'restyle'} for feature in features],
        'direction': 'down', 'pad': {'r': 10, 't': 10}, 'showactive': True, 'x': 0.3, 'xanchor': 'left', 'y': 1.1, 'yanchor': 'top', 'bgcolor': '#CCCCCC', 'bordercolor': '#FFFFFF', 'font': dict(size=11)
    }]

    # Set the first trace to visible and update layout with dropdowns
    fig.data[0].visible = True

    fig.update_layout(
        mapbox={'accesstoken': mapbox_access_token, 'style': 'outdoors', 'center': initial_center, 'zoom': initial_zoom},
        updatemenus=activity_dropdown + feature_dropdown,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
    )

    fig.show()


def plot_activity_map_intensity(running_df, cycling_df, mapbox_access_token, map_style='outdoors', zoom=12):
    """
    Plots a Mapbox map with markers for running and cycling activities,
    colored based on intensity levels.

    Parameters:
    - running_df: DataFrame containing columns ['position_long_degrees', 'position_lat_degrees', 'intensity'] for running activities.
    - cycling_df: DataFrame containing similar columns for cycling activities.
    - mapbox_access_token: String, your Mapbox access token.
    - map_style: String, style of the Mapbox map. Defaults to 'outdoors'.
    - zoom: Int, initial zoom level of the map. Defaults to 12.
    """
    # Define a color map for intensity levels
    intensity_color_scale = {
        'Low': 'green',
        'Moderate': 'yellow',
        'High': 'red'
    }

    # Create a Plotly figure
    fig = go.Figure()

    # Add running trace with color mapping based on intensity
    fig.add_trace(go.Scattermapbox(
        name='Running',
        mode='markers',
        lon=running_df['position_long_degrees'],
        lat=running_df['position_lat_degrees'],
        marker={'size': 8, 'color': [intensity_color_scale[i] for i in running_df['intensity']]},
        text=running_df['intensity'],  # Hover text
        hoverinfo='text'
    ))

    # Add cycling trace with color mapping based on intensity
    fig.add_trace(go.Scattermapbox(
        name='Cycling',
        mode='markers',
        lon=cycling_df['position_long_degrees'],
        lat=cycling_df['position_lat_degrees'],
        marker={'size': 8, 'color': [intensity_color_scale[i] for i in cycling_df['intensity']]},
        text=cycling_df['intensity'],  # Hover text
        hoverinfo='text'
    ))

    # Update the layout to adjust map and legend properties
    fig.update_layout(
        mapbox={
            'accesstoken': mapbox_access_token,
            'style': map_style,
            'zoom': zoom,
            'center': {'lon': (running_df['position_long_degrees'].mean() + cycling_df['position_long_degrees'].mean()) / 2,
                       'lat': (running_df['position_lat_degrees'].mean() + cycling_df['position_lat_degrees'].mean()) / 2}
        },
        legend={'title': 'Activity Type', 'bgcolor': 'rgba(255, 255, 255, 0.5)'},
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
    )

    fig.show()



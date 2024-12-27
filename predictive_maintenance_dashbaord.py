import shap
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Set Streamlit layout
st.set_page_config(layout="wide", page_title="NASA Jet Engine Dataset")

st.title("NASA Jet Engine Predictive Maintenance")
st.markdown("_Prototype v0.4.1_")

def RUL_calculator(df, df_max_cycles):
    max_cycle = df_max_cycles["cycle"]
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='id', right_index=True)
    result_frame["RUL"] = result_frame["max_cycle"] - result_frame["cycle"] 
    result_frame.drop(['max_cycle'], axis=1, inplace=True)
    return result_frame

@st.cache_data
def load_data(path: str):  # Data loading function
    data = pd.read_csv(path, sep=" ", header=None)
    data.columns = ["id", "cycle", "op1", "op2", "op3",
                    "sensor1", "sensor2", "sensor3",
                    "sensor4", "sensor5", "sensor6",
                    "sensor7", "sensor8", "sensor9",
                    "sensor10", "sensor11", "sensor12",
                    "sensor13", "sensor14", "sensor15",
                    "sensor16", "sensor17", "sensor18",
                    "sensor19", "sensor20", "sensor21",
                    "sensor22", "sensor23"]
    
    data = data.drop(['sensor22', 'sensor23',
                      "op1", "op2", "op3", "sensor1",
                      "sensor5", "sensor6", "sensor10",
                      "sensor16", "sensor18", "sensor19",
                      "sensor14", "sensor13", "sensor12",
                      "sensor11"], axis=1)
    
    jet_id_and_rul = data.groupby(['id'])[["id", "cycle"]].max()
    jet_id_and_rul.set_index('id', inplace=True)
    jet_data = RUL_calculator(data, jet_id_and_rul)
    return jet_data

# Generate random latitude and longitude for engine IDs 1 to 100
np.random.seed(4)  # For reproducibility
engine_ids = range(1, 101)
latitudes = np.random.uniform(24.396308, 49.384358, 100)  # US latitude range
longitudes = np.random.uniform(-125.0, -66.93457, 100)    # US longitude range

# Create a separate DataFrame for locations
location_data = pd.DataFrame({
    'id': engine_ids,
    'latitude': latitudes,
    'longitude': longitudes
})

with st.sidebar:
    st.image("file.png", use_column_width=True)  # Replace 'file.png' with your image path or URL
    
    uploaded_file = st.file_uploader("Choose a file")  # Gets the path only

    if uploaded_file is None:
        st.info("Upload the file through config", icon="ℹ️")
        st.stop()

df = load_data(uploaded_file)  # Data loads here

with st.expander('Data Preview'):
    st.dataframe(df, use_container_width=True)

# Sidebar: Select engine ID
engine_ids = df['id'].unique()
selected_engine = st.sidebar.selectbox("Select Jet Engine ID", engine_ids)

# Filter location data for the selected engine
selected_engine_location = location_data[location_data['id'] == selected_engine]

# Function to visualize the engine on map
def display_engine_location(selected_engine, selected_engine_location):
    """
    Displays the location of the selected jet engine on a map using Plotly, with annotations for better visualization.
    
    Parameters:
        selected_engine (int): The ID of the selected engine.
        selected_engine_location (DataFrame): A DataFrame containing 'latitude' and 'longitude' of the selected engine.
    """
    st.subheader(f"Location for Jet Engine ID: {selected_engine}")
    if not selected_engine_location.empty:
        lat = selected_engine_location.iloc[0]['latitude']
        lon = selected_engine_location.iloc[0]['longitude']
        
        # Create a scattermapbox plot
        fig = go.Figure(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14,  # Adjust size for better visibility
                color='red',  # Highlight the marker with a distinct color
            ),
            text=[f"Engine ID: {selected_engine}"],  # Tooltip text
        ))

        # Add annotation to the map
        fig.add_annotation(
            x=lon,  # Longitude for annotation
            y=lat,  # Latitude for annotation
            text=f"<b>Engine ID: {selected_engine}</b>",  # Annotated text
            showarrow=True,
            arrowhead=1,
            arrowsize=1.5,
            arrowcolor="blue",
            ax=-30,  # X offset for the arrow
            ay=-50,  # Y offset for the arrow
            bgcolor="white",
            font=dict(size=12)
        )

        # Update layout for zoomed-out view
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=lat, lon=lon),  # Center on the selected engine
                zoom=6  # Keep the map zoomed out
            ),
            margin=dict(l=0, r=0, t=0, b=0),  # Remove margins for fullscreen view
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Location data not available for the selected engine.")
        
display_engine_location(selected_engine, selected_engine_location)

################# Old Map Visualization #####################
# Map visualization for the selected engine
# st.subheader(f"Location for Jet Engine ID: {selected_engine}")
# if not selected_engine_location.empty:
#     st.map(selected_engine_location[['latitude', 'longitude']])
# else:
#     st.warning("Location data not available for the selected engine.")

# Function to plot Sensor vs. RUL
def plot_sensor_vs_rul(sensor_name, engine_id, data, threshold=30):
    """
    Plots Sensor values vs. RUL for a specific engine ID and sensor, with a shaded area beyond a dynamic threshold.

    Parameters:
        sensor_name (str): The selected sensor name.
        engine_id (int): The selected engine ID.
        data (DataFrame): The dataset containing sensors, RUL, and IDs.
        threshold (int): The RUL threshold to mark critical zones (default is 30).
    """
    # Filter data for the selected engine ID
    filtered_data = data[data['id'] == engine_id]

    # Create the Plotly figure
    fig = px.line(
        filtered_data,
        x='RUL',
        y=sensor_name,
        title=f"{sensor_name} vs Remaining Useful Life (RUL) for Engine {engine_id}",
        labels={'RUL': 'Remaining Useful Life (RUL)', sensor_name: f'{sensor_name}'}
    )

    # Add a vertical line for the threshold
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Critical Threshold (RUL={threshold})")

    # Add a red transparent fill region after the threshold
    fig.add_shape(
        type="rect",
        x0=threshold,
        x1=min(filtered_data['RUL']),
        y0=min(filtered_data[sensor_name]),
        y1=max(filtered_data[sensor_name]),
        fillcolor="red",
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    # Reverse the x-axis for RUL to count down
    fig.update_xaxes(autorange="reversed")

    return fig


# Sidebar: Select sensor
sensors = [col for col in df.columns if "sensor" in col]
selected_sensor = st.sidebar.selectbox("Select Sensor", sensors)

# Plot Sensor vs. RUL for the selected engine and sensor
st.subheader(f"RUL for Jet Engine ID: {selected_engine} vs {selected_sensor}")
fig = plot_sensor_vs_rul(selected_sensor, selected_engine, df)
st.plotly_chart(fig, use_container_width=True)

left_column, middle_column, right_column = st.columns(3)

with left_column:
    fig = plot_sensor_vs_rul("sensor4", selected_engine, df)
    st.plotly_chart(fig, use_container_width=True)

with middle_column:
    fig = plot_sensor_vs_rul("sensor7", selected_engine, df)
    st.plotly_chart(fig, use_container_width=True)

with right_column:
    fig = plot_sensor_vs_rul("sensor15", selected_engine, df)
    st.plotly_chart(fig, use_container_width=True)

# Optionally display raw data for debugging or exploration
if st.checkbox("Show raw data for selected engine"):
    st.write(df[df['id'] == selected_engine])
    
#-------------------------------------------------------------------------

# Load your trained model here
try:
    loaded_model = joblib.load('models\logistic_regression_model.pkl') 
    loaded_scaler = joblib.load('models\minmax_scaler.pkl') 
except:
    st.warning("Please upload a trained model file.")
    st.stop()

# Prepare data for prediction
def prepare_data_for_prediction(df, engine_id, scaler):
    """
    Prepares data for prediction by scaling and retaining feature names.

    Parameters:
        df: DataFrame containing the full dataset
        engine_id: ID of the engine for which data is prepared
        scaler: Scaler object to normalize data

    Returns:
        Scaled DataFrame with feature names preserved
    """
    engine_data = df[df['id'] == engine_id]

    features = ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8',
                'sensor15', 'sensor17', 'sensor20', 'sensor21']

    # Extract features as a DataFrame
    X = engine_data[features]

    # Scale and return as a DataFrame
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)

    return X_scaled

def predict_probabilities(model, X):
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities

# Function to plot probabilities
def plot_probabilities(engine_id, df, model):
    X = prepare_data_for_prediction(df, engine_id, loaded_scaler)
    probabilities = predict_probabilities(model, X)
    engine_data = df[df['id'] == engine_id]

    # Create a color scale based on probability
    colors = px.colors.sequential.Viridis

    fig = go.Figure()

    # Add probability line
    fig.add_trace(go.Scatter(
        x=engine_data['RUL'],
        y=probabilities,
        mode='lines',
        name="Failure Probability",
        line=dict(color='rgba(0,100,80,0.8)', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,100,80,0.2)'
    ))

    # Add markers with color scale
    fig.add_trace(go.Scatter(
        x=engine_data['RUL'],
        y=probabilities,
        #mode='markers',
        marker=dict(
            size=8,
            color=probabilities,
            colorscale=colors,
            showscale=True,
            colorbar=dict(title="Probability")
        ),
        name="Data Points"
    ))

    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': f"Failure Probability Over RUL for Engine {engine_id}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Remaining Useful Life (RUL)",
        yaxis_title="Failure Probability",
        xaxis=dict(autorange="reversed", title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        hovermode='x unified',
        #plot_bgcolor='rgba(240,240,240,0.8)',
        #paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
    )

    # Add shapes for probability thresholds
    fig.add_shape(
        type="line",
        x0=min(engine_data['RUL']), x1=max(engine_data['RUL']),
        y0=0.5, y1=0.5,
        line=dict(color="red", width=2, dash="dash"),
        name="50% Threshold"
    )

    fig.add_annotation(
        x=max(engine_data['RUL']),
        y=0.5,
        text="50% Threshold",
        showarrow=False,
        yshift=10,
        font=dict(color="red")
    )

    # Make the chart responsive
    fig.update_layout(autosize=True)

    return fig

@st.cache_data
def compute_global_importance(_model, X):
    """
    Computes global feature importance using SHAP values.
    
    Parameters:
        _model: Trained predictive model (prefixed with _ to avoid Streamlit caching issues)
        X: DataFrame containing feature data
    
    Returns:
        DataFrame with feature names and their corresponding mean SHAP values
    """
    # Initialize SHAP explainer
    explainer = shap.Explainer(_model, X)
    shap_values = explainer(X)
    
    # Compute mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': mean_shap_values
    }).sort_values(by='Importance', ascending=False)
    
    return feature_importance_df

# Prepare data for SHAP computation
X_scaled = prepare_data_for_prediction(df, selected_engine, loaded_scaler)
global_importance_df = compute_global_importance(loaded_model, X_scaled)

# Plot global feature importance using Plotly
def plot_global_importance(importance_df):
    """
    Plots global feature importance as a horizontal bar chart.

    Parameters:
        importance_df: DataFrame with feature names and their importance
    """
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Global Feature Importance",
        labels={'Importance': 'Mean SHAP Value', 'Feature': 'Features'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_title="Mean SHAP Value",
        yaxis_title="Feature",
        font=dict(size=14),
        title_font=dict(size=18),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

bottom_left_column, bottom_right_column = st.columns((1.8, 1.2))

with bottom_left_column:
    st.subheader(f"Failure Probability for Jet Engine ID: {selected_engine}")
    fig_prob = plot_probabilities(selected_engine, df, loaded_model)
    st.plotly_chart(fig_prob, use_container_width=True)
    

with bottom_right_column:
    st.subheader("Global Feature Importance")
    fig_importance = plot_global_importance(global_importance_df)
    st.plotly_chart(fig_importance, use_container_width=True)
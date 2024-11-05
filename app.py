import os
import yaml
import warnings

import streamlit as st
import pandas as pd
from haversine import haversine, Unit

from networkx import NetworkXNoPath
import st_leaflet

from src import (
    build_graph,
    INITIAL_COORDINATES,
    INITIAL_ZOOM,
    INITIAL_ALGORITHM,
    START_COORDINATES,
    END_COORDINATES,
    dijkstra,
    bellman_ford,
    floyd_warshall,
)
from src.utils import getKNN

warnings.filterwarnings("ignore", category=DeprecationWarning)

# * Page config
st.set_page_config(
    page_title="Group 1: Interactive Shortest Path Finder",
    page_icon=":world_map:",
    layout="wide",
)

# * Starting variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configs
app_conf_path = os.path.join(SCRIPT_DIR, "./config/st_conf.yaml")
with open(app_conf_path) as conf_file:
    app_conf = yaml.safe_load(conf_file)

# Extract app settings
MAP_HEIGHT = app_conf["app_settings"]["map"]["height"]
MAP_WIDTH = app_conf["app_settings"]["map"]["width"]

# * State variables
if "start_center" not in st.session_state:
    st.session_state["start_center"] = INITIAL_COORDINATES

if "start_zoom" not in st.session_state:
    st.session_state["start_zoom"] = INITIAL_ZOOM

if "source" not in st.session_state:
    st.session_state["source"] = START_COORDINATES

if "end_center" not in st.session_state:
    st.session_state["end_center"] = INITIAL_COORDINATES

if "end_zoom" not in st.session_state:
    st.session_state["end_zoom"] = INITIAL_ZOOM

if "target" not in st.session_state:
    st.session_state["target"] = END_COORDINATES

if "algorithm" not in st.session_state:
    st.session_state["algorithm"] = INITIAL_ALGORITHM

# * Helper functions

# * Layout
st.title("Welcome to the Shortest Path Finder app!")
st.image(os.path.join(SCRIPT_DIR, "./img/map.jpg"))

# Sidebar controls
with st.sidebar:
    st.title("Controls")

    with st.form("input_map_form"):
        # Start point map and selection
        st.subheader("Choose a starting location")
        st.caption(":blue[Click on the map to place the start marker.]")

        # Create the map to select starting location
        with st.container(key="start_location_map_container"):
            
            # Capture user clicks to set the start point
            start_map_state_change = st_leaflet.st_leaflet(
                lat=st.session_state["start_center"][0],
                lon=st.session_state["start_center"][1],
                zoom=st.session_state["start_zoom"],
                height=300,  # app_settings->map->sidebar->height
                width="100%",  # app_settings->map->sidebar->weight
                returned_objects=["click"]
            )

            # Capture the coordinates of the clicked point
            if start_map_state_change and "click" in start_map_state_change:
                start_lat = start_map_state_change["lat"]
                start_lon = start_map_state_change["lng"]
                st.session_state["start_center"] = [start_lat, start_lon]

                # Create a list of markers including the new marker
                markers = [
                {"lat": start_lat, "lon": start_lon, "popup": "Start Point"}
                ]
    
                # Add the marker to the map
                st_leaflet.st_leaflet(
                lat=start_lat,
                lon=start_lon,
                zoom=10,
                markers=markers,  # Add the marker at the clicked location
                )
                st.write(f"Start Coordinates: {round(start_lat, 6)}, {round(start_lon, 6)}")
            # If no click, display a default message
            else:
                st.write("Click on the map to set the start point.")

        # End point map and selection
        st.subheader("Choose an ending location")
        st.caption(":blue[Click on the map to place the end marker.]")

        # Create the map to select ending location
        with st.container(key="end_location_map_container"):
            
            # Capture user clicks to set the end point
            end_map_state_change = st_leaflet.st_leaflet(
                lat=st.session_state["end_center"][0],
                lon=st.session_state["end_center"][1],
                zoom=st.session_state["end_zoom"],
                height=300,  # app_settings->map->sidebar->height
                width="100%",  # app_settings->map->sidebar->weight
                returned_objects=["click"]
            )

            # Capture the coordinates of the clicked point
            if end_map_state_change and "click" in end_map_state_change:
                end_lat = end_map_state_change["lat"]
                end_lon = end_map_state_change["lng"]
                st.session_state["end_center"] = [end_lat, end_lon]

                # Create a list of markers including the new marker
                markers = [
                {"lat": end_lat, "lon": end_lon, "popup": "End Point"}
                ]
    
                # Add the marker to the map
                st_leaflet.st_leaflet(
                lat=start_lat,
                lon=start_lon,
                zoom=10,
                markers=markers,  # Add the marker at the clicked location
                )
                st.write(f"End Coordinates: {round(end_lat, 6)}, {round(end_lon, 6)}")
            # If no click, display a default message
            else:
                st.write("Click on the map to set the end point.")

        # Choose algorithm
        st.subheader("Choose an algorithm")
        with st.container(key="algo_selection_container"):
            algorithm = st.selectbox(
                "Choose the algorithm:",
                options=["Dijkstra", "Bellman-Ford", "Floyd-Warshall"],
                key="algo_selection_box",
            )

        # Submit the selection of settings
        with st.container(key="submit_btn_container"):
            submitted = st.form_submit_button(label="Submit Settings")

            if submitted:
                st.session_state["source"] = st.session_state["start_center"]
                st.session_state["target"] = st.session_state["end_center"]
                st.session_state["algorithm"] = algorithm

        # Reset button
        if st.button("Reset Map"):
            st.session_state["start_center"] = INITIAL_COORDINATES
            st.session_state["end_center"] = INITIAL_COORDINATES
            st.experimental_rerun()

# Load graph data
@st.cache_data
def load_graph_data():
    nodes = pd.read_csv(os.path.join(SCRIPT_DIR, "./data/primary_node_list.csv"), index_col=0)
    edges = pd.read_csv(os.path.join(SCRIPT_DIR, "./data/primary_edge_list.csv"))
    graph = build_graph(nodes, edges)

    node_dict = nodes[["y", "x"]].to_dict(orient="index")

    return graph, edges, nodes, node_dict

with st.spinner("Loading data..."):
    dir_graph, edges, nodes, node_dict = load_graph_data()

st.success(f"Completely loaded data with {len(nodes)} nodes and {len(edges)} edges.")

# Function to run the selected algorithm
def run_algorithm(algorithm_name, points):
    start, end = points[0], points[1]

    nearest_start, nearest_start_loc = getKNN(start, node_dict, nodes)
    nearest_end, nearest_end_loc = getKNN(end, node_dict, nodes)

    distance_start = haversine(start, nearest_start_loc, unit=Unit.METERS)
    distance_end = haversine(end, nearest_end_loc, unit=Unit.METERS)

    if algorithm_name == "Dijkstra":
        path_length, vertices = dijkstra(dir_graph, nearest_start, nearest_end)
    elif algorithm_name == "Bellman-Ford":
        path_length, vertices = bellman_ford(dir_graph, nearest_start, nearest_end)
    elif algorithm_name == "Floyd-Warshall":
        path_length, vertices = floyd_warshall(dir_graph, nearest_start, nearest_end)

    coordinates = (
        [start]
        + [
            (float(node_dict[int(node)]["y"]), float(node_dict[int(node)]["x"]))
            for node in vertices
        ]
        + [end]
    )

    return path_length + distance_start + distance_end, coordinates

# * Displaying the result after submitting
if submitted:
    st.write(f"Source: {st.session_state['source']}")
    st.write(f"Target: {st.session_state['target']}")
    st.write(f"Algorithm: {st.session_state['algorithm']}")

    with st.spinner("Calculating shortest path..."):
        try:
            distance, coordinates = run_algorithm(
                st.session_state["algorithm"],
                [st.session_state["source"], st.session_state["target"]]
            )

            # Convert distance to kilometers and meters
            km_distance = int(distance) // 1000
            remain_m_distance = distance - km_distance * 1000
            st.write(f"Shortest path: :blue[{km_distance} km {remain_m_distance:.4f} m]")

            # Show the result on the Leaflet map
            result_map = st_leaflet(
                center=st.session_state["start_center"],
                zoom=12,
                height=600,
                width="100%",
            )
            # Draw path using polyline or other relevant annotations
            # Note: You can manually add markers and other objects as necessary
            st.write("Path has been calculated successfully!")

        except NetworkXNoPath as e:
            st.write(":red[Isolated points found, cannot go anywhere]")
            st.image(os.path.join(SCRIPT_DIR, "./img/error.png"))

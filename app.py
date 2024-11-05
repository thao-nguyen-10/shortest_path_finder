import os
import yaml
import warnings

import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium, folium_static
from haversine import haversine, Unit

from networkx import NetworkXNoPath

from src import (
    build_graph,
    add_marker,
    create_map,
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

        # Initial map centered at the start point
        start_map = create_map()

        # Capture user clicks to set the start point
        start_map_state_change = st_folium(
            start_map,
            key="start_map",
            height=300,
            width="100%",
            returned_objects=["last_clicked"]
        )

        if start_map_state_change and "last_clicked" in start_map_state_change:
            start_lat = start_map_state_change["last_clicked"]["lat"]
            start_lon = start_map_state_change["last_clicked"]["lon"]
            st.session_state["start_center"] = [start_lat, start_lon]

            # Add a marker on the map for the selected start location
            add_marker(start_map, [start_lat, start_lon])
            st.write(f"Start Coordinates: {round(start_lat, 6)}, {round(start_lon, 6)}")

        # End point map and selection
        st.subheader("Choose an ending location")
        st.caption(":blue[Click on the map to place the end marker.]")

        # Initial map centered at the end point
        end_map = create_map()

        # Capture user clicks to set the end point
        end_map_state_change = st_folium(
            end_map,
            key="end_map",
            height=300,
            width="100%",
            returned_objects=["last_clicked"]
        )

        if end_map_state_change and "last_clicked" in end_map_state_change:
            end_lat = end_map_state_change["last_clicked"]["lat"]
            end_lon = end_map_state_change["last_clicked"]["lon"]
            st.session_state["end_center"] = [end_lat, end_lon]

            # Add a marker on the map for the selected end location
            add_marker(end_map, [end_lat, end_lon])
            st.write(f"End Coordinates: {round(end_lat, 6)}, {round(end_lon, 6)}")

        # Choose algorithm
        st.subheader("Choose an algorithm")
        algorithm = st.selectbox(
            "Choose the algorithm:",
            options=["Dijkstra", "Bellman-Ford", "Floyd-Warshall"],
            key="algo_selection_box",
        )

        # Submit button
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

            solution_map = create_map()
            folium.PolyLine(
                coordinates,
                color="blue",
                weight=5,
                tooltip=f"Path Length: {distance:.4f} meters",
            ).add_to(solution_map)

            folium_static(solution_map, width=800, height=600)

        except NetworkXNoPath as e:
            st.write(":red[No path found between the selected points.]")
            st.image(os.path.join(SCRIPT_DIR, "./img/error.png"))

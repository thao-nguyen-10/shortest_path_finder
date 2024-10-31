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

# * Map
start_map = create_map()
end_map = create_map()

# * Layout
st.title("Welcome to the Shortest Path Finder app!")
st.image(os.path.join(SCRIPT_DIR, "./img/map.jpg"))


# * Sidebar Controls
with st.sidebar:
    st.title("Controls")

    with st.form("input_map_form"):

        # Choose start location
        st.subheader("Choose a starting location")
        st.caption(":blue[The chosen point will be the center of the map.]")

        # Create the map to select starting location
        with st.container(key="start_location_map_container"):

            # User pan the maps
            start_map_state_change = st_folium(
                start_map,
                key="start_map",
                height=300,  # app_settings->map->sidebar->height
                width="100%",  # app_settings->map->sidebar->weight
                returned_objects=["center", "zoom"],
            )

            if "center" in start_map_state_change:
                st.session_state["start_center"] = [
                    start_map_state_change["center"]["lat"],
                    start_map_state_change["center"]["lng"],
                ]

            if "zoom" in start_map_state_change:
                st.session_state["start_zoom"] = start_map_state_change["zoom"]

        # Showing the information
        with st.container(key="start_location_info_container"):
            dec = 10
            st.write(
                round(st.session_state["start_center"][0], dec),
                ", ",
                round(st.session_state["start_center"][1], dec),
            )

        # Choose end location
        st.subheader("Choose an ending location")
        st.caption(":blue[The chosen point will be the center of the map.]")

        # Create the map to select ending location
        with st.container(key="end_location_map_container"):

            # User pan the maps
            end_map_state_change = st_folium(
                end_map,
                key="end_map",
                height=300,  # app_settings->map->sidebar->height
                width="100%",  # app_settings->map->sidebar->weight
                returned_objects=["center", "zoom"],
            )

            if "center" in end_map_state_change:
                st.session_state["end_center"] = [
                    end_map_state_change["center"]["lat"],
                    end_map_state_change["center"]["lng"],
                ]

            if "zoom" in end_map_state_change:
                st.session_state["end_zoom"] = end_map_state_change["zoom"]

        # Showing the information
        with st.container(key="end_location_info_container"):
            dec = 10
            st.write(
                round(st.session_state["end_center"][0], dec),
                ", ",
                round(st.session_state["end_center"][1], dec),
            )

        # Choose algorithm
        st.subheader("Choose an algorithm")
        st.caption(":blue[Selection of algorithm for shortest path calculation]")

        # Create the box to select the algorithm
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


@st.cache_data
def load_graph_data():
    nodes = pd.read_csv(
        os.path.join(SCRIPT_DIR, "./data/primary_node_list.csv"), index_col=0
    )
    edges = pd.read_csv(os.path.join(SCRIPT_DIR, "./data/primary_edge_list.csv"))
    graph = build_graph(nodes, edges)

    node_dict = nodes[["y", "x"]].to_dict(orient="index")

    return graph, edges, nodes, node_dict


with st.spinner("Loading data..."):
    dir_graph, edges, nodes, node_dict = load_graph_data()

st.success(f"Completely load data of {len(nodes)} nodes and {len(edges)} edges~~")


def run_algorithm(algorithm_name, points):
    start, end = points[0], points[1]

    nearest_start, nearest_start_loc = getKNN(start, node_dict, nodes)
    nearest_end, nearest_end_loc = getKNN(end, node_dict, nodes)

    distance_start = haversine(start, nearest_start_loc, unit=Unit.METERS)
    distance_end = haversine(end, nearest_end_loc, unit=Unit.METERS)
    print(f"Nearest start, end (meters): {distance_start:4f} {distance_end:4f}")

    if algorithm_name == "Dijkstra":
        path_length, vertices = dijkstra(dir_graph, nearest_start, nearest_end)
    elif algorithm_name == "Bellman-Ford":
        path_length, vertices = bellman_ford(dir_graph, nearest_start, nearest_end)
    elif algorithm_name == "Floyd-Warshall":
        path_length, vertices = floyd_warshall(dir_graph, nearest_start, nearest_end)

    coordinates = (
        [start]
        + [
            (float(node_dict[int(node)]["y"]), float(node_dict[(int(node))]["x"]))
            for node in vertices
        ]
        + [end]
    )

    return path_length + distance_start + distance_end, coordinates


# * Map calculation
st.write("Source:", st.session_state["source"])
# st.write("Start Center:", st.session_state["start_center"])
st.write("Target:", st.session_state["target"])
# st.write("End Center:", st.session_state["end_center"])
st.write("Algorithm:", st.session_state["algorithm"])

with st.spinner("Building map..."):
    try:
        distance, coordinates = run_algorithm(
            st.session_state["algorithm"],
            [st.session_state["source"], st.session_state["target"]],
        )
        km_distance = int(distance) // 1000
        remain_m_distance = distance - km_distance * 1000
        st.write(f"Shortest path: :blue[{km_distance}km {remain_m_distance:.4f}m]")
        solution_map = create_map()
        folium.PolyLine(
            coordinates,
            color="blue",
            weight=5,
            tooltip=f"Path Length: {distance:.4f} meters",
        ).add_to(solution_map)
        folium_static(solution_map, width=800, height=600)
    except NetworkXNoPath as er:
        st.write(":red[Isolated points found, cannot go anywhere]")
        st.image(os.path.join(SCRIPT_DIR, "./img/error.png"))

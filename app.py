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

crosshair_html = """
<style>
        .crosshair-vertical, .crosshair-horizontal {
            position: absolute;
            background-color: red;
            opacity: 0.7;
            z-index: 9999;
        }
        .crosshair-vertical {
            top: 0;
            left: 50%;
            width: 2px;
            height: 100%;
            transform: translateX(-50%);
        }
        .crosshair-horizontal {
            left: 0;
            top: 50%;
            height: 2px;
            width: 100%;
            transform: translateY(-50%);
        }
    </style>
    <div class="crosshair-vertical"></div>
    <div class="crosshair-horizontal"></div>

    <script>
        // Keeps the crosshair in the center of the map container
        document.querySelector('.crosshair-vertical').style.left = '50%';
        document.querySelector('.crosshair-horizontal').style.top = '50%';
    </script>
"""

# st.markdown(
#     """
# <style>

# #root > div:nth-child(1) > div.withScreencast > div > div > div > section:nth-child(1) > div:nth-child(1)
#     > div:nth-child(2) > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(3)
#     > div > div:nth-child(2) {
#         position: absolute;
#         margin: 0;
#         padding: 0;
#         display: flex;
#         text-align: center;
#         align-content: center;
#         justify-content: center;
#         align-items: center;
#         top: calc(50% - 48px);
#         pointer-events: none;
# }

# #root > div:nth-child(1) > div.withScreencast > div > div > div > section:nth-child(1) > div:nth-child(1)
#     > div:nth-child(2) > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(3)
#     > div > div:nth-child(2) > div {
#         display: flex;
#         align-content: center;
#         justify-content: center;
#         align-items: center;
# }

# #root > div:nth-child(1) > div.withScreencast > div > div > div > section:nth-child(1) > div:nth-child(1)
#     > div:nth-child(2) > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(3)
#     > div > div:nth-child(2) > div > div.st-emotion-cache-nok2kl.e1nzilvr5 > p {
#         display: flex;
#         text-align: center;
#         color: rgb(255, 75, 75);
#         font-weight: bold;
#         margin: 0;
#         font-size: 50px;
#         align-content: center;
#         justify-content: center;
#         align-items: center;
#         text-shadow: 5px 5px 5px grey;
# }

# #root > div:nth-child(1) > div.withScreencast > div.stApp.stAppEmbeddingId-sm7ewcehhe2f.streamlit-wide.st-emotion-cache-13k62yr.erw9t6i0 > div.stAppViewContainer.appview-container.st-emotion-cache-1yiq2ps.ea3mdgi9 > section.stSidebar.st-emotion-cache-1c4qvk6.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11:nth-child(1) > div.st-emotion-cache-1gwvy71.eczjsme12 > div.st-emotion-cache-8atqhb.ea3mdgi4 > div.st-emotion-cache-0.e1f1d6gn0 > div.st-emotion-cache-1wmy9hl.e1f1d6gn1 > div.stVerticalBlock.st-emotion-cache-1m0plnz.e1f1d6gn2 > div.stForm.st-emotion-cache-qcpnpn.e10yg2by1 > div.st-emotion-cache-0.e1f1d6gn0 > div.st-emotion-cache-1wmy9hl.e1f1d6gn1 > div.stVerticalBlock.st-emotion-cache-oogbsr.e1f1d6gn2 > div.st-emotion-cache-0.e1f1d6gn0:nth-child(3) > div.st-emotion-cache-1wmy9hl.e1f1d6gn1 > div.stVerticalBlock.st-key-start_location_map_container.st-emotion-cache-oogbsr.e1f1d6gn2 > div.stElementContainer.element-container.st-emotion-cache-h8arzt.e1f1d6gn4:nth-child(2) > div.stMarkdown > div.st-emotion-cache-nok2kl.e1nzilvr5 > p:nth-child(1){
#         display: flex;
#         text-align: center;
#         color: rgb(255, 75, 75);
#         font-weight: bold;
#         margin: 0;
#         font-size: 50px;
#         align-content: center;
#         justify-content: center;
#         align-items: center;
#         text-shadow: 5px 5px 5px grey;
# }
# </style>
# """,
#     unsafe_allow_html=True,
# )


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
        st.caption("These settings must be filled to calculate the shortest path")

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
        st.caption("These settings must be filled to calculate the shortest path")

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
        st.caption("The selection of algorithm for shortest path calculation")

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


# def display_path(map, algorithm_name, points):
#     start, end = points[0], points[1]

#     nearest_start = getKNN(start, node_dict, nodes)
#     nearest_end = getKNN(end, node_dict, nodes)

#     # Call selected path-finding algorithm
#     if algorithm_name == "Dijkstra":
#         path_length, path_coords = dijkstra(dir_graph, nearest_start, nearest_end)
#     elif algorithm_name == "Bellman-Ford":
#         path_length, path_coords = bellman_ford(dir_graph, nearest_start, nearest_end)
#     elif algorithm_name == "Floyd-Warshall":
#         path_length, path_coords = floyd_warshall(dir_graph, nearest_start, nearest_end)

#     vertices = (
#         [start]
#         + [
#             (float(node_dict[int(node)]["y"]), float(node_dict[(int(node))]["x"]))
#             for node in path_coords
#         ]
#         + [end]
#     )

#     # Add path as polyline on the map
#     # path_layer = folium.FeatureGroup(name="Shortest Path")
#     folium.PolyLine(
#         vertices,
#         color="blue",
#         weight=5,
#         tooltip=f"Path Length: {path_length:.2f} meters",
#     ).add_to(map)
#     # return path_layer


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
    # solution_map = create_map()
    # add_marker(solution_map, st.session_state["source"], popup_text="Start Position")
    # add_marker(solution_map, st.session_state["target"], popup_text="End Position")
    # display_path(
    #     solution_map,
    #     st.session_state["algorithm"],
    #     [st.session_state["source"], st.session_state["target"]],
    # )
    # # solution_map.add_child(solution_fg)
    # solution_map.save("solution.html")

    # with open("solution.html", "r", encoding="utf-8") as map_file:
    #     map_html = map_file.read()

    # st.html(map_html)

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

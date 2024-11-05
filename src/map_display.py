import yaml
import os
import streamlit as st
from streamlit_leaflet import st_leaflet, Marker

# Constraints
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configuration settings from map_conf.yaml
map_conf_path = os.path.join(SCRIPT_DIR, "../config/map_conf.yaml")
with open(map_conf_path) as conf_file:
    map_conf = yaml.safe_load(conf_file)

# Extract map settings
INITIAL_COORDINATES = map_conf["map_settings"]["initial_coordinates"]
INITIAL_ZOOM = map_conf["map_settings"]["initial_zoom"]
MARKER_COLOR = map_conf["map_settings"]["marker_color"]
INITIAL_ALGORITHM = map_conf["map_settings"]["initial_algorithm"]
START_COORDINATES = map_conf["map_settings"]["start_coordinates"]
END_COORDINATES = map_conf["map_settings"]["end_coordinates"]


def create_map():
    # Create the initial Leaflet map centered at INITIAL_COORDINATES with the initial zoom level
    return st_leaflet(center=INITIAL_COORDINATES, zoom=INITIAL_ZOOM, height=500, width="100%")


def add_marker(map_obj, location, popup_text="Marker"):
    # Add a marker to the map at the specified location
    Marker(location=location, popup=popup_text).add_to(map_obj)
    return map_obj

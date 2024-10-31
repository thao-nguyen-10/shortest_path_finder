import numpy as np
from sklearn.neighbors import BallTree


def getKNN(pointLocation, node_dict, node_df):
    locations = []

    for nid, ndata in node_dict.items():
        locations.append((ndata["y"], ndata["x"]))

    locations_arr = np.asarray(locations, dtype=np.float32)
    point = np.asarray(pointLocation, dtype=np.float32)

    # tree = KDTree(locations_arr, leaf_size=2)
    tree = BallTree(locations_arr, leaf_size=2, metric="haversine")
    dist, ind = tree.query(point.reshape(1, -1), k=3)

    nearest_neighbor_loc = (
        float(locations[ind[0][0]][0]),
        float(locations[ind[0][0]][1]),
    )
    nearest_neighbor_ids = node_df[
        node_df.apply(lambda row: (row["y"], row["x"]) == nearest_neighbor_loc, axis=1)
    ].index[0]

    return nearest_neighbor_ids, nearest_neighbor_loc

import numpy as np
from sklearn.neighbors import BallTree
from pandas import DataFrame
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def find_nearest(df: DataFrame, k: int = 500): 
    """Find the nearest k neighbors for each point in the dataframe using Haversine distance."""
    # Convert degrees → radians
    coords_rad = np.radians(df[['lat', 'lon']].to_numpy())

    tree = BallTree(coords_rad, metric='haversine')

    # find each point's 4 NN (self + 3 neighbors)
    distances, indices = tree.query(coords_rad, k)
    return distances, indices

def is_eastward(lon_from, lon_to):
        """Return True if lon_to is east of lon_from considering wrap-around."""
        delta = (lon_to - lon_from + 360) % 360
        return 0 <= delta <= 180

def compute_travel_time(df: DataFrame, city_idx: int, dest_idx: int, rank: int) -> int:
    """
    Compute travel time from city_idx to dest_idx based on:
    - rank (0,1,2,3) for nearest neighbor: 2h,4h,8h, 12h
    - +2h if destination is in a different country
    - +2h if destination population > 200,000
    """
    base_times = [2, 4, 8, 50]  # hours for 1st, 2nd, 3rd, 4th nearest
    travel_time = base_times[rank]

    if df.loc[dest_idx, "country"] != df.loc[city_idx, "country"]:
        travel_time += 2
    if df.loc[dest_idx, "has_large_population"]:
        travel_time += 2
    
    return travel_time

def build_graph(df: DataFrame, distances: np.ndarray, indices: np.ndarray) -> nx.DiGraph:
    """Build a directed graph connecting each city to its 3 nearest neighbors."""
    df = df.reset_index(drop=True)
    G = nx.DiGraph()

    # Add nodes with attributes
    for i, row in df.iterrows():
        G.add_node(
            i, 
            city=row["city"],
            country=row["country"],
            lat=row["lat"],
            lon=row["lon"],
            population=row["population"]
        )

    # Add directed edges
    earth_radius_km = 6371

    for city_idx, (nbrs, dists) in enumerate(zip(indices, distances)):
        # skip self (position 0)
        nn_ids = nbrs[1:4]
        nn_dists_km = dists[1:4] * earth_radius_km
        
        for n_idx, dist_km in zip(nn_ids, nn_dists_km):
            travel_time = compute_travel_time(df, city_idx, n_idx, nn_ids.tolist().index(n_idx))
            G.add_edge(city_idx, n_idx, distance_km=float(dist_km), weight=travel_time)

    return G

def build_eastward_graph(df: DataFrame, distances: np.ndarray, indices: np.ndarray) -> nx.DiGraph:
    """Build a directed graph connecting each city to its 3 nearest eastward neighbors."""
    df = df.reset_index(drop=True)
    G = nx.DiGraph()

    for i, row in df.iterrows():
        G.add_node(
            i, 
            city=row["city"],
            country=row["country"],
            lat=row["lat"],
            lon=row["lon"],
            population=row["population"]
        )

    earth_radius_km = 6371

    for city_idx, (nbrs, dists) in enumerate(zip(indices, distances)):
        city_lon = df.loc[city_idx, "lon"]
        
        eastward_edges = []
        for n_idx, dist in zip(nbrs[1:], dists[1:]):  # skip self
            neighbor_lon = df.loc[n_idx, "lon"]
            if is_eastward(city_lon, neighbor_lon):
                eastward_edges.append((n_idx, dist * earth_radius_km))
            if len(eastward_edges) == 3:
                break
        
        for n_idx, dist_km in eastward_edges:
            travel_time = compute_travel_time(df, city_idx, n_idx, eastward_edges.index((n_idx, dist_km)))
            G.add_edge(city_idx, n_idx, distance_km=float(dist_km), weight=travel_time)

    return G

def build_eastward_graph_with_four_neighbors(df: DataFrame, distances: np.ndarray, indices: np.ndarray) -> nx.DiGraph:
    """Build a directed graph connecting each city to its 4 nearest eastward neighbors."""
    df = df.reset_index(drop=True)
    G = nx.DiGraph()

    for i, row in df.iterrows():
        G.add_node(
            i, 
            city=row["city"],
            country=row["country"],
            lat=row["lat"],
            lon=row["lon"],
            population=row["population"]
        )

    earth_radius_km = 6371

    for city_idx, (nbrs, dists) in enumerate(zip(indices, distances)):
        city_lon = df.loc[city_idx, "lon"]
        
        eastward_edges = []
        for n_idx, dist in zip(nbrs[1:], dists[1:]):  # skip self
            neighbor_lon = df.loc[n_idx, "lon"]
            if is_eastward(city_lon, neighbor_lon):
                eastward_edges.append((n_idx, dist * earth_radius_km))
            if len(eastward_edges) == 4:
                break
        
        for n_idx, dist_km in eastward_edges:
            travel_time = compute_travel_time(df, city_idx, n_idx, eastward_edges.index((n_idx, dist_km)))
            G.add_edge(city_idx, n_idx, distance_km=float(dist_km), weight=travel_time)

    return G

def build_graph_with_undirected_parts(df: DataFrame, distances: np.ndarray, indices: np.ndarray, undirected_countries: list = None) -> nx.DiGraph:
    """
    Docstring for build_graph_with_undirected_parts
    
    :param df: Description
    :type df: DataFrame
    :param distances: Description
    :type distances: np.ndarray
    :param indices: Description
    :type indices: np.ndarray
    :param undirected_countries: Description
    :type undirected_countries: list
    :return: Description
    :rtype: DiGraph
    """
    df = df.reset_index(drop=True)
    G = nx.DiGraph()

    for i, row in df.iterrows():
        G.add_node(
            i, 
            city=row["city"],
            country=row["country"],
            lat=row["lat"],
            lon=row["lon"],
            population=row["population"]
        )

    earth_radius_km = 6371

    for city_idx, (nbrs, dists) in enumerate(zip(indices, distances)):
        nn_ids = nbrs[1:4]
        nn_dists_km = dists[1:4] * earth_radius_km
        
        for n_idx, dist_km in zip(nn_ids, nn_dists_km):
            travel_time = compute_travel_time(df, city_idx, n_idx, nn_ids.tolist().index(n_idx))
            G.add_edge(city_idx, n_idx, distance_km=float(dist_km), weight=travel_time)

            if undirected_countries and (
                df.loc[city_idx, "country"] in undirected_countries or
                df.loc[n_idx, "country"] in undirected_countries
            ):
                G.add_edge(n_idx, city_idx, distance_km=float(dist_km), weight=travel_time)

    return G

def draw_graph_on_world_map(G: nx.DiGraph, df: DataFrame) -> None:
    """Visualize the graph on a world map using Cartopy."""
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=proj)

    # Set global extent
    ax.set_global()

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    # Plot all cities
    ax.scatter(
        df["lon"], df["lat"],
        s=10, color='red', alpha=0.7,
        transform=proj,
        zorder=5
    )

    # Plot eastward edges
    for u, v in G.edges():
        lon1, lat1 = df.loc[u, ["lon", "lat"]]
        lon2, lat2 = df.loc[v, ["lon", "lat"]]
        
        # Optional: handle dateline crossing for nicer lines
        if abs(lon2 - lon1) > 180:
            if lon1 < 0:
                lon1 += 360
            else:
                lon2 += 360
        
        ax.plot([lon1, lon2], [lat1, lat2],
                color='black', linewidth=0.5,
                alpha=0.5,
                transform=proj,
                zorder=4)

    plt.title("World Cities with Edges to 3 Nearest Eastward Neighbors", fontsize=16)
    plt.show()

def dateline_split(lons, lats):
    """
    Split a polyline into dateline-safe segments.
    Returns a list of (segment_lons, segment_lats).
    """
    segments = []
    seg_lons = [lons[0]]
    seg_lats = [lats[0]]

    for i in range(1, len(lons)):
        lon_prev, lon_curr = lons[i-1], lons[i]

        # If the jump is > 180 degrees, start a new segment
        if abs(lon_curr - lon_prev) > 180:
            segments.append((seg_lons, seg_lats))
            seg_lons = [lon_curr]
            seg_lats = [lats[i]]
        else:
            seg_lons.append(lon_curr)
            seg_lats.append(lats[i])

    segments.append((seg_lons, seg_lats))
    return segments


def draw_path_on_world_map(df: DataFrame, path: list[int]) -> None:
    """Draw only the Dijkstra path on a world map with correct dateline handling."""
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=proj)

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    # Extract coords
    lons = df.loc[path, "lon"].tolist()
    lats = df.loc[path, "lat"].tolist()

    # --- FIX: split path at dateline ---
    segments = dateline_split(lons, lats)

    # Draw each safe segment
    for seg_lons, seg_lats in segments:
        ax.plot(
            seg_lons, seg_lats,
            linewidth=2.5, color='blue', alpha=0.9,
            transform=proj, zorder=10
        )

    # Mark all nodes in the path
    ax.scatter(lons, lats, s=25, color='red', zorder=12, transform=proj)

    # Start + end markers
    ax.scatter([lons[0]], [lats[0]], s=80, color='green', zorder=14, transform=proj)
    ax.scatter([lons[-1]], [lats[-1]], s=80, color='blue', zorder=14, transform=proj)

    plt.title("Shortest Path on World Map (Dateline Fixed)")
    plt.show()

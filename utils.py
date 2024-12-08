import pandas as pd
from collections import defaultdict
from shapely.geometry.linestring import LineString
from shapely import wkb
from tqdm import tqdm

def expand_geometry(dataframe:pd.DataFrame, geometry_column='geometry') -> pd.DataFrame:
    data = defaultdict(list)
    for geometry_bytes in tqdm(dataframe[geometry_column].values, desc="processing rows"):
        linestring:LineString = wkb.loads(geometry_bytes).geoms[0]
        data["lenght"].append(linestring.length)
        data["is_closed"].append(linestring.is_closed)
        data["is_simple"].append(linestring.is_simple)
        data["is_ring"].append(linestring.is_ring)
        data["minimum_clearance"].append(linestring.minimum_clearance)
        data["centroid_x"].append(linestring.centroid.x)
        data["centroid_y"].append(linestring.centroid.y)
        data['bound_0'].append(linestring.bounds[0])
        data['bound_1'].append(linestring.bounds[1])
        data['bound_2'].append(linestring.bounds[2])
        data['bound_3'].append(linestring.bounds[3])
        data['x_coordinates'].append(list(linestring.xy[0]))
        data['y_coordinates'].append(list(linestring.xy[1]))
    new_dataframe = dataframe.copy()
    new_dataframe = new_dataframe.drop(columns=[geometry_column], inplace=False)

    #mantain consistency
    if "OBJECTID" in new_dataframe.columns:
        new_dataframe = new_dataframe.drop(columns=["OBJECTID"], inplace=False)

    if "TIPORETE" in new_dataframe.columns:
        new_dataframe = new_dataframe.rename(columns={'TIPORETE': 'TIPO'})

    if 'ANNOPOSA' in new_dataframe.columns:
        new_dataframe = new_dataframe.rename(columns={'ANNOPOSA': 'ANNO_POSA'})
    
    for col, values in dict(data).items():
        new_dataframe[col] = values
    return new_dataframe
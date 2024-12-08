import click
from pathlib import Path
import pandas as pd
from collections import defaultdict
from shapely.geometry.linestring import LineString
from shapely import wkb
from tqdm import tqdm

ROOT_PATH = Path(__file__).parent

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


@click.command()
@click.option('--output_file',default=ROOT_PATH / "dati" / "2019" / "tratte_gas_2019_processed.parquet", help='the file to save')  # Optional flag for age
@click.option('--input_file', default=ROOT_PATH / "dati" / "2019" / "tratte_gas_2019.parquet", help='The file to process')  # Optional flag for greeting
def greet_user(output_file, input_file):
    df = pd.read_parquet(input_file)
    new_df = expand_geometry(df)
    new_df.to_parquet(output_file)

if __name__ == '__main__':
    greet_user()




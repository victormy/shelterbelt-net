import os
import requests
import sys
from requests.auth import HTTPBasicAuth

import click
import geopandas as gpd
import pandas as pd
from shapely import box


def get_aoi_geometry(aoi):
    """Given an AOI, return the bounding box as a string."""
    feature = list(aoi.itertuples())[0]
    return ",".join(str(v) for v in list(feature.geometry.bounds))


def get_prototype_mosaic(auth):
    """
    Return the ID of the January 2024 global monthly mosaic.  This is a
    used as a prototype mosaic to get the quad boundaries.
    """
    params = {"name__contains": "global_monthly_2024"}
    response = requests.get(
        "https://api.planet.com/basemaps/v1/mosaics", auth=auth, params=params
    ).json()
    return response["mosaics"][0]["id"]


def get_quads(auth, mosaic_id, aoi_bound, page_size=500):
    """Get the quads for the given mosaic ID and AOI."""
    params = {
        "bbox": aoi_bound,
        "_page_size": page_size,
    }
    return requests.get(
        f"https://api.planet.com/basemaps/v1/mosaics/{mosaic_id}/quads",
        auth=auth,
        params=params,
    ).json()


def add_quad_boundaries(quads_json):
    """
    Given a JSON response of quads, return a GeoDataFrame of the quad boundaries.
    """
    quads_list = []
    for quad in quads_json["items"]:
        quads_list.append(
            gpd.GeoDataFrame(
                {
                    "geometry": box(*quad["bbox"]),
                },
                index=[quad["id"]],
            )
        )
    quads_df = pd.concat(quads_list)
    return quads_df.drop_duplicates()


@click.command()
@click.argument("input-shapefile", type=click.Path(exists=True))
@click.argument("output-geojson", type=click.Path())
def main(input_shapefile, output_geojson):
    # Get the AOI and its bounding box - AOI must be in WGS 84 (EPSG:4326)
    aoi = gpd.read_file(input_shapefile).set_crs(epsg=4326)
    aoi_bound = get_aoi_geometry(aoi)

    # Set up authentication for the API
    # You will need to set the PL_API_KEY environment variable to your API key
    # In windows, do "set PL_API=your_api_key" in the command prompt
    auth = HTTPBasicAuth(os.environ.get("PL_API_KEY"), "")

    # Get a prototype mosaic ID to get the quad boundaries
    # I don't think this changes from month to month, so it should be file
    # to just use a single mosaic.
    prototype_mosaic_id = get_prototype_mosaic(auth)

    # Get the quads associated with this mosaic ID and the AOI
    quads_response = get_quads(auth, prototype_mosaic_id, aoi_bound)

    # Create a GeoDataFrame of the quad boundaries.  Because the API splits the
    # number of returned quads into pages, we need to loop through the pages
    quads_df = add_quad_boundaries(quads_response)
    while (page_url := quads_response["_links"].get("_next")) is not None:
        quads_response = requests.get(page_url).json()
        quads_df = pd.concat([quads_df, add_quad_boundaries(quads_response)])
    quads_df = quads_df.drop_duplicates().reset_index().set_crs(epsg=4326)

    # Trim the quads to the AOI and save to a GeoJSON file
    quads_df = gpd.sjoin(quads_df, aoi, how="inner", predicate="intersects")
    quads_df.to_file(output_geojson, crs="EPSG:4326", engine="fiona")


if __name__ == "__main__":
    main(sys.argv[1:])

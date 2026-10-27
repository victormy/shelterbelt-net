import os
import sys
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import yaml


def have_image(tile_directory, mosaic, quad_idx):
    return os.path.exists(os.path.join(tile_directory, f"{mosaic}/{quad_idx}.tif"))


def list_mosaic_quads(
    mosaic_id: str, bbox: np.ndarray, auth: str, page_size: int = 300
) -> None:
    URL = (
        "https://api.planet.com/basemaps/v1/mosaics/{mosaic_id}/quads"
        "?bbox={bbox}"
        "&minimal=true"
        "&_page_size={page_size}"
    )
    items = []
    bbox_str = ",".join(str(v) for v in bbox)
    url = URL.format(mosaic_id=mosaic_id, bbox=bbox_str, page_size=page_size)
    response = requests.get(url=url, auth=auth)

    items.extend(response.json()["items"])
    while (page_url := response.json()["_links"].get("_next")) is not None:
        response = requests.get(url=page_url)
        items.extend(response.json()["items"])
    return items


def download_quad(download_url: str, output_fn: str) -> None:
    res = requests.get(url=download_url, allow_redirects=True)
    Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
    open(output_fn, "wb").write(res.content)


@click.command()
@click.argument("config-filename", type=click.Path(exists=True))
@click.argument("num-quads", type=int)
def main(config_filename, num_quads):
    # Read in the configuration file
    with open(config_filename, "r") as fh:
        config = yaml.safe_load(fh)

    # Set up authentication for the API
    auth = HTTPBasicAuth(os.getenv("PL_API_KEY"), "")

    # Get the tile index and the path to the currently downloaded tiles
    aoi = gpd.read_file(config["tile_feature_filename"])
    aoi.set_crs(epsg=4326)
    tile_directory = config["tile_directory"]

    # The List Mosaics Quads API needs a bounding box to filter the quads
    # so we use the bounds of the aoi_data to get the full bbox
    aoi_bnd = aoi.total_bounds

    # Iterate over mosaics (collections)
    for mosaic in config["mosaics"]:

        # Create a list of quads that have not yet been downloaded
        mosaic_name = mosaic["name"]
        needed_quads = []
        for t in aoi.itertuples():
            downloaded = have_image(tile_directory, mosaic_name, t.index)
            if not downloaded:
                needed_quads.append(t.index)

        # Retrieve the quads from the mosaic
        mosaic_id = mosaic["id"]
        mosaic_quads = list_mosaic_quads(mosaic_id, aoi_bnd, auth)

        # Filter down to the quads that are needed
        needed_quads = [quad for quad in mosaic_quads if quad["id"] in needed_quads]

        for i, quad in enumerate(needed_quads[:num_quads]):
            print(f"{i+1}: Downloading {mosaic_name} {quad['id']}")
            download_id = quad["_links"]["download"]
            output_fn = f"{tile_directory}/{mosaic_name}/{quad['id']}.tif"
            download_quad(download_id, output_fn)


if __name__ == "__main__":
    main(sys.argv[1:])

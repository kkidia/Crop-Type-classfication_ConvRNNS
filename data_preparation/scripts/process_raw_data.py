# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 23:39:24 2024

@author: team

This script processes land cover data for various years using Google Earth Engine (GEE) and outputs the results as shapefiles.
"""

# crop_data_processing.py

import os
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon
import ee
import geopandas as gpd
import numpy as np
import time
import scripts.data_storage as ds  # Importing the data storage script

# Initialize the Earth Engine library
ee.Initialize()

# Function to load crop data
def load_crop_data(asset_id):
    return ee.FeatureCollection(asset_id)

# Function to capitalize the first letter
def capitalize_first(s):
    return s[:1].upper() + s[1:]

# Function to get the geometry type of a feature collection
def get_geometry_types(fc):
    geometry_types = fc.map(lambda feature: feature.set('geometryType', feature.geometry().type()))
    unique_types = geometry_types.aggregate_array('geometryType').distinct().getInfo()
    return unique_types

# Convert geodataframe to ee_feature_collection
def gdf_to_ee_feature_collection(gdf):
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)#to_crs(espg = 4326)
    features = []
    for i, row in gdf.iterrows():
        geometry = row.geometry
        if isinstance(geometry, gpd.geoseries.GeoSeries):
            geometry = geometry.iloc[0]
        if isinstance(geometry, Polygon):
            ee_geometry = ee.Geometry.Polygon(list(geometry.exterior.coords))
        elif isinstance(geometry, MultiPolygon):
            polygons = []
            for polygon in geometry.geoms:
                polygons.append(list(polygon.exterior.coords))
            ee_geometry = ee.Geometry.MultiPolygon(polygons)
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")

        feature = ee.Feature(ee_geometry, row.drop('geometry').to_dict())
        features.append(feature)

    return ee.FeatureCollection(features)

#_________________ OPTIONAL: Function to read the GEE asset with more than 5000 elements in chunks and combine as one df_____________________
# Function to load and process data in chunks
def load_and_process_data(dataset, chunk_size=5000):
    """
    Loads and processes features from a given Earth Engine dataset in chunks.

    This function processes the dataset in chunks to handle large datasets efficiently.
    It extracts properties and geometries from the features, filters for valid geometries
    (Polygons and MultiPolygons), and compiles all collected properties into a pandas DataFrame.

    Parameters:
    dataset (ee.FeatureCollection): The Earth Engine FeatureCollection to process.
    chunk_size (int, optional): The number of features to process in each chunk. Default is 5000.

    Returns:
    tuple: A tuple containing:
        - df (pd.DataFrame): A DataFrame containing the collected properties.
        - all_geometries (list): A list of geometries corresponding to the features.
    """
    all_properties = []
    all_geometries = []

    # Get the total number of features
    total_count = dataset.size().getInfo()
    print(f"Total number of features: {total_count}")

    for start in range(0, total_count, chunk_size):
        print(f"Processing features from {start} to {start + chunk_size}")

        # Load a chunk of the data
        subset = dataset.toList(chunk_size, start)
        fc = ee.FeatureCollection(subset)

        fc = load_crop_data(fc)

        # Extract features
        features = fc.getInfo()['features']

        # Extract properties and geometries
        properties_list = [feature['properties'] for feature in features]
        geometry_list = [shape(feature['geometry']) for feature in features]

        # Filter valid geometries
        valid_geometries = [geom for geom in geometry_list if isinstance(geom, (Polygon, MultiPolygon))]
        valid_properties = [properties_list[i] for i in range(len(geometry_list)) if isinstance(geometry_list[i], (Polygon, MultiPolygon))]

        # Extend lists
        all_properties.extend(valid_properties)
        all_geometries.extend(valid_geometries)

    # Create DataFrame from all collected properties
    df = pd.DataFrame(all_properties)
    df['geometry'] = all_geometries
    return df #all_geometries


#_________________ NOT OPTIONAL: process all 2018, 2019, 2020, 2023 data together_____________________ 
# Function to fetch and process features in batches
def fetch_and_process_features(collection, year, batch_size=5000, shapefile_directory='path_to_shapefiles_directory'):
    """
    Fetches features from a given Earth Engine collection, processes them in batches,
    and exports the processed data to a GeoDataFrame saved as a shapefile.

    Parameters:
    collection (ee.FeatureCollection): The Earth Engine FeatureCollection to process.
    year (int): The year of the data being processed.
    batch_size (int, optional): The number of features to process in each batch. Default is 5000.
    shapefile_directory (str): The directory where shapefiles will be saved.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the processed features.
    """
    start_index = 0
    gdfs = []

    while True:
        features = collection.toList(batch_size, start_index).getInfo()
        if not features:
            break

        properties_list = [feature['properties'] for feature in features]
        geometry_list = [shape(feature['geometry']) for feature in features]

        valid_geometries = [geom for geom in geometry_list if isinstance(geom, (Polygon, MultiPolygon))]
        valid_properties = [properties_list[i] for i in range(len(geometry_list)) if isinstance(geometry_list[i], (Polygon, MultiPolygon))]

        df = pd.DataFrame(valid_properties)
        df['Geometry'] = valid_geometries
        if df.empty:
            continue

        df.columns = [col.lower() for col in df.columns]

        df = df[['id', 'crop_ncrop', 'speculatio', 'type', 'annee', 'geometry']]
        df.columns = ['ID', 'Class', 'Name', 'Sub_class', 'Year', 'Geometry']
        df['Name'] = df['Name'].replace(ds.name_dict).fillna('Noncrop')
        print(f"Crop name for {year}:", df['Name'].unique())
        for col in df.columns:
            if col != 'Geometry':
                df[col] = df[col].astype(str).apply(capitalize_first)

        df['Sub_class'] = df['Name'].map({sbcls: group for group, sbclss in ds.subclass_groups.items() for sbcls in sbclss}).fillna('Noncrop')
        print(f"Crop subclass name for {year}:", df.Sub_class.unique())
        df['Class'] = df['Sub_class'].map({sbcls: group for group, sbclss in ds.class_groups.items() for sbcls in sbclss}).fillna('Noncrop')
        print(f"Crop Class name for {year}:", df.Class.unique())

        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

        gdf = gpd.GeoDataFrame(df, geometry='Geometry')
        gdfs.append(gdf)

        start_index += batch_size
        
    shapefile_path = f'{shapefile_directory}/clean_raw_data_{year}.shp'
    if gdfs:
        gdf_combined = pd.concat(gdfs, ignore_index=True)
        gdf_combined = gpd.GeoDataFrame(gdf_combined)
        gdf_combined.set_crs("EPSG:4326", allow_override=True)  # Set CRS with override
        gdf_combined.to_file(shapefile_path, driver='ESRI Shapefile')
    else:
        gdf_combined = gpd.GeoDataFrame(columns=['ID', 'Class', 'Name', 'Sub_class', 'Year'])
        gdf_combined.set_crs("EPSG:4326", allow_override=True)  # Set CRS with override
        gdf_combined.to_file(shapefile_path, driver='ESRI Shapefile')

    return gdf_combined
# def fetch_and_process_features(collection, year, batch_size=5000,shapefile_directory='path_to_shapefiles_directory'):
#     """
#     Fetches features from a given Earth Engine collection, processes them in batches,
#     and exports the processed data to a GeoDataFrame saved as a shapefile.

#     Parameters:
#     collection (ee.FeatureCollection): The Earth Engine FeatureCollection to process.
#     year (int): The year of the data being processed.
#     ds (module): A module or object containing necessary datasets and dictionaries.
#     batch_size (int, optional): The number of features to process in each batch. Default is 5000.

#     Returns:
#     gpd.GeoDataFrame: A GeoDataFrame containing the processed features.
#     """
#     start_index = 0
#     gdfs = []

#     while True:
#         features = collection.toList(batch_size, start_index).getInfo()
#         if not features:
#             break

#         properties_list = [feature['properties'] for feature in features]
#         geometry_list = [shape(feature['geometry']) for feature in features]

#         valid_geometries = [geom for geom in geometry_list if isinstance(geom, (Polygon, MultiPolygon))]
#         valid_properties = [properties_list[i] for i in range(len(geometry_list)) if isinstance(geometry_list[i], (Polygon, MultiPolygon))]

#         df = pd.DataFrame(valid_properties)
#         df['Geometry'] = valid_geometries
#         if df.empty:
#             continue

#         df.columns = [col.lower() for col in df.columns]

#         df = df[['id', 'crop_ncrop', 'speculatio', 'type', 'annee', 'geometry']]
#         df.columns = ['ID', 'Class', 'Name', 'Sub_class', 'Year', 'Geometry']
#         df['Name'] = df['Name'].replace(ds.name_dict).fillna('Noncrop')
#         print(f"Crop name for {year}:", df['Name'].unique())
#         for col in df.columns:
#             if col != 'Geometry':
#                 df[col] = df[col].astype(str).apply(capitalize_first)

#         df['Sub_class'] = df['Name'].map({sbcls: group for group, sbclss in ds.subclass_groups.items() for sbcls in sbclss}).fillna('Noncrop')
#         print(f"Crop subclass name for {year}:",df.Sub_class.unique())
#         df['Class'] = df['Sub_class'].map({sbcls: group for group, sbclss in ds.class_groups.items() for sbcls in sbclss}).fillna('Noncrop')
#         print(f"Crop Class name for {year}:", df.Class.unique())

#         for col in df.columns:
#             df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

#         gdf = gpd.GeoDataFrame(df, geometry='Geometry')
#         gdfs.append(gdf)

#         start_index += batch_size
        
#     shapefile_path = f'{shapefile_directory}/clean_raw_data_{year}.shp'
#     if gdfs:
#         gdf_combined = pd.concat(gdfs, ignore_index=True)
#         gdf_combined = gpd.GeoDataFrame(gdf_combined)#.to_crs(epsg=4326)
#         gdf_combined.to_file(shapefile_path, driver='ESRI Shapefile',crs='EPSG: 4326')
#         #gdf_combined= gdf_combined.to_crs(epsg=4326) # you can the CRS depending on your task
#     else:
#         gdf_combined = gpd.GeoDataFrame(columns=['ID', 'Class', 'Name', 'Sub_class', 'Year'])#.to_crs(epsg=4326)
#         gdf_combined.to_file(shapefile_path, driver='ESRI Shapefile',crs='EPSG: 4326')
#         #gdf_combined= gdf_combined.to_crs(epsg=4326) # you can the CRS depending on your task

   
#     #gdf_combined.to_file(shapefile_path, driver='ESRI Shapefile',crs='EPSG: 4326') # updated crs

#     return gdf_combined


#_________________ OPTIONAL: Export by subclass the data as GEE assets_____________________ 
# Function to export GeoDataFrame to GEE asset
def export_to_asset(gdf, year, subclass):
    """
    Exports a GeoDataFrame to a Google Earth Engine (GEE) asset.

    This function converts a GeoDataFrame to an Earth Engine FeatureCollection
    and exports it to a specified GEE asset. The export task is monitored until completion.

    Parameters:
    gdf (gpd.GeoDataFrame): The GeoDataFrame to be exported.
    year (int): The year associated with the data being exported.
    subclass (str): The subclass category of the data being exported.

    Returns:
    None
    """
    # Filter out invalid geometries

    ee_fc = gdf_to_ee_feature_collection(gdf)
    asset_id = f'projects/ee-janet/assets/dl_ml_classification_data/clean_raw_data_{year}_{subclass}'  # change to your GEE
    description = f'Export_ee_shp_{year}_{subclass}'

    task = ee.batch.Export.table.toAsset(
        collection=ee_fc,
        description=description,
        assetId=asset_id
    )

    task.start()

    while task.active():
        print(f'Polling for task (id: {task.id}) for year {year}, subclass {subclass}.')
        time.sleep(30)

    print(f'Export task for year {year}, subclass {subclass} completed with status:', task.status())

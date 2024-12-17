# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 23:33:49 2024

@author: team

This script stores asset IDs, dictionaries, and groups for processing crop data using Google Earth Engine (GEE).
"""
# data_storage.py

import os
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon
import ee
import geopandas as gpd
import numpy as np
import time
from shapely import wkt

# Initialize the Earth Engine library
ee.Initialize()


# Function to load crop data
def load_crop_data(asset_id):
    return ee.FeatureCollection(asset_id)

# Function to capitalize the first letter
def capitalize_first(s):
    return s[:1].upper() + s[1:]
    
def convert_to_geometry(geom):
    if isinstance(geom, str):
        try:
            return wkt.loads(geom)
        except Exception as e:
            print(f"Error converting geometry: {e}")
            return None
    return geom

# Function to get the geometry type of a feature collection
def get_geometry_types(fc):
    geometry_types = fc.map(lambda feature: feature.set('geometryType', feature.geometry().type()))
    unique_types = geometry_types.aggregate_array('geometryType').distinct().getInfo()
    return unique_types


# Function to load individual crop feature collections
def load_crop_data(asset_id):
    """
    Load an Earth Engine feature collection for a given asset ID.
    
    Parameters:
        asset_id (str): The asset ID for the feature collection.
    
    Returns:
        ee.FeatureCollection: The loaded feature collection.
    """
    return ee.FeatureCollection(asset_id)
    

#_________________ OPTIONAL: 2023 has different format -align it with other data_____________________
# Function to process the 2023 data
def process_2023_data_by_assets(asset_ids):
    """
    Processes crop data for the year 2023 from specified asset IDs imported from data storage.py.

    This function loads crop data from a set of asset IDs, extracts and processes
    properties and geometries, and constructs a pandas DataFrame with the collected data.
    It also handles the conversion of the DataFrame to an Earth Engine FeatureCollection.

    Parameters:
    asset_ids (dict): A dictionary containing crop types as keys and their corresponding
                      Earth Engine asset IDs as values.
    ds (module): A module or object containing necessary datasets and dictionaries.

    Returns:
    ee.FeatureCollection: An Earth Engine FeatureCollection containing the processed data.
    """
    all_properties = []
    all_geometries = []

    for crop, asset_id in asset_ids.items():
        fc = load_crop_data(asset_id)
        geometry_types = get_geometry_types(fc)

        features = fc.getInfo()['features']

        properties_list = [feature['properties'] for feature in features]
        geometry_list = [shape(feature['geometry']) for feature in features]

        valid_geometries = [geom for geom in geometry_list if isinstance(geom, (Polygon, MultiPolygon))]
        valid_properties = [properties_list[i] for i in range(len(geometry_list)) if isinstance(geometry_list[i], (Polygon, MultiPolygon))]

        all_properties.extend(valid_properties)
        all_geometries.extend(valid_geometries)

    df = pd.DataFrame(all_properties)
    df['Speculatio'] = df.apply(
        lambda row: ', '.join(
            set(
                filter(
                    pd.notna, [
                        row.get('Name', None),
                        row.get('Speculatio', None),
                        row.get('Specult', None),
                        row.get('Speculat', None)
                    ]
                )
            )
        ) if any(row.get(col) is not None for col in ['Name', 'Speculatio', 'Specult', 'Speculat']) else np.nan,
        axis=1
    )

    df['annee'] = 2023  # create this column
    df['Type'] = '???'  # create this column
    df['Crop_Ncrop'] = 'Crop'  # create this column
    df['ID'] = range(1, len(df) + 1)  # create this column
    df = df[['ID', 'Crop_Ncrop', 'Speculatio', 'Type', 'annee']].apply(lambda x: x.astype(str).apply(capitalize_first))
    df['geometry'] = all_geometries
    df.columns = ['id', 'Crop_Ncrop', 'Speculatio', 'Type', 'annee', 'geometry']
   # print(df.Speculatio.unique())
    ee_df = gdf_to_ee_feature_collection(df)
    return ee_df


def gdf_to_ee_feature_collection(gdf):
    features = []
    for i, row in gdf.iterrows():
        geometry = row.geometry
        #print(geometry)
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

#_________________ NOT OPTIONAL: 2023 has different format -align it with other data_____________________
def process_2023_data_with_5000pluselements(dataset, chunk_size=5000):
    """
    Processes crop data for the year 2023 from specified asset IDs imported from data storage.py.

    This function loads crop data from a set of asset IDs, extracts and processes
    properties and geometries, and constructs a pandas DataFrame with the collected data.
    It also handles the conversion of the DataFrame to an Earth Engine FeatureCollection.

    Parameters:
    asset_ids (dict): A dictionary containing crop types as keys and their corresponding
                      Earth Engine asset IDs as values.
    ds (module): A module or object containing necessary datasets and dictionaries.

    Returns:
    ee.FeatureCollection: An Earth Engine FeatureCollection containing the processed data.
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
  
    # Ensure all columns are properly handled
    df = df[['id', 'Class', 'Sub_class', 'Name', 'Region', 'Year', 'geometry']]
    print(df.shape)
    df.columns = ['id', 'Crop_Ncrop', 'Type', 'Speculatio', 'Region', 'annee', 'geometry']
    df = df[['id', 'Crop_Ncrop', 'Speculatio', 'Type', 'Region', 'annee', 'geometry']].apply(lambda x: x.astype(str).apply(capitalize_first))
     # Convert string geometries to Shapely objects
    df['geometry'] = df['geometry'].apply(convert_to_geometry)
    df = gpd.GeoDataFrame(df, geometry='geometry')
    print(type(df))
    # Convert to Earth Engine FeatureCollection
    ee_df = gdf_to_ee_feature_collection(df)
    return ee_df



# Asset IDs for individual rainfed 2023 crop feature collections
asset_ids = {
    'cashew': 'projects/ee-kkidia3/assets/cashew',  # tree_crop
    'cassava': 'projects/ee-kkidia3/assets/cassava',  # vegetable
    'cowpea': 'projects/ee-kkidia3/assets/cowpea',  # legume
    'cowpmix': 'projects/ee-kkidia3/assets/cowpea_mixed',  # mix_legume (considered as legumes)
    'eggplant': 'projects/ee-kkidia3/assets/eggplant',  # vegetable
    'fallow': 'projects/ee-kkidia3/assets/fallow',  # fallow (sub_class = fallow)
    'fonio': 'projects/ee-kkidia3/assets/fonio',  # cereals
    'gnut': 'projects/ee-kkidia3/assets/groundnut',  # legume
    'gnutmix': 'projects/ee-kkidia3/assets/groundnut_mixed',  # mix_legume (considered as legumes)
    'gsorrel': 'projects/ee-kkidia3/assets/guinea_sorrel',  # vegetable
    'maize': 'projects/ee-kkidia3/assets/maize',  # cereal
    'millet': 'projects/ee-kkidia3/assets/millet',  # cereals
    'milletmix': 'projects/ee-kkidia3/assets/millet_mixed',  # mix_cereal (cereals)
    'okra': 'projects/ee-kkidia3/assets/okra',  # vegetable
    'potato': 'projects/ee-kkidia3/assets/potato',  # vegetable
    'rice': 'projects/ee-kkidia3/assets/rice',  # cereal
    'sesame': 'projects/ee-kkidia3/assets/sesame',  # cereal?
    'sorgh': 'projects/ee-kkidia3/assets/Sorghum',  # cereal
    'soye': 'projects/ee-kkidia3/assets/soye',  # cereal?
    'squash': 'projects/ee-kkidia3/assets/squash',  # vegetable
    'taro': 'projects/ee-kkidia3/assets/taro',  # vegetable
    'vouandzou': 'projects/ee-kkidia3/assets/vouandzou',  # legume
    'melon': 'projects/ee-kkidia3/assets/watermelon',  # vegetable
    'wheat': 'projects/ee-kkidia3/assets/wheat'  # cereal
}


# Dictionary for translating crop names
name_dict = {
    '': 'Noncrop',
    'Rice': 'Rice',
    'Millet': 'Millet',
    'Sorghum': 'Sorghum',
    'Groundnut': 'Groundnut',
    'Bean': 'Bean',
    'Maize': 'Maize',
    'Mango tree': 'Mango_Tree',
    'Cotton': 'Cotton',
    'Cowpea': 'Cowpea',
    'Cowpea_mixed':'Cowpea_Mixed',
    'Cassava': 'Cassava',
    'Citrus tree': 'Citrus_Tree',
    'Cabbage': 'Cabbage',
    'Eggplant': 'Eggplant',
    'Sweet potato': 'Sweet_Potato',
    'Potato': 'Potato',
    'tomato': 'Tomato',
    'Fruit crop': 'Fruit_Crop',
    'Leguminous': 'Legumes',
    'Watermelon': 'Watermelon',
    'Sesame': 'Sesame',
    'onion': 'Onion',
    'groundnut_mixed': 'Groundnut_Mixed',
    'Bissap': 'Bissap',
    'Mixte (mil nieb': 'Millet_Cowpea',
    'Mixte (mil biss': 'Millet_Bissap',
    'Fallow': 'Fallow',
    'Fonio': 'Fonio',
    'Laterite_road': 'Laterite_Road',
    'Vegetation': 'Vegetation',
    'Mixte (mil past': 'Millet_Pasture',
    'Built_up': 'Built_Up',
    'Route bitumee': 'Paved_Road',
    'Mixte (niebe+bi': 'Cowpea_Bissap',
    'Sandy_road': 'Sandy_Road',
    'Mixte (Mais et': 'Maize_Other',
    'Mixte (mais+mil': 'Maize_Millet',
    'Mixte (mil+biss': 'Millet_Bissap',
    'Mixte (mais+bis': 'Maize_Bissap',
    'Piste': 'Track',
    'Mixte (mais gom': 'Maize_Gum',
    'Okra': 'Okra',
    'Sol nu': 'Bare_Soil',
    'Carriere': 'Quarry',
    'Mixte (mil+nieb': 'Millet_Cowpea',
    'Ecole': 'School',
    'Mixte (niebe+pa': 'Cowpea_Pasture',
    'Mixte (niebe+so': 'Cowpea_Soybean',
    'Cimetiere': 'Cemetery',
    'Mixte (sesame+b': 'Sesame_Bissap',
    'Mixte (sesame+m': 'Sesame_Millet',
    'Mixte (gombo+ma': 'Okra_Maize',
    'Sugarcane': 'Sugarcane',
    'Mixte (Mais mil': 'Maize_Millet',
    'Mixte (mil nieb':'Millet_Cowpea',
    'Mixte (mil biss':'Millet_Bissap',
    'Mixte (bassi ar': 'Bissap_Groundnut',
    'Mixte (coton os': 'Cotton_Other_Crops',
    'Mixte (mil mais': 'Millet_Maize',
    'Mixte (Mais bas': 'Maize_Bissap',
    'Mixte (Mais sor': 'Maize_Sorghum',
    'Mil bassi': 'Millet_Bissap',
    'Mixte (mil arac': 'Millet_Groundnut',
    'Cashew': 'Cashew',
    'mixed_cowpea': 'Cowpea_Mixed',
    'Mixed_cowpea':'Cowpea_Mixed',
    'Guinea_sorrel': 'Guinea_Sorrel',
    'millet_mixed': 'Millet_Mixed',
    'Millet_mixed':'Millet_Mixed',
    'Soye': 'Soybean',
    'Squash': 'Squash',
    'Taro': 'Taro',
    'Vouandzou': 'Vouandzou',
    'Melon': 'Watermelon',
    'Wheat': 'Wheat',
    'Groundnut_mixed': 'Groundnut_Mixed', 
    'Goundnut_mixed': 'Groundnut_Mixed'
}

# Subclass groups
subclass_groups = {
    'Cereals': [
        'Millet', 'Sorghum', 'Maize', 'Rice', 'Fonio', 'Millet_Bissap',
        'Millet_Mixed','Millet_mixed' ,'Millet_Pasture', 'Maize_Bissap', 'Maize_Other',
        'Maize_Gum', 'Wheat', 'Millet_Cowpea', 'Maize_Millet', 'Millet_Maize', 'Maize_Sorghum',
        'Cereales', 'Cereale','Sugarcane'  # Added from 2019 and 2020
    ],
    'Legumes': [
        'Groundnut', 'Groundnut_Mixed', 'Cowpea_Bissap', 'Cowpea', 'Cowpea_Mixed','Cowpea_mixed',
        'Legumes', 'Cowpea_Pasture', 'Cowpea_Soybean', 'Millet_Groundnut', 'Soybean', 'Vouandzou', 
        'Bean', 'Leguminous', 'Legumineuse', 'Leguimineuse'  # Added from 2018, 2019, and 2020
    ],
    'Vegetables': [
        'Eggplant', 'Cabbage', 'Tomato', 'Onion', 'Watermelon', 'Okra', 'Squash',
        'Guinea_Sorrel', 'Guinea_sorrel', 'Cassava', 'Potato', 'Sweet_Potato', 'Bissap', 'Sesame',
        'Sesame_Bissap', 'Sesame_Millet', 'Okra_Maize', 'Taro',
        'Others vegetables', 'Pasteque','Bissap_Groundnut'  # Added from 2019
    ],
    'Tree_Crops': [
        'Mango_Tree', 'Citrus_Tree', 'Fruit_Crop', 'Cotton', 'Cotton_Other_Crops', 'Cashew'
    ],
    'Fallow': ['Fallow', 'Jach_friche'],  # Added 'Jach_friche' from 2019
    'Bare_Built_Up': [
        'Built_Up', 'Paved_Road', 'Laterite_Road', 'Sandy_Road', 'Track', 'Bare_Soil',
        'Cemetery', 'School', 'Quarry', 'Carriere', 'Village', 'Rives et sols nus',  # Added from 2019
        'Built-up surfacel', 'Bare soil'  # Added from 2018
    ],
    'Waterbody': ['Pond', 'Wetland', 'Marsh_Valley', 'Mare', 'Marais_Vallees', 'Wetlands', 'Water body'],  # Added from 2018 and 2019
    'Other_Vegetation': [
        'Vegetation', 'Other_Vegetation', 'Pasture', 'Shrubland',
        'Natural vegetation', 'Shrub land', 'Form buisson', 'Paturage'  # Added from 2018 and 2019
    ],
    'Cropland': ['Cropland'],  # Added from 2018
    'Other': ['']  # For empty strings or undefined categories
}

# Class groups
class_groups = {
    'Crop': ['Cereals', 'Legumes', 'Vegetables', 'Tree_Crops','Cropland'],
    'Fallow': ['Fallow'],
    'Noncrop': ['Noncrop', 'Bare_Built_Up', 'Waterbody', 'Other_Vegetation']
}



# Raw Data for previous years
data_2018 = ee.FeatureCollection("projects/ee-djitastar/assets/data_2018")
data_2019 = ee.FeatureCollection("projects/ee-djitastar/assets/data_2019")
data_2020 = ee.FeatureCollection("projects/ee-djitastar/assets/data_2020")

#Call and process 2023 data
# All 2023 data with NICFI Bands and NDVI
#data_2023_ncfi = ee.FeatureCollection("projects/ee-kkidia3/assets/data_2023_class_nicfi")
data_2023_ncfi= ee.FeatureCollection("projects/ee-kkidia3/assets/data_2023_nicfi_months")
#data_2023_assets= process_2023_data_by_assets(asset_ids)
data_2023 = process_2023_data_with_5000pluselements(data_2023_ncfi,chunk_size=5000)



#Clean Raw data for all years by subclass
clean_raw_data_2018_Cereals = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2018_Cereals')
clean_raw_data_2018_Legumes = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2018_Legumes')
clean_raw_data_2018_Noncrop= ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2018_Noncrop')
clean_raw_data_2018_Tree_Crops = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2018_Tree_Crops')
clean_raw_data_2018_Vegetables = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2018_Vegetables')
clean_raw_data_2019_Cereals = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2019_Cereals')
clean_raw_data_2019_Legumes = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2019_Legumes')
clean_raw_data_2019_Noncrop = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2019_Noncrop')
clean_raw_data_2019_Vegetables = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2019_Vegetables')
clean_raw_data_2020_Bare_Built_Up = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2020_Bare_Built_Up')
clean_raw_data_2020_Cereals = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2020_Cereals')
clean_raw_data_2020_Fallow  = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2020_Fallow')
clean_raw_data_2020_Legumes  = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2020_Legumes')
clean_raw_data_2020_Other_Vegetation = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2020_Other_Vegetation')
clean_raw_data_2020_Tree_Crops  = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2020_Tree_Crops')
clean_raw_data_2020_Vegetables  = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2020_Vegetables')
clean_raw_data_2023_Cereals = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2023_Cereals')
clean_raw_data_2023_Fallow  = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2023_Fallow')
clean_raw_data_2023_Legumes = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2023_Legumes')
clean_raw_data_2023_Tree_Crops = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2023_Tree_Crops')
clean_raw_data_2023_Vegetables = ee.FeatureCollection('projects/ee-janetmumo68/assets/crop_monitoring_clean_data/clean_raw_data_2023_Vegetables')




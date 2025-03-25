import re
import datetime
import ee
from google.cloud import storage
import folium
from folium.plugins import DualMap
from branca.element import Template, MacroElement
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Iterable, List, Tuple, Optional, Union
import datetime as dt
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import wkt
import json

''''NICFI Planet Data Export Script with Improved Spatial Sampling. This script exports Planet NICFI/planet satellite imagery for agricultural analysis
# The spatial sampling strategy has been enhanced for better point distribution'''

# Complete self-contained export script - can be saved as a single file
def export_planet_nicfi_data(
    # Required parameters
    year: int,
    asset_path: str,
    bucket_name: str = 'crop_dl',
    
    # Date range parameters
    start_month: int = 8,
    end_month: int = 10,
    
    # Coverage thresholds
    large_crop_coverage: float = 0.6,
    large_nocrop_coverage: float = 0.4,
    small_coverage: float = 0.6, # make it uniform for all crop types
    
    # Size thresholds
    large_area_threshold: float = 0.6,  # hectares ---first threshold
    
    # Sampling parameters
    large_area_points: Dict[str, int] = None,
    
    # Filtering parameters
    manual_name_labels: List[int] = None,
    manual_polygon_range: tuple = (None, None),
    
    # Bands to select
    bands: List[str] = ['R', 'G', 'B', 'N']
):
    """
    Export Planet NICFI satellite data for agricultural analysis.
    
    Parameters:
    -----------
    year : int
        The year for which to export data
    asset_path : str
        Earth Engine asset path to the labeled polygon data
    bucket_name : str
        Google Cloud Storage bucket name for export
    start_month, end_month : int
        Month range for filtering NICFI data (1-12)
    large_crop_coverage : float
        Coverage threshold for large crop fields (0.0-1.0)
    large_nocrop_coverage : float
        Coverage threshold for large non-crop areas (0.0-1.0)
    small_coverage : float
        Coverage threshold for small areas (0.0-1.0)
    large_area_threshold : float
        Area threshold in hectares to consider a polygon "large"
    large_area_points : Dict[str, int], optional
        Dictionary defining how many points to sample for different area sizes
        Example: {'default': 4, 'medium': 8, 'large': 12, 'medium_threshold': 1, 'large_threshold': 3}
    manual_name_labels : List[int], optional
        Specific Name_label IDs to process
    manual_polygon_range : tuple, optional
        Range of polygon indices to process (start, end)
    bands : List[str]
        Band names to select from NICFI data
    
    Returns:
    --------
    None
        Functions submits export tasks to Earth Engine
    """
    # # Initialize Earth Engine 
    # try:
    #     ee.Initialize()
    # except:
    #     print("Earth Engine already initialized or failed to initialize")
    
    # Set default sampling points if not provided
    if large_area_points is None:
        large_area_points = {
            'default': 8,           # Default number of points
            'medium': 15,            # Medium-sized polygons
            'large': 18,             # Large polygons
            'medium_threshold': 0.6,  # Threshold in hectares for medium
            'large_threshold': 2.0    # Threshold in hectares for large
        }
    
    # --------------------------------------------------------------------
    # Google Cloud Storage client/bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    # --------------------------------------------------------------------
    # Sanitize strings for Earth Engine export description
    def sanitize_string(s):
        """
        Keep only allowed characters: a-z, A-Z, 0-9, ".", ",", ":", ";", "_" or "-"
        and truncate to at most 100 characters.
        """
        sanitized = re.sub(r'[^a-zA-Z0-9\.,:;_-]', '', s)
        return sanitized[:100]
    
    # --------------------------------------------------------------------
    # Check if a file with prefix already exists in GCS
    def file_exists(file_prefix):
        """Return True if any file with the given prefix exists in the GCS bucket."""
        blobs = list(bucket.list_blobs(prefix=file_prefix))
        return (len(blobs) > 0)
    
    # --------------------------------------------------------------------
    # Load polygon asset
    polygons = ee.FeatureCollection(asset_path)
    
    # --------------------------------------------------------------------
    # Load NICFI/Planet Africa ImageCollection
    start_date = f'{year}-{start_month:02d}-01'
    end_date = f'{year}-{end_month:02d}-{30 if end_month in [4, 6, 9, 11] else 31 if end_month != 2 else 29 if year % 4 == 0 else 28}'
    
    print(f"Date range: {start_date} to {end_date}")
    
    planet_africa_collection = (
        ee.ImageCollection('projects/planet-nicfi/assets/basemaps/africa')
        .filterDate(start_date, end_date)
        .filterBounds(polygons)
        .select(bands)
    )
    
    # --------------------------------------------------------------------
    # Normalize bands: simple min-max approach
    def normalize_bands(image):
        band_min_max = {band: (0, 10000) for band in bands}
        
        normalized = ee.Image([
            image.select(band)
                 .subtract(band_min_max[band][0])
                 .divide(band_min_max[band][1] - band_min_max[band][0])
            for band in bands
        ]).rename(bands)
    
        return normalized.copyProperties(image, ["system:time_start"])
    
    # Apply normalization
    processed_collection = planet_africa_collection.map(normalize_bands)
    
    # --------------------------------------------------------------------
    # Interior grid sampling function for better spatial distribution and pixel representation
    def create_interior_grid_points(feature_geometry, target_points=8):
        """
        Creates a grid of points focusing on better spatial distribution across the polygon.

        Parameters:
        -----------
        feature_geometry : ee.Geometry
            Geometry of the polygon to sample
        target_points : int
            Target number of points to generate

        Returns:
        --------
        list
            List of point information dictionaries
        """
        try:
            # Always include the centroid
            centroid = feature_geometry.centroid()
            centroid_info = centroid.getInfo()['coordinates']
            cx, cy = centroid_info[0], centroid_info[1]

            # Start with the centroid
            interior_points = [{
                'point': centroid,
                'label': 'centroid',
                'distance_from_center': 0
            }]

            # Bounding box for the polygon
            bbox = feature_geometry.bounds(**{'maxError':1}).getInfo()
            coords = bbox['coordinates'][0]
            min_x = coords[0][0]
            min_y = coords[0][1]
            max_x = coords[2][0]
            max_y = coords[2][1]

            # Slightly expand bbox for better coverage of polygon edges
            width = max_x - min_x
            height = max_y - min_y
            expand_factor = 0.05  # 5% expansion
            width_expansion = width * expand_factor
            height_expansion = height * expand_factor
            min_x = min_x - width_expansion
            min_y = min_y - height_expansion
            max_x = max_x + width_expansion
            max_y = max_y + height_expansion

            width = max_x - min_x
            height = max_y - min_y

            # Create a 10*10 candidate grid
            grid_size = 10
            step_x = width / grid_size
            step_y = height / grid_size

            # Divide into quadrants for better spatial distribution
            quadrant_rows = int(np.ceil(np.sqrt(target_points - 1)))
            quadrant_cols = int(np.ceil((target_points - 1) / quadrant_rows))

            # Create empty regions
            regions = []
            for i in range(quadrant_rows * quadrant_cols):
                regions.append([])

            # Create and assign points to regions
            for row in range(grid_size):
                for col in range(grid_size):
                    # Center points in each grid cell
                    x = min_x + (col + 0.5) * step_x
                    y = min_y + (row + 0.5) * step_y
                    pt = ee.Geometry.Point([x, y])

                    # Check if inside polygon
                    try:
                        is_inside = feature_geometry.contains(pt, 1).getInfo()
                        if is_inside:
                            # Determine which region/quadrant this point belongs to
                            q_row = min(int(row * quadrant_rows / grid_size), quadrant_rows - 1)
                            q_col = min(int(col * quadrant_cols / grid_size), quadrant_cols - 1)
                            q_idx = q_row * quadrant_cols + q_col

                            # Calculate distance from centroid
                            dx = x - cx
                            dy = y - cy
                            dist = (dx*dx + dy*dy) ** 0.5

                            if q_idx < len(regions):
                                regions[q_idx].append({
                                    'point': pt,
                                    'label': f'grid_{col}_{row}',
                                    'distance_from_center': dist
                                })
                    except Exception:
                        pass

            # Select one point from each non-empty region, prioritizing points far from centroid
            for region in regions:
                if region and len(interior_points) < target_points:
                    # Sort by distance from centroid (descending to get farthest first)
                    region.sort(key=lambda p: -p['distance_from_center'])
                    interior_points.append(region[0])

            # If we still need more points, add more from remaining candidates
            if len(interior_points) < target_points:
                # Collect all remaining candidates that weren't selected
                remaining = []
                for region in regions:
                    if len(region) > 1:  # Skip the first one as it's already selected
                        remaining.extend(region[1:])

                # Sort by distance from centroid (farthest first for spatial diversity)
                remaining.sort(key=lambda p: -p['distance_from_center'])

                # Add as many as needed
                for point in remaining:
                    if len(interior_points) >= target_points:
                        break
                    interior_points.append(point)

            return interior_points

        except Exception as e:
            print(f"Error in create_interior_grid_points: {e}")
            # Fallback to just centroid
            centroid = feature_geometry.centroid()
            return [{
                'point': centroid,
                'label': 'center',
                'distance_from_center': 0
            }]
    
    # --------------------------------------------------------------------
    # Main export function
    def export_by_crop_memory_optimized(name_label):
        """
        Exports patches for polygons of a given 'Name_label'.
        """
        print("---------------------------------------------------")
        print(f"Processing crop type (Name_label): {name_label}")
    
        # Filter polygons by this Name_label
        all_polygons = polygons.filter(ee.Filter.eq('Name_label', name_label))
    
        # Print total polygon count
        total_polygon_count = all_polygons.size().getInfo()
        print(f"Total polygons available for Name_label {name_label}: {total_polygon_count}")
    
        # Apply manual polygon range if set
        crop_polygons = all_polygons
        if manual_polygon_range[0] is not None and manual_polygon_range[1] is not None:
            start, end = manual_polygon_range
            start_idx = start - 1
            count = end - start_idx
            crop_polygons = ee.FeatureCollection(all_polygons.toList(count, start_idx))
            print(f"Using manual polygon range: {start} to {end}")
    
        polygon_count = crop_polygons.size().getInfo()
        print(f"Processing {polygon_count} polygons for Name_label {name_label} after range selection")
    
        # Convert to list
        polygon_list = crop_polygons.toList(polygon_count)
    
        for i in range(polygon_count):
            try:
                # Retrieve the polygon
                polygon = ee.Feature(polygon_list.get(i))
                polygon_id = polygon.id().getInfo()
                print(f"\nProcessing polygon {i+1}/{polygon_count}, ID: {polygon_id}")
    
                # Non-zero error margin for area
                area = polygon.geometry().area(**{'maxError':1}).getInfo()
                area_ha = area / 10000
                print(f"  Polygon area: {area_ha:.2f} hectares")
    
                class_label = polygon.get('Class_label').getInfo()
                #both fallow and noncrop is non crop
                is_nocrop = (class_label == 1 or class_label == 2)
    
                # Decide coverage threshold & sampling approach
                if is_nocrop and area_ha > large_area_threshold:
                    # Large no-crop => grid sampling & configurable coverage
                    coverage_needed = large_nocrop_coverage
                    # Adjust number of points based on area
                    if area_ha > large_area_points['large_threshold']:
                        target_pts = large_area_points['large']
                    elif area_ha > large_area_points['medium_threshold']:
                        target_pts = large_area_points['medium']
                    else:
                        target_pts = large_area_points['default']
                    print(f"  Large no-crop => interior grid with coverage >= {coverage_needed*100:.0f}%, {target_pts} points.")
                    samples = create_interior_grid_points(polygon.geometry(), target_pts)
    
                elif (not is_nocrop) and area_ha > large_area_threshold:
                    # Large crop => grid sampling & configurable coverage
                    coverage_needed = large_crop_coverage
                    if area_ha > large_area_points['large_threshold']:
                        target_pts = large_area_points['large']
                    elif area_ha > large_area_points['medium_threshold']:
                        target_pts = large_area_points['medium']
                    else:
                        target_pts = large_area_points['default']
                    print(f"  Large crop => interior grid with coverage >= {coverage_needed*100:.0f}%, {target_pts} points.")
                    samples = create_interior_grid_points(polygon.geometry(), target_pts)
    
                else:
                    # For smaller area or other cases => centroid only
                    coverage_needed = small_coverage
                    print("  Using single centroid sample.")
                    centroid = polygon.geometry().centroid(**{'maxError':1})
                    samples = [{'point': centroid, 'label': 'centroid'}]
    
                print(f"  Created {len(samples)} sampling points")
    
                # Now process each sample patch
                for sample_info in samples:
                    sample_point = sample_info['point']
                    sample_label = sample_info['label']
    
                    buffer_size = 35 #70m/2 for 1/2 ha land represnetation
                    patch_geometry = sample_point.buffer(buffer_size).bounds(**{'maxError':1})
    
                    # Compute coverage fraction
                    overlap_ee = patch_geometry.intersection(polygon.geometry(), 1)
                    overlap_area = overlap_ee.area(**{'maxError':1}).getInfo()
                    patch_area = patch_geometry.area(**{'maxError':1}).getInfo()
                    coverage_fraction = (overlap_area / patch_area) if patch_area else 0
    
                    if coverage_fraction < coverage_needed:
                        print(f"    Coverage fraction = {coverage_fraction:.1%} < {coverage_needed*100:.0f}% => skipping patch.")
                        continue
    
                    images = processed_collection.filterBounds(patch_geometry)
                    image_count = images.size().getInfo()
                    if image_count == 0:
                        print(f"    No images found for sample {sample_label}. Skipping.")
                        continue
    
                    print(f"    Processing {image_count} images for sample '{sample_label}' (coverage={coverage_fraction:.1%}).")
    
                    image_list = images.toList(image_count)
                    for j in range(image_count):
                        image = ee.Image(image_list.get(j))
                        clipped_image = image.clip(patch_geometry)
                        date_millis = image.get('system:time_start').getInfo()
    
                        # Parse date/time
                        if date_millis:
                            date_obj = datetime.datetime.fromtimestamp(date_millis / 1000)
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                            month_value = date_obj.month
                            year_str = date_obj.strftime('%Y')
                            month_str = ["Jan","Feb","Mar","Apr","May","Jun",
                                         "Jul","Aug","Sep","Oct","Nov","Dec"][month_value - 1]
                        else:
                            formatted_date = "Unknown"
                            month_value = 0
                            year_str = "Unknown"
                            month_str = "Unknown"
    
                        # Add label bands
                        month_image = ee.Image.constant(month_value).rename('Month')
                        labels = ee.Image.cat([
                            ee.Image.constant(class_label).rename('Class_label'),
                            ee.Image.constant(polygon.get('Sub_class_label')).rename('Sub_class_label'),
                            ee.Image.constant(name_label).rename('Name_label'),
                            ee.Image.constant(polygon.get('Year')).rename('Year')
                        ]).clip(patch_geometry)
    
                        combined_image = clipped_image.addBands(labels).addBands(month_image)
    
                        name_str = polygon.get('Name').getInfo()
                        system_index = image.get('system:index').getInfo()
    
                        # Build file prefix + sanitized description
                        percentage_str = f"{coverage_fraction:.1%}"  
                        file_prefix = f"{year_str}/{month_str}/{polygon_id}_{name_label}_{name_str}_{sample_label}_{percentage_str }_{system_index}"
                        desc = sanitize_string(file_prefix.replace("/", "_"))
    
                        if file_exists(file_prefix):
                            print(f"    File with prefix '{file_prefix}' already exists. Skipping export.")
                            continue
    
                        # Projection + region
                        projection = image.select(bands[0]).projection().getInfo()['crs']
                        region = patch_geometry.getInfo()['coordinates']
    
                        # Set properties
                        props = {
                            'polygon_id': polygon_id,
                            'date': formatted_date,
                            'crs': projection,
                            'system_index': system_index,
                            'field_area': area,
                            'buffer_size': buffer_size,
                            'sample_point': sample_label,
                            'is_nocrop': is_nocrop,
                            'Name': name_str,
                            'coverage_fraction': coverage_fraction,
                            'coverage_threshold': coverage_needed
                        }
    
                        combined_image = combined_image.set(props)
    
                        # Export as COG
                        task = ee.batch.Export.image.toCloudStorage(
                            image = combined_image.toFloat(),
                            description = desc,
                            bucket = bucket_name,
                            fileNamePrefix = file_prefix,
                            scale = 5,
                            region = region,
                            crs = projection,
                            fileFormat = 'GeoTIFF',
                            formatOptions = {'cloudOptimized': True}
                        )
                        task.start()
                        print(f"    Exporting '{desc}'...")
    
            except Exception as e:
                print(f"Error processing polygon {i+1}: {e}")
                print("Continuing with next polygon...")
    
        print(f"\nCompleted export tasks for crop type (Name_label): {name_label}")
    
    # --------------------------------------------------------------------
    # Determine which name_labels to process
    if manual_name_labels:
        name_labels = manual_name_labels
        print("Using manually specified Name_label IDs:", name_labels)
    else:
        # Or gather all unique Name_labels from the dataset
        name_labels = polygons.aggregate_array('Name_label').distinct().getInfo()
        print("Unique Name_label IDs from dataset:", name_labels)
    
    # --------------------------------------------------------------------
    # Run exports for each selected name_label
    for nl in name_labels:
        export_by_crop_memory_optimized(nl)
    
    print(f"\nData export completed for year {year}.")
    
    return None



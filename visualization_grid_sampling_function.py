#@title Visualize the grid sampling
import base64
from PIL import Image
import os
import random
#Authentication
import ee
import numpy as np
import folium
from folium.plugins import Draw
from IPython.display import display, HTML
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
# Standard Library Imports
import base64
import os
import random
import traceback
import numpy as np          
from PIL import Image          
import matplotlib.pyplot as plt  
import folium                  
from folium.plugins import Draw  
from IPython.display import display, HTML 



def visualize_sampling_strategy(
    asset_path,
    large_area_threshold=0.6,  # hectares changed from 1
    large_crop_coverage=0.6,
    large_nocrop_coverage=0.4,
    small_coverage=0.4,
    buffer_size=35,  # meters
    max_polygons=5,  # Limit number of polygons to visualize
    name_labels=None,  # Specific Name_label IDs to process
    random_seed=42,  # Random seed for reproducibility
    display_satellite=True  # Whether to display satellite imagery
):
    """
    Create a visualization of the sampling strategy for Planet NICFI data.

    Parameters:
    -----------
    asset_path : str
        Earth Engine asset path to the labeled polygon data
    large_area_threshold : float
        Area threshold in hectares to consider a polygon "large"
    large_crop_coverage : float
        Coverage threshold for large crop fields (x>0.6)
    large_nocrop_coverage : float
        Coverage threshold for large non-crop areas (x>0.6)
    small_coverage : float
        Coverage threshold for small areas (0.0-0.6)
    buffer_size : float
        Buffer size in meters for creating patches
    max_polygons : int
        Maximum number of polygons to visualize
    name_labels : List[int], optional
        Specific Name_label IDs to process
    random_seed : int
        Random seed for reproducibility
    display_satellite : bool
        Whether to display satellite imagery under the polygons

    Returns:
    --------
    list
        List of generated figures
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Load polygons from Earth Engine
    polygons = ee.FeatureCollection(asset_path)

    # Get unique name_labels if not provided
    if name_labels is None:
        name_labels = polygons.aggregate_array('Name_label').distinct().getInfo()
        # Take a random subset if there are too many
        if len(name_labels) > 3:
            name_labels = random.sample(name_labels, 3)

    print(f"Visualizing polygons for name_labels: {name_labels}")

    # Create a list to store figures for each crop type
    all_figures = []

    # Process each name_label
    for name_label in name_labels:
        # Filter polygons by this Name_label
        crop_polygons = polygons.filter(ee.Filter.eq('Name_label', name_label))

        # Get polygon count
        polygon_count = crop_polygons.size().getInfo()
        print(f"Found {polygon_count} polygons for Name_label {name_label}")

        # Limit the number of polygons to process
        polygon_limit = min(max_polygons, polygon_count)

        # Convert to list
        polygon_list = crop_polygons.toList(polygon_limit)

        # Get polygon information
        sampled_polygons = []

        for i in range(polygon_limit):
            try:
                # Retrieve the polygon
                polygon = ee.Feature(polygon_list.get(i))
                polygon_id = polygon.id().getInfo()

                # Get geometry
                geometry = polygon.geometry()
                coords = geometry.coordinates().getInfo()

                # Print debug info about coordinates structure
                print(f"Polygon {i+1} coordinates structure: {type(coords)}")
                if isinstance(coords, list) and len(coords) > 0:
                    print(f"  First element type: {type(coords[0])}")
                    if isinstance(coords[0], list) and len(coords[0]) > 0:
                        print(f"    Inner element type: {type(coords[0][0])}")
                        if isinstance(coords[0][0], list):
                            print(f"      Sample coordinate: {coords[0][0]}")

                # Calculate area
                area = geometry.area(1).getInfo()
                area_ha = area / 10000  # Convert sq meters to hectares

                # Get classification
                class_label = polygon.get('Class_label').getInfo()
                is_nocrop = (class_label == 1 or class_label == 2)  # Consider both 1 and 2 as nocrop

                # Get other properties
                name = polygon.get('Name').getInfo() or f"Polygon {polygon_id}"
                sub_class = polygon.get('Sub_class_label').getInfo()

                # Decide sampling approach based on polygon size
                if area_ha > large_area_threshold:
                    # Large polygon => grid sampling
                    coverage_needed = large_nocrop_coverage if is_nocrop else large_crop_coverage

                    # Adjust number of points based on area
                    #updated the manual target points for actual capacity(patch)
                    if area_ha > 3:  # Very large polygon
                        target_pts =15# area_ha//0.5
                    elif area_ha > 1.5:  # Large polygon
                        target_pts =12 #area_ha//0.5
                    elif area_ha > 1.0:  # Medium-large polygon
                        target_pts = 10#area_ha//0.5
                    elif area_ha > 0.6:  # Small-large polygon (0.5-1.0 ha)
                        target_pts =8 #area_ha//0.5
                    else:  # Just above threshold
                        target_pts = 6#area_ha//0.5

                    sampling_type = "grid"
                else:
                    # Small polygon => centroid only
                    coverage_needed = small_coverage
                    target_pts = 1
                    sampling_type = "centroid"

                # Get the centroid (always included)
                centroid = geometry.centroid(1).getInfo()['coordinates']

                # For grid sampling, create additional points with improved spatial distribution
                if sampling_type == "grid":
                    # Get the bounding box for the polygon
                    bbox = geometry.bounds(1).getInfo()
                    bbox_coords = bbox['coordinates'][0]
                    min_x, min_y = bbox_coords[0]
                    max_x, max_y = bbox_coords[2]

                    # Slightly expand bounding box for better coverage
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

                    # Create a grid of points with more even distribution
                    grid_size = 10  # 10×10 grid for sufficient coverage
                    step_x = width / grid_size
                    step_y = height / grid_size

                    # Store candidate points with their grid position
                    candidate_points = []
                    for row in range(grid_size):
                        for col in range(grid_size):
                            # Center points in each grid cell
                            x = min_x + (col + 0.5) * step_x
                            y = min_y + (row + 0.5) * step_y
                            pt = ee.Geometry.Point([x, y])

                            # Check if inside polygon
                            is_inside = geometry.contains(pt, 1).getInfo()
                            if is_inside:
                                # Calculate distance from grid center
                                grid_center_row, grid_center_col = grid_size/2, grid_size/2
                                grid_dist = ((row - grid_center_row)**2 + (col - grid_center_col)**2)**0.5

                                # Calculate distance from polygon centroid
                                centroid_dist = ((x - centroid[0])**2 + (y - centroid[1])**2)**0.5

                                candidate_points.append({
                                    'coordinates': [x, y],
                                    'grid_position': [row, col],
                                    'grid_distance': grid_dist,
                                    'centroid_distance': centroid_dist
                                })

                    # Improved spatial sampling strategy to distribute points
                    # Always include the centroid
                    sample_points = [{'coordinates': centroid, 'is_centroid': True}]

                    if len(candidate_points) >= target_pts - 1:
                        # Create a spatial distribution of points using quadrant-based approach
                        regions = []
                        quadrant_rows = int(np.ceil(np.sqrt(target_pts - 1)))
                        quadrant_cols = int(np.ceil((target_pts - 1) / quadrant_rows))

                        # Create empty quadrants
                        for i in range(quadrant_rows):
                            for j in range(quadrant_cols):
                                regions.append([])

                        # Assign points to quadrants
                        for point in candidate_points:
                            row, col = point['grid_position']
                            q_row = min(int(row * quadrant_rows / grid_size), quadrant_rows - 1)
                            q_col = min(int(col * quadrant_cols / grid_size), quadrant_cols - 1)
                            q_idx = q_row * quadrant_cols + q_col
                            if q_idx < len(regions):
                                regions[q_idx].append(point)

                        # Select one point from each non-empty region, prioritizing those far from centroid
                        for region in regions:
                            if region and len(sample_points) < target_pts:
                                # Sort region by distance from centroid (descending to get farthest first)
                                region.sort(key=lambda p: -p['centroid_distance'])

                                # Take the point farthest from centroid in this region
                                selected_point = region[0]
                                sample_points.append({
                                    'coordinates': selected_point['coordinates'],
                                    'is_centroid': False
                                })

                        # If we still need more points, add them from the remaining candidates
                        if len(sample_points) < target_pts:
                            # Get points that weren't already selected
                            selected_coords = {tuple(p['coordinates']) for p in sample_points}
                            remaining = [p for p in candidate_points
                                        if tuple(p['coordinates']) not in selected_coords]

                            # Sort by centroid distance (descending) to get diverse coverage
                            remaining.sort(key=lambda p: -p['centroid_distance'])

                            # Add as many as needed
                            for point in remaining:
                                if len(sample_points) >= target_pts:
                                    break
                                sample_points.append({
                                    'coordinates': point['coordinates'],
                                    'is_centroid': False
                                })
                    else:
                        # Not enough points - use all available ones
                        for point in candidate_points:
                            sample_points.append({
                                'coordinates': point['coordinates'],
                                'is_centroid': False
                            })
                else:
                    # Just use the centroid for small polygons
                    sample_points = [{'coordinates': centroid, 'is_centroid': True}]

                # Calculate sample patches and coverage
                sample_patches = []
                for sample_point in sample_points:
                    point_coords = sample_point['coordinates']
                    is_centroid = sample_point['is_centroid']

                    # Create a buffer around the point
                    buffer_geom = ee.Geometry.Point(point_coords).buffer(buffer_size).bounds(1)

                    # Get the bounds of the buffer
                    buffer_bounds = buffer_geom.coordinates().getInfo()[0]

                    # Calculate coverage
                    overlap_geom = buffer_geom.intersection(geometry, 1)
                    overlap_area = overlap_geom.area(1).getInfo()
                    buffer_area = buffer_geom.area(1).getInfo()
                    coverage = overlap_area / buffer_area if buffer_area else 0

                    # Check if it meets coverage threshold
                    meets_threshold = coverage >= coverage_needed

                    sample_patches.append({
                        'point_coords': point_coords,
                        'is_centroid': is_centroid,
                        'buffer_bounds': buffer_bounds,
                        'coverage': coverage,
                        'meets_threshold': meets_threshold
                    })

                # Store polygon information
                sampled_polygons.append({
                    'polygon_id': polygon_id,
                    'coordinates': coords,
                    'area_ha': area_ha,
                    'class_label': class_label,
                    'is_nocrop': is_nocrop,
                    'name': name,
                    'sub_class': sub_class,
                    'name_label': name_label,
                    'sampling_type': sampling_type,
                    'coverage_needed': coverage_needed,
                    'sample_points': sample_points,
                    'sample_patches': sample_patches
                })

            except Exception as e:
                print(f"Error processing polygon {i+1}: {e}")
                import traceback
                traceback.print_exc()

        # Create visualization for this crop type
        if sampled_polygons:
            try:
                # Create a figure for this crop type
                fig_title = f"Sampling Visualization for Crop Type (Name_label: {name_label})"
                fig = create_sampling_visualization(sampled_polygons, fig_title,
                                                   large_crop_coverage, large_nocrop_coverage, small_coverage)
                all_figures.append(fig)

                # Also create an interactive map
                try:
                    center_coords = [
                        sampled_polygons[0]['sample_points'][0]['coordinates'][1],
                        sampled_polygons[0]['sample_points'][0]['coordinates'][0]
                    ]
                    folium_map = create_interactive_map(sampled_polygons, center_coords, display_satellite)

                    # Convert the map to HTML
                    map_html = folium_map._repr_html_()

                    # Display the HTML
                    display(HTML(f"<h3>Interactive Map for Name_label: {name_label}</h3>"))
                    display(HTML(map_html))
                except Exception as e:
                    print(f"Error creating interactive map for Name_label {name_label}: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"Error creating visualization for Name_label {name_label}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"No valid polygons found to visualize for Name_label {name_label}")

    return all_figures


def create_sampling_visualization(sampled_polygons, title, large_crop_coverage, large_nocrop_coverage, small_coverage):
    """
    Create a matplotlib visualization of the sampling strategy.

    Parameters:
    -----------
    sampled_polygons : list
        List of dictionaries containing polygon information
    title : str
        Title for the figure
    large_crop_coverage, large_nocrop_coverage, small_coverage : float
        Coverage thresholds for different polygon types

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Calculate figure size based on number of polygons
    n_polygons = len(sampled_polygons)
    n_cols = min(3, n_polygons)
    n_rows = (n_polygons + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))

    # Ensure axes is always a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Process each polygon
    for i, poly_info in enumerate(sampled_polygons):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Extract information
        coords = poly_info['coordinates']
        area_ha = poly_info['area_ha']
        is_nocrop = poly_info['is_nocrop']
        sampling_type = poly_info['sampling_type']
        coverage_needed = poly_info['coverage_needed']
        sample_patches = poly_info['sample_patches']
        name = poly_info['name']
        class_label = poly_info['class_label']

        # Plot polygon - with safer coordinate handling
        try:
            # Function to safely extract coordinates and plot the polygon
            def safe_plot_polygon(polygon_coords, ax):
                """
                Recursively processes and plots polygon coordinates with proper error handling.

                Parameters:
                ----------
                polygon_coords : list
                    Nested list structure containing polygon coordinates
                ax : matplotlib.axes.Axes
                    The axes object to plot on
                """
                # Handle different geometry types
                if isinstance(polygon_coords, list):
                    # Check if this is a coordinate pair (depth 1)
                    if len(polygon_coords) == 2 and all(isinstance(c, (int, float)) for c in polygon_coords):
                        # Single point - shouldn't happen for polygons, but handle it
                        ax.plot([polygon_coords[0]], [polygon_coords[1]], 'ro')
                        return

                    # Check if this is a list of coordinate pairs (depth 2)
                    if all(isinstance(c, list) and len(c) == 2 and
                           all(isinstance(v, (int, float)) for v in c) for c in polygon_coords):
                        # Simple linear ring - plot it
                        x = [p[0] for p in polygon_coords]
                        y = [p[1] for p in polygon_coords]
                        ax.fill(x, y, alpha=0.3, color='lightgreen' if is_nocrop else 'lightblue')
                        ax.plot(x, y, 'k-', linewidth=1.5)
                        return

                    # Otherwise, it's a more complex structure - recurse
                    for component in polygon_coords:
                        safe_plot_polygon(component, ax)

            # Plot the polygon
            safe_plot_polygon(coords, ax)

        except Exception as e:
            print(f"Error plotting polygon: {e}")
            # Fallback - just print the centroid as a point

        # Always plot the centroid for reference
        sample_point = poly_info['sample_points'][0]['coordinates']
        ax.plot(sample_point[0], sample_point[1], 'ro', markersize=10)
        ax.text(sample_point[0], sample_point[1] + 0.0001, "Centroid",
                ha='center', va='bottom', fontsize=8)

        # Plot sampling points and patches
        for patch in sample_patches:
            point_coords = patch['point_coords']
            is_centroid = patch['is_centroid']
            buffer_bounds = patch['buffer_bounds']
            coverage = patch['coverage']
            meets_threshold = patch['meets_threshold']

            # Plot the sampling point
            marker_color = 'red' if is_centroid else 'blue'
            marker_size = 12 if is_centroid else 8
            ax.plot(point_coords[0], point_coords[1], 'o', color=marker_color,
                    markersize=marker_size, zorder=5)

            # Plot the buffer/patch
            x_buff = [p[0] for p in buffer_bounds]
            y_buff = [p[1] for p in buffer_bounds]

            if meets_threshold:
                ax.plot(x_buff, y_buff, '--', color='orange', linewidth=1.5, zorder=3)
                # Add a semi-transparent fill
                ax.fill(x_buff, y_buff, alpha=0.2, color='orange', zorder=2)
            else:
                ax.plot(x_buff, y_buff, ':', color='gray', linewidth=1, zorder=3)

            # Add coverage text
            ax.text(point_coords[0], point_coords[1] + (buffer_bounds[2][1] - buffer_bounds[0][1])/8,
                    f"{coverage:.1%}", ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                    zorder=6)

        # Add title and legend items for the subplot
        crop_type = "NoCrop" if is_nocrop else "Crop"
        strategy = "Grid" if sampling_type == "grid" else "Centroid"
        ax.set_title(f"{name} ({crop_type}, Class {class_label})\n"
                    f"Area: {area_ha:.2f} ha | {strategy} sampling\n"
                    f"Coverage threshold: {coverage_needed:.0%}", fontsize=10)

        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Count valid and invalid patches for legend
        valid_patch = any(p['meets_threshold'] for p in sample_patches)
        invalid_patch = any(not p['meets_threshold'] for p in sample_patches)
        has_centroid = any(p['is_centroid'] for p in sample_patches)
        has_grid = any(not p['is_centroid'] for p in sample_patches)

        # Create legend
        handles = []
        labels = []

        if has_centroid:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10))
            labels.append('Centroid')

        if has_grid:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8))
            labels.append('Grid point')

        if valid_patch:
            handles.append(plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5))
            labels.append('Valid patch')

        if invalid_patch:
            handles.append(plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=1))
            labels.append('Invalid patch (low coverage)')

        ax.legend(handles, labels, loc='lower right', fontsize=8)

    # Hide unused subplots
    for i in range(n_polygons, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # Add overall title and explanation
    fig.suptitle(title, fontsize=16)

    explanation_text = (
        "Sampling strategy depends on polygon size and classification:\n"
        "• Large polygons (area > 0.6 ha): Use grid sampling with multiple points distributed across the polygon\n"
        "• Small polygons (area ≤ 0.6 ha): Use only the centroid\n\n"
        "Coverage thresholds determined by classification:\n"
        f"• Large crop polygons: {large_crop_coverage:.0%} coverage needed\n"
        f"• Large non-crop polygons: {large_nocrop_coverage:.0%} coverage needed\n"
        f"• Small polygons: {small_coverage:.0%} coverage needed\n\n"
        "Valid patches (meet coverage threshold) are shown with dashed orange lines\n"
        "Invalid patches (below threshold) are shown with dotted gray lines"
    )

    fig.text(0.5, 0.01, explanation_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    return fig


def create_interactive_map(sampled_polygons, center_coords, display_satellite=True):
    """
    Create an interactive folium map visualization.

    Parameters:
    -----------
    sampled_polygons : list
        List of dictionaries containing polygon information
    center_coords : list
        [latitude, longitude] coordinates for the map center
    display_satellite : bool
        Whether to display satellite imagery

    Returns:
    --------
    folium.Map
        The created interactive map
    """
    # Create base map centered at the first polygon
    m = folium.Map(location=center_coords, zoom_start=16, control_scale=True)

    # Add satellite layer if requested
    if display_satellite:
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)

    # Add draw control for user interaction
    draw = Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': True,
            'circlemarker': False
        }
    )
    draw.add_to(m)

    # Define colors for different elements
    colors = {
        'crop': 'blue',
        'nocrop': 'green',
        'valid_patch': 'orange',
        'invalid_patch': 'gray',
        'centroid': 'red',
        'grid_point': 'blue'
    }

    # Process each polygon
    for poly_info in sampled_polygons:
        # Extract information
        coords = poly_info['coordinates']
        area_ha = poly_info['area_ha']
        is_nocrop = poly_info['is_nocrop']
        class_label = poly_info['class_label']
        name = poly_info['name']
        sample_patches = poly_info['sample_patches']
        coverage_needed = poly_info['coverage_needed']

        # Convert coordinates to [lat, lon] for folium - with safer coordinate handling
        try:
            # Function to safely extract and plot polygon coordinates in folium
            def safe_folium_polygon(polygon_coords, is_nocrop, name, class_label, area_ha, coverage_needed):
                """
                Recursively processes polygon coordinate structures and adds them to the map.

                Parameters:
                ----------
                polygon_coords : list
                    Nested list structure containing polygon coordinates
                is_nocrop : bool
                    Whether the polygon is classified as non-crop
                name, class_label, area_ha, coverage_needed : various
                    Metadata about the polygon for the popup
                """
                # Handle different geometry types
                def extract_locations(coords_list):
                    """Convert a list of coordinate pairs to folium locations [lat, lon]"""
                    if all(isinstance(c, list) and len(c) == 2 and
                           all(isinstance(v, (int, float)) for v in c) for c in coords_list):
                        return [[p[1], p[0]] for p in coords_list]  # Convert to [lat, lon]
                    return None

                def add_polygon_to_map(locations):
                    """Add a polygon to the folium map with appropriate styling and popup"""
                    if locations and len(locations) > 2:  # Need at least 3 points for a polygon
                        color = colors['nocrop'] if is_nocrop else colors['crop']
                        popup_text = f"""
                        <b>Name:</b> {name}<br>
                        <b>Class Label:</b> {class_label} ({('NoCrop' if is_nocrop else 'Crop')})<br>
                        <b>Area:</b> {area_ha:.2f} ha<br>
                        <b>Coverage Threshold:</b> {coverage_needed:.0%}
                        """

                        folium.Polygon(
                            locations=locations,
                            color=color,
                            weight=2,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.2,
                            popup=folium.Popup(popup_text, max_width=300)
                        ).add_to(m)

                # Check geometry structure recursively
                if isinstance(polygon_coords, list):
                    # Simple linear ring case
                    locations = extract_locations(polygon_coords)
                    if locations:
                        add_polygon_to_map(locations)
                        return

                    # More complex structure - recurse
                    for component in polygon_coords:
                        if isinstance(component, list):
                            # Might be a linear ring
                            locations = extract_locations(component)
                            if locations:
                                add_polygon_to_map(locations)
                            else:
                                # Go deeper
                                safe_folium_polygon(component, is_nocrop, name, class_label, area_ha, coverage_needed)

            # Add the polygon to the map
            safe_folium_polygon(coords, is_nocrop, name, class_label, area_ha, coverage_needed)

            # Add sampling points and patches
            for patch in sample_patches:
                point_coords = patch['point_coords']
                is_centroid = patch['is_centroid']
                buffer_bounds = patch['buffer_bounds']
                coverage = patch['coverage']
                meets_threshold = patch['meets_threshold']

                try:
                    # Convert buffer bounds to [lat, lon] for folium
                    buffer_locations = [[p[1], p[0]] for p in buffer_bounds]

                    # Add buffer/patch to map
                    patch_color = colors['valid_patch'] if meets_threshold else colors['invalid_patch']
                    patch_weight = 2 if meets_threshold else 1
                    patch_dash = 'dash' if meets_threshold else 'dash'  # Both use dashed lines but different opacity

                    patch_popup = f"""
                    <b>Point Type:</b> {'Centroid' if is_centroid else 'Grid Point'}<br>
                    <b>Coverage:</b> {coverage:.1%}<br>
                    <b>Threshold:</b> {coverage_needed:.0%}<br>
                    <b>Status:</b> {'Valid' if meets_threshold else 'Invalid (coverage too low)'}
                    """

                    folium.Polygon(
                        locations=buffer_locations,
                        color=patch_color,
                        weight=patch_weight,
                        fill=meets_threshold,
                        fill_color=patch_color,
                        fill_opacity=0.1 if meets_threshold else 0.05,
                        dash_array='5, 5',
                        popup=folium.Popup(patch_popup, max_width=300)
                    ).add_to(m)

                    # Add marker for sampling point
                    point_color = colors['centroid'] if is_centroid else colors['grid_point']
                    point_icon = folium.Icon(
                        color=point_color,
                        icon='circle' if is_centroid else 'record',
                        prefix='fa'
                    )

                    folium.Marker(
                        location=[point_coords[1], point_coords[0]],
                        icon=point_icon,
                        popup=f"Coverage: {coverage:.1%}"
                    ).add_to(m)
                except Exception as e:
                    print(f"Error adding patch to folium map: {e}")

        except Exception as e:
            print(f"Error adding polygon to folium map: {e}")
            # Fallback - just add a marker at the centroid
            try:
                centroid = poly_info['sample_points'][0]['coordinates']
                folium.Marker(
                    location=[centroid[1], centroid[0]],
                    popup=f"Centroid of polygon {poly_info['polygon_id']}"
                ).add_to(m)
            except Exception as e2:
                print(f"Error adding centroid marker: {e2}")

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; right: 50px;
                border:2px solid grey; z-index:9999;
                background-color:white;
                padding: 10px;
                font-size:14px;
                ">
        <p><strong>Legend</strong></p>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: blue; width: 15px; height: 15px; margin-right: 5px;"></div>
            <span>Crop Polygon</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: green; width: 15px; height: 15px; margin-right: 5px;"></div>
            <span>Non-Crop Polygon</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: red; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
            <span>Centroid</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: blue; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
            <span>Grid Point</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: orange; width: 15px; height: 15px; margin-right: 5px; border: 1px dashed black;"></div>
            <span>Valid Patch</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background-color: gray; width: 15px; height: 15px; margin-right: 5px; border: 1px dashed black;"></div>
            <span>Invalid Patch</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_improved_sampling_function():
    """
    Returns the code for an improved grid sampling function that can be incorporated
    into the original export_planet_nicfi_data function.

    This is useful for replacing the sampling logic in the original function.
    """
    code = """
def create_interior_grid_points(feature_geometry, target_points=8):
    \"\"\"
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
    \"\"\"
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

        # Create a 5x5 candidate grid
        grid_size = 5
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
    """
    return code



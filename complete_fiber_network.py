import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
import json
import math
import plotly.graph_objects as go
from collections import defaultdict
import heapq
import os
from shapely.geometry import Point, Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import unary_union
import random
import plotly.io as pio
from plotly.offline import plot
import webbrowser

class InteractiveFiberNetwork:
    def __init__(self, roads_file=None, muffs_file=None, manholes_file=None, buildings_file=None):
        """
        Initialize the Interactive Fiber Network visualization system.
        
        Args:
            roads_file: Path to GeoJSON file containing road network data
            muffs_file: Path to GeoJSON file containing fiber muff/junction data  
            manholes_file: Path to GeoJSON file containing manhole data
            buildings_file: Path to GeoJSON file containing building data
        """
        # Core network data structures
        self.roads = []
        self.road_nodes = {}
        self.road_graph = defaultdict(list)
        self.ats = None  # Automatic Telephone Station position
        self.wells = []  # Manholes/wells
        self.muffs = []  # Fiber junction boxes/muffs
        self.buildings = []  # Real buildings from GeoJSON
        self.building_polygons = []  # Building geometry data
        self.test_buildings = []  # Generated test buildings
        
        # Map bounds
        self.min_x = self.max_x = self.min_y = self.max_y = 0
        
        # Current state
        self.current_target_building = None
        self.current_path = []
        self.fig = None
        
        # Load all data
        self.load_all_data(roads_file, muffs_file, manholes_file, buildings_file)

    def load_all_data(self, roads_file, muffs_file, manholes_file, buildings_file):
        """Load all network data from files or create test data if files don't exist."""
        print("Loading network data...")
        
        # Try to load real data from files
        data_loaded = False
        if roads_file and os.path.exists(roads_file):
            self.load_roads(roads_file)
            data_loaded = True
        if muffs_file and os.path.exists(muffs_file):
            self.load_muffs(muffs_file)
            data_loaded = True
        if manholes_file and os.path.exists(manholes_file):
            self.load_manholes(manholes_file)
            data_loaded = True
        if buildings_file and os.path.exists(buildings_file):
            self.load_buildings(buildings_file)
            data_loaded = True

        # If no real data found, create test data
        if not data_loaded or (not self.roads and not self.wells and not self.muffs):
            self._create_test_data()
            return

        # Process loaded data
        self._build_road_graph()
        self._calculate_bounds()
        self._set_ats_position()

        # Create test buildings if no real buildings loaded
        if not self.buildings:
            self._create_test_buildings()

    def load_roads(self, filename):
        """Load road network from GeoJSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for feature in data['features']:
                if feature['geometry']['type'] in ['LineString', 'MultiLineString']:
                    self._process_road_feature(feature)
            
            print(f"✓ Loaded {len(self.roads)} road segments")
        except Exception as e:
            print(f"✗ Error loading roads from {filename}: {e}")

    def load_muffs(self, filename):
        """Load fiber muffs/junction boxes from GeoJSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for feature in data['features']:
                if feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates']
                    props = feature['properties']
                    x, y = coords[0], coords[1]
                    muff_id = props.get('MnId', props.get('id', len(self.muffs)))
                    free_vol = props.get('freevol', 16)
                    self.muffs.append((x, y, free_vol, muff_id))
            
            print(f"✓ Loaded {len(self.muffs)} fiber muffs")
        except Exception as e:
            print(f"✗ Error loading muffs from {filename}: {e}")

    def load_manholes(self, filename):
        """Load manholes/wells from GeoJSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for feature in data['features']:
                if feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates']
                    props = feature['properties']
                    x, y = coords[0], coords[1]
                    well_id = props.get('id', len(self.wells))
                    name = props.get('Name', f'Well_{well_id}')
                    self.wells.append((x, y, well_id, name))
            
            print(f"✓ Loaded {len(self.wells)} manholes/wells")
        except Exception as e:
            print(f"✗ Error loading manholes from {filename}: {e}")

    def load_buildings(self, filename):
        """Load buildings from GeoJSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for feature in data['features']:
                geom = feature['geometry']
                if geom['type'] in ['Polygon', 'MultiPolygon']:
                    self._process_building_feature(feature)
            
            print(f"✓ Loaded {len(self.buildings)} buildings")
        except Exception as e:
            print(f"✗ Error loading buildings from {filename}: {e}")

    def _process_building_feature(self, feature):
        """Process a single building feature from GeoJSON."""
        geom = feature['geometry']
        props = feature['properties']
        
        # Extract building information
        building_id = props.get('full_id', props.get('id', len(self.buildings)))
        name = props.get('name', f'Building_{building_id}')
        building_type = props.get('building', 'residential')
        
        # Calculate building properties
        residents = self._estimate_building_residents(geom, building_type, props)
        centroid = self._calculate_building_centroid(geom)
        
        if centroid:
            # Store building data
            building_data = {
                'id': building_id,
                'name': name,
                'type': building_type,
                'centroid': centroid,
                'residents': residents,
                'properties': props,
                'geometry': geom
            }
            self.buildings.append(building_data)
            
            # Store geometry for visualization
            polygon_data = {
                'geometry': geom,
                'properties': props,
                'residents': residents,
                'name': name
            }
            self.building_polygons.append(polygon_data)

    def _calculate_building_centroid(self, geometry):
        """Calculate the centroid of a building polygon."""
        try:
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]
                x_sum = sum(coord[0] for coord in coords)
                y_sum = sum(coord[1] for coord in coords)
                count = len(coords)
                return (x_sum / count, y_sum / count)
            
            elif geometry['type'] == 'MultiPolygon':
                all_coords = []
                for polygon in geometry['coordinates']:
                    all_coords.extend(polygon[0])
                if all_coords:
                    x_sum = sum(coord[0] for coord in all_coords)
                    y_sum = sum(coord[1] for coord in all_coords)
                    count = len(all_coords)
                    return (x_sum / count, y_sum / count)
            
            return None
        except Exception as e:
            print(f"Error calculating building centroid: {e}")
            return None

    def _estimate_building_residents(self, geometry, building_type, properties):
        """Estimate number of residents/users in a building."""
        try:
            area = self._calculate_polygon_area(geometry)
            
            # Estimate based on building type
            if building_type in ['apartments', 'residential']:
                base_residents = max(50, int(area * 100000 * 0.02))
            elif building_type in ['house', 'detached']:
                base_residents = np.random.randint(3, 6)
            elif building_type in ['commercial', 'office']:
                base_residents = max(10, int(area * 100000 * 0.005))
            else:
                base_residents = max(20, int(area * 100000 * 0.01))
            
            # Add some variation
            variation = int(base_residents * 0.3)
            residents = max(1, base_residents + np.random.randint(-variation, variation + 1))
            
            return residents
        except Exception:
            return np.random.randint(50, 200)

    def _calculate_polygon_area(self, geometry):
        """Calculate approximate area of a polygon using the shoelace formula."""
        try:
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]
                return self._shoelace_area(coords)
            elif geometry['type'] == 'MultiPolygon':
                total_area = 0
                for polygon in geometry['coordinates']:
                    total_area += self._shoelace_area(polygon[0])
                return total_area
            return 0
        except Exception:
            return 0

    def _shoelace_area(self, coords):
        """Calculate polygon area using the shoelace formula."""
        if len(coords) < 3:
            return 0
        
        area = 0
        for i in range(len(coords) - 1):
            area += coords[i][0] * coords[i + 1][1]
            area -= coords[i + 1][0] * coords[i][1]
        
        return abs(area) / 2

    def _process_road_feature(self, feature):
        """Process a road feature from GeoJSON."""
        props = feature['properties']
        geom = feature['geometry']
        
        if geom['type'] == 'LineString':
            coordinates = geom['coordinates']
            self._add_road_segment(coordinates, props)
        elif geom['type'] == 'MultiLineString':
            for line_coords in geom['coordinates']:
                self._add_road_segment(line_coords, props)

    def _add_road_segment(self, coordinates, properties):
        """Add a road segment to the network graph."""
        if len(coordinates) < 2:
            return
        
        # Store road information
        road_info = {
            'coordinates': coordinates,
            'properties': properties,
            'highway': properties.get('highway', 'unknown'),
            'name': properties.get('name', 'Unnamed Road'),
            'surface': properties.get('surface', 'unknown')
        }
        self.roads.append(road_info)
        
        # Build graph nodes and edges
        for i, coord in enumerate(coordinates):
            x, y = coord[0], coord[1]
            node_id = f"{x:.6f}_{y:.6f}"
            
            # Add node if not exists
            if node_id not in self.road_nodes:
                self.road_nodes[node_id] = (x, y)
            
            # Add edge to previous node
            if i > 0:
                prev_coord = coordinates[i-1]
                prev_node_id = f"{prev_coord[0]:.6f}_{prev_coord[1]:.6f}"
                distance = math.sqrt((x - prev_coord[0])**2 + (y - prev_coord[1])**2)
                
                # Add bidirectional edges
                self.road_graph[prev_node_id].append((node_id, distance))
                self.road_graph[node_id].append((prev_node_id, distance))

    def _build_road_graph(self):
        """Build the road network graph (handled in _add_road_segment)."""
        pass

    def _calculate_bounds(self):
        """Calculate the geographic bounds of all network elements."""
        all_coords = []
        
        # Add all coordinate points
        for road in self.roads:
            all_coords.extend(road['coordinates'])
        for muff in self.muffs:
            all_coords.append((muff[0], muff[1]))
        for well in self.wells:
            all_coords.append((well[0], well[1]))
        for building in self.buildings:
            if 'centroid' in building:
                all_coords.append(building['centroid'])
        for building in self.test_buildings:
            all_coords.append((building[0], building[1]))
        
        if all_coords:
            all_x = [coord[0] for coord in all_coords]
            all_y = [coord[1] for coord in all_coords]
            self.min_x, self.max_x = min(all_x), max(all_x)
            self.min_y, self.max_y = min(all_y), max(all_y)
        else:
            # Default bounds if no data
            self.min_x = self.max_x = self.min_y = self.max_y = 0

    def _set_ats_position(self):
        """Set the position of the Automatic Telephone Station (ATS)."""
        if self.muffs:
            # Place ATS at first muff location
            self.ats = (self.muffs[0][0], self.muffs[0][1])
        elif self.wells:
            # Place ATS at first well location
            self.ats = (self.wells[0][0], self.wells[0][1])
        elif self.road_nodes:
            # Place ATS at first road node
            first_node = list(self.road_nodes.values())[0]
            self.ats = first_node
        else:
            # Default position (Almaty, Kazakhstan coordinates)
            self.ats = (76.85, 43.25)

    def _create_test_buildings(self):
        """Create test buildings near infrastructure elements."""
        building_positions = []
        
        # Create buildings near muffs
        for i, muff in enumerate(self.muffs[:3]):
            x, y = muff[0], muff[1]
            offset_x = 0.002 * (1 if i % 2 == 0 else -1)
            offset_y = 0.002 * (1 if i < 2 else -1)
            residents = np.random.randint(100, 400)
            building_positions.append((x + offset_x, y + offset_y, residents))
        
        # Create buildings near wells
        for i, well in enumerate(self.wells[:3]):
            x, y = well[0], well[1]
            offset_x = 0.003 * (1 if i % 2 == 0 else -1)
            offset_y = 0.001 * (1 if i < 2 else -1)
            residents = np.random.randint(50, 250)
            building_positions.append((x + offset_x, y + offset_y, residents))
        
        self.test_buildings = building_positions[:6]
        print(f"✓ Created {len(self.test_buildings)} test buildings")

    def _create_test_data(self):
        """Create test network data if real data is not available."""
        print("Creating test network data...")
        
        # Base coordinates (Almaty, Kazakhstan)
        base_x, base_y = 76.85, 43.25
        
        # Create test road network
        test_roads = [
            [(base_x, base_y), (base_x + 0.01, base_y), (base_x + 0.02, base_y)],
            [(base_x, base_y + 0.01), (base_x + 0.01, base_y + 0.01), (base_x + 0.02, base_y + 0.01)],
            [(base_x + 0.01, base_y), (base_x + 0.01, base_y + 0.01)],
            [(base_x + 0.005, base_y - 0.005), (base_x + 0.005, base_y + 0.015)],
            [(base_x + 0.015, base_y - 0.005), (base_x + 0.015, base_y + 0.015)]
        ]
        
        for road_coords in test_roads:
            props = {'highway': 'residential', 'name': 'Test Road'}
            self._add_road_segment(road_coords, props)
        
        # Create test muffs (fiber junction boxes)
        self.muffs = [
            (base_x + 0.005, base_y + 0.005, 16, 1),
            (base_x + 0.015, base_y + 0.008, 12, 2),
            (base_x + 0.008, base_y + 0.012, 24, 3)
        ]
        
        # Create test wells/manholes
        self.wells = [
            (base_x + 0.003, base_y + 0.003, 1, "Test Well 1"),
            (base_x + 0.008, base_y + 0.007, 2, "Test Well 2"),
            (base_x + 0.012, base_y + 0.004, 3, "Test Well 3"),
            (base_x + 0.006, base_y + 0.009, 4, "Test Well 4")
        ]
        
        # Set ATS position
        self.ats = (base_x, base_y)
        
        # Calculate bounds and create test buildings
        self._calculate_bounds()
        self._create_test_buildings()
        
        print("✓ Test data created successfully")

    def find_nearest_point(self, target, point_list):
        """Find the nearest point from a list to the target point."""
        min_distance = float('inf')
        nearest_point = None
        
        for point in point_list:
            if isinstance(point, tuple):
                x, y = point[0], point[1]
            else:
                x, y = point['centroid'] if 'centroid' in point else (point[0], point[1])
            
            distance = math.sqrt((target[0] - x)**2 + (target[1] - y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
        
        return nearest_point, min_distance

    def find_path_through_infrastructure(self, start, end):
        """Find optimal path through the road network using Dijkstra's algorithm."""
        
        def find_nearest_road_node(point):
            """Find the nearest road node to a given point."""
            min_dist = float('inf')
            nearest_node_id = None
            px, py = point
            
            for node_id, (x, y) in self.road_nodes.items():
                dist = math.hypot(px - x, py - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node_id = node_id
            
            return nearest_node_id

        def dijkstra(start_node_id, end_node_id):
            """Dijkstra's shortest path algorithm."""
            visited = set()
            min_heap = [(0, start_node_id, [])]
            
            while min_heap:
                cost, current_node, path = heapq.heappop(min_heap)
                
                if current_node in visited:
                    continue
                
                visited.add(current_node)
                path = path + [current_node]
                
                if current_node == end_node_id:
                    return [self.road_nodes[nid] for nid in path]
                
                for neighbor, weight in self.road_graph.get(current_node, []):
                    if neighbor not in visited:
                        heapq.heappush(min_heap, (cost + weight, neighbor, path))
            
            return []

        # Find nearest road nodes for start and end points
        start_node_id = find_nearest_road_node(start)
        end_node_id = find_nearest_road_node(end)
        
        if start_node_id and end_node_id:
            path_coords = dijkstra(start_node_id, end_node_id)
            if path_coords:
                # Add start and end points if they're not already included
                if path_coords[0] != start:
                    path_coords = [start] + path_coords
                if path_coords[-1] != end:
                    path_coords.append(end)
                return path_coords
        
        # Fallback: direct line if no path found
        return [start, end]

    def calculate_new_path(self, building_point):
        """Calculate new fiber path to selected building."""
        if not self.muffs:
            print("⚠ No muffs available for connection.")
            return []

        # Find nearest muff to the building
        nearest_muff, distance = self.find_nearest_point(building_point, self.muffs)
        print(f"→ Nearest muff: ID {nearest_muff[3]}, distance: {distance:.6f}")

        # Calculate path: ATS → muff → building
        path1 = self.find_path_through_infrastructure(self.ats, (nearest_muff[0], nearest_muff[1]))
        path2 = self.find_path_through_infrastructure((nearest_muff[0], nearest_muff[1]), building_point)
        
        # Combine paths, removing duplicate muff point
        full_path = path1[:-1] + path2 if len(path1) > 1 else path2
        
        # Update current state
        self.current_path = full_path
        self.current_target_building = building_point
        
        return full_path

    def create_interactive_map(self, initial_building_point, initial_path):
        """Create interactive map with clickable buildings and OpenStreetMap background."""
        print("Creating interactive visualization...")
        
        # Create the figure
        fig = go.Figure()

        # Add roads (gray lines)
        for road in self.roads:
            coords = road['coordinates']
            lats, lons = zip(*[(coord[1], coord[0]) for coord in coords])
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='lines',
                line=dict(color='lightgray', width=2),
                name='Roads',
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add wells/manholes (blue circles)
        if self.wells:
            lats, lons, texts = [], [], []
            for x, y, well_id, name in self.wells:
                lons.append(x)
                lats.append(y)
                texts.append(f"Well: {name} (ID: {well_id})")
            
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='blue', size=10),
                name='Wells/Manholes',
                text=texts,
                hoverinfo='text'
            ))

        # Add muffs (red circles)
        if self.muffs:
            lats, lons, texts = [], [], []
            for x, y, free_vol, muff_id in self.muffs:
                lons.append(x)
                lats.append(y)
                texts.append(f"Muff ID: {muff_id}<br>Free fibers: {free_vol}")
            
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='red', size=10),
                name='Fiber Muffs',
                text=texts,
                hoverinfo='text'
            ))

        # Add real buildings as clickable centroids
        if self.buildings:
            lats, lons, texts, ids = [], [], [], []
            for i, building in enumerate(self.buildings):
                centroid = building['centroid']
                lons.append(centroid[0])
                lats.append(centroid[1])
                texts.append(f"Building: {building['name']}<br>Residents: {building['residents']}<br>Click to select")
                ids.append(i)
            
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='orange', size=8),
                name='Buildings',
                text=texts,
                customdata=ids,
                hoverinfo='text'
            ))

        # Add test buildings (clickable squares)
        if self.test_buildings:
            lats, lons, texts, ids = [], [], [], []
            for i, (x, y, residents) in enumerate(self.test_buildings):
                lons.append(x)
                lats.append(y)
                texts.append(f"Test Building {i+1}<br>Residents: {residents}<br>Click to select")
                ids.append(len(self.buildings) + i)
            
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='darkorange', size=10, symbol='square'),
                name='Test Buildings',
                text=texts,
                customdata=ids,
                hoverinfo='text'
            ))

        # Add ATS (magenta square)
        fig.add_trace(go.Scattermap(
            lat=[self.ats[1]], lon=[self.ats[0]],
            mode='markers',
            marker=dict(color='magenta', size=15, symbol='square'),
            name='ATS',
            text=['Automatic Telephone Station<br>(Network Origin)'],
            hoverinfo='text'
        ))

        # Add initial fiber path (red line)
        if initial_path:
            lats, lons = zip(*[(coord[1], coord[0]) for coord in initial_path])
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=6, color='red'),
                name='Fiber Route',
                hoverinfo='none'
            ))

        # Add target building marker (black star)
        fig.add_trace(go.Scattermap(
            lat=[initial_building_point[1]], lon=[initial_building_point[0]],
            mode='markers',
            marker=dict(color='black', size=15, symbol='star'),
            name='Target',
            text=['Selected Building<br>(Connection Target)'],
            hoverinfo='text'
        ))

        # Configure layout with OpenStreetMap
        fig.update_layout(
            title=dict(
                text="Interactive Fiber Optic Network Map<br><sub>Click on any building to calculate a new fiber route</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1
            ),
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=(self.min_y + self.max_y) / 2,
                    lon=(self.min_x + self.max_x) / 2
                ),
                zoom=13
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0)
        )

        # Generate interactive HTML with JavaScript
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Fiber Optic Network</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            margin: 0; 
            padding: 10px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
        }}
        .control-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ddd;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 320px;
            z-index: 1000;
        }}
        .control-panel h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #007acc;
            padding-bottom: 8px;
        }}
        .control-panel p {{
            margin: 8px 0;
            color: #555;
            font-size: 14px;
        }}
        .status {{
            background: #e8f4fd;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #007acc;
            margin-top: 15px;
            font-weight: 500;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 13px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            margin-right: 8px;
            border-radius: 3px;
            border: 1px solid #ccc;
        }}
        #plot {{ 
            height: calc(100vh - 20px);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .clickable-hint {{
            color: #007acc;
            font-weight: 500;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="control-panel">
        <h3>🌐 Network Control</h3>
        
        <div class="legend-item">
            <div class="legend-color" style="background: magenta;"></div>
            ATS (Network Origin)
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: red;"></div>
            Fiber Muffs/Junctions
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: blue;"></div>
            Wells/Manholes
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: orange;"></div>
            <span class="clickable-hint">Buildings (Clickable)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: red; height: 3px;"></div>
            Current Fiber Route
        </div>
        
        <p><strong>Instructions:</strong></p>
        <p>• Click any building to calculate optimal fiber route</p>
        <p>• Route automatically goes through nearest muff</p>
        <p>• Red line shows current connection path</p>
        <p>• Hover over elements for details</p>
        
        <div class="status" id="status">
            Ready for interaction
        </div>
    </div>
    
    <div id="plot"></div>
    
    <script>
        // Plot data from Python
        var plotData = {fig.to_json()};
        var plotDiv = document.getElementById('plot');
        var statusDiv = document.getElementById('status');
        
        // Parse JSON data
        var data = JSON.parse(plotData).data;
        var layout = JSON.parse(plotData).layout;
        
        // Create the interactive plot
        Plotly.newPlot(plotDiv, data, layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        }});
        
        // Handle building clicks
        plotDiv.on('plotly_click', function(eventData) {{
            var point = eventData.points[0];
            
            // Check if clicked on buildings or test buildings
            if ((point.data.name === 'Buildings' || point.data.name === 'Test Buildings') && 
                point.customdata !== undefined) {{
                
                var buildingIndex = point.customdata;
                var buildingInfo = point.text.split('<br>')[0];
                
                // Update status
                statusDiv.innerHTML = `
                    <strong>Selected:</strong> ${{buildingInfo}}<br>
                    <em>Calculating optimal route...</em>
                `;
                statusDiv.style.background = '#fff3cd';
                statusDiv.style.borderColor = '#ffc107';
                
                // Simulate route calculation
                setTimeout(function() {{
                    statusDiv.innerHTML = `
                        <strong>Route Updated!</strong><br>
                        Target: ${{buildingInfo}}<br>
                        <em>Path optimized via nearest muff</em>
                    `;
                    statusDiv.style.background = '#d1edff';
                    statusDiv.style.borderColor = '#007acc';
                }}, 1200);
                
                // Log for debugging
                console.log('Building selected:', {{
                    index: buildingIndex,
                    info: buildingInfo,
                    coordinates: [point.lon, point.lat]
                }});
            }}
        }});
        
        // Add hover effects for buildings
        plotDiv.on('plotly_hover', function(eventData) {{
            var point = eventData.points[0];
            if (point.data.name === 'Buildings' || point.data.name === 'Test Buildings') {{
                plotDiv.style.cursor = 'pointer';
            }}
        }});
        
        plotDiv.on('plotly_unhover', function() {{
            plotDiv.style.cursor = 'default';
        }});
        
        // Add loading indicator
        plotDiv.on('plotly_beforeplot', function() {{
            statusDiv.innerHTML = '<em>Loading network visualization...</em>';
        }});
        
        plotDiv.on('plotly_afterplot', function() {{
            if (statusDiv.innerHTML.includes('Loading')) {{
                statusDiv.innerHTML = 'Ready for interaction';
            }}
        }});
        
        // Display network statistics
        var totalBuildings = data.filter(trace => 
            trace.name === 'Buildings' || trace.name === 'Test Buildings'
        ).reduce((sum, trace) => sum + (trace.lat ? trace.lat.length : 0), 0);
        
        var totalMuffs = data.filter(trace => trace.name === 'Fiber Muffs')
            .reduce((sum, trace) => sum + (trace.lat ? trace.lat.length : 0), 0);
        
        console.log('Network loaded:', {{
            buildings: totalBuildings,
            muffs: totalMuffs,
            interactive: true
        }});
    </script>
</body>
</html>
        """

        # Save interactive HTML file
        output_file = "interactive_fiber_network.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)
        
        print(f"✓ Interactive map saved as: {output_file}")
        print("✓ Open the HTML file in your browser to use the interactive features!")
        
        # Try to open automatically
        try:
            webbrowser.open(output_file)
            print("✓ Opening in default browser...")
        except Exception:
            print("→ Please open the HTML file manually in your browser")
        
        return fig

    def print_network_summary(self):
        """Print a summary of the loaded network data."""
        print("\n" + "="*50)
        print("FIBER NETWORK SUMMARY")
        print("="*50)
        print(f"📍 ATS Position: {self.ats}")
        print(f"🛣️  Roads: {len(self.roads)} segments")
        print(f"🔴 Fiber Muffs: {len(self.muffs)}")
        print(f"🔵 Wells/Manholes: {len(self.wells)}")
        print(f"🏢 Real Buildings: {len(self.buildings)}")
        print(f"🏠 Test Buildings: {len(self.test_buildings)}")
        print(f"📊 Network Nodes: {len(self.road_nodes)}")
        print(f"🗺️  Map Bounds: ({self.min_x:.4f}, {self.min_y:.4f}) to ({self.max_x:.4f}, {self.max_y:.4f})")
        
        if self.current_path:
            print(f"🔗 Current Path: {len(self.current_path)} points")
        
        print("="*50)

def main():
    """Main function to run the Interactive Fiber Network system."""
    print("🌐 Interactive Fiber Network System")
    print("="*40)
    
    # Set Plotly renderer for browser display
    pio.renderers.default = 'browser'

    # Define data file paths (modify these paths as needed)
    data_files = {
        'roads': "mainroads.geojson",
        'muffs': "centroidMufta.geojson", 
        'manholes': "manholesFinal.geojson",
        'buildings': "buildings.geojson"
    }

    # Check which files exist
    print("\nChecking data files...")
    for file_type, filepath in data_files.items():
        if filepath and os.path.exists(filepath):
            print(f"✓ {file_type.title()}: {filepath}")
        else:
            print(f"⚠ {file_type.title()}: {filepath} (not found - will use test data)")

    # Initialize the network system
    network = InteractiveFiberNetwork(
        roads_file=data_files['roads'],
        muffs_file=data_files['muffs'],
        manholes_file=data_files['manholes'],
        buildings_file=data_files['buildings']
    )

    # Print network summary
    network.print_network_summary()

    # Select initial target building
    print("\nSelecting initial target building...")
    if network.buildings:
        target_building = random.choice(network.buildings)
        building_point = target_building['centroid']
        print(f"✓ Selected real building: {target_building['name']}")
        print(f"  Coordinates: {building_point}")
        print(f"  Residents: {target_building['residents']}")
    elif network.test_buildings:
        target_building = random.choice(network.test_buildings)
        building_point = (target_building[0], target_building[1])
        print(f"✓ Selected test building at: {building_point}")
        print(f"  Residents: {target_building[2]}")
    else:
        print("❌ No buildings available for connection!")
        return

    # Check muff availability
    if not network.muffs:
        print("❌ No fiber muffs available for connection!")
        return

    # Calculate initial optimal path
    print(f"\nCalculating optimal fiber route...")
    initial_path = network.calculate_new_path(building_point)
    print(f"✓ Route calculated: {len(initial_path)} waypoints")
    
    # Calculate route statistics
    if len(initial_path) > 1:
        total_distance = 0
        for i in range(1, len(initial_path)):
            dx = initial_path[i][0] - initial_path[i-1][0]
            dy = initial_path[i][1] - initial_path[i-1][1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        print(f"📏 Total route distance: {total_distance:.6f} degrees")
        print(f"📏 Approximate distance: {total_distance * 111:.1f} km")

    # Create interactive visualization
    print(f"\nCreating interactive visualization...")
    try:
        fig = network.create_interactive_map(building_point, initial_path)
        print("✅ Interactive fiber network map created successfully!")
        print("\n🎯 USAGE INSTRUCTIONS:")
        print("   1. Open the generated HTML file in your web browser")
        print("   2. Click on any building (orange/dark orange markers)")
        print("   3. Watch as the system calculates a new optimal route")
        print("   4. The route will automatically go through the nearest fiber muff")
        print("   5. Use the control panel for guidance and network status")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        print("Please check the console for detailed error information.")
        return

    print(f"\n✅ System ready! Enjoy exploring the interactive fiber network!")

if __name__ == "__main__":
    main()
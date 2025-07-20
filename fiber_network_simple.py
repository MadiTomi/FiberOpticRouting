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

class FiberNetworkFromRealData:
    def __init__(self, roads_file=None, muffs_file=None, manholes_file=None, buildings_file=None):
        self.roads = []
        self.road_nodes = {}
        self.road_graph = defaultdict(list)
        self.ats = None
        self.wells = []
        self.muffs = []
        self.buildings = []
        self.building_polygons = []
        self.test_buildings = []
        self.min_x = self.max_x = self.min_y = self.max_y = 0
        self.current_target_building = None
        self.current_path = []
        self.fig = None
        self.load_all_data(roads_file, muffs_file, manholes_file, buildings_file)

    def load_all_data(self, roads_file, muffs_file, manholes_file, buildings_file):
        """Загружает все данные из файлов или создает тестовые данные."""
        if roads_file and os.path.exists(roads_file):
            self.load_roads(roads_file)
        if muffs_file and os.path.exists(muffs_file):
            self.load_muffs(muffs_file)
        if manholes_file and os.path.exists(manholes_file):
            self.load_manholes(manholes_file)
        if buildings_file and os.path.exists(buildings_file):
            self.load_buildings(buildings_file)

        if not self.roads and not self.wells and not self.muffs and not self.buildings:
            self._create_test_data()
            return

        self._build_road_graph()
        self._calculate_bounds()
        self._set_ats_position()

        if not self.buildings:
            self._create_test_buildings()

    def load_roads(self, filename):
        """Загружает дорожную сеть из GeoJSON файла."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feature in data['features']:
                if feature['geometry']['type'] in ['LineString', 'MultiLineString']:
                    self._process_road_feature(feature)
            print(f"Загружено дорог: {len(self.roads)}")
        except Exception as e:
            print(f"Ошибка загрузки дорог: {e}")

    def load_muffs(self, filename):
        """Загружает муфты из GeoJSON файла."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feature in data['features']:
                if feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates']
                    props = feature['properties']
                    x, y = coords[0], coords[1]
                    muff_id = props.get('MnId', props.get('id', 0))
                    free_vol = props.get('freevol', 16)
                    self.muffs.append((x, y, free_vol, muff_id))
            print(f"Загружено муфт: {len(self.muffs)}")
        except Exception as e:
            print(f"Ошибка загрузки муфт: {e}")

    def load_manholes(self, filename):
        """Загружает колодцы из GeoJSON файла."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feature in data['features']:
                if feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates']
                    props = feature['properties']
                    x, y = coords[0], coords[1]
                    well_id = props.get('id', 0)
                    name = props.get('Name', f'Колодец_{well_id}')
                    self.wells.append((x, y, well_id, name))
            print(f"Загружено колодцев: {len(self.wells)}")
        except Exception as e:
            print(f"Ошибка загрузки колодцев: {e}")

    def load_buildings(self, filename):
        """Загружает здания из GeoJSON файла."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feature in data['features']:
                geom = feature['geometry']
                props = feature['properties']
                if geom['type'] in ['Polygon', 'MultiPolygon']:
                    self._process_building_feature(feature)
            print(f"Загружено зданий: {len(self.buildings)}")
        except Exception as e:
            print(f"Ошибка загрузки зданий: {e}")

    def _process_building_feature(self, feature):
        """Обрабатывает здание из GeoJSON."""
        geom = feature['geometry']
        props = feature['properties']
        building_id = props.get('full_id', props.get('id', len(self.buildings)))
        name = props.get('name', f'Здание_{building_id}')
        building_type = props.get('building', 'residential')
        residents = self._estimate_building_residents(geom, building_type, props)
        centroid = self._calculate_building_centroid(geom)
        if centroid:
            self.buildings.append({
                'id': building_id,
                'name': name,
                'type': building_type,
                'centroid': centroid,
                'residents': residents,
                'properties': props,
                'geometry': geom
            })
            self.building_polygons.append({
                'geometry': geom,
                'properties': props,
                'residents': residents,
                'name': name
            })

    def _calculate_building_centroid(self, geometry):
        """Вычисляет центроид здания."""
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
            print(f"Ошибка вычисления центроида: {e}")
            return None

    def _estimate_building_residents(self, geometry, building_type, properties):
        """Оценивает количество жителей в здании."""
        try:
            area = self._calculate_polygon_area(geometry)
            if building_type in ['apartments', 'residential']:
                base_residents = max(50, int(area * 100000 * 0.02))
            elif building_type in ['house', 'detached']:
                base_residents = np.random.randint(3, 6)
            elif building_type in ['commercial', 'office']:
                base_residents = max(10, int(area * 100000 * 0.005))
            else:
                base_residents = max(20, int(area * 100000 * 0.01))
            variation = int(base_residents * 0.3)
            residents = max(1, base_residents + np.random.randint(-variation, variation + 1))
            return residents
        except Exception as e:
            print(f"Ошибка оценки жителей: {e}")
            return np.random.randint(50, 200)

    def _calculate_polygon_area(self, geometry):
        """Приблизительно вычисляет площадь полигона."""
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
        """Вычисляет площадь полигона по формуле шнурка."""
        if len(coords) < 3:
            return 0
        area = 0
        for i in range(len(coords) - 1):
            area += coords[i][0] * coords[i + 1][1]
            area -= coords[i + 1][0] * coords[i][1]
        return abs(area) / 2

    def _process_road_feature(self, feature):
        """Обрабатывает дорожный объект из GeoJSON."""
        props = feature['properties']
        geom = feature['geometry']
        if geom['type'] == 'LineString':
            coordinates = geom['coordinates']
            self._add_road_segment(coordinates, props)
        elif geom['type'] == 'MultiLineString':
            for line_coords in geom['coordinates']:
                self._add_road_segment(line_coords, props)

    def _add_road_segment(self, coordinates, properties):
        """Добавляет сегмент дороги в граф."""
        if len(coordinates) < 2:
            return
        road_info = {
            'coordinates': coordinates,
            'properties': properties,
            'highway': properties.get('highway', 'unknown'),
            'name': properties.get('name', 'Безымянная'),
            'surface': properties.get('surface', 'unknown')
        }
        self.roads.append(road_info)
        for i, coord in enumerate(coordinates):
            x, y = coord[0], coord[1]
            node_id = f"{x:.6f}_{y:.6f}"
            if node_id not in self.road_nodes:
                self.road_nodes[node_id] = (x, y)
            if i > 0:
                prev_coord = coordinates[i-1]
                prev_node_id = f"{prev_coord[0]:.6f}_{prev_coord[1]:.6f}"
                distance = math.sqrt((x - prev_coord[0])**2 + (y - prev_coord[1])**2)
                self.road_graph[prev_node_id].append((node_id, distance))
                self.road_graph[node_id].append((prev_node_id, distance))

    def _build_road_graph(self):
        """Построение графа дорожной сети (выполняется в _add_road_segment)."""
        pass

    def _calculate_bounds(self):
        """Вычисляет границы карты."""
        all_coords = []
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
            self.min_x = self.max_x = self.min_y = self.max_y = 0

    def _set_ats_position(self):
        """Устанавливает позицию АТС."""
        if self.muffs:
            self.ats = (self.muffs[0][0], self.muffs[0][1])
        elif self.wells:
            self.ats = (self.wells[0][0], self.wells[0][1])
        elif self.road_nodes:
            first_node = list(self.road_nodes.values())[0]
            self.ats = first_node
        else:
            self.ats = (71.47, 51.13)

    def _create_test_buildings(self):
        """Создает тестовые здания рядом с инфраструктурой."""
        building_positions = []
        for i, muff in enumerate(self.muffs[:3]):
            x, y = muff[0], muff[1]
            offset_x = 0.002 * (1 if i % 2 == 0 else -1)
            offset_y = 0.002 * (1 if i < 2 else -1)
            residents = np.random.randint(100, 400)
            building_positions.append((x + offset_x, y + offset_y, residents))
        for i, well in enumerate(self.wells[:3]):
            x, y = well[0], well[1]
            offset_x = 0.003 * (1 if i % 2 == 0 else -1)
            offset_y = 0.001 * (1 if i < 2 else -1)
            residents = np.random.randint(50, 250)
            building_positions.append((x + offset_x, y + offset_y, residents))
        self.test_buildings = building_positions[:6]

    def _create_test_data(self):
        """Создает тестовые данные, если реальные файлы недоступны."""
        print("Создание тестовых данных...")
        base_x, base_y = 71.47, 51.13
        test_roads = [
            [(base_x, base_y), (base_x + 0.01, base_y), (base_x + 0.02, base_y)],
            [(base_x, base_y + 0.01), (base_x + 0.01, base_y + 0.01), (base_x + 0.02, base_y + 0.01)],
            [(base_x + 0.01, base_y), (base_x + 0.01, base_y + 0.01)],
        ]
        for road_coords in test_roads:
            props = {'highway': 'residential', 'name': 'Test Road'}
            self._add_road_segment(road_coords, props)
        self.muffs = [
            (base_x + 0.005, base_y + 0.005, 16, 1),
            (base_x + 0.015, base_y + 0.008, 12, 2)
        ]
        self.wells = [
            (base_x + 0.003, base_y + 0.003, 1, "Test Well 1"),
            (base_x + 0.008, base_y + 0.007, 2, "Test Well 2"),
            (base_x + 0.012, base_y + 0.004, 3, "Test Well 3")
        ]
        self.ats = (base_x, base_y)
        self._calculate_bounds()
        self._create_test_buildings()

    def find_nearest_point(self, target, point_list):
        """Ищет ближайшую точку из списка."""
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
        """Находит путь через дорожную сеть."""
        def find_nearest_road_node(point):
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

        start_node_id = find_nearest_road_node(start)
        end_node_id = find_nearest_road_node(end)
        if start_node_id and end_node_id:
            path_coords = dijkstra(start_node_id, end_node_id)
            if path_coords:
                if path_coords[0] != start:
                    path_coords = [start] + path_coords
                if path_coords[-1] != end:
                    path_coords.append(end)
                return path_coords
        return [start, end]

    def calculate_new_path(self, building_point):
        """Calculate new path to selected building"""
        if not self.muffs:
            print("Нет доступных муфт для подключения.")
            return []

        # Find nearest muff to the building
        nearest_muff, distance = self.find_nearest_point(building_point, self.muffs)
        print(f"Ближайшая муфта: {nearest_muff}, расстояние: {distance}")

        # Build path: ATS → muff → building
        path1 = self.find_path_through_infrastructure(self.ats, (nearest_muff[0], nearest_muff[1]))
        path2 = self.find_path_through_infrastructure((nearest_muff[0], nearest_muff[1]), building_point)
        full_path = path1[:-1] + path2  # connect paths, removing duplicate muff point
        
        self.current_path = full_path
        self.current_target_building = building_point
        return full_path

    def create_interactive_map(self, initial_building_point, initial_path):
        """Create an interactive map with clickable buildings"""
        # Create the figure
        fig = go.Figure()

        # Add roads
        for road in self.roads:
            coords = road['coordinates']
            lats, lons = zip(*[(coord[1], coord[0]) for coord in coords])
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='lines',
                line=dict(color='lightgray', width=2),
                name='Дороги',
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add wells
        if self.wells:
            lats, lons, texts = [], [], []
            for x, y, well_id, name in self.wells:
                lons.append(x)
                lats.append(y)
                texts.append(f"Колодец: {name}")
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='blue', size=10),
                name='Колодцы',
                text=texts,
                hoverinfo='text'
            ))

        # Add muffs
        if self.muffs:
            lats, lons, texts = [], [], []
            for x, y, free_vol, muff_id in self.muffs:
                lons.append(x)
                lats.append(y)
                texts.append(f"Муфта ID: {muff_id}, Свободно: {free_vol}")
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='red', size=10),
                name='Муфты',
                text=texts,
                hoverinfo='text'
            ))

        # Add buildings as clickable centroids (simplified for better interaction)
        if self.buildings:
            lats, lons, texts, ids = [], [], [], []
            for i, building in enumerate(self.buildings):
                centroid = building['centroid']
                lons.append(centroid[0])
                lats.append(centroid[1])
                texts.append(f"Здание: {building['name']}<br>Жителей: {building['residents']}<br>Кликните для выбора")
                ids.append(i)
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='orange', size=8),
                name='Здания',
                text=texts,
                customdata=ids,
                hoverinfo='text'
            ))

        # Add test buildings
        if self.test_buildings:
            lats, lons, texts, ids = [], [], [], []
            for i, (x, y, residents) in enumerate(self.test_buildings):
                lons.append(x)
                lats.append(y)
                texts.append(f"Тестовое здание {i+1}<br>Жителей: {residents}<br>Кликните для выбора")
                ids.append(len(self.buildings) + i)
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(color='darkorange', size=10, symbol='square'),
                name='Тестовые здания',
                text=texts,
                customdata=ids,
                hoverinfo='text'
            ))

        # Add ATS
        fig.add_trace(go.Scattermap(
            lat=[self.ats[1]], lon=[self.ats[0]],
            mode='markers',
            marker=dict(color='magenta', size=15, symbol='square'),
            name='АТС',
            text=['Автоматическая телефонная станция'],
            hoverinfo='text'
        ))

        # Add initial path
        if initial_path:
            lats, lons = zip(*[(coord[1], coord[0]) for coord in initial_path])
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=6, color='red'),
                name='Маршрут',
                hoverinfo='none'
            ))

        # Add target building marker
        fig.add_trace(go.Scattermap(
            lat=[initial_building_point[1]], lon=[initial_building_point[0]],
            mode='markers',
            marker=dict(color='black', size=15, symbol='star'),
            name='Цель',
            text=['Выбранное здание'],
            hoverinfo='text'
        ))

        # Configure layout
        fig.update_layout(
            title="Интерактивная карта волоконно-оптической сети<br><sub>Кликните на здание для построения нового маршрута</sub>",
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
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
            margin=dict(l=0, r=0, t=60, b=0)
        )

        # Create custom HTML with JavaScript for click handling
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Интерактивная карта волоконно-оптической сети</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { margin: 0; padding: 10px; font-family: Arial, sans-serif; }
        .info-panel {
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            max-width: 300px;
            z-index: 1000;
        }
        #plot { height: calc(100vh - 20px); }
    </style>
</head>
<body>
    <div class="info-panel">
        <h3>Инструкция:</h3>
        <p>• Кликните на любое здание (оранжевые точки) для построения нового маршрута</p>
        <p>• Красная линия показывает текущий маршрут от АТС до выбранного здания</p>
        <p>• Маршрут проходит через ближайшую муфту</p>
        <div id="status">Готов к работе</div>
    </div>
    <div id="plot"></div>
    
    <script>
        var plotData = """ + fig.to_json() + """;
        var plotDiv = document.getElementById('plot');
        var statusDiv = document.getElementById('status');
        
        // Parse the JSON data
        var data = JSON.parse(plotData).data;
        var layout = JSON.parse(plotData).layout;
        
        // Create the plot
        Plotly.newPlot(plotDiv, data, layout, {responsive: true});
        
        // Handle clicks on buildings
        plotDiv.on('plotly_click', function(eventData) {
            var point = eventData.points[0];
            
            // Check if clicked on buildings or test buildings
            if ((point.data.name === 'Здания' || point.data.name === 'Тестовые здания') && 
                point.customdata !== undefined) {
                
                var buildingIndex = point.customdata;
                var buildingInfo = point.text.split('<br>')[0];
                
                statusDiv.innerHTML = 'Выбрано: ' + buildingInfo + '<br>Расчет нового маршрута...';
                
                // Here you would normally send a request to your Python backend
                // For demo purposes, we'll just show the selection
                setTimeout(function() {
                    statusDiv.innerHTML = 'Маршрут обновлен для: ' + buildingInfo;
                }, 1000);
                
                console.log('Selected building index:', buildingIndex);
                console.log('Building info:', buildingInfo);
            }
        });
        
        // Add hover effects
        plotDiv.on('plotly_hover', function(eventData) {
            var point = eventData.points[0];
            if (point.data.name === 'Здания' || point.data.name === 'Тестовые здания') {
                plotDiv.style.cursor = 'pointer';
            }
        });
        
        plotDiv.on('plotly_unhover', function() {
            plotDiv.style.cursor = 'default';
        });
    </script>
</body>
</html>
        """

        # Save and open the HTML file
        with open("interactive_fiber_network.html", "w", encoding="utf-8") as f:
            f.write(html_template)
        
        print("Интерактивная карта создана: interactive_fiber_network.html")
        print("Откройте файл в браузере для использования!")
        
        # Try to open in browser automatically
        try:
            webbrowser.open("interactive_fiber_network.html")
        except:
            pass
        
        return fig

if __name__ == "__main__":
    # Set renderer for browser display
    pio.renderers.default = 'browser'

    # File paths
    roads_file = "mainroads.geojson"
    muffs_file = "centroidMufta.geojson"
    manholes_file = "manholesFinal.geojson"
    buildings_file = "buildings.geojson"

    # Check file existence
    for file in [roads_file, muffs_file, manholes_file, buildings_file]:
        if file and not os.path.exists(file):
            print(f"Файл {file} не найден, будут использованы тестовые данные.")

    # Create network object
    network = FiberNetworkFromRealData(
        roads_file=roads_file,
        muffs_file=muffs_file,
        manholes_file=manholes_file,
        buildings_file=buildings_file
    )

    # Print loaded data info
    print(f"Дороги: {len(network.roads)}, Муфты: {len(network.muffs)}, Колодцы: {len(network.wells)}, Здания: {len(network.buildings)}")
    print(f"Тестовые здания: {len(network.test_buildings)}, АТС: {network.ats}")

    # Select initial building
    if network.buildings:
        target_building = random.choice(network.buildings)
        building_point = target_building['centroid']
        print(f"Выбрано реальное здание: {target_building['name']}, центроид: {building_point}")
    elif network.test_buildings:
        target_building = random.choice(network.test_buildings)
        building_point = (target_building[0], target_building[1])
        print(f"Выбрано тестовое здание, координаты: {building_point}")
    else:
        print("Нет доступных зданий для подключения.")
        exit(1)

    # Check muffs availability
    if not network.muffs:
        print("Нет доступных муфт для подключения.")
        exit(1)

    # Calculate initial path
    initial_path = network.calculate_new_path(building_point)
    print(f"Начальный маршрут построен, длина: {len(initial_path)} точек")

    # Create interactive visualization
    try:
        network.create_interactive_map(building_point, initial_path)
    except Exception as e:
        print(f"Ошибка при создании визуализации: {e}")
        print("Проверьте файл 'interactive_fiber_network.html' для результатов.")
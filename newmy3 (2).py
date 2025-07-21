import matplotlib.pyplot as plt
import numpy as np
import json
import math
import plotly.graph_objects as go
from collections import defaultdict
import heapq
import os
import random
import plotly.io as pio

class FiberNetworkFromRealData:
    def __init__(self, roads_file=None, muffs_file=None, manholes_file=None, buildings_file=None, building_centers_file=None):
        self.roads = []
        self.road_nodes = {}
        self.road_graph = defaultdict(list)
        self.wells = []
        self.muffs = []
        self.buildings = []
        self.building_polygons = []
        self.test_buildings = []
        self.min_x = self.max_x = self.min_y = self.max_y = 0
        self.potential_manholes = []  # Store identified intersections as potential manholes
        self.load_all_data(roads_file, muffs_file, manholes_file, buildings_file, building_centers_file)

    def load_all_data(self, roads_file, muffs_file, manholes_file, buildings_file, building_centers_file):
        """Загружает все данные из файлов или создает тестовые данные."""
        if roads_file and os.path.exists(roads_file):
            self.load_roads(roads_file)
        if muffs_file and os.path.exists(muffs_file):
            self.load_muffs(muffs_file)
        if manholes_file and os.path.exists(manholes_file):
            self.load_manholes(manholes_file)
        if buildings_file and os.path.exists(buildings_file):
            self.load_buildings(buildings_file)
        if building_centers_file and os.path.exists(building_centers_file):
            self.load_building_centers(building_centers_file)

        if not self.roads and not self.wells and not self.muffs and not self.buildings:
            self._create_test_data()
            return

        self._build_road_graph()
        self._calculate_bounds()
        self._identify_intersections()  # Identify road intersections as potential manholes

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
        """Загружает здания из GeoJSON файла (для полигонов)."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feature in data['features']:
                geom = feature['geometry']
                props = feature['properties']
                if geom['type'] in ['Polygon', 'MultiPolygon']:
                    self._process_building_feature(feature)
            print(f"Загружено зданий (полигоны): {len(self.building_polygons)}")
        except Exception as e:
            print(f"Ошибка загрузки зданий: {e}")

    def load_building_centers(self, filename):
        """Загружает центроиды зданий из GeoJSON файла."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feature in data['features']:
                if feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates']
                    props = feature['properties']
                    x, y = coords[0], coords[1]
                    building_id = props.get('full_id', props.get('id', len(self.buildings)))
                    name = props.get('name', f'Здание_{building_id}')
                    building_type = props.get('building', 'residential')
                    apartments = props.get('KvartiraNum', np.random.randint(50, 200))
                    penetration_rate = props.get('penetration', 0.5)
                    fibers_needed = props.get('fibers_needed', self._calculate_fibers_needed(apartments, penetration_rate))
                    # Проверка, что центроид не совпадает с муфтой
                    is_mufta = False
                    for muff in self.muffs:
                        if abs(muff[0] - x) < 1e-6 and abs(muff[1] - y) < 1e-6:
                            print(f"Предупреждение: центроид здания {name} совпадает с муфтой ID {muff[3]}")
                            is_mufta = True
                            break
                    if not is_mufta:
                        self.buildings.append({
                            'id': building_id,
                            'name': name,
                            'type': building_type,
                            'centroid': (x, y),
                            'apartments': apartments,
                            'penetration_rate': penetration_rate,
                            'fibers_needed': fibers_needed,
                            'properties': props
                        })
            print(f"Загружено зданий (центроиды): {len(self.buildings)}")
        except Exception as e:
            print(f"Ошибка загрузки центроидов зданий: {e}")

    def _process_building_feature(self, feature):
        """Обрабатывает здание из GeoJSON (для полигонов)."""
        geom = feature['geometry']
        props = feature['properties']
        building_id = props.get('full_id', props.get('id', len(self.building_polygons)))
        name = props.get('name', f'Здание_{building_id}')
        apartments = props.get('KvartiraNum', np.random.randint(50, 200))
        self.building_polygons.append({
            'geometry': geom,
            'properties': props,
            'apartments': apartments,
            'name': name
        })

    def _calculate_fibers_needed(self, apartments, penetration_rate):
        """Вычисляет необходимое количество волокон: ceil(apartments * penetration_rate / 64) * 2."""
        try:
            ports_needed = apartments * penetration_rate
            fibers = math.ceil(ports_needed / 64) * 2
            return fibers
        except Exception as e:
            print(f"Ошибка вычисления волокон: {e}")
            return 2

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

    def _identify_intersections(self):
        """Идентифицирует пересечения дорог как потенциальные колодцы."""
        print("\n=== АНАЛИЗ ПЕРЕСЕЧЕНИЙ ДОРОГ ===")
        
        # Count connections for each node to identify intersections
        node_connections = defaultdict(int)
        for node_id, neighbors in self.road_graph.items():
            node_connections[node_id] = len(neighbors)
        
        # Find intersections (nodes with 3+ connections)
        intersection_count = 0
        for node_id, connection_count in node_connections.items():
            if connection_count >= 3:  # Intersection point
                x, y = self.road_nodes[node_id]
                
                # Check if it's already a manhole or mufta
                is_existing_infrastructure = False
                for well in self.wells:
                    if abs(well[0] - x) < 1e-5 and abs(well[1] - y) < 1e-5:
                        is_existing_infrastructure = True
                        break
                for muff in self.muffs:
                    if abs(muff[0] - x) < 1e-5 and abs(muff[1] - y) < 1e-5:
                        is_existing_infrastructure = True
                        break
                
                if not is_existing_infrastructure:
                    intersection_count += 1
                    manhole_id = f"POTENTIAL_MH_{intersection_count:03d}"
                    self.potential_manholes.append({
                        'id': manhole_id,
                        'coordinates': (x, y),
                        'connections': connection_count,
                        'node_id': node_id
                    })
                    print(f"Потенциальный колодец {manhole_id}: координаты ({x:.6f}, {y:.6f}), соединений: {connection_count}")
        
        print(f"\nВсего найдено потенциальных колодцев: {len(self.potential_manholes)}")
        print("=" * 50)

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

    def _create_test_buildings(self):
        """Создает тестовые здания рядом с инфраструктурой."""
        building_positions = []
        for i, muff in enumerate(self.muffs[:3]):
            x, y = muff[0], muff[1]
            offset_x = 0.002 * (1 if i % 2 == 0 else -1)
            offset_y = 0.002 * (1 if i < 2 else -1)
            apartments = np.random.randint(100, 400)
            penetration_rate = random.uniform(0.3, 0.7)
            building_positions.append((x + offset_x, y + offset_y, apartments, penetration_rate))
        for i, well in enumerate(self.wells[:3]):
            x, y = well[0], well[1]
            offset_x = 0.003 * (1 if i % 2 == 0 else -1)
            offset_y = 0.001 * (1 if i < 2 else -1)
            apartments = np.random.randint(50, 250)
            penetration_rate = random.uniform(0.3, 0.7)
            building_positions.append((x + offset_x, y + offset_y, apartments, penetration_rate))
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
        self._calculate_bounds()
        self._create_test_buildings()

    def find_nearest_mufta(self, building_point, fibers_needed, max_distance=500):
        """Ищет ближайшую муфту в радиусе max_distance с достаточным количеством свободных волокон."""
        min_distance = float('inf')
        nearest_mufta = None
        for muff in self.muffs:
            x, y, free_vol, muff_id = muff
            distance = math.sqrt((building_point[0] - x)**2 + (building_point[1] - y)**2) * 111139
            if distance <= max_distance and free_vol >= fibers_needed:
                if distance < min_distance:
                    min_distance = distance
                    nearest_mufta = muff
        return nearest_mufta, min_distance

    def find_path_through_roads_only(self, start, end):
        """Находит путь строго по дорожной сети, не пересекая жилые зоны."""
        print(f"\n=== ПОИСК ПУТИ ПО ДОРОГАМ ===")
        print(f"От: ({start[0]:.6f}, {start[1]:.6f})")
        print(f"До: ({end[0]:.6f}, {end[1]:.6f})")
        
        def find_nearest_road_node(point):
            """Находит ближайший узел дорожной сети."""
            min_dist = float('inf')
            nearest_node_id = None
            px, py = point
            for node_id, (x, y) in self.road_nodes.items():
                dist = math.hypot(px - x, py - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node_id = node_id
            return nearest_node_id, min_dist

        def dijkstra_with_path_info(start_node_id, end_node_id):
            """Алгоритм Дейкстры с детальной информацией о пути."""
            visited = set()
            min_heap = [(0, start_node_id, [])]
            distances = defaultdict(lambda: float('inf'))
            distances[start_node_id] = 0
            
            while min_heap:
                cost, current_node, path = heapq.heappop(min_heap)
                
                if current_node in visited:
                    continue
                    
                visited.add(current_node)
                path = path + [current_node]
                
                if current_node == end_node_id:
                    # Конвертируем node_id обратно в координаты
                    road_path = [self.road_nodes[nid] for nid in path]
                    return road_path, cost
                
                for neighbor, weight in self.road_graph.get(current_node, []):
                    if neighbor not in visited:
                        new_cost = cost + weight
                        if new_cost < distances[neighbor]:
                            distances[neighbor] = new_cost
                            heapq.heappush(min_heap, (new_cost, neighbor, path))
            
            return [], float('inf')

        # Найти ближайшие узлы дорожной сети
        start_node_id, start_distance = find_nearest_road_node(start)
        end_node_id, end_distance = find_nearest_road_node(end)
        
        print(f"Ближайший узел к началу: {start_node_id}, расстояние: {start_distance*111139:.2f} м")
        print(f"Ближайший узел к концу: {end_node_id}, расстояние: {end_distance*111139:.2f} м")
        
        if not start_node_id or not end_node_id:
            print("Не удалось найти подходящие узлы дорожной сети!")
            return [start, end], 0
        
        # Найти путь по дорогам
        road_path, total_road_distance = dijkstra_with_path_info(start_node_id, end_node_id)
        
        if not road_path:
            print("Путь по дорогам не найден!")
            return [start, end], 0
        
        # Добавить соединения от/до зданий
        final_path = [start] + road_path + [end]
        
        # Убрать дубликаты соседних точек
        cleaned_path = [final_path[0]]
        for i in range(1, len(final_path)):
            if final_path[i] != cleaned_path[-1]:
                cleaned_path.append(final_path[i])
        
        # Вычислить общую длину пути
        total_distance = start_distance  # От здания до дороги
        total_distance += total_road_distance  # По дорогам
        total_distance += end_distance  # От дороги до муфты
        total_distance_meters = total_distance * 111139  # Конвертация в метры
        
        print(f"Путь найден! Общая длина: {total_distance_meters:.2f} м")
        print(f"Количество точек в пути: {len(cleaned_path)}")
        
        # Анализ пересечений в пути
        intersections_in_path = []
        for i, point in enumerate(cleaned_path[1:-1], 1):  # Исключаем начало и конец
            # Проверяем, является ли эта точка пересечением
            for potential_mh in self.potential_manholes:
                if abs(point[0] - potential_mh['coordinates'][0]) < 1e-5 and \
                   abs(point[1] - potential_mh['coordinates'][1]) < 1e-5:
                    intersections_in_path.append({
                        'position_in_path': i,
                        'coordinates': point,
                        'manhole_id': potential_mh['id'],
                        'connections': potential_mh['connections']
                    })
                    break
        
        if intersections_in_path:
            print(f"\nПересечения на пути (потенциальные колодцы):")
            for intersection in intersections_in_path:
                print(f"  {intersection['manhole_id']}: позиция {intersection['position_in_path']}, "
                      f"координаты ({intersection['coordinates'][0]:.6f}, {intersection['coordinates'][1]:.6f}), "
                      f"соединений: {intersection['connections']}")
        else:
            print("\nНа пути нет пересечений дорог.")
        
        print("=" * 50)
        
        return cleaned_path, total_distance_meters

    def calculate_path_distance(self, path_coords):
        """Вычисляет общее расстояние пути в метрах."""
        if len(path_coords) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(path_coords) - 1):
            x1, y1 = path_coords[i]
            x2, y2 = path_coords[i + 1]
            # Приблизительное расстояние в метрах (для небольших расстояний)
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 111139
            total_distance += distance
        
        return total_distance

    def visualize_path_plotly(self, start, end, path_coords, fibers_needed, mufta, total_distance):
        """Визуализирует сеть и маршрут с помощью Plotly."""
        print(f"\n=== ВИЗУАЛИЗАЦИЯ МАРШРУТА ===")
        print(f"Общая длина маршрута: {total_distance:.2f} м")
        
        fig = go.Figure()

        # Дороги (серые линии)
        for road in self.roads:
            coords = road['coordinates']
            xs, ys = zip(*coords)
            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='lines',
                line=dict(color='lightgray', width=2),
                name='Дорога',
                hoverinfo='skip',
                showlegend=False
            ))

        # Колодцы (синие круги)
        xs, ys, labels = [], [], []
        for x, y, well_id, name in self.wells:
            xs.append(x)
            ys.append(y)
            labels.append(f"Колодец: {name} (ID: {well_id})")
        if xs:
            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='markers',
                marker=dict(color='blue', size=10, line=dict(width=1, color='darkblue')),
                name='Существующий колодец',
                text=labels,
                hoverinfo='text'
            ))

        # Потенциальные колодцы (фиолетовые ромбы)
        if self.potential_manholes:
            xs, ys, labels = [], [], []
            for mh in self.potential_manholes:
                x, y = mh['coordinates']
                xs.append(x)
                ys.append(y)
                labels.append(f"Потенциальный колодец: {mh['id']}<br>Соединений: {mh['connections']}")
            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='markers',
                marker=dict(color='blueviolet', size=8, symbol='diamond', line=dict(width=1, color='purple')),
                name='Потенциальный колодец',
                text=labels,
                hoverinfo='text'
            ))

        # Муфты (красные круги)
        xs, ys, labels = [], [], []
        for x, y, free_vol, muff_id in self.muffs:
            xs.append(x)
            ys.append(y)
            labels.append(f"Муфта ID: {muff_id}<br>Свободно волокон: {free_vol}")
        if xs:
            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='markers',
                marker=dict(color='red', size=10, line=dict(width=1, color='darkred')),
                name='Муфта',
                text=labels,
                hoverinfo='text'
            ))

        # Здания (оранжевые полигоны)
        for b in self.building_polygons:
            geom = b['geometry']
            name = b['name']
            apartments = b['apartments']
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
                xs, ys = zip(*coords)
                fig.add_trace(go.Scattergl(
                    x=xs, y=ys,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(255, 165, 0, 0.3)',
                    line=dict(color='brown', width=1.5),
                    name='Здание',
                    text=[f"{name}<br>Квартир: {apartments}"] * len(xs),
                    hoverinfo='text',
                    showlegend=False
                ))
            elif geom['type'] == 'MultiPolygon':
                for poly_coords in geom['coordinates']:
                    xs, ys = zip(*poly_coords[0])
                    fig.add_trace(go.Scattergl(
                        x=xs, y=ys,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 165, 0, 0.3)',
                        line=dict(color='brown', width=1.5),
                        name='Здание',
                        text=[f"{name}<br>Квартир: {apartments}"] * len(xs),
                        hoverinfo='text',
                        showlegend=False
                    ))

        # Маршрут (толстая красная линия)
        if path_coords and len(path_coords) > 1:
            xs, ys = zip(*path_coords)
            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(color='red', size=6),
                name=f'Маршрут ({total_distance:.0f}м)',
                hoverinfo='none'
            ))

        # Целевая муфта (зеленый квадрат)
        fig.add_trace(go.Scattergl(
            x=[mufta[0]], y=[mufta[1]],
            mode='markers',
            marker=dict(color='green', size=16, symbol='square', line=dict(width=2, color='black')),
            name=f'Целевая муфта (ID: {mufta[3]})',
            hoverinfo='text',
            text=[f'Муфта ID: {mufta[3]}<br>Свободно волокон: {mufta[2]}']
        ))

        # Целевое здание (черный квадрат)
        fig.add_trace(go.Scattergl(
            x=[start[0]], y=[start[1]],
            mode='markers',
            marker=dict(color='black', size=16, symbol='square', line=dict(width=2, color='white')),
            name='Целевое здание',
            hoverinfo='text',
            text=[f'Здание<br>Требуется волокон: {fibers_needed}<br>Расстояние до муфты: {total_distance:.0f}м']
        ))

        # Вычисляем границы для начального зума
        all_coords = path_coords + [(mufta[0], mufta[1]), start]
        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]
        x_range = [min(x_coords) - 0.001, max(x_coords) + 0.001]
        y_range = [min(y_coords) - 0.001, max(y_coords) + 0.001]

        # Настройка графика
        fig.update_layout(
            title=f"Оптимальный маршрут подключения (Длина: {total_distance:.0f}м)",
            xaxis_title="Долгота",
            yaxis_title="Широта",
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.9)'),
            hovermode='closest',
            dragmode='pan',
            uirevision='constant',
            margin=dict(l=0, r=0, t=60, b=0),
            xaxis=dict(
                range=x_range,
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
            ),
            yaxis=dict(
                range=y_range,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        # Сохраняем график в HTML для отладки
        fig.write_html("network_visualization.html")
        print("График сохранен в 'network_visualization.html'")
        print("=" * 50)

        # Показываем график
        fig.show()

if __name__ == "__main__":
    import random
    import plotly.io as pio

    pio.renderers.default = 'browser'

    roads_file = "superroads.geojson"
    muffs_file = "mufcentroids.geojson"
    manholes_file = "finalfinalmn.geojson"
    buildings_file = "buildings.geojson"
    building_centers_file = "buildingscenter.geojson"

    for file in [roads_file, muffs_file, manholes_file, buildings_file, building_centers_file]:
        if file and not os.path.exists(file):
            print(f"Файл {file} не найден, будут использованы тестовые данные.")

    print("=" * 60)
    print("СИСТЕМА ПЛАНИРОВАНИЯ ОПТИЧЕСКИХ СЕТЕЙ")
    print("=" * 60)

    network = FiberNetworkFromRealData(
        roads_file=roads_file,
        muffs_file=muffs_file,
        manholes_file=manholes_file,
        buildings_file=buildings_file,
        building_centers_file=building_centers_file
    )

    print(f"\n=== ЗАГРУЖЕННЫЕ ДАННЫЕ ===")
    print(f"Дороги: {len(network.roads)}")
    print(f"Узлы дорожной сети: {len(network.road_nodes)}")
    print(f"Муфты: {len(network.muffs)}")
    print(f"Существующие колодцы: {len(network.wells)}")
    print(f"Здания (полигоны): {len(network.building_polygons)}")
    print(f"Здания (центроиды): {len(network.buildings)}")
    print(f"Тестовые здания: {len(network.test_buildings)}")
    print(f"Потенциальные колодцы: {len(network.potential_manholes)}")

    # Выбор здания для подключения
    if network.buildings:
        target_building = random.choice(network.buildings)
        building_point = target_building['centroid']
        fibers_needed = target_building['fibers_needed']
        print(f"\n=== ВЫБРАННОЕ ЗДАНИЕ ===")
        print(f"Название: {target_building['name']}")
        print(f"Координаты: ({building_point[0]:.6f}, {building_point[1]:.6f})")
        print(f"Квартир: {target_building['apartments']}")
        print(f"Процент проникновения: {target_building['penetration_rate']:.1%}")
        print(f"Требуется волокон: {fibers_needed}")
    elif network.test_buildings:
        target_building = random.choice(network.test_buildings)
        building_point = (target_building[0], target_building[1])
        apartments = target_building[2]
        penetration_rate = target_building[3]
        fibers_needed = network._calculate_fibers_needed(apartments, penetration_rate)
        print(f"\n=== ВЫБРАННОЕ ТЕСТОВОЕ ЗДАНИЕ ===")
        print(f"Координаты: ({building_point[0]:.6f}, {building_point[1]:.6f})")
        print(f"Квартир: {apartments}")
        print(f"Процент проникновения: {penetration_rate:.1%}")
        print(f"Требуется волокон: {fibers_needed}")
    else:
        print("\nОШИБКА: Нет доступных зданий для подключения.")
        exit(1)

    if not network.muffs:
        print("\nОШИБКА: Нет доступных муфт для подключения.")
        exit(1)

    # Поиск ближайшей подходящей муфты
    print(f"\n=== ПОИСК МУФТЫ ===")
    nearest_mufta, distance = network.find_nearest_mufta(building_point, fibers_needed, max_distance=500)
    if nearest_mufta is None:
        print(f"ОШИБКА: Не найдено муфт в радиусе 500 метров с достаточным количеством волокон (>= {fibers_needed}).")
        # Попробуем найти любую ближайшую муфту
        nearest_mufta = min(network.muffs, key=lambda m: math.sqrt((building_point[0] - m[0])**2 + (building_point[1] - m[1])**2))
        distance = math.sqrt((building_point[0] - nearest_mufta[0])**2 + (building_point[1] - nearest_mufta[1])**2) * 111139
        print(f"Использую ближайшую доступную муфту (может не хватить волокон):")
    
    print(f"Выбранная муфта:")
    print(f"  ID: {nearest_mufta[3]}")
    print(f"  Координаты: ({nearest_mufta[0]:.6f}, {nearest_mufta[1]:.6f})")
    print(f"  Прямое расстояние: {distance:.2f} м")
    print(f"  Свободно волокон: {nearest_mufta[2]}")
    print(f"  Достаточно волокон: {'Да' if nearest_mufta[2] >= fibers_needed else 'НЕТ'}")

    # Поиск оптимального пути по дорогам
    path, total_distance = network.find_path_through_roads_only(building_point, (nearest_mufta[0], nearest_mufta[1]))
    
    print(f"\n=== ИТОГОВЫЙ МАРШРУТ ===")
    print(f"Количество точек в маршруте: {len(path)}")
    print(f"Общая длина маршрута: {total_distance:.2f} м")
    print(f"Экономия по сравнению с прямой линией: {abs(total_distance - distance):.2f} м")
    
    # Детализация маршрута
    print(f"\nДетализация маршрута:")
    for i, point in enumerate(path):
        if i == 0:
            print(f"  {i+1}. СТАРТ - Здание: ({point[0]:.6f}, {point[1]:.6f})")
        elif i == len(path) - 1:
            print(f"  {i+1}. ФИНИШ - Муфта ID {nearest_mufta[3]}: ({point[0]:.6f}, {point[1]:.6f})")
        else:
            # Проверяем, является ли точка существующим колодцем
            is_existing_well = False
            for well in network.wells:
                if abs(point[0] - well[0]) < 1e-5 and abs(point[1] - well[1]) < 1e-5:
                    print(f"  {i+1}. Существующий колодец '{well[3]}': ({point[0]:.6f}, {point[1]:.6f})")
                    is_existing_well = True
                    break
            
            if not is_existing_well:
                # Проверяем, является ли точка потенциальным колодцем
                is_potential_mh = False
                for mh in network.potential_manholes:
                    if abs(point[0] - mh['coordinates'][0]) < 1e-5 and abs(point[1] - mh['coordinates'][1]) < 1e-5:
                        print(f"  {i+1}. {mh['id']} (пересечение дорог): ({point[0]:.6f}, {point[1]:.6f})")
                        is_potential_mh = True
                        break
                
                if not is_potential_mh:
                    print(f"  {i+1}. Узел дороги: ({point[0]:.6f}, {point[1]:.6f})")

    # Визуализация
    try:
        network.visualize_path_plotly(
            start=building_point, 
            end=(nearest_mufta[0], nearest_mufta[1]), 
            path_coords=path, 
            fibers_needed=fibers_needed, 
            mufta=nearest_mufta,
            total_distance=total_distance
        )
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
        print("Проверьте 'network_visualization.html' для результатов.")

    print(f"\n{'='*60}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*60}")
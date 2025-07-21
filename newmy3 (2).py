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

    def find_path_through_infrastructure(self, start, end):
        """Находит путь через дорожную сеть и колодцы."""
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

        path_coords = [start]
        for well in self.wells:
            wx, wy, _, _ = well
            dist_to_start = math.hypot(start[0] - wx, start[1] - wy)
            dist_to_end = math.hypot(end[0] - wx, end[1] - wy)
            if dist_to_start < 0.005 and dist_to_end < 0.005:
                path_coords.append((wx, wy))
        path_coords.append(end)

        if self.road_nodes:
            start_node_id = find_nearest_road_node(start)
            end_node_id = find_nearest_road_node(end)
            if start_node_id and end_node_id:
                road_path = dijkstra(start_node_id, end_node_id)
                if road_path:
                    if road_path[0] != start:
                        road_path = [start] + road_path
                    if road_path[-1] != end:
                        road_path.append(end)
                    return road_path
        return path_coords

    def visualize_path_plotly(self, start, end, path_coords, fibers_needed, mufta):
        """Визуализирует сеть и маршрут с помощью Plotly."""
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
        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            mode='markers',
            marker=dict(color='blue', size=10, line=dict(width=1, color='darkblue')),
            name='Колодец',
            text=labels,
            hoverinfo='text'
        ))

        # Муфты (красные круги)
        xs, ys, labels = [], [], []
        for x, y, free_vol, muff_id in self.muffs:
            xs.append(x)
            ys.append(y)
            labels.append(f"Муфта ID: {muff_id}<br>Свободно волокон: {free_vol}")
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
                    fillcolor='rgba(255, 165, 0, 0.5)',
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
                        fillcolor='rgba(255, 165, 0, 0.5)',
                        line=dict(color='brown', width=1.5),
                        name='Здание',
                        text=[f"{name}<br>Квартир: {apartments}"] * len(xs),
                        hoverinfo='text',
                        showlegend=False
                    ))

        # Маршрут (красная линия)
        if path_coords:
            xs, ys = zip(*path_coords)
            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='lines',
                line=dict(color='red', width=4),
                name='Маршрут',
                hoverinfo='none'
            ))

            # Add blueviolet dots for intersection nodes (excluding start, end, manholes, muftas)
            intersection_nodes = []
            for node in path_coords[1:-1]:  # Skip start and end
                is_special_node = False
                # Check if node is a manhole
                for well in self.wells:
                    if abs(node[0] - well[0]) < 1e-6 and abs(node[1] - well[1]) < 1e-6:
                        is_special_node = True
                        break
                # Check if node is a mufta
                for muff in self.muffs:
                    if abs(node[0] - muff[0]) < 1e-6 and abs(node[1] - muff[1]) < 1e-6:
                        is_special_node = True
                        break
                if not is_special_node:
                    intersection_nodes.append(node)
            
            if intersection_nodes:
                xs_inter, ys_inter = zip(*intersection_nodes)
                fig.add_trace(go.Scattergl(
                    x=xs_inter, y=ys_inter,
                    mode='markers',
                    marker=dict(color='blueviolet', size=8),
                    name='Пересечения',
                    hoverinfo='none'
                ))

        # Муфта (зеленый квадрат)
        fig.add_trace(go.Scattergl(
            x=[mufta[0]], y=[mufta[1]],
            mode='markers',
            marker=dict(color='green', size=14, symbol='square', line=dict(width=1, color='black')),
            name=f'Муфта (ID: {mufta[3]})',
            hoverinfo='text',
            text=[f'Муфта ID: {mufta[3]}<br>Свободно волокон: {mufta[2]}']
        ))

        # Здание (черный квадрат)
        fig.add_trace(go.Scattergl(
            x=[start[0]], y=[start[1]],
            mode='markers',
            marker=dict(color='black', size=14, symbol='square', line=dict(width=1, color='white')),
            name='Здание (старт)',
            hoverinfo='text',
            text=[f'Здание<br>Требуется волокон: {fibers_needed}']
        ))

        # Вычисляем границы для начального зума
        all_coords = path_coords + [(mufta[0], mufta[1]), start]
        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]
        x_range = [min(x_coords) - 0.001, max(x_coords) + 0.001]
        y_range = [min(y_coords) - 0.001, max(y_coords) + 0.001]

        # Настройка графика
        fig.update_layout(
            title="Интерактивный маршрут подключения",
            xaxis_title="Долгота",
            yaxis_title="Широта",
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
            hovermode='closest',
            dragmode='pan',
            uirevision='constant',
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(
                range=x_range,
                scaleanchor="y",
                scaleratio=1,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=y_range,
                showgrid=False,
                zeroline=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        # Сохраняем график в HTML для отладки
        fig.write_html("network_visualization.html")
        print("График сохранен в 'network_visualization.html'")

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

    network = FiberNetworkFromRealData(
        roads_file=roads_file,
        muffs_file=muffs_file,
        manholes_file=manholes_file,
        buildings_file=buildings_file,
        building_centers_file=building_centers_file
    )

    print(f"Дороги: {len(network.roads)}, Муфты: {len(network.muffs)}, Колодцы: {len(network.wells)}, Здания: {len(network.buildings)}")
    print(f"Тестовые здания: {len(network.test_buildings)}")

    if network.buildings:
        target_building = random.choice(network.buildings)
        building_point = target_building['centroid']
        fibers_needed = target_building['fibers_needed']
        print(f"Выбрано реальное здание: {target_building['name']}, центроид: {building_point}, требуется волокон: {fibers_needed}")
    elif network.test_buildings:
        target_building = random.choice(network.test_buildings)
        building_point = (target_building[0], target_building[1])
        apartments = target_building[2]
        penetration_rate = target_building[3]
        fibers_needed = network._calculate_fibers_needed(apartments, penetration_rate)
        print(f"Выбрано тестовое здание, координаты: {building_point}, требуется волокон: {fibers_needed}")
    else:
        print("Нет доступных зданий для подключения.")
        exit(1)

    if not network.muffs:
        print("Нет доступных муфт для подключения.")
        exit(1)

    nearest_mufta, distance = network.find_nearest_mufta(building_point, fibers_needed, max_distance=500)
    if nearest_mufta is None:
        print(f"Не найдено муфт в радиусе 500 метров с достаточным количеством волокон (>= {fibers_needed}).")
        exit(1)
    print(f"Ближайшая муфта: ID {nearest_mufta[3]}, координаты: ({nearest_mufta[0]}, {nearest_mufta[1]}), расстояние: {distance:.2f} м, свободно волокон: {nearest_mufta[2]}")

    path = network.find_path_through_infrastructure(building_point, (nearest_mufta[0], nearest_mufta[1]))
    print(f"Маршрут (здание → муфта): {path}")

    try:
        network.visualize_path_plotly(start=building_point, end=(nearest_mufta[0], nearest_mufta[1]), path_coords=path, fibers_needed=fibers_needed, mufta=nearest_mufta)
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
        print("Проверьте 'network_visualization.html' для результатов.")
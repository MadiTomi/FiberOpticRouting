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
        self.fig = None  # Store the figure for updates
        self.load_all_data(roads_file, muffs_file, manholes_file, buildings_file)

    # ... existing code ...

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

    def on_building_click(self, trace, points, selector):
        """Handle building click events"""
        if points.point_inds:
            # Get the clicked building index
            clicked_idx = points.point_inds[0]
            
            # Find which building was clicked based on the trace
            if 'building_idx' in trace.customdata[clicked_idx]:
                building_idx = trace.customdata[clicked_idx]['building_idx']
                if building_idx < len(self.buildings):
                    building = self.buildings[building_idx]
                    building_point = building['centroid']
                    print(f"Выбрано здание: {building['name']}")
                elif building_idx - len(self.buildings) < len(self.test_buildings):
                    test_idx = building_idx - len(self.buildings)
                    building_point = (self.test_buildings[test_idx][0], self.test_buildings[test_idx][1])
                    print(f"Выбрано тестовое здание: {building_point}")
                
                # Calculate new path
                new_path = self.calculate_new_path(building_point)
                
                # Update visualization
                self.update_path_visualization(building_point, new_path)

    def update_path_visualization(self, end_point, path_coords):
        """Update the path visualization without recreating the entire plot"""
        if self.fig is None:
            return
            
        # Remove old path and target marker
        traces_to_remove = []
        for i, trace in enumerate(self.fig.data):
            if trace.name in ['Маршрут', 'Здание (финиш)']:
                traces_to_remove.append(i)
        
        # Remove traces in reverse order to maintain indices
        for i in reversed(traces_to_remove):
            self.fig.data = list(self.fig.data[:i]) + list(self.fig.data[i+1:])

        # Add new path
        if path_coords:
            xs, ys = zip(*path_coords)
            self.fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=8, color='red'),
                name='Маршрут',
                hoverinfo='none'
            ))

        # Add new target building marker
        self.fig.add_trace(go.Scattergl(
            x=[end_point[0]], y=[end_point[1]],
            mode='markers',
            marker=dict(color='black', size=14, symbol='square', line=dict(width=1, color='white')),
            name='Здание (финиш)',
            hoverinfo='text',
            text=['Выбранное здание']
        ))

    def visualize_path_plotly(self, start, end, path_coords):
        """Визуализирует сеть и маршрут с помощью Plotly с OpenStreetMap фоном."""
        self.fig = go.Figure()

        # Add OpenStreetMap background
        self.fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=(self.min_y + self.max_y)/2, lon=(self.min_x + self.max_x)/2),
                zoom=12
            )
        )

        # Дороги (серые линии)
        for road in self.roads:
            coords = road['coordinates']
            xs, ys = zip(*coords)
            self.fig.add_trace(go.Scattergl(
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
            self.fig.add_trace(go.Scattergl(
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
        if xs:
            self.fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='markers',
                marker=dict(color='red', size=10, line=dict(width=1, color='darkred')),
                name='Муфта',
                text=labels,
                hoverinfo='text'
            ))

        # Здания (оранжевые полигоны) - clickable
        building_idx = 0
        for b in self.building_polygons:
            geom = b['geometry']
            name = b['name']
            residents = b['residents']
            customdata = [{'building_idx': building_idx}] * len(geom['coordinates'][0] if geom['type'] == 'Polygon' else geom['coordinates'][0][0])
            
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
                xs, ys = zip(*coords)
                trace = go.Scattergl(
                    x=xs, y=ys,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(255, 165, 0, 0.5)',
                    line=dict(color='brown', width=1.5),
                    name='Здание',
                    text=[f"{name}<br>Жителей: {residents}<br>Кликните для выбора"] * len(xs),
                    hoverinfo='text',
                    showlegend=False,
                    customdata=customdata
                )
                trace.on_click(self.on_building_click)
                self.fig.add_trace(trace)
            elif geom['type'] == 'MultiPolygon':
                for poly_coords in geom['coordinates']:
                    xs, ys = zip(*poly_coords[0])
                    trace = go.Scattergl(
                        x=xs, y=ys,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 165, 0, 0.5)',
                        line=dict(color='brown', width=1.5),
                        name='Здание',
                        text=[f"{name}<br>Жителей: {residents}<br>Кликните для выбора"] * len(xs),
                        hoverinfo='text',
                        showlegend=False,
                        customdata=customdata
                    )
                    trace.on_click(self.on_building_click)
                    self.fig.add_trace(trace)
            building_idx += 1

        # Test buildings (clickable)
        for i, (x, y, residents) in enumerate(self.test_buildings):
            customdata = [{'building_idx': len(self.buildings) + i}]
            trace = go.Scattergl(
                x=[x], y=[y],
                mode='markers',
                marker=dict(color='orange', size=12, symbol='square', line=dict(width=1, color='darkorange')),
                name='Тестовое здание',
                text=[f"Тестовое здание {i+1}<br>Жителей: {residents}<br>Кликните для выбора"],
                hoverinfo='text',
                showlegend=False,
                customdata=customdata
            )
            trace.on_click(self.on_building_click)
            self.fig.add_trace(trace)

        # Маршрут (красная линия)
        if path_coords:
            xs, ys = zip(*path_coords)
            self.fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=8, color='red'),
                name='Маршрут',
                hoverinfo='none'
            ))

        # АТС (фиолетовый квадрат)
        self.fig.add_trace(go.Scattergl(
            x=[start[0]], y=[start[1]],
            mode='markers',
            marker=dict(color='magenta', size=14, symbol='square', line=dict(width=1, color='black')),
            name='АТС (старт)',
            hoverinfo='text',
            text=['АТС']
        ))

        # Здание (черный квадрат)
        self.fig.add_trace(go.Scattergl(
            x=[end[0]], y=[end[1]],
            mode='markers',
            marker=dict(color='black', size=14, symbol='square', line=dict(width=1, color='white')),
            name='Здание (финиш)',
            hoverinfo='text',
            text=['Выбранное здание']
        ))

        # Настройка графика с OpenStreetMap
        self.fig.update_layout(
            title="Интерактивный маршрут подключения - Кликните на здание для выбора",
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
            hovermode='closest',
            dragmode='pan',
            margin=dict(l=0, r=0, t=50, b=0),
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=(self.min_y + self.max_y)/2,
                    lon=(self.min_x + self.max_x)/2
                ),
                zoom=13
            ),
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Store current path and target
        self.current_path = path_coords
        self.current_target_building = end

        # Сохраняем график в HTML для отладки
        self.fig.write_html("network_visualization.html")
        print("График сохранен в 'network_visualization.html'")
        print("Кликните на любое здание на карте для выбора нового маршрута!")

        # Показываем график
        self.fig.show()

    # ... existing code ...
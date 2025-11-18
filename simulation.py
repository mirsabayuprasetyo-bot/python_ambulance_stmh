import random
import webbrowser
import osmnx as ox
import pandas as pd
import os
from datetime import datetime
import folium
from folium.plugins import MarkerCluster, TimestampedGeoJson
from folium.map import Popup, Icon
import map_downloader as map
import hospital as hosp
import ambulance as amb
import patient_caller as patient
from folium.plugins import Timeline, TimelineSlider
import ga_routing as genetics
import djikstra_routing as djikstra
import astar_routing as astar
import copy
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt


class simulation():
    def __init__(self):
        self.map_nodes = {}
        self.downloader = map.map_downloader()
        self.caller_nodes = []
        self.hospital_nodes = []
        self.map_graph_with_traffic = {}
        pass

    def run_simulation(self, location_name):
        self.__define_hospital_and_ambulance_agents(location_name)
        self.__define_caller_agents(location_name,10)
        self.__define_traffic_condition(location_name)

        out_dir = Path.cwd() / f"simulation_{location_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        map_ox_path = out_dir / f"map_ox_{location_name}.html"
        map_djikstra_path = out_dir / f"map_djikstra_{location_name}.html"
        map_astar_path = out_dir / f"map_astar_{location_name}.html"
        map_ga_path = out_dir / f"map_ga_{location_name}.html"
        plot_path = out_dir / f"mean_response_time_{location_name}.png"
        
        simulation_records_ox = self.__simulate_ambulance_movement(location_name, simulation_time_in_minute=5, algorithm="ox")
        simulation_records_djikstra = self.__simulate_ambulance_movement(location_name, simulation_time_in_minute=5, algorithm="djikstra")
        simulation_records_astar = self.__simulate_ambulance_movement(location_name, simulation_time_in_minute=5, algorithm="astar")
        simulation_records_ga = self.__simulate_ambulance_movement(location_name, simulation_time_in_minute=5, algorithm="ga")

        folium_map_ox = self.__visualize_simulation(location_name, simulation_records_ox)
        folium_map_djikstra = self.__visualize_simulation(location_name, simulation_records_djikstra)
        folium_map_astar = self.__visualize_simulation(location_name, simulation_records_astar)
        folium_map_ga = self.__visualize_simulation(location_name, simulation_records_ga)

        folium_map_ox.save(map_ox_path)
        folium_map_djikstra.save(map_djikstra_path)
        folium_map_astar.save(map_astar_path)
        folium_map_ga.save(map_ga_path)

        mean_response_time_djikstra = self.__calculate_mean_ambulance_response_time(simulation_records_djikstra)
        mean_response_time_astar = self.__calculate_mean_ambulance_response_time(simulation_records_astar)
        mean_response_time_ox = self.__calculate_mean_ambulance_response_time(simulation_records_ox)
        mean_response_time_ga = self.__calculate_mean_ambulance_response_time(simulation_records_ga)


        self.__save_bar_plot_mean_response_time(plot_path, mean_response_time_djikstra, mean_response_time_astar, mean_response_time_ox, mean_response_time_ga)

        self.__combine_maps_and_graph(location_name, map_ox_path, map_djikstra_path, map_astar_path, map_ga_path, plot_path)

    def __define_hospital_and_ambulance_agents(self, location_name):
        map_nodes = self.downloader.get_map_nodes(location_name)  # Use self.downloader
        hospital_df = self.downloader.get_hospital_location_dataframe(location_name)  # Use self.downloader
        hospital_nodes = []
        for idx, row in hospital_df.iterrows():
            hospital_id = row.get('id')
            hospital_lat = row.get('lat')
            hospital_lon = row.get('lon')
            hospital_name = row.get('name')
            hospital_address = row.get('address')
            hospital_type = row.get('type')
            hospital_class = row.get('kelas')
            hospital_ambulance = row.get('ambulance')
            nearest_node = ox.nearest_nodes(map_nodes, hospital_lon, hospital_lat)
            hospital_node = hosp.hospital()
            hospital_node.setup_hospital(hospital_id, hospital_name, hospital_address, hospital_type,
                                         hospital_class, hospital_lat, hospital_lon, nearest_node)
            
            for i in range (hospital_ambulance):
                ambulance_node = amb.ambulance()
                ambulance_id = "ambulance_"+hospital_name+"_"+str(random.randint(0,100))
                ambulance_node.setup_ambulance(ambulance_id, hospital_id, nearest_node)
                hospital_node.add_ambulance(ambulance_node)

            hospital_nodes.append(hospital_node)
        
        self.hospital_nodes = hospital_nodes
        pass

    def __define_caller_agents(self, location_name, max_caller):
        map_graph = self.downloader.get_map_nodes(location_name)
        all_nodes = list(map_graph.nodes())
        total_caller = random.randrange(1, max_caller)
        caller_nodes = []
        for i in range(total_caller):
            caller_id = "patient_"+str(random.randint(0,1000))
            severity = ["low", "medium", "high"]
            severity_caller = random.choice(severity)
            idx_nodes = random.choice(all_nodes)
            caller_lat = map_graph.nodes[idx_nodes]["y"]
            caller_lon = map_graph.nodes[idx_nodes]["x"]
            caller_inst = patient.patient_caller(caller_id, idx_nodes, caller_lat,
                                                 caller_lon, severity_caller)
            caller_inst.set_responded(False)
            caller_nodes.append(caller_inst)
        self.caller_nodes = caller_nodes

    def __define_traffic_condition(self, location_name):
        map_graph = self.downloader.get_map_nodes(location_name)  # Use self.downloader
        map_edge = self.downloader.get_gdf_edge(location_name)  # Use self.downloader

        for idx, row in map_edge.iterrows():
            u = row['u']
            v = row['v']
            k = row['key']
            # Generate random traffic condition (0.1 = light traffic, 1.5 = heavy congestion)
            traffic_factor = random.uniform(0.1, 1.5)
            # Update the edge in map_graph with traffic factor
            if map_graph.has_edge(u, v, k):
                map_graph[u][v][k]['maxspeed'] = 50  # assuming default speed 50 m/s
                map_graph[u][v][k]['traffic_factor'] = traffic_factor
                map_graph[u][v][k]['time_s'] = map_graph[u][v][k]['length'] / 50  # assuming default speed 50 m/s
                map_graph[u][v][k]['time_per_edge'] = map_graph[u][v][k]['time_s'] * traffic_factor 

        self.map_graph_with_traffic = map_graph
        
        pass

    def __simulate_ambulance_movement(self, location_name, simulation_time_in_minute=60,algorithm="ox"):
        map_graph = copy.deepcopy(self.map_graph_with_traffic)
        caller_nodes = copy.deepcopy(self.caller_nodes)
        hospital_nodes = copy.deepcopy(self.hospital_nodes)

        time_step_s = 1  # seconds
        max_simulation_duration_s = simulation_time_in_minute * 60  # convert minutes to seconds
        simulation_elapsed_time = 0
        
        simulation_records = []
        
        
        # Initial snapshot
        current_positions_snapshot = {}
        for hospital in hospital_nodes:
            for ambulance in hospital.get_ambulance_agents():
                ambulance_id = ambulance.get_ambulance_id()
                current_node = ambulance.get_current_location_node()
                current_positions_snapshot[ambulance_id] = {
                    'lat': map_graph.nodes[current_node]['y'],
                    'lon': map_graph.nodes[current_node]['x'],
                    'hospital_id': hospital.get_hospital_id(),
                    'status': ambulance.is_available()
                }

        simulation_records.append(
            {'time': simulation_elapsed_time, 
             'positions': current_positions_snapshot.copy()
            }
            )
        bIsResponseRecorded = False
        response_time = 0
        # Simulation loop
        while simulation_elapsed_time < max_simulation_duration_s:
            simulation_elapsed_time += time_step_s
            
            for caller in caller_nodes:
                if caller.is_responded():
                    continue

                # Find nearest available ambulance
                nearest_ambulance = None
                min_distance = float('inf')
            
                for hospital in hospital_nodes:
                    for ambulance in hospital.get_ambulance_agents():
                        if ambulance.is_available():
                            path = ox.shortest_path(map_graph, ambulance.get_current_location_node(), 
                            caller.get_node(), weight='length')

                            distance = len(path)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_ambulance = ambulance

                # Dispatch nearest ambulance
                if nearest_ambulance and min_distance != float('inf'):
                    nearest_ambulance.set_available(False)
                    nearest_ambulance.set_destination_node(caller.get_node())
                    caller.set_responded(True)
            
            # Move all responding ambulances to caller
            current_positions_snapshot = {}
            for hospital in hospital_nodes:
                for ambulance in hospital.get_ambulance_agents():
                    current_node = ambulance.get_current_location_node()
                    
                    if not ambulance.is_available():
                        destination = ambulance.get_destination_node()

                        path = None
                        if algorithm == "ga":
                            path = genetics.ga_shortest_path(map_graph, current_node, destination, weight='time_per_edge', population_size=10, num_generations=10)
                        elif algorithm == "ox":
                            path = ox.shortest_path(map_graph, current_node, destination, weight='time_per_edge')
                        elif algorithm == "astar":
                            router = astar.AStarRouter(default_weight='time_per_edge')
                            path = router.shortest_path(map_graph, current_node, destination)
                        elif algorithm == "djikstra":
                            router = djikstra.DijkstraRouter(map_graph, current_node, destination, weight='time_per_edge')
                            path = router.shortest_path()

                        if path is None:
                            continue
                        if len(path) > 1 :
                            ambulance.set_current_location_node(path[1])
                            current_node = path[1]
                        else : 
                            ambulance.set_current_location_node(destination)
                            current_node = destination

                        if current_node == destination:
                            if current_node == ambulance.get_origin_node() and not ambulance.is_available():
                                current_positions_snapshot[ambulance.get_ambulance_id()] = {
                                    'lat': map_graph.nodes[current_node]['y'],
                                    'lon': map_graph.nodes[current_node]['x'],
                                    'hospital_id': ambulance.get_ambulance_id(),
                                    'status': ambulance.is_returned(),
                                    'response_time': simulation_elapsed_time/2
                                    }
                                simulation_records.append({'time': simulation_elapsed_time, 'positions': current_positions_snapshot.copy()})
                                ambulance.set_available(True)

                            ambulance.set_current_location_node(destination)
                            ambulance.set_destination_node(ambulance.get_origin_node())
                            ambulance.set_take_patient_to_hospital()

                        current_positions_snapshot[ambulance.get_ambulance_id()] = {
                                    'lat': map_graph.nodes[current_node]['y'],
                                    'lon': map_graph.nodes[current_node]['x'],
                                    'hospital_id': ambulance.get_ambulance_id(),
                                    'status': ambulance.is_returned()
                                    }
                            
                        
                        
            simulation_records.append({'time': simulation_elapsed_time, 'positions': current_positions_snapshot.copy()})
            print(f"simulation_elapsed_time for {algorithm}: {simulation_elapsed_time}")

        return simulation_records

    def __visualize_simulation(self, location_name, simulation_records):
        map_edge = self.downloader.get_gdf_edge(location_name)  # Use self.downloader
        map_graph = self.downloader.get_map_nodes(location_name)
        centroid = map_edge.unary_union.centroid
        print(centroid)
        
        folium_map = folium.Map(location=[centroid.y, centroid.x], zoom_start = 15)

        feature_group_hospital = folium.FeatureGroup(name="Rumah Sakit", show=True).add_to(folium_map)
        feature_group_ambulance = folium.FeatureGroup(name="ambulans", show=True).add_to(folium_map)
        feature_group_caller = folium.FeatureGroup(name="pasien", show=True).add_to(folium_map)
        for hospital in self.hospital_nodes:
            folium.Marker(
                location=[hospital.get_latitude(), hospital.get_longitude()], 
                popup=Popup(hospital.get_hospital_name()), 
                tooltip=hospital.get_address(),
                icon=Icon(color="red", icon_color="black", icon="hospital", prefix="fa")
                ).add_to(feature_group_hospital)
            for ambulance in hospital.get_ambulance_agents():
                amb_lat = map_graph.nodes[ambulance.get_current_location_node()]["y"] + random.uniform(-0.0003, 0.0003) 
                amb_lon = map_graph.nodes[ambulance.get_current_location_node()]["x"] + random.uniform(-0.0003, 0.0003)
                folium.Marker(
                    location=[amb_lat, amb_lon], 
                    popup=Popup(ambulance.get_ambulance_id()), 
                    tooltip=hospital.get_hospital_name(),
                    icon=Icon(color="blue",icon_color="black", icon="truck-medical", prefix="fa")
                    ).add_to(feature_group_ambulance)
                
        for caller in self.caller_nodes:
            folium.Marker(
                location=[caller.get_latitude(), caller.get_longitude()], 
                popup=Popup(caller.get_caller_id()), 
                tooltip=caller.get_severity_level(),
                icon=Icon(color=caller.get_severity_color(), icon="person-falling-burst", prefix="fa")
                ).add_to(feature_group_caller)
            
        folium.LayerControl().add_to(folium_map)        

        self.__create_polygon_to_graph_edges_traffic(self.map_graph_with_traffic, folium_map)

        # Create timeline and timeslider using simulation records
        
         # Fix: Create proper timeline data structure
        timeline_features = []
        
        for record in simulation_records:
            timestamp = record['time']
            time_str = (datetime(2000, 1, 1) + pd.to_timedelta(timestamp, unit='s')).isoformat()
            
            # Only add features if positions exist and are valid
            if record['positions']:
                for amb_key, amb_val in record['positions'].items():
                    # Validate coordinates
                    if 'lat' in amb_val and 'lon' in amb_val:
                        lat, lon = amb_val['lat'], amb_val['lon']
                        
                        # Check if coordinates are valid numbers
                        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                            fill_color = 'blue'
                            if amb_val['status']:
                                fill_color = 'red'

                            feature = {
                                'type': 'Feature',
                                'geometry': {
                                    'type': 'Point',
                                    'coordinates': [lon, lat]  # GeoJSON: [longitude, latitude]
                                },
                                'properties': {
                                    'time': time_str,
                                    'popup': f"Ambulance {amb_val['hospital_id']}",
                                    'icon': 'circle',
                                    'iconstyle': {
                                        'fillColor': fill_color,
                                        'color': 'black',
                                        'radius': 6,
                                        'fillOpacity': 0.8
                                    }
                                }
                            }
                            timeline_features.append(feature)

        # Create GeoJSON FeatureCollection
        if timeline_features:  # Only create timeline if we have valid features
            geojson_data = {
            'type': 'FeatureCollection',
            'features': timeline_features
            }

            TimestampedGeoJson(
                geojson_data,
                period='PT1S',  # 2 second intervals matching your time_step_s
                duration='PT10M',
                auto_play=False,
                loop=False,
                max_speed=5,
                loop_button=True,
                date_options='YYYY-MM-DD HH:mm:ss',
                time_slider_drag_update=True
            ).add_to(folium_map)
        return folium_map
    
    def __create_polygon_to_graph_edges_traffic(self, map_graph_with_traffic, folium_map):
        for u, v, k, data in map_graph_with_traffic.edges(keys=True, data=True):
            if 'traffic_factor' in data:
                start_lat = map_graph_with_traffic.nodes[u]['y']
                start_lon = map_graph_with_traffic.nodes[u]['x']
                end_lat = map_graph_with_traffic.nodes[v]['y']
                end_lon = map_graph_with_traffic.nodes[v]['x']
                
                folium.PolyLine(
                    locations=[(start_lat, start_lon), (end_lat, end_lon)],
                    color=self.__get_color_based_on_traffic(data['traffic_factor']),
                    weight=2,
                    opacity=0.7
                ).add_to(folium_map)
        pass
    
    def __get_color_based_on_traffic(self, traffic_factor):
        if traffic_factor < 0.5:
            return 'green'  # Light traffic
        elif 0.5 <= traffic_factor < 1.0:
            return 'orange'  # Moderate traffic
        else:
            return 'red'  # Heavy traffic

    def __calculate_mean_ambulance_response_time(self, simulation_records):
        total_response_time = 0
        response_count = 0

        for record in simulation_records:
            for ambulance_id, ambulance_value in record['positions'].items():
                if 'response_time' in ambulance_value:
                    total_response_time += ambulance_value['response_time']
                    response_count += 1

        if response_count == 0:
            return 0  # Avoid division by zero

        mean_response_time = total_response_time / response_count
        return mean_response_time
    
    def __save_bar_plot_mean_response_time(self, plot_path, mean_response_time_djikstra, mean_response_time_astar, mean_response_time_ox, mean_response_time_ga):
        matplotlib.use("Agg")
        algos = ["Dijkstra", "A*", "OSMnx", "Genetic Algorithm"]
        means = [
            mean_response_time_djikstra,
            mean_response_time_astar,
            mean_response_time_ox,
            mean_response_time_ga,
        ]

        fig, ax = plt.subplots(figsize=(4, 2))
        bars = ax.bar(range(len(algos)), means, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

        ax.set_title(f"Mean Response Time by Algorithm")
        ax.set_ylabel("Time (minutes)")
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

        for bar, val in zip(bars, means):
            ax.annotate(f"{val:.1f}",
                        (bar.get_x() + bar.get_width() / 2, val),
                        textcoords="offset points",
                        xytext=(0, 3),
                        ha="center",
                        fontsize=5)

        plt.tight_layout()

        
        plt.savefig(str(plot_path), dpi=150)
        plt.close(fig)

    def __combine_maps_and_graph(self, location_name, map_ox_path, map_djikstra_path, map_astar_path, map_ga_path, plot_path):
        out_dir = os.path.abspath(f"simulation_{location_name}")
        os.makedirs(out_dir, exist_ok=True)

        wrapper_path = os.path.join(out_dir, f"subplot_{location_name}.html")
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <title>Ambulance Simulation - {location_name}</title>
            <style>
            body {{ margin: 0; font-family: Arial, sans-serif; }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                grid-auto-rows: 50vh;
                gap: 8px;
                padding: 8px;
                box-sizing: border-box;
            }}
            .cell {{ position: relative; border: 1px solid #ddd; }}
            .cell h3 {{
                position: absolute;
                top: 6px;
                left: 8px;
                margin: 0;
                z-index: 2;
                background: rgba(255, 255, 255, 0.85);
                padding: 2px 6px;
                font-size: 14px;
                border-radius: 4px;
            }}
            .cell iframe {{ width: 100%; height: 100%; border: 0; }}
            @media (max-width: 1200px) {{
                .grid {{ grid-template-columns: 1fr; grid-auto-rows: 60vh; }}
            }}
            </style>
            </head>
            <body>
            <div class="grid">
                <div class="cell">
                <h3>using osm algorithm</h3>
                <iframe src="{os.path.basename(map_ox_path)}"></iframe>
                </div>
                <div class="cell">
                <h3>Djikstra Algorithm</h3>
                <iframe src="{os.path.basename(map_djikstra_path)}"></iframe>
                </div>
                <div class="cell">
                <h3>A* Algorithm</h3>
                <iframe src="{os.path.basename(map_astar_path)}"></iframe>
                </div>
                <div class="cell">
                <h3>Genetics Algorithm</h3>
                <iframe src="{os.path.basename(map_ga_path)}"></iframe>
                </div>
                <div class="cell">
                <h3>Comparison of Mean response Time</h3>
                <iframe src="{os.path.basename(plot_path)}"></iframe>
                </div>
            </div>
            </body>
            </html>""")

        webbrowser.open(f"file://{wrapper_path}")
        pass
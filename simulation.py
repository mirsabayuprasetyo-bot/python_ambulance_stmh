import random
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
import ga_routing


class simulation():
    def __init__(self):
        self.map_nodes = {}
        self.downloader = map.map_downloader()
        self.caller_nodes = []
        self.hospital_nodes = []
        self.map_graph_with_traffic = {}
        self.simulation_records = []
        pass
    def run_simulation(self, location_name):
        self.__define_hospital_and_ambulance_agents(location_name)
        self.__define_caller_agents(location_name,5)
        self.__define_traffic_condition(location_name)
        self.__simulate_ambulance_movement(location_name, simulation_time_in_minute=5)
        return self.__visualize_simulation(location_name)
        
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
                map_graph[u][v][k]['traffic_factor'] = traffic_factor
                map_graph[u][v][k]['time_s'] = map_graph[u][v][k]['length'] / 50  # assuming default speed 50 m/s
                map_graph[u][v][k]['maxspeed'] = 50  # assuming default speed 50 m/s

        self.map_graph_with_traffic = map_graph
        
        pass

    def __simulate_ambulance_movement(self, location_name, simulation_time_in_minute=60):
        map_graph = self.map_graph_with_traffic
        time_step_s = 1  # seconds
        max_simulation_duration_s = simulation_time_in_minute * 60  # convert minutes to seconds
        simulation_elapsed_time = 0
        
        simulation_records = []
        
        # Initial snapshot
        current_positions_snapshot = {}
        for hospital in self.hospital_nodes:
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

        # Simulation loop
        while simulation_elapsed_time < max_simulation_duration_s:
            simulation_elapsed_time += time_step_s
            
            for caller in self.caller_nodes:
                if caller.is_responded():
                    continue
                
                # Find nearest available ambulance
                nearest_ambulance = None
                min_distance = float('inf')
            
                for hospital in self.hospital_nodes:
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
            for hospital in self.hospital_nodes:
                for ambulance in hospital.get_ambulance_agents():
                    current_node = ambulance.get_current_location_node()
                    
                    if not ambulance.is_available():
                        destination = ambulance.get_destination_node()

                        path = ox.shortest_path(map_graph, current_node, destination, weight='time_s')
                        # path = ga_routing.ga_shortest_path(map_graph, current_node, destination, weight='time_s', population_size=10,generations=10)
                        if path is None:
                            continue
                        if len(path) > 1 :
                            ambulance.set_current_location_node(path[1])
                            current_node = path[1]
                        else : 
                            ambulance.set_current_location_node(destination)
                            current_node = destination

                        if current_node == destination:
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
                        # remaining_time = time_step_s

                        # for i in range(len(path) - 1):
                        #     u, v = path[i], path[i + 1]
                        #     # edge_data = map_graph.get_edge_data(u, v)

                        #     # if edge_data:
                        #     #     edge = edge_data[0]
                        #     #     travel_time = edge.get('time_s')
                        #     #     traffic_factor = edge.get('traffic_factor', 1.0)
                        #     #     adjusted_time = travel_time
                                        
                            
                                
                                # Check if reached destination
            
            
            print("simulation_elapsed_time:", simulation_elapsed_time)
        
        self.simulation_records = simulation_records

    def __visualize_simulation(self, location_name):
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
                amb_lat = map_graph.nodes[ambulance.get_current_location_node()]["y"] 
                amb_lon = map_graph.nodes[ambulance.get_current_location_node()]["x"]
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

        # Create timeline and timeslider using simulation records
        
         # Fix: Create proper timeline data structure
        timeline_features = []
        
        for record in self.simulation_records:
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
        
    def draw_polyline_to_map(self, map_graph, ambulance_agent, caller_agent):
        start_node = ambulance_agent.get_current_location_node()
        end_node = caller_agent.get_node()
        path = ox.shortest_path(map_graph, start_node, end_node, weight='length')
        locations = []
        for node in path:
            node_lat = map_graph.nodes[node]["y"]
            node_lon = map_graph.nodes[node]["x"]
            locations.append((node_lat, node_lon))
        return locations
   
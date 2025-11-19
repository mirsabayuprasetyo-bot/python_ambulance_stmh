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
import ga_router as genetics
# removed djikstra import (no longer used)
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
        print(f"[SIM] Starting simulation for: {location_name}")
        print(f"[SIM] Hospitals loaded: {len(self.hospital_nodes)}")
        print(f"[SIM] Callers loaded: {len(self.caller_nodes)}")

        # NOTE: callers now may request more than 1 ambulance (demand 1..2)
        self.__define_hospital_and_ambulance_agents(location_name)
        self.__define_caller_agents(location_name, max_caller=5, max_demand=2)
        self.__define_traffic_condition(location_name)

        out_dir = Path.cwd() / f"simulation_{location_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        map_ga_path = out_dir / f"map_ga_{location_name}.html"
        plot_path = out_dir / f"mean_response_time_{location_name}.png"
        
        # Only run GA + A* routing (single backend)
        simulation_records_ga = self.__simulate_ambulance_movement(location_name, simulation_time_in_minute=10, algorithm="ga")

        folium_map_ga = self.__visualize_simulation(location_name, simulation_records_ga)

        folium_map_ga.save(map_ga_path)

        mean_response_time_ga = self.__calculate_mean_ambulance_response_time(simulation_records_ga)

        # Save simple bar showing single algorithm (GA) mean response time
        self.__save_single_bar_plot_mean_response_time(plot_path, mean_response_time_ga)

        # Combine maps/plot (only GA map + plot)
        self.__combine_maps_and_graph(location_name, map_ga_path, plot_path)

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
                # setup_ambulance signature unchanged
                ambulance_node.setup_ambulance(ambulance_id, hospital_id, nearest_node)
                hospital_node.add_ambulance(ambulance_node)

            hospital_nodes.append(hospital_node)
        
        self.hospital_nodes = hospital_nodes
        pass

    def __define_caller_agents(self, location_name, max_caller, max_demand=2):
        """
        Define callers. Each caller is assigned:
         - node (graph node)
         - severity level (low/medium/high)
         - dynamic attributes:
            * demand (1..max_demand) : how many ambulances requested
            * assigned_ambulances : list of ambulance ids assigned
            * remaining_demand : how many still needed
        """
        map_graph = self.downloader.get_map_nodes(location_name)
        all_nodes = list(map_graph.nodes())
        total_caller = max_caller
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
            # dynamic demand: 1 .. max_demand ambulances per caller
            demand = random.randint(1, max(1, int(max_demand)))
            # attach extra attributes to caller instance (safe in Python)
            caller_inst.demand = demand
            caller_inst.assigned_ambulances = []  # store ambulance ids assigned
            caller_inst.remaining_demand = demand
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
                # preserve length; compute time based on assumed default speed 50 (m/s)
                map_graph[u][v][k]['maxspeed'] = 50  # assuming default speed 50 m/s
                map_graph[u][v][k]['traffic_factor'] = traffic_factor
                map_graph[u][v][k]['time_s'] = map_graph[u][v][k]['length'] / 50  # base time in seconds
                map_graph[u][v][k]['time_per_edge'] = map_graph[u][v][k]['time_s'] * traffic_factor 

        self.map_graph_with_traffic = map_graph
        
        pass

    def __simulate_ambulance_movement(self, location_name, simulation_time_in_minute=60, algorithm="ga"):
        """
        Main simulation:
        - Build deep copies of graph, callers, hospitals
        - Dispatch phase: use greedy assignment BUT respect caller.demand (assign nearest available ambulances until demand satisfied).
          Then compute GA/A*-based path for each dispatched ambulance from origin -> caller and caller -> hospital.
        - Movement phase: step through paths using time_per_edge computed on edges (traffic-aware).
        """
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
        
        # ---------- Dispatch logic ----------
        # Use priority (severity) order; high severity first.
        # For each caller, assign up to caller.demand nearest available ambulances.
        caller_shortlist = sorted(caller_nodes, key=lambda x: x.get_severity_number(), reverse=True)

        # Build a flattened list of all ambulances (hospital reference kept)
        all_ambulances = []
        for hospital in hospital_nodes:
            for ambulance in hospital.get_ambulance_agents():
                all_ambulances.append((hospital, ambulance))

        for caller in caller_shortlist:
            # for each caller, try to assign ambulances until demand met or no available ambulances remain
            while getattr(caller, "remaining_demand", 0) > 0:
                nearest_ambulance = None
                min_distance = float('inf')
                # find nearest available ambulance by travel-time-aware path length using A* (fast) or GA path length fallback
                for hospital, ambulance in all_ambulances:
                    if ambulance.is_available():
                        try:
                            # instantiate a router bound to the current map graph
                            router = astar.AStarRouter(map_graph, weight='time_per_edge')
                            path_est = router.shortest_path(ambulance.get_current_location_node(), caller.get_node())
                            # path_est may be None if no path; handle that
                            if path_est is None:
                                dist_est = float('inf')
                            else:
                                # compute time cost along path using time_per_edge
                                dist_est = 0.0
                                for u, v in zip(path_est[:-1], path_est[1:]):
                                    edge_data = map_graph.get_edge_data(u, v)
                                    if isinstance(edge_data, dict):
                                        first_key = next(iter(edge_data))
                                        dist_est += edge_data[first_key].get('time_per_edge', edge_data[first_key].get('length', 0))
                                    else:
                                        dist_est += edge_data.get('time_per_edge', edge_data.get('length', 0))
                            # use time_per_edge sum as distance metric (better than hops)
                            dist_est = 0.0
                            for u, v in zip(path_est[:-1], path_est[1:]):
                                # pick first edge if multigraph
                                edge_data = map_graph.get_edge_data(u, v)
                                if isinstance(edge_data, dict):
                                    first_key = next(iter(edge_data))
                                    dist_est += edge_data[first_key].get('time_per_edge', 0)
                                else:
                                    dist_est += edge_data.get('time_per_edge', 0)
                        except Exception:
                            # fallback to simple networkx shortest path by hops
                            try:
                                path_est = ox.shortest_path(map_graph, ambulance.get_current_location_node(), caller.get_node(), weight='length')
                                dist_est = len(path_est)
                            except Exception:
                                dist_est = float('inf')

                        if dist_est < min_distance:
                            min_distance = dist_est
                            nearest_ambulance = (hospital, ambulance)

                if nearest_ambulance is None:
                    # no available ambulances left
                    break

                hospital_assigned, amb_assigned = nearest_ambulance
                # assign this ambulance
                amb_assigned.set_available(False)
                amb_assigned.set_destination_node(caller.get_node())
                # record assignment on caller
                caller.assigned_ambulances.append(amb_assigned.get_ambulance_id())
                caller.remaining_demand -= 1
                if caller.remaining_demand <= 0:
                    caller.set_responded(True)

        # ---------- Path generation: use GA + A* for each dispatched ambulance ----------
        for hospital in hospital_nodes:
            for ambulance in hospital.get_ambulance_agents():
                if not ambulance.is_available():
                    origin_node = ambulance.get_origin_node()
                    destination_node = ambulance.get_destination_node()
                    # We use GA-based routing (genetics.ga_shortest_path) which will call your astar router for segments
                    path_to_caller = self.__generate_path_from_node(map_graph, origin_node, destination_node, algorithm)
                    path_to_hospital = self.__generate_path_from_node(map_graph, destination_node, origin_node, algorithm)

                    if path_to_caller is not None:
                        ambulance.set_path_to_caller(path_to_caller)
                    if path_to_hospital is not None:
                        ambulance.set_path_to_hospital(path_to_hospital)

        # ---------- Simulation loop: move ambulances along assigned paths ----------
        while simulation_elapsed_time < max_simulation_duration_s:
            simulation_elapsed_time += time_step_s

            # Move all responding ambulances to caller
            current_positions_snapshot = {}
            for hospital in hospital_nodes:
                for ambulance in hospital.get_ambulance_agents():
                    current_node = ambulance.get_current_location_node()
                    
                    if not ambulance.is_available():
                        destination = ambulance.get_destination_node()
                        path = None
                        if  not ambulance.is_returned():
                            path = self.__remaining_node_from_path(ambulance.get_path_to_caller(), current_node) 
                        else :
                            path = self.__remaining_node_from_path(ambulance.get_path_to_hospital(), current_node)

                        if path is None:
                            # nothing to do (maybe path not computed); skip
                            current_positions_snapshot[ambulance.get_ambulance_id()] = {
                                'lat': map_graph.nodes[current_node]['y'],
                                'lon': map_graph.nodes[current_node]['x'],
                                'hospital_id': ambulance.get_ambulance_id(),
                                'status': ambulance.is_returned()
                            }
                            continue
                        if len(path) > 1 :
                            # compute travel time along single edge (u->v)
                            ambulance_time = self.__get_time_from_node(path[0], path[1], map_graph)
                            ambulance.add_time(ambulance_time)
                            # move ambulance to next node
                            ambulance.set_current_location_node(path[1])
                            current_node = path[1]
                        else : 
                            ambulance.set_current_location_node(destination)
                            current_node = destination
                        
                        # If ambulance reached the destination node
                        if current_node == destination:
                            # If arrived back at origin (hospital) and was not available (meaning trip finished)
                            if current_node == ambulance.get_origin_node() and not ambulance.is_available():
                                current_positions_snapshot[ambulance.get_ambulance_id()] = {
                                    'lat': map_graph.nodes[current_node]['y'],
                                    'lon': map_graph.nodes[current_node]['x'],
                                    'hospital_id': ambulance.get_ambulance_id(),
                                    'status': ambulance.is_returned(),
                                    'response_time': ambulance.get_total_time()/2
                                    }
                                simulation_records.append({'time': simulation_elapsed_time, 'positions': current_positions_snapshot.copy()})
                                ambulance.set_available(True)

                            # set return-to-hospital behavior after servicing caller
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
            print(f"simulation_elapsed_time (GA+A*) : {simulation_elapsed_time}")

            # Stop early if all ambulances are available again (i.e., all trips finished)
            all_idle = True
            for hospital in hospital_nodes:
                for ambulance in hospital.get_ambulance_agents():
                    if not ambulance.is_available():
                        all_idle = False
                        break
                if not all_idle:
                    break
            if all_idle:
                # final snapshot at end time
                simulation_records.append({'time': simulation_elapsed_time, 'positions': current_positions_snapshot.copy()})
                break

        return simulation_records
    
    def __generate_path_from_node(self, map_graph, origin_node, destination_node, algorithm="ga"):
        """
        Generate a full node list path from origin_node to destination_node.
        Only GA+A* is used; GA will call astar segments internally via ga_routing.
        Default GA parameters set small for speed; tune population_size / num_generations if needed.
        """
        path = None
        # Use GA-based routing which delegates to astar_routing segments
        if algorithm == "ga":
            # keep population small for simulation speed; tune as required
            path = genetics.ga_shortest_path(map_graph, origin_node, destination_node,
                                             weight='time_per_edge', population_size=12, num_generations=40, allow_revisit=True)
        else:
            # fallback single-segment A* router (should rarely be used)
            router = astar.AStarRouter(default_weight='time_per_edge', default_heuristic="manhattan")
            path = router.shortest_path(map_graph, origin_node, destination_node)
        return path
    
    def __get_time_from_node(self, u, v, map_graph):
        """
        Return time cost for edge u->v using 'time_per_edge' attribute.
        Works with MultiDiGraph by selecting the first parallel edge.
        """
        if map_graph.has_edge(u, v):
            edge_data = map_graph.get_edge_data(u, v)
            # If there are multiple edges, take the first one
            if isinstance(edge_data, dict):
                first_key = next(iter(edge_data))
                return edge_data[first_key].get('time_per_edge', 0)
            else:
                return edge_data.get('time_per_edge', 0)
        return 0
    
    def __remaining_node_from_path(self, path, current_node):
        if path is None:
            return None
        try:
            current_index = path.index(current_node)
            return path[current_index:]
        except ValueError:
            # If current node not in path, return full path (start from beginning)
            return path

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
                tooltip=f"{caller.get_severity_level()} (demand={getattr(caller,'demand',1)})",
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
                period='PT1S',  # 1-second intervals matching your time_step_s
                duration='PT20M',
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

        if not simulation_records:
            return 0
        for record in simulation_records:
            for ambulance_id, ambulance_value in record['positions'].items():
                if 'response_time' in ambulance_value:
                    total_response_time += ambulance_value['response_time']
                    response_count += 1

        if response_count == 0:
            return 0  # Avoid division by zero

        mean_response_time = total_response_time / response_count
        return mean_response_time
    
    def __save_single_bar_plot_mean_response_time(self, plot_path, mean_response_time_ga):
        matplotlib.use("Agg")
        algos = ["Genetic Algorithm (GA + A*)"]
        means = [mean_response_time_ga]

        fig, ax = plt.subplots(figsize=(4, 2))
        bars = ax.bar(range(len(algos)), means, color=["#d62728"])

        ax.set_title(f"Mean Response Time (GA + A*)")
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
                        fontsize=7)

        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        plt.close(fig)

    def __combine_maps_and_graph(self, location_name, map_ga_path, plot_path):
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
                grid-template-columns: repeat(2, 1fr);
                grid-auto-rows: 60vh;
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
                <h3>Genetic Algorithm + A*</h3>
                <iframe src="{os.path.basename(map_ga_path)}"></iframe>
                </div>
                <div class="cell">
                <h3>Comparison: Mean Response Time</h3>
                <iframe src="{os.path.basename(plot_path)}"></iframe>
                </div>
            </div>
            </body>
            </html>""")
        webbrowser.open(f"file://{wrapper_path}")
        pass

if __name__ == "__main__":
    print("[SIMULATION] Running simulation...")

    sim = simulation()
    sim.run_simulation("sukabumi")   # or the region name you already use

    print("[SIMULATION] Finished.")

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


class simulate_agent():
    def __init__(self):
        self.map_nodes = {}
        self.downloader = map.map_downloader()
        self.caller_nodes = []
        self.hospital_nodes = []
        self.map_graph_with_traffic = {}
        self.simulation_records = []
        pass
    def run_simulation(self):
        self.__define_hospital_and_ambulance_agents("sukabumi")
        self.__define_caller_agents("sukabumi",5)
        self.__define_traffic_condition("sukabumi")
        self.__simulate_ambulance_movement("sukabumi", simulation_time_in_minute=5)
        self.__visualize_simulation("sukabumi")
        return
        self.__define_ambulance_agents()
        self.__simulate_road_condition()
        self.__visualize_map()
        for idx in range(max_iteration):
            print("simulate road congestien")
            print("simulate ambulance driving")
        
        pass

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
       

        folium_map.save("simulation_"+location_name+".html")
        
        # nearest_hospital_nodes = []
        # for hospital in self.hospital_nodes :
        #     nearest_node = hospital.get_osmnx_node()
        #     nearest_hospital_nodes.append(nearest_node)

        # node_colors = []
        # for node in map_nodes.nodes():
        #     if node in nearest_hospital_nodes:
        #         node_colors.append('red')
        #     else:
        #         node_colors.append('blue')

        # node_sizes = []
        # for node in map_nodes.nodes():
        #     if node in nearest_hospital_nodes:
        #         node_sizes.append(100)
        #     else:
        #         node_sizes.append(10)

        # fig,ax = ox.plot_graph(
        #     map_nodes, 
        #     node_color=node_colors, 
        #     node_size=node_sizes, 
        #     node_zorder=1, 
        #     edge_color='gray', 
        #     edge_linewidth=0.5
        #     )
        
        # fig.show()



        # ox.plot_graph(map_nodes)
        pass

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

    def __simulate_road_condition(self):
        map_nodes = self.downloader.get_map_nodes("sukabumi")  # Use self.downloader

        pass

    def __visualize_map(self):
        map_nodes = self.downloader.get_map_nodes("sukabumi")  # Use self.downloader
        pass
    
    def define_ambulance_agents(self, hospitals, map_nodes):
        # 1. Set the number of ambulances
        num_ambulances = 10
        print(f"Number of ambulances to simulate: {num_ambulances}")

        # 2. Extract all node IDs from the loaded map_nodes
        all_graph_nodes = list(map_nodes.nodes())
        print(f"Total nodes in graph: {len(all_graph_nodes)}")

        # 3. For each hospital, find its nearest node in the graph G
        hospital_nodes = []
        for idx, row in hospitals.iterrows():
            hospital_lat = row['lat']
            hospital_lon = row['lon']
            # osmnx.nearest_nodes expects (lat, lon)
            nearest_node = ox.nearest_nodes(map_nodes, hospital_lon, hospital_lat)
            hospital_nodes.append(nearest_node)

        # Remove duplicates from hospital_nodes if any hospital maps to the same node
        hospital_nodes = list(set(hospital_nodes))
        print(f"Total unique hospital-associated nodes: {len(hospital_nodes)}")

        # Ensure we have enough nodes for the simulation
        if num_ambulances > len(all_graph_nodes):
            print(f"Warning: num_ambulances ({num_ambulances}) is greater than available graph nodes ({len(all_graph_nodes)}). Adjusting num_ambulances.")
            num_ambulances = len(all_graph_nodes)

        if num_ambulances > len(hospital_nodes):
            print(f"Warning: num_ambulances ({num_ambulances}) is greater than available hospital nodes ({len(hospital_nodes)}). Adjusting num_ambulances.")
            num_ambulances = len(hospital_nodes)

        # 4. Randomly select num_ambulances unique starting node IDs
        starting_nodes = random.sample(all_graph_nodes, num_ambulances)

        # 5. Randomly select num_ambulances unique destination node IDs (from hospital-associated nodes)
        destination_nodes = random.sample(hospital_nodes, num_ambulances)

        # 6. Create ambulance_agents list
        ambulance_agents = []
        for i in range(num_ambulances):
            agent = {
                'id': i,
                'current_node': starting_nodes[i],
                'destination_node': destination_nodes[i],
                'path': None, # To be calculated later
                'current_path_index': 0, # To track progress along the path
                'traveled_time': 0.0 # To accumulate travel time
            }
            ambulance_agents.append(agent)

        print(f"\nGenerated {len(ambulance_agents)} ambulance agents:")
        for agent in ambulance_agents:
            print(f"  Agent {agent['id']}: Start Node={agent['current_node']}, Dest Node={agent['destination_node']}")

        print('>> Calculating shortest paths for each ambulance agent...')
        for agent in ambulance_agents:
            start_node = agent['current_node']
            end_node = agent['destination_node']

            try:
                # Calculate shortest path using 'time_s' as the weight
                path = ox.shortest_path(map_nodes, start_node, end_node, weight='length')
                agent['path'] = path
                print(f"  Agent {agent['id']}: Path calculated (length={len(path) if path else 0} nodes)")
            except Exception as e:
                agent['path'] = None
                print(f"  Warning: Could not find path for Agent {agent['id']} from {start_node} to {end_node}. Error: {e}")

        print('\n[OK] Shortest paths calculated for all agents.')
        print('Example agent after path calculation:')
        print(ambulance_agents[0])

        print('>> Initializing simulation ...')

        # Augment ambulance_agents with simulation-specific fields
        initial_ambulance_agents = []
        for agent in ambulance_agents:
            new_agent = agent.copy()
            if new_agent['path'] is not None and len(new_agent['path']) > 0:
                new_agent['current_node_idx'] = 0
                new_agent['current_node_osmnx'] = new_agent['path'][0] # Actual OSMnx node ID where agent starts
                new_agent['remaining_time_on_edge'] = 0.0 # Time left to traverse the *current* edge
                new_agent['arrived'] = False
                new_agent['traveled_time'] = 0.0 # Re-initialize total traveled time
            else:
                new_agent['current_node_idx'] = 0
                new_agent['current_node_osmnx'] = new_agent['current_node'] # Keep initial node as current
                new_agent['remaining_time_on_edge'] = 0.0
                new_agent['arrived'] = True # Mark as arrived immediately if no path found
                new_agent['traveled_time'] = float('inf') # Mark as impossible if no path
            initial_ambulance_agents.append(new_agent)

        # Replace the original list with the augmented one
        ambulance_agents = initial_ambulance_agents
        print("Ambulance agents augmented with simulation specific fields.")

        time_step_s = 10  # seconds
        max_simulation_duration_s = 3 * 60  # 3 hours in seconds
        simulation_time = 0  # Total elapsed simulation time

        completed_agents_data = [] # To store details of agents once they arrive
        simulation_records = [] # To store snapshots of agent positions for visualization

        # Initial snapshot for visualization
        current_agent_positions_snapshot = {}
        for agent in ambulance_agents:
            if not agent['arrived']:
                current_node_id = agent['current_node_osmnx']
                if current_node_id in map_nodes.nodes:
                    current_agent_positions_snapshot[agent['id']] = {
                        'lat': map_nodes.nodes[current_node_id]['y'],
                        'lon': map_nodes.nodes[current_node_id]['x']
                    }
                else:
                    print(f"Warning: Initial node {current_node_id} for agent {agent['id']} not found in graph.")
            else:
                # If agent is already marked arrived (e.g., no path), record its destination for snapshot
                dest_node_id = agent['destination_node']
                if dest_node_id in map_nodes.nodes:
                    current_agent_positions_snapshot[agent['id']] = {
                        'lat': map_nodes.nodes[dest_node_id]['y'],
                        'lon': map_nodes.nodes[dest_node_id]['x']
                    }
                else:
                    print(f"Warning: Destination node {dest_node_id} for agent {agent['id']} not found in graph.")
        simulation_records.append({'time': simulation_time, 'positions': current_agent_positions_snapshot.copy()})


        # Simulation loop
        while simulation_time < max_simulation_duration_s:
            simulation_time += time_step_s

            for agent in ambulance_agents:
                if agent['arrived']:
                    continue

                if agent['path'] is None or len(agent['path']) < 2: # No path or path is just one node (start = dest)
                    agent['arrived'] = True
                    agent['traveled_time'] = 0.0 if agent['path'] is not None and len(agent['path']) == 1 else float('inf')
                    completed_agents_data.append(agent.copy())
                    continue

                current_travel_time_budget = time_step_s

                while current_travel_time_budget > 0:
                    # Check if agent is already at destination
                    if agent['current_node_osmnx'] == agent['destination_node']:
                        agent['arrived'] = True
                        completed_agents_data.append(agent.copy())
                        break # Exit inner while loop, agent finished

                    # Check if agent has traversed all nodes in path (but not necessarily at destination_node yet)
                    if agent['current_node_idx'] >= len(agent['path']) - 1:
                        # This state means the agent is at the very last node of its path.
                        # If this last node is also the destination, it should have been caught by the check above.
                        # If it's not the destination, then path is incomplete.
                        if agent['current_node_osmnx'] != agent['destination_node']:
                            print(f"Warning: Agent {agent['id']} reached end of path but not destination {agent['destination_node']} vs {agent['current_node_osmnx']}. Marking as failed.")
                            agent['traveled_time'] = float('inf') # Mark as failed
                        agent['arrived'] = True
                        completed_agents_data.append(agent.copy())
                        break # Exit inner while loop

                    # Identify current edge
                    u = agent['current_node_osmnx']
                    v = agent['path'][agent['current_node_idx'] + 1]

                    # Get edge data. Handle potential missing edge or 'time_s'
                    edge_data = map_nodes.get_edge_data(u, v)
                    if not edge_data:
                        print(f"Warning: No edge data found between {u} and {v} for Agent {agent['id']}. Marking as failed.")
                        agent['arrived'] = True
                        agent['traveled_time'] = float('inf')
                        completed_agents_data.append(agent.copy())
                        break

                    # Take the first (or only) edge if multiple exist
                    edge_attributes = edge_data[0]
                    edge_time = edge_attributes.get('time_s')

                    if edge_time is None or edge_time <= 0: # Handle cases where time_s is missing or invalid
                        print(f"Warning: Edge {u}-{v} has invalid 'time_s' ({edge_time}) for Agent {agent['id']}. Marking as failed.")
                        agent['arrived'] = True
                        agent['traveled_time'] = float('inf')
                        completed_agents_data.append(agent.copy())
                        break

                    # Initialize remaining_time_on_edge if starting a new edge
                    if agent['remaining_time_on_edge'] <= 0: # Use <= 0 to handle potential floating point inaccuracies
                        agent['remaining_time_on_edge'] = edge_time

                    # Determine time to spend on current edge within this time_step
                    time_to_spend_on_current_edge = min(current_travel_time_budget, agent['remaining_time_on_edge'])

                    agent['traveled_time'] += time_to_spend_on_current_edge
                    agent['remaining_time_on_edge'] -= time_to_spend_on_current_edge
                    current_travel_time_budget -= time_to_spend_on_current_edge

                    # If current edge is completed
                    if agent['remaining_time_on_edge'] <= 0: # Use <= 0 to handle potential floating point inaccuracies
                        agent['current_node_idx'] += 1
                        agent['current_node_osmnx'] = v # Agent is now at node v
                        agent['remaining_time_on_edge'] = 0.0 # Reset for next edge

                        # Check again if destination is reached after moving to next node
                        if agent['current_node_osmnx'] == agent['destination_node']:
                            agent['arrived'] = True
                            completed_agents_data.append(agent.copy())
                            break # Agent finished
                    # Else, agent is still on the current edge, and current_travel_time_budget should be 0, so inner loop exits

            # Store current positions for all active agents for visualization (snapshot)
            current_agent_positions_snapshot = {}
            for agent in ambulance_agents:
                current_pos_node_id = None
                if agent['arrived']:
                    current_pos_node_id = agent['destination_node']
                elif agent['path'] is not None and agent['current_node_idx'] < len(agent['path']):
                    current_pos_node_id = agent['current_node_osmnx']

                if current_pos_node_id is not None and current_pos_node_id in map_nodes.nodes:
                    current_agent_positions_snapshot[agent['id']] = {
                        'lat': map_nodes.nodes[current_pos_node_id]['y'],
                        'lon': map_nodes.nodes[current_pos_node_id]['x']
                    }
                elif agent['id'] not in current_agent_positions_snapshot and not agent['arrived']:
                    print(f"Warning: Agent {agent['id']} is active but its position could not be determined for snapshot.")

            # Only add if there are active agents or this is a meaningful final state
            if current_agent_positions_snapshot or (simulation_records and simulation_records[-1]['time'] < simulation_time):
                simulation_records.append({'time': simulation_time, 'positions': current_agent_positions_snapshot.copy()})
                if not current_agent_positions_snapshot and not all(a['arrived'] for a in ambulance_agents): # If no active agents for plotting but some not arrived
                    break


        # If simulation ended due to max_simulation_duration_s, mark remaining as failed
        for agent in ambulance_agents:
            if not agent['arrived']:
                print(f"Agent {agent['id']} did not arrive within max simulation duration ({max_simulation_duration_s}s). Marking as failed.")
                agent['traveled_time'] = float('inf')
                agent['arrived'] = True # Mark as arrived/failed to ensure it's processed
                completed_agents_data.append(agent.copy()) # Add to completed_agents_data for summary


        # Simulation summary
        print(f"\nSimulation finished after {simulation_time} seconds.")
        print(f"Total agents: {len(ambulance_agents)}")
        successful_agents_count = len([a for a in completed_agents_data if a['traveled_time'] != float('inf')])
        print(f"Agents successfully arrived: {successful_agents_count} out of {len(ambulance_agents)}")

        total_successful_travel_time = 0
        for agent_data in completed_agents_data:
            if agent_data['traveled_time'] != float('inf'):
                total_successful_travel_time += agent_data['traveled_time']

        if successful_agents_count > 0:
            avg_travel_time = total_successful_travel_time / successful_agents_count
            print(f"Average travel time for successful agents: {avg_travel_time:.2f} seconds ({avg_travel_time / 60:.2f} minutes)")
        else:
            print("No agents completed their journey successfully.")

        # Output simulation_records so it's available in the kernel for subsequent steps.
        print("\nFirst few simulation records (snapshots for visualization):")
        for i, rec in enumerate(simulation_records[:min(3, len(simulation_records))]):
            print(f"Time {rec['time']}: {rec['positions']}")


        print('>> Visualizing simulation results on a Folium map...')

        # 1. Initialize Folium map centered on Yogyakarta
        center_yogya = [-7.7956, 110.3695]  # Approximate center of Yogyakarta
        folium_map = folium.Map(location=center_yogya, zoom_start=11, tiles='cartodbpositron')

        # 2. Add hospital markers
        marker_cluster = MarkerCluster(name="Hospitals").add_to(folium_map)
        # hospitals_df was loaded in a previous step
        for idx, row in hospitals.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=row.get('name', 'Hospital'),
                icon=folium.Icon(color='red', icon='plus-sign', prefix='fa')
            ).add_to(marker_cluster)

        print('[OK] Folium map initialized and hospital markers added.')

        # Prepare GeoJSON features for TimestampedGeoJson
        geojson_features = []

        # Define unique colors for each agent for better visualization
        agent_colors = {
            0: 'blue', 1: 'green', 2: 'purple', 3: 'orange', 4: 'darkred',
            5: 'cadetblue', 6: 'darkgreen', 7: 'darkblue', 8: 'lightred', 9: 'pink'
        }

        for record in simulation_records:
            timestamp = record['time']
            # Convert seconds to a human-readable datetime string, required by TimestampedGeoJson
            # The base datetime doesn't matter, only the relative time for the slider
            time_str = (datetime(2000, 1, 1) + pd.to_timedelta(timestamp, unit='s')).isoformat()

            for agent_id, pos_data in record['positions'].items():
                lat = pos_data['lat']
                lon = pos_data['lon']

                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lon, lat] # GeoJSON standard is [longitude, latitude]
                    },
                    'properties': {
                        'time': time_str,
                        'popup': f"Agent {agent_id} at {timestamp}s",
                        'color': agent_colors.get(agent_id, 'gray'), # Directly set color for Leaflet
                        'fillColor': agent_colors.get(agent_id, 'gray'), # Directly set fillColor for Leaflet
                        'radius': 6, # Directly set radius for Leaflet circle markers
                        'weight': 1,
                        'opacity': 0.8,
                        'icon' : 'car'
                    }
                }
                geojson_features.append(feature)

        # Create a FeatureCollection
        geojson_data = {
            'type': 'FeatureCollection',
            'features': geojson_features
        }

        # Add TimestampedGeoJson layer to the map
        time_slider_geojson = TimestampedGeoJson(
            geojson_data,
            period='PT10S', # Period for animation (e.g., 'PT10S' for 10 seconds steps, matching time_step_s)
            duration='PT10S', # Duration to display each point
            auto_play=False,
            loop=False,
            transition_time=200, # ms for transition animation
            add_last_point=True
        ).add_to(folium_map)

        # Add LayerControl to toggle layers (e.g., Hospitals, Ambulances)
        folium.LayerControl().add_to(folium_map)

        # Display the map
        print('[OK] TimestampedGeoJson layer added to map.')
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ambulance_simulation_map.html")
        folium_map.save(output_path)
        print(f"[OK] Map saved to {output_path}")
        pass
    # def 

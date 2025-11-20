import random
import json

class traffic_data_creator:
    def __init__(self, map_graph, map_edge,  path, interval=5):
        self.map_graph = map_graph
        self.map_edge = map_edge
        self.output_path = path
        self.interval = interval
        self.traffic_data = None

    def get_traffic_data(self):
        print("start create json traffic")
        map_edge = self.map_edge
        traffic_list = []
        for i in range(self.interval):
            # Fetch traffic data from downloader
            traffic_data_list = []
            for idx, row in map_edge.iterrows():
                u = row['u']
                v = row['v']
                k = row['key']

                # Generate random traffic condition (0.1 = light traffic, 1.5 = heavy congestion)
                traffic_factor = random.uniform(0.1, 1.5)
                traffic_data = {"edge_id":idx, "traffic_factor":traffic_factor}
                traffic_data_list.append(traffic_data)

            if traffic_data_list is not None:
                traffic_list.append({"interval":i, "traffic_data":traffic_data_list})
    
        with open(self.output_path, 'w') as f:
            print("create json file")
            json.dump(traffic_list, f, indent=4)
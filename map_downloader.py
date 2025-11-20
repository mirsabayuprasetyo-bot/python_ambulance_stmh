import os
import json
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from datetime import datetime
import re
import traffic_data_creator as td

# Map Downloader Class
# this class handles downloading map data and hospital POIs from OpenStreetMap
# after download, then it would save the data into local files for later use.
class map_downloader():
    def __init__(self):
        ox.settings.log_console = True
        ox.settings.use_cache = True
        self.DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ambulance_nav_data")
        os.makedirs(self.DIRECTORY, exist_ok=True)
        self.map_graph = {}

    def download_map(self, location):
        location_name = location[0].split(",")[0].replace(" ","_").lower()
        self.__download_and_save_to_file_polygon_from_location(location, location_name)
        map_polygon = gpd.read_file(os.path.join(self.DIRECTORY, location_name + "_polygon.gpkg"))
        self.__generate_graphml_and_save_to_file(map_polygon, location_name)
        graphml_path = os.path.join(self.DIRECTORY, "roads_"+location_name+".graphml")
        map_graph = ox.load_graphml(graphml_path)
        self.__generate_and_save_to_file_gdfs_nodes_and_edges_from_graphml(map_graph, location_name)
        self.__generate_hospital_data_from_polygon(map_polygon, location_name)
        return
        

      

      
        # =========================
        # 4) (Opsional cepat) Hitung bobot waktu tempuh sederhana per edge
        #    - Jika 'maxspeed' kosong, pakai default per highway class
        #    - Simpan kembali edges beratribut time_s
        # =========================
        print(">> Menetapkan kecepatan default & estimasi waktu tempuh edge ...")
        edges = gdf_edges.copy()
        edges["length_m"] = edges["length"]
        edges["speed_kmh"] = edges.apply(self.__estimate_speed(edges), axis=1)
        edges["speed_ms"] = edges["speed_kmh"] * 1000 / 3600.0
        edges["time_s"]   = edges["length_m"] / edges["speed_ms"]
        edges_out = os.path.join(self.DIRECTORY, "roads_diy_edges_with_time.gpkg")
        edges.to_file(edges_out, driver="GPKG")
        print(f"[OK] Edges + time_s ke GPKG: {edges_out}")

        # =========================
        # 5) Ringkasan
        # =========================
        print("\n=== RINGKASAN FILE OUTPUT ===")
        for p in [graphml_path, edges_gpkg, nodes_gpkg, edges_out, hosp_geojson, hosp_csv]:
            print(p)

        print("\nSelesai. Data siap untuk modul GA (kromosom=rute; fitness=total time_s).")
    
    def __download_and_save_to_file_polygon_from_location(self, location_list, location_name):
        print("download polygon from location {location_list}")
        multi_gdf = ox.geocode_to_gdf(location_list)  # setiap place → polygon
        combined_poly = multi_gdf.unary_union
        # gdf = gpd.GeoDataFrame([{"geometry": combined_poly}], crs="EPSG:4326")
        output_path = os.path.join(self.DIRECTORY, location_name +"_polygon.gpkg")
        multi_gdf.to_file(output_path, driver="GPKG")


    def __generate_graphml_and_save_to_file(self, map_polygon, location_name):
        print(">> Mengunduh jaringan jalan dalam bentuk GraphML ...")
        polygon_geom = {}
        if isinstance(map_polygon, gpd.GeoDataFrame):
            polygon_geom = map_polygon.geometry.unary_union
        elif isinstance(map_polygon, gpd.GeoSeries):
            polygon_geom = map_polygon.unary_union
        else:
            polygon_geom = map_polygon

        graph = ox.graph_from_polygon(polygon_geom, network_type="drive", simplify=True)

        # Simpan graf
        graphml_path = os.path.join(self.DIRECTORY, "roads_"+ location_name +".graphml")
        ox.save_graphml(graph, graphml_path)
        print(f"[OK] GraphML disimpan: {graphml_path}")
        pass

    def __generate_and_save_to_file_gdfs_nodes_and_edges_from_graphml(self, map_graph, map_name):
        print(">> Mengonversi graf ke GeoDataFrames nodes & edges ...")
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(map_graph)
        edges_gpkg = os.path.join(self.DIRECTORY, "roads_"+map_name+"_edges.gpkg")
        nodes_gpkg = os.path.join(self.DIRECTORY, "roads_"+map_name+"_nodes.gpkg")
        gdf_edges.to_file(edges_gpkg, driver="GPKG")
        gdf_nodes.to_file(nodes_gpkg, driver="GPKG")
        print(f"[OK] Edges ke GPKG: {edges_gpkg}")
        print(f"[OK] Nodes ke GPKG: {nodes_gpkg}")
        pass

    def __generate_hospital_data_from_polygon(self, map_polygon, map_name):
        print(">> Mengambil POI Rumah Sakit (amenity=hospital) ...")

        polygon_geom = {}
        if isinstance(map_polygon, gpd.GeoDataFrame):
            polygon_geom = map_polygon.geometry.unary_union
        elif isinstance(map_polygon, gpd.GeoSeries):
            polygon_geom = map_polygon.unary_union
        else:
            polygon_geom = map_polygon

        tags = {"amenity": "hospital"}
        hosp_gdf = ox.features_from_polygon(polygon_geom, tags)

        # Beberapa elemen adalah polygon/area; kita buat centroid untuk titik
        hosp_gdf = hosp_gdf.reset_index(drop=True)
        if "geometry" in hosp_gdf.columns:
            hosp_points = hosp_gdf.copy()
            # Pastikan semua geometri menjadi titik (untuk keperluan routing/nearest)
            hosp_points["geometry"] = hosp_points.geometry.centroid
        else:
            hosp_points = gpd.GeoDataFrame(columns=["name", "geometry"], geometry="geometry", crs="EPSG:4326")

        # Pilih kolom penting jika tersedia
        keep_cols = [c for c in ["name","addr:full","addr:city","addr:district","operator","phone","emergency","healthcare","amenity"] if c in hosp_points.columns]
        hosp_points = hosp_points[keep_cols + ["geometry"]] if keep_cols else hosp_points[["geometry"]]

        # Simpan ke GeoJSON & CSV ringkas (lon/lat)
        hosp_geojson = os.path.join(self.DIRECTORY, "hospitals_diy_ext.geojson")
        hosp_points.to_file(hosp_geojson, driver="GeoJSON")
        print(f"[OK] RS GeoJSON: {hosp_geojson}")

        # Buat CSV dengan koordinat
        hosp_pts_wgs = hosp_points.to_crs(epsg=4326).copy()
        hosp_pts_wgs["lon"] = hosp_pts_wgs.geometry.x
        hosp_pts_wgs["lat"] = hosp_pts_wgs.geometry.y
        hosp_csv = os.path.join(self.DIRECTORY, "hospitals_"+map_name+".csv")
        hosp_pts_wgs.drop(columns="geometry").to_csv(hosp_csv, index=False)
        print(f"[OK] RS CSV: {hosp_csv}")

    def __estimate_speed(self, row):
          # default kasar (km/h) per kelas jalan
        DEFAULT_SPEEDS = {
            "motorway": 80,
            "trunk": 70,
            "primary": 60,
            "secondary": 50,
            "tertiary": 40,
            "unclassified": 30,
            "residential": 30,
            "service": 20
        }
        # 1) pakai maxspeed jika ada
        max_speed = row.get("maxspeed")
        velocity = self.__
        
        # 2) pakai default per 'highway'
        high_way = row.get("highway")
        if isinstance(high_way, list):
            high_way = high_way[0]
        return DEFAULT_SPEEDS.get(high_way, 30.0)
    
    def __parse_maxspeed(sef, val):
        # menerima string/daftar, kembalikan km/h (float) jika mungkin
        if val is None:
            return None
        if isinstance(val, list):
            val = val[0]
        if isinstance(val, (int, float)):
            return float(val)
            # contoh format: "40", "50 km/h", "60;80", "maxspeed=50"
        m = re.findall(r"\d+\.?\d*", str(val))
        if not m:
            return None
        return float(m[0])

    def get_map_nodes(self, location_name):
        graphml_path = os.path.join(self.DIRECTORY, "roads_"+location_name+".graphml")
        self.map_nodes = ox.load_graphml(graphml_path)
        print(f"[OK] Graph loaded from: {graphml_path}")
        return self.map_nodes
    
    def get_hospital_location_dataframe(self, location_name):
        hosp_csv_path = os.path.join(self.DIRECTORY, "hospitals_"+location_name+".csv")
        self.hospital_dataframe = pd.read_csv(hosp_csv_path)
        print(f"[OK] Hospitals data loaded from: {hosp_csv_path}")
        return self.hospital_dataframe
    
    def get_gdf_edge(self, location_name):
        gdf_path = os.path.join(self.DIRECTORY, "roads_"+location_name+"_edges.gpkg")
        gdf_edge = gpd.read_file(gdf_path)
        return gdf_edge
    
    def get_caller(self):
        patient_json_path = os.path.join(self.DIRECTORY, "patient.json")
        with open(patient_json_path, "r") as f:
            patient_data = json.load(f)
        return patient_data
    
    def get_traffic(self):
        traffic_json_path = os.path.join(self.DIRECTORY, "traffic.json")
        with open(traffic_json_path, "r") as f:
            traffic_data = json.load(f)
        return traffic_data
    
    def create_traffic_data(self):
        map_graph = self.get_map_nodes("sukabumi")
        map_edge = self.get_gdf_edge("sukabumi")
        path = os.path.join(self.DIRECTORY,"traffic.json")
        traffic = td.traffic_data_creator(map_graph, map_edge, path, 120)
        traffic.get_traffic_data()
    

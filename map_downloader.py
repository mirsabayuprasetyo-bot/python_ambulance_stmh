import os
import json
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from datetime import datetime


class map_downloader():
    def __init__(self):
        ox.settings.log_console = True
        ox.settings.use_cache = True
        self.OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ambulance_nav_data")
        os.makedirs(self.OUT_DIR, exist_ok=True)

    def download_map(self, location):
        print(f"Downloading map for {location}")
        print(">> Mengambil boundary gabungan wilayah ...")
        multi_gdf = ox.geocode_to_gdf(location)  # setiap place → polygon
        # Gabungkan semua poligon:
        combined_poly = multi_gdf.unary_union
        print(">> Mengunduh jaringan jalan (mode=drive) ... (bisa beberapa menit)")
        graph = ox.graph_from_polygon(combined_poly, network_type="drive", simplify=True)

        # Simpan graf
        graphml_path = os.path.join(self.OUT_DIR, "roads_diy.graphml")
        ox.save_graphml(graph, graphml_path)
        print(f"[OK] GraphML disimpan: {graphml_path}")

        # Juga ekspor edges/nodes ke GeoPackage (siap GIS)
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
        edges_gpkg = os.path.join(self.OUT_DIR, "roads_diy_edges.gpkg")
        nodes_gpkg = os.path.join(self.OUT_DIR, "roads_diy_nodes.gpkg")
        gdf_edges.to_file(edges_gpkg, driver="GPKG")
        gdf_nodes.to_file(nodes_gpkg, driver="GPKG")
        print(f"[OK] Edges ke GPKG: {edges_gpkg}")
        print(f"[OK] Nodes ke GPKG: {nodes_gpkg}")

        print(">> Mengambil POI Rumah Sakit (amenity=hospital) ...")
        tags = {"amenity": "hospital"}
        hosp_gdf = ox.features_from_polygon(combined_poly, tags)

        # Beberapa elemen adalah polygon/area; kita buat centroid untuk titik
        hosp_gdf = hosp_gdf.reset_index(drop=True)
        if "geometry" in hosp_gdf.columns:
            hosp_points = hosp_gdf.copy()
            # Pastikan semua geometri menjadi titik (untuk keperluan routing/nearest)
            hosp_points["geometry"] = hosp_points.geometry.centroid
        else:
            # Jika tidak ada kolom geometry (sangat jarang), buat GeoDataFrame kosong
            hosp_points = gpd.GeoDataFrame(columns=["name", "geometry"], geometry="geometry", crs="EPSG:4326")

        # Pilih kolom penting jika tersedia
        keep_cols = [c for c in ["name","addr:full","addr:city","addr:district","operator","phone","emergency","healthcare","amenity"] if c in hosp_points.columns]
        hosp_points = hosp_points[keep_cols + ["geometry"]] if keep_cols else hosp_points[["geometry"]]

        # Simpan ke GeoJSON & CSV ringkas (lon/lat)
        hosp_geojson = os.path.join(self.OUT_DIR, "hospitals_diy_ext.geojson")
        hosp_points.to_file(hosp_geojson, driver="GeoJSON")
        print(f"[OK] RS GeoJSON: {hosp_geojson}")

        # Buat CSV dengan koordinat
        hosp_pts_wgs = hosp_points.to_crs(epsg=4326).copy()
        hosp_pts_wgs["lon"] = hosp_pts_wgs.geometry.x
        hosp_pts_wgs["lat"] = hosp_pts_wgs.geometry.y
        hosp_csv = os.path.join(self.OUT_DIR, "hospitals_diy_ext.csv")
        hosp_pts_wgs.drop(columns="geometry").to_csv(hosp_csv, index=False)
        print(f"[OK] RS CSV: {hosp_csv}")

        # =========================
        # 4) (Opsional cepat) Hitung bobot waktu tempuh sederhana per edge
        #    - Jika 'maxspeed' kosong, pakai default per highway class
        #    - Simpan kembali edges beratribut time_s
        # =========================
        print(">> Menetapkan kecepatan default & estimasi waktu tempuh edge ...")
        import re

        def parse_maxspeed(val):
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

        edges = gdf_edges.copy()
        edges["length_m"] = edges["length"]
        
        # estimasi kecepatan
        def estimate_speed(row):
            # 1) pakai maxspeed jika ada
            ms = row.get("maxspeed")
            v = parse_maxspeed(ms)
            if v:
                return v
            # 2) pakai default per 'highway'
            hw = row.get("highway")
            if isinstance(hw, list):
                hw = hw[0]
            return DEFAULT_SPEEDS.get(hw, 30.0)

        edges["speed_kmh"] = edges.apply(estimate_speed, axis=1)
        edges["speed_ms"] = edges["speed_kmh"] * 1000 / 3600.0
        edges["time_s"]   = edges["length_m"] / edges["speed_ms"]
        edges_out = os.path.join(OUT_DIR, "roads_diy_edges_with_time.gpkg")
        edges.to_file(edges_out, driver="GPKG")
        print(f"[OK] Edges + time_s ke GPKG: {edges_out}")

        # =========================
        # 5) Ringkasan
        # =========================
        print("\n=== RINGKASAN FILE OUTPUT ===")
        for p in [graphml_path, edges_gpkg, nodes_gpkg, edges_out, hosp_geojson, hosp_csv]:
            print(p)

        print("\nSelesai. Data siap untuk modul GA (kromosom=rute; fitness=total time_s).")
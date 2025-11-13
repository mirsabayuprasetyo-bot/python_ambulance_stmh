import map_downloader as map
import simulate_agent as sim

class main() : 
    def __init__(self) : 
        print("This is main class")
        self.map_downloader = map.map_downloader()
        self.simulate_agent = sim.simulate_agent()

    def run(self) : 
        print("Running main class")
        PLACES_POLY = [
            "Sukabumi, West Java, Java, Indonesia"   # Nusantara
        ]
        self.map_downloader.download_map(PLACES_POLY)
        return
        polygon = self.map_downloader.get_polygon()
        
        # Print key properties
        print(f"Polygon Type: {polygon.geom_type}")
        print(f"Is Valid: {polygon.is_valid}")
        print(f"Area: {polygon.area}")
        print(f"Perimeter: {polygon.length}")
        print(f"Centroid: {polygon.centroid}")
        print(f"Bounds: {polygon.bounds}")
        # print(f"Exterior Coordinates Count: {len(polygon.exterior.coords)}")
        print("=" * 50)
        # self.simulate_agent.run_simulation()


if __name__ == "__main__" :
    obj = main()
    obj.run()
    
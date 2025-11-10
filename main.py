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
            "Daerah Istimewa Yogyakarta, Indonesia",   # provinsi DIY
            "Kabupaten Sleman, DI Yogyakarta, Indonesia",
            "Kabupaten Bantul, DI Yogyakarta, Indonesia",
            "Kabupaten Kulon Progo, DI Yogyakarta, Indonesia",
            "Kabupaten Gunungkidul, DI Yogyakarta, Indonesia",
            "Kota Magelang, Jawa Tengah, Indonesia",
            "Kabupaten Magelang, Jawa Tengah, Indonesia",
            "Kabupaten Klaten, Jawa Tengah, Indonesia"
        ]
        # self.map_downloader.download_map(PLACES_POLY)
        self.simulate_agent.run_simulation()


if __name__ == "__main__" :
    obj = main()
    obj.run()
    
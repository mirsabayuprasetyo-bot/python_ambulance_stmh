import map_downloader as map
import simulation as sim


class main() : 
    def __init__(self) : 
        print("This is main class")
        self.map_downloader = map.map_downloader()
        self.simulation = sim.simulation()

    def run(self) : 
        self.simulation.run_simulation("sukabumi")
        return


if __name__ == "__main__" :
    obj = main()
    obj.run()
    
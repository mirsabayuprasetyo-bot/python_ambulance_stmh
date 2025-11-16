import map_downloader as map
import simulate_agent as sim

class main() : 
    def __init__(self) : 
        print("This is main class")
        self.map_downloader = map.map_downloader()
        self.simulate_agent = sim.simulate_agent()

    def run(self) : 
        self.simulate_agent.run_simulation()
        return


if __name__ == "__main__" :
    obj = main()
    obj.run()
    
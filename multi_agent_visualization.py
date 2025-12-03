

#this class used to visualize the simulation using the streamlit, 
# also making the visualization of the multiple run seamless and easy
class multi_agent_visualizer:
    def __init__(self, simulation=None):
        print("Multi-Agent Visualizer initialized")
        self.simulation = simulation

    def visualize(self):
        print("Visualizing multi-agent simulation")
        st.set_page_config(page_title="Multi-Agent Hospital System", layout="wide")
        st.title("Simulation of Ambulance Routing")

    def set_dependencies(self, simulation):
        self.simulation = simulation
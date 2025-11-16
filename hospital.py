class hospital:
    def __init__(self):
        self.name = ""
        self.hospital_id = ""
        self.address = ""
        self.type = ""
        self.hospital_class = ""
        self.lat = 0.0
        self.lon = 0.0
        self.osmx_node = {}
        self.ambulances = []
        pass

    def setup_hospital(self, hospital_id, hospital_name, address, type, hospital_class, lat, lon, osmnx_node):
        self.hospital_id = hospital_id
        self.name = hospital_name
        self.address = address
        self.type = type
        self.hospital_class = hospital_class
        self.lat = lat
        self.lon = lon
        self.osmx_node = osmnx_node
        pass

    def add_ambulance(self, ambulance_agent):
        self.ambulances.append(ambulance_agent)
        pass

    def get_ambulance_agents(self):
        return self.ambulances

    def get_hospital_name(self):
        return self.name

    def get_hospital_id(self):
        return self.hospital_id

    def get_latitude(self):
        return self.lat
    
    def get_longitude(self):
        return self.lon
    
    def get_address(self): 
        return self.address
    
    def get_type(self):
        return self.type

    def get_hospital_class(self):
        return self.hospital_class
    
    def get_osmnx_node(self):
        return self.osmx_node
    

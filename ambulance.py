class ambulance:
    def __init__(self):
        self.ambulance_id = ""
        self.origin_hospital_id = {}
        self.origin_node = {}
        self.destination_node = {}
        self.current_location_node = {}
        pass

    def setup_ambulance(self, ambulance_id, origin_hospital_id, origin_node):
        self.ambulance_id = ambulance_id
        self.origin_hospital_id = origin_hospital_id
        self.origin_node = origin_node
        self.current_location_node = self.origin_node
        pass

    def is_arrived(self):
        return self.current_location_node == self.destination_node

    def get_ambulance_id(self):
        return self.ambulance_id
    
    def get_hospital_id(self):
        return self.origin_hospital_id

    def get_current_location_node(self):
        return self.current_location_node
    
    def set_destination_node(self, destination_node):
        self.destination_node = destination_node
        pass

    def set_current_location_node(self, curret_node):
        self.current_location_node = curret_node


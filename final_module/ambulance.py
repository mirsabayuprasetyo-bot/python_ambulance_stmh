class ambulance:
    def __init__(self):
        self.ambulance_id = ""
        self.origin_hospital_id = {}
        self.origin_node = {}
        self.destination_node = {}
        self.current_location_node = {}
        self.available = True
        self.path_to_caller = None
        self.path_to_hospital = None
        pass

    def setup_ambulance(self, ambulance_id, origin_hospital_id, origin_node):
        self.ambulance_id = ambulance_id
        self.origin_hospital_id = origin_hospital_id
        self.origin_node = origin_node
        self.current_location_node = self.origin_node
        self.available = True
        self.isReturned = False
        self.total_time = 0
        pass

    def add_time(self, time):
        self.total_time += time

    def get_total_time(self):
        return self.total_time

    def get_path_to_caller(self):
        return self.path_to_caller
    
    def set_path_to_caller(self, path):
        self.path_to_caller = path

    def get_path_to_hospital(self):
        return self.path_to_hospital
    
    def set_path_to_hospital(self, path):
        self.path_to_hospital = path

    def get_origin_node(self):
        return self.origin_node
    
    def set_take_patient_to_hospital(self):
        self.isReturned = True

    def is_returned(self):
        return self.isReturned

    def get_ambulance_id(self):
        return self.ambulance_id
    
    def get_hospital_id(self):
        return self.origin_hospital_id

    def get_current_location_node(self):
        return self.current_location_node
    
    def get_destination_node(self):
        return self.destination_node
    
    def set_destination_node(self, destination_node):
        self.destination_node = destination_node
        pass

    def set_current_location_node(self, curret_node):
        self.current_location_node = curret_node

    def is_available(self):
        return self.available
    
    def set_available(self, availability_status):
        self.available = availability_status
        pass


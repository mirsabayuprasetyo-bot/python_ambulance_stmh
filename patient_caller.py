class patient_caller:
    def __init__(self, caller_id,  node_id, latitude, longitude, severity_level):
        self.caller_id = caller_id
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        self.severity_level = severity_level  # e.g., 'low', 'medium', 'high'
        self.responded = False

    def get_caller_id(self):
        return self.caller_id

    def get_latitude(self):
        return self.latitude
    
    def get_longitude(self):
        return self.longitude
    
    def get_node(self):
        return self.node_id
    
    def get_severity_level(self):
        return self.severity_level
    
    def get_severity_color(self):
        if self.get_severity_level() == "low":
            return "lightgreen"
        elif self.get_severity_level() == "medium":
            return "beige"
        else :
            return "lightred"
        
    def get_severity_number(self):
        if self.get_severity_level() == "low":
            return 1
        elif self.get_severity_level() == "medium":
            return 2
        else :
            return 3
        
    def is_responded(self):
        return self.responded
        
    def set_responded(self, responded_status):
        self.responded = responded_status


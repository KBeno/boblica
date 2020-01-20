import uuid

class Heating:
    def __init__(self, name: str, efficiency: float, energy_source: str):
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.efficiency = efficiency
        self.energy_source = energy_source


class Cooling:
    def __init__(self, name: str, efficiency: float, energy_source: str):
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.efficiency = efficiency
        self.energy_source = energy_source


class HVAC:
    def __init__(self, name: str, heating: Heating, cooling: Cooling):
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.Heating = heating
        self.Cooling = cooling
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


class Lighting:
    def __init__(self, name: str, inefficiency: float, energy_source: str):
        """

        :param name:
        :param inefficiency: a correction factor that the demand will be multiplied with
            e=1 - no correction, e=0 - infinitely efficient, e>1 - inefficient
        :param energy_source:
        """
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.inefficiency = inefficiency
        self.energy_source = energy_source


class HVAC:
    def __init__(self, name: str, heating: Heating, cooling: Cooling, lighting: Lighting):
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.Heating = heating
        self.Cooling = cooling
        self.Lighting = lighting

import uuid

class Heating:
    def __init__(self, name: str, efficiency: float, energy_source: str, aux_energy_source: str = None,
                 aux_energy_rate: float = 0, set_point: float = 20, number_of_units: float = 1):
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.efficiency = efficiency
        self.energy_source = energy_source
        self.aux_energy_source = aux_energy_source
        self.aux_energy_rate = aux_energy_rate
        self.set_point = set_point
        self.n_units = number_of_units

    @property
    def set_point(self):
        return self._set_point

    @set_point.setter
    def set_point(self, temperature):
        if temperature < -273:
            raise Exception('Temperature cannot be lower as -273 °C')
        else:
            self._set_point = temperature


class Cooling:
    def __init__(self, name: str, efficiency: float, energy_source: str, set_point: float = 26,
                 number_of_units: float = 1):
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.efficiency = efficiency
        self.energy_source = energy_source
        self.set_point = set_point
        self.n_units = number_of_units

    @property
    def set_point(self):
        return self._set_point

    @set_point.setter
    def set_point(self, temperature):
        if temperature < -273:
            raise Exception('Temperature cannot be lower as -273 °C')
        else:
            self._set_point = temperature


class NaturalVentilation:
    def __init__(self, name: str, ach: float = 0, summer_night_ach: float = 4, summer_night_duration: float = 7):
        self.Name = name
        self.ach = ach  # [1/h]
        self.summer_night_ach = summer_night_ach  # [1/h]
        self.summer_night_duration = summer_night_duration  # [h]


class Lighting:
    def __init__(self, name: str, inefficiency: float, energy_source: str, power_density: float = 1,
                 number_of_units: float = 1):
        """

        :param name:
        :param inefficiency: a correction factor that the demand will be multiplied with
            e=1 - no correction, e=0 - infinitely efficient, e>1 - inefficient
        :param power_density: in W/m2
        :param energy_source:
        """
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.inefficiency = inefficiency
        self.energy_source = energy_source
        self.power_density = power_density  # W/m2
        self.n_units = number_of_units


class HVAC:
    def __init__(self, name: str, heating: Heating, cooling: Cooling, nat_vent: NaturalVentilation, lighting: Lighting,
                 infiltration: float, internal_gain: float = 5, required_ach: float = 0.5):
        self.Name = name
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.Heating = heating
        self.Cooling = cooling
        self.NaturalVentilation = nat_vent
        self.Lighting = lighting
        self.internal_gain = internal_gain  # [W/m2]
        self.required_ach = required_ach  # [1/h]
        self.infiltration_ach = infiltration  # [1/h]
from pathlib import Path

from firepy.model.building import *
# from firepy.tools.database import SqlDB
import pandas as pd
from eppy.modeleditor import IDF


class JsonSerializer:

    def __init__(self):
        pass

    def dictify(self, obj):
        """
        Prepare Firepy object to be serialized into json

        :param obj: Any type of Firepy object
        :return: json serializable dictionary structure
        """

        # initiate the target dictionary, include object type to make deserialization easier
        obj_dict = {
            'ObjType': obj.__class__.__name__
        }

        # let's create the structure recursively, because we love short code
        for attr_name, attr_value in obj.__dict__.items():
            # we can serialize these as they are
            if isinstance(attr_value, (int, float, str)) or attr_value is None:
                obj_dict[attr_name] = attr_value

            # we need to serialize element-wise in a list
            elif isinstance(attr_value, list):
                obj_dict[attr_name] = [
                    self.dictify(o)
                    for o in attr_value
                ]

            # same for dictionaries
            elif isinstance(attr_value, dict):
                obj_dict[attr_name] = {
                    key: self.dictify(o)
                    for key, o in attr_value.items()
                }

            # we expect everything other to be another Firepy object
            else:
                obj_dict[attr_name] = self.dictify(attr_value)

        return obj_dict

    def dumps(self, obj):
        import json
        # simply create the json string from the dictified object
        return json.dumps(self.dictify(obj))

    def to_file(self, obj, file_path: str):
        """
        write out json formatted obj to the specified file_path

        :param obj: Firepy object to jsonify
        :param file_path: the file_path to write the object into
        :return: None
        """
        import json

        with open(file_path) as json_file:
            json.dump(self.dumps(obj), json_file)

    def loads(self, s):
        """
        Deserialize a json formatted string into Firepy object

        :param s: json formatted string
        :return: Firepy object
        """
        # TODO use these:
        # elif getattr(value, '__module__') in ['model.building', 'model.results']:
        #     building_module = importlib.import_module('model.building')
        #     instance = getattr(building_module, 'Building')
        #     print(type(instance()))


class IdfSerializer:

    def __init__(self, idd_path: Union[str, Path], idf: Union[IDF, Path, str] = None,
                 parse_lca_data: bool = False, life_cycle_data: Union[str, pd.DataFrame] = None,
                 db = None, matching_col: str = 'DbId', matching_property: str = 'DbId'):
        # set path to EnergyPlus idd file
        self.idd_path = str(idd_path)

        # parse the idf using eppy
        self.idf = idf

        self.library = ObjectLibrary(default_key='Name')
        # self.opaque_materials = {}  # use name as key
        # self.window_materials = {}
        # self.shading_materials = {}
        # self.shadings = {}
        # self.constructions = {}

        # weather to add further data to the model that is not included in the idf
        # if parse_lca_data is True, we need either:
        #   - LifeCycleData
        #   - DB
        # and optionally a column in the LifeCycleData or in the DB to match the materials based on
        self.parse_lca = parse_lca_data
        # how the model objects will be matched with the LifeCycle Data (column name in the life_cycle_data or DB)
        self.match_col = matching_col
        # what property of the model objects (e.g. materials) to use for the matching ('DbId' or 'Name')
        self.match_prop = matching_property

        # Life Cycle Data can be given:
        #   - directly by a DataFrame or a csv address
        #   - from a database if DbId-s are supplied within the model (database_connection) # TODO
        self.LifeCycleData = life_cycle_data  # DataFrame
        self._db = db  # SqlDB instance to connect to the database if no ready results are supplied # TODO

        # collect initial values from idf for objects where the list of properties is more detailed than in firepy
        self.idf_values = {
            'window_materials': {},
            'shade_materials': {},
            'blind_materials': {},
            'shadings': {},
            'constructions': {}
        }

    @property
    def idf(self) -> IDF:
        return self._idf

    @idf.setter
    def idf(self, idf_source: Union[IDF, Path, str, None]):
        """
        Set original idf model
        :param idf_source: either eppy IDF instance or file path to the idf or idf string
        :return:
        """
        if idf_source is not None:
            if isinstance(idf_source, IDF):
                self._idf = idf_source
                # self.idf_idd_info = self._idf.__class__.idd_info
                # self.idf_block = self._idf.__class__.block
            elif isinstance(idf_source, Path):
                self._idf = self.from_file(path=str(idf_source), idd_path=self.idd_path)
                # self.idf_idd_info = self._idf.__class__.idd_info
                # self.idf_block = self._idf.__class__.block
            elif isinstance(idf_source, str):
                self._idf = self.from_text(idf_string=idf_source, idd_path=self.idd_path)

    @property
    def LifeCycleData(self) -> pd.DataFrame:
        """
        pandas DataFrame with all the life cycle information
        Columns:
          - Name, DbId, Unit, openLCAname, openLCAid, ProductionId, TransportId, WasteTreatmentId,
            WasteTreatmentTransportId, LifeTime, CuttingWaste
        Rows:
          - Materials
        Index: DbId

        :return: DataFrame
        """

        return self._life_cycle_data

    @LifeCycleData.setter
    def LifeCycleData(self, source: Union[str, pd.DataFrame]):
        if isinstance(source, str):
            # Read from path
            self._life_cycle_data = pd.read_csv(source, index_col=self.match_col)
        elif isinstance(source, pd.DataFrame):
            source.set_index(self.match_col)
            self._life_cycle_data = source

    def new(self, idd_path: str):
        from eppy.modeleditor import IDF
        from io import StringIO
        IDF.setiddname(idd_path)
        self.idf = IDF(StringIO(""))
        return self

    def set_defaults(self):
        self.idf.newidfobject(
            key='VERSION',
            Version_Identifier=str(8.9)
        )

    def set_simulation_options(self, time_step: int = 1):
        self.idf.newidfobject(
            key='TIMESTEP',
            Number_of_Timesteps_per_Hour=time_step
        )

    def disable_sizing(self):
        # this does not work
        simulation_control = self.idf.idfobjects['SimulationControl'.upper()][0]

        simulation_control.Do_Zone_Sizing_Calculation = 'No'
        simulation_control.Do_System_Sizing_Calculation = 'No'
        simulation_control.Do_Plant_Sizing_Calculation = 'No'
        simulation_control.Run_Simulation_for_Sizing_Periods = 'No'
        # simulation_control.Run_Simulation_for_Weather_File_Run_Periods = 'Yes'

    def from_file(self, path: str, idd_path: str) -> IDF:
        from eppy.modeleditor import IDF
        IDF.setiddname(idd_path)
        with open(path, 'r') as idf_file:
            idf = IDF(idf_file)
        return idf

    def from_text(self, idf_string: str, idd_path: str) -> IDF:
        from eppy.modeleditor import IDF
        IDF.setiddname(idd_path)
        idf = IDF()
        idf.initreadtxt(idf_string)
        return idf

    def idf_opaque_material(self, material: OpaqueMaterial):
        self.idf.newidfobject(
            key='MATERIAL'.upper(),
            Name=material.Name,
            Roughness=material.Roughness,
            Thickness=material.Thickness,
            Conductivity=material.Conductivity,
            Density=material.Density,
            Specific_Heat=material.SpecificHeat,
            Thermal_Absorptance=material.ThermalAbsorptance,
            Solar_Absorptance=material.SolarAbsorptance,
            Visible_Absorptance=material.VisibleAbsorptance
        )

    def update_idf_opaque_material(self, material: OpaqueMaterial):

        idf_material = self.idf.getobject('MATERIAL', material.Name)

        idf_material.Roughness = material.Roughness
        idf_material.Thickness = material.Thickness
        idf_material.Conductivity = material.Conductivity
        idf_material.Density = material.Density
        idf_material.Specific_Heat = material.SpecificHeat
        idf_material.Thermal_Absorptance = material.ThermalAbsorptance
        idf_material.Solar_Absorptance = material.SolarAbsorptance
        idf_material.Visible_Absorptance = material.VisibleAbsorptance

    def fp_opaque_material(self, ep_material):  # MATERIAL
        if self.parse_lca:
            # TODO use matching (like in the lca module)
            if '#' in ep_material.Name:
                name, db_id = ep_material.Name.split('#')
                db_material = self._db.get(table='materials', keyword=db_id, by='Dataset-ID')
                if db_material.empty:
                    raise Exception('Cannot assign database material to {mat}'.format(mat=ep_material.Name))
            else:
                raise Exception('No database id indicated for {mat}'.format(mat=ep_material.Name))
            transport_scenario = db_material['Transport scenario']
            disposal_scenario = db_material['Disposal scenario']
            cutting_waste = db_material['cutting waste']
            life_time = db_material['Life Time']

        else:
            db_id = None
            transport_scenario = None
            disposal_scenario = None
            cutting_waste = None
            life_time = None

        material = OpaqueMaterial(
            name=ep_material.Name,
            db_id=db_id,
            thickness=ep_material.Thickness,
            conductivity=ep_material.Conductivity,
            density=ep_material.Density,
            specific_heat=ep_material.Specific_Heat,
            transport_scenario=transport_scenario,
            disposal_scenario=disposal_scenario,
            cutting_waste=cutting_waste,
            life_time=life_time,
            roughness=ep_material.Roughness,
            thermal_absorptance=ep_material.Thermal_Absorptance,
            solar_absorptance=ep_material.Solar_Absorptance,
            visible_absorptance=ep_material.Visible_Absorptance
        )

        # if ep_material.Name not in self.opaque_materials:
        #     self.opaque_materials[ep_material.Name] = material

        self.library.add(material)

        return material

    def idf_window_material(self, window_material: WindowMaterial):
        self.idf.newidfobject(
            key='WindowMaterial:SimpleGlazingSystem'.upper(),
            Name=window_material.Name,
            UFactor=window_material.UValue,
            Solar_Heat_Gain_Coefficient=window_material.gValue,
            Visible_Transmittance=self.idf_values['window_materials'][window_material.Name]['Visible_Transmittance']
        )

    def update_idf_window_material(self, window_material: WindowMaterial):

        idf_material = self.idf.getobject('WindowMaterial:SimpleGlazingSystem'.upper(), window_material.Name)

        idf_material.UFactor = window_material.UValue
        idf_material.Solar_Heat_Gain_Coefficient = window_material.gValue

    def fp_window_material(self, ep_simple_glazing):  # WindowMaterial:SimpleGlazingSystem
        if self.parse_lca:
            # TODO update this by using matching (like in the lca module)
            if '#' in ep_simple_glazing.Name:
                name, ids = ep_simple_glazing.Name.split('#')
                if '+' in ids:
                    # glazing+frame
                    glz_id, frm_id = ids.split('+')
                    db_material_frame = self._db.get(table='materials', keyword=frm_id, by='Dataset-ID')
                    db_material_glazing = self._db.get(table='materials', keyword=glz_id, by='Dataset-ID')
                    if db_material_frame.empty:
                        raise Exception('Cannot assign database material to frame in {mat}'.format(mat=ep_simple_glazing.Name))
                    if db_material_glazing.empty:
                        raise Exception('Cannot assign database material to glazing in {mat}'.format(
                            mat=db_material_glazing.Name))
                    db_id = None
                    transport_scenario_glazing = db_material_glazing['Transport scenario']
                    disposal_scenario_glazing = db_material_glazing['Disposal scenario']
                    cutting_waste_glazing = db_material_glazing['cutting waste']
                    life_time_glazing = db_material_glazing['Life Time']
                    transport_scenario_frame = db_material_frame['Transport scenario']
                    disposal_scenario_frame = db_material_frame['Disposal scenario']
                    cutting_waste_frame = db_material_frame['cutting waste']
                    life_time_frame = db_material_frame['Life Time']
                    transport_scenario_window = None
                    disposal_scenario_window = None
                    cutting_waste_window = None
                    life_time_window = None
                    surface_weight = db_material_frame['Surface Weight']  # TODO invalid
                else:
                    # window (inc. glazing and frame)
                    db_material = self._db.get(table='materials', keyword=ids, by='Dataset-ID')
                    if db_material.empty:
                        raise Exception(
                            'Cannot assign database material to frame in {mat}'.format(mat=ep_simple_glazing.Name))
                    db_id = ids
                    glz_id = None
                    frm_id = None
                    transport_scenario_window = db_material['Transport scenario']
                    disposal_scenario_window = db_material['Disposal scenario']
                    cutting_waste_window = db_material['cutting waste']
                    life_time_window = db_material['Life Time']
                    transport_scenario_glazing, transport_scenario_frame = None, None
                    disposal_scenario_glazing, disposal_scenario_frame = None, None
                    cutting_waste_glazing, cutting_waste_frame = None, None
                    life_time_glazing, life_time_frame = None, None
                    surface_weight = None

            else:
                raise Exception('No database id indicated for {mat}'.format(mat=ep_simple_glazing.Name))

        else:
            glz_id = None
            frm_id = None
            transport_scenario_glazing = None
            disposal_scenario_glazing = None
            cutting_waste_glazing = None
            life_time_glazing = None
            transport_scenario_frame = None
            disposal_scenario_frame = None
            cutting_waste_frame = None
            life_time_frame = None
            transport_scenario_window = None
            disposal_scenario_window = None
            cutting_waste_window = None
            life_time_window = None
            surface_weight = None

        material = WindowMaterial(
            name=ep_simple_glazing.Name,
            typ='SimpleGlazingSystem',
            glazing_id=glz_id,
            frame_id=frm_id,
            u_value=ep_simple_glazing.UFactor,
            g_value=ep_simple_glazing.Solar_Heat_Gain_Coefficient,
            surface_weight=surface_weight,
            transport_scenario_glazing=transport_scenario_glazing,
            disposal_scenario_glazing=disposal_scenario_glazing,
            cutting_waste_glazing=cutting_waste_glazing,
            life_time_glazing=life_time_glazing,
            transport_scenario_frame=transport_scenario_frame,
            disposal_scenario_frame=disposal_scenario_frame,
            cutting_waste_frame=cutting_waste_frame,
            life_time_frame=life_time_frame,
            transport_scenario_window=transport_scenario_window,
            disposal_scenario_window=disposal_scenario_window,
            cutting_waste_window=cutting_waste_window,
            life_time_window=life_time_window
        )
        # if ep_simple_glazing.Name not in self.window_materials:
        #     self.window_materials[ep_simple_glazing.Name] = material
        #
        self.library.add(material)

        if ep_simple_glazing.Name not in self.idf_values['window_materials']:
            self.idf_values['window_materials'][ep_simple_glazing.Name] = {
                'Visible_Transmittance': ep_simple_glazing.Visible_Transmittance
            }

        return material

    def idf_shade_material(self, shade_mat: ShadeMaterial): #, shading: Shading = None

        # collect original values from idf
        idf_values = self.idf_values['shade_materials'][shade_mat.Name]
        idf_args = [
            'Visible_Transmittance',
            'Visible_Reflectance',
            'Infrared_Transmittance',
            # 'Shade_to_Glass_Distance',
            'Top_Opening_Multiplier',
            'Bottom_Opening_Multiplier',
            'LeftSide_Opening_Multiplier',
            'RightSide_Opening_Multiplier',
            'Airflow_Permeability'
        ]
        idf_kwargs = {kw: idf_values[kw] for kw in idf_args}

        # # override idf info if we have it from the shading arg
        # if shading is not None:
        #     idf_kwargs['Shade_to_Glass_Distance'] = shading.DistanceToGlass

        self.idf.newidfobject(
            key='WindowMaterial:Shade'.upper(),
            Name=shade_mat.Name,
            Solar_Transmittance=shade_mat.Transmittance,
            Solar_Reflectance=shade_mat.Reflectance,
            Infrared_Hemispherical_Emissivity=shade_mat.Emissivity,
            Thickness=shade_mat.Thickness,
            Conductivity=shade_mat.Conductivity,
            Shade_to_Glass_Distance=shade_mat.DistanceToGlass
            **idf_kwargs
        )

    def update_idf_shade_material(self, shade_mat: ShadeMaterial): # , shading: Shading = None

        idf_material = self.idf.getobject('WindowMaterial:Shade'.upper(), shade_mat.Name)

        idf_material.Solar_Transmittance = shade_mat.Transmittance
        idf_material.Solar_Reflectance = shade_mat.Reflectance
        idf_material.Infrared_Hemispherical_Emissivity = shade_mat.Emissivity
        idf_material.Thickness = shade_mat.Thickness
        idf_material.Conductivity = shade_mat.Conductivity
        idf_material.Shade_to_Glass_Distance = shade_mat.DistanceToGlass

        # if shading is not None:
        #     idf_material.Shade_to_Glass_Distance = shading.DistanceToGlass

    def fp_shade_material(self, ep_shade_mat):  # WindowMaterial:Shade
        if self.parse_lca:
            # TODO use matching (like in the lca module) only the density is missing
            if '#' in ep_shade_mat.Name:
                name, db_id = ep_shade_mat.Name.split('#')
                db_id = db_id.split('-')[0]  # eliminate additional info generated by e.g. honeybee
                db_shading_material = self._db.get(table='materials', keyword=db_id, by='Dataset-ID')
                if db_shading_material.empty:
                    raise Exception('Cannot assign database material to {mat}'.format(mat=ep_shade_mat.Name))
                else:
                    db_shading_mat_prop = self._db.get(table='properties',
                                                       keyword=db_shading_material['Auricon ID'],
                                                       by='GUID')
            else:
                raise Exception('No database id indicated for {mat}'.format(mat=ep_shade_mat.Name))
            density = db_shading_mat_prop['Suruseg']
            transport_scenario = db_shading_material['Transport scenario']
            disposal_scenario = db_shading_material['Disposal scenario']
            cutting_waste = db_shading_material['cutting waste']
            life_time = db_shading_material['Life Time']

        else:
            db_id = None
            density = None
            transport_scenario = None
            disposal_scenario = None
            cutting_waste = None
            life_time = None

        material = ShadeMaterial(
            name=ep_shade_mat.Name,
            db_id=db_id,
            reflectance=ep_shade_mat.Solar_Reflectance,
            transmittance=ep_shade_mat.Solar_Transmittance,
            emissivity=ep_shade_mat.Infrared_Hemispherical_Emissivity,
            thickness=ep_shade_mat.Thickness,
            conductivity=ep_shade_mat.Conductivity,
            density=density,
            distance_to_glass=ep_shade_mat.Shade_to_Glass_Distance
            # transport_scenario=transport_scenario,
            # disposal_scenario=disposal_scenario,
            # cutting_waste=cutting_waste,
            # life_time=life_time,
        )
        # if ep_shading_mat.Name not in self.shading_materials:
        #     self.shading_materials[ep_shading_mat.Name] = material
        self.library.add(material)

        if ep_shade_mat.Name not in self.idf_values['shade_materials']:
            self.idf_values['shade_materials'][ep_shade_mat.Name] = {
                'Visible_Transmittance': ep_shade_mat.Visible_Transmittance,
                'Visible_Reflectance': ep_shade_mat.Visible_Reflectance,
                'Infrared_Transmittance': ep_shade_mat.Infrared_Transmittance,
                'Top_Opening_Multiplier': ep_shade_mat.Top_Opening_Multiplier,
                'Bottom_Opening_Multiplier': ep_shade_mat.Bottom_Opening_Multiplier,
                'LeftSide_Opening_Multiplier': ep_shade_mat.LeftSide_Opening_Multiplier,
                'RightSide_Opening_Multiplier': ep_shade_mat.RightSide_Opening_Multiplier,
                'Airflow_Permeability': ep_shade_mat.Airflow_Permeability
            }

        return material

    def idf_blind_material(self, blind_mat: BlindMaterial): # , shading: Shading = None

        idf_values = self.idf_values['blind_materials'][blind_mat.Name]
        idf_args = [
            'Slat_Orientation',
            # 'Slat_Width',
            # 'Slat_Separation',
            # 'Slat_Angle',
            'Back_Side_Slat_Beam_Solar_Reflectance',
            'Slat_Diffuse_Solar_Transmittance',
            'Front_Side_Slat_Diffuse_Solar_Reflectance',
            'Back_Side_Slat_Diffuse_Solar_Reflectance',
            'Slat_Beam_Visible_Transmittance',
            'Front_Side_Slat_Beam_Visible_Reflectance',
            'Back_Side_Slat_Beam_Visible_Reflectance',
            'Slat_Diffuse_Visible_Transmittance',
            'Front_Side_Slat_Diffuse_Visible_Reflectance',
            'Back_Side_Slat_Diffuse_Visible_Reflectance',
            'Slat_Infrared_Hemispherical_Transmittance',
            'Back_Side_Slat_Infrared_Hemispherical_Emissivity',
            # 'Blind_to_Glass_Distance',
            'Blind_Top_Opening_Multiplier',
            'Blind_Bottom_Opening_Multiplier',
            'Blind_Left_Side_Opening_Multiplier',
            'Blind_Right_Side_Opening_Multiplier',
            'Minimum_Slat_Angle',
            'Maximum_Slat_Angle'
        ]
        idf_kwargs = {kw: idf_values[kw] for kw in idf_args}

        # # override idf info if we have it from the shading arg
        # if shading is not None:
        #     idf_kwargs['Slat_Width'] = blind_mat.SlatWidth
        #     idf_kwargs['Slat_Separation'] = blind_mat.SlatSeparation
        #     idf_kwargs['Slat_Angle'] = blind_mat.SlatAngle
        #     idf_kwargs['Blind_to_Glass_Distance'] = blind_mat.DistanceToGlass

        self.idf.newidfobject(
            key='WindowMaterial:Blind'.upper(),
            Name=blind_mat.Name,
            Slat_Thickness=blind_mat.Thickness,
            Slat_Conductivity=blind_mat.Conductivity,
            Slat_Beam_Solar_Transmittance=blind_mat.Transmittance,
            Front_Side_Slat_Beam_Solar_Reflectance=blind_mat.Reflectance,
            Front_Side_Slat_Infrared_Hemispherical_Emissivity=blind_mat.Emissivity,
            Slat_Width=blind_mat.SlatWidth,
            Slat_Separation=blind_mat.SlatSeparation,
            Slat_Angle=blind_mat.SlatAngle,
            Blind_to_Glass_Distance=blind_mat.DistanceToGlass,
            **idf_kwargs
        )

    def update_idf_blind_material(self, blind_mat: BlindMaterial): # , shading: Shading = None

        idf_material = self.idf.getobject('WindowMaterial:Blind'.upper(), blind_mat.Name)

        idf_material.Slat_Thickness = blind_mat.Thickness
        idf_material.Slat_Conductivity = blind_mat.Conductivity
        idf_material.Slat_Beam_Solar_Transmittance = blind_mat.Transmittance
        idf_material.Front_Side_Slat_Beam_Solar_Reflectance = blind_mat.Reflectance
        idf_material.Front_Side_Slat_Infrared_Hemispherical_Emissivity = blind_mat.Emissivity
        idf_material.Slat_Width = blind_mat.SlatWidth
        idf_material.Slat_Separation = blind_mat.SlatSeparation
        idf_material.Slat_Angle = blind_mat.SlatAngle
        idf_material.Blind_to_Glass_Distance = blind_mat.DistanceToGlass

        # if shading is not None:
        #     idf_material.Slat_Width = shading.SlatWidth
        #     idf_material.Slat_Separation = shading.SlatSeparation
        #     idf_material.Slat_Angle = shading.SlatAngle
        #     idf_material.Blind_to_Glass_Distance = shading.DistanceToGlass

    def fp_blind_material(self, ep_blind_mat):  # WindowMaterial:Blind
        if self.parse_lca:
            # TODO use matching (like in the lca module)
            if '#' in ep_blind_mat.Name:
                name, db_id = ep_blind_mat.Name.split('#')
                db_id = db_id.split('-')[0]  # eliminate additional info generated by e.g. honeybee

                db_shading_material = self._db.get(table='materials', keyword=db_id, by='Dataset-ID')
                if db_shading_material.empty:
                    raise Exception('Cannot assign database material to {mat}'.format(mat=ep_blind_mat.Name))
                else:
                    db_shading_mat_prop = self._db.get(table='properties',
                                                       keyword=db_shading_material['Auricon ID'],
                                                       by='GUID')
            else:
                raise Exception('Cannot assign database material to {mat}'.format(mat=ep_blind_mat.Name))
            density = db_shading_mat_prop['Suruseg']
            transport_scenario = db_shading_material['Transport scenario']
            disposal_scenario = db_shading_material['Disposal scenario']
            cutting_waste = db_shading_material['cutting waste']
            life_time = db_shading_material['Life Time']

        else:
            db_id = None
            density = None
            transport_scenario = None
            disposal_scenario = None
            cutting_waste = None
            life_time = None

        material = BlindMaterial(
            name=ep_blind_mat.Name,
            db_id=db_id,
            reflectance=ep_blind_mat.Front_Side_Slat_Beam_Solar_Reflectance,
            transmittance=ep_blind_mat.Slat_Beam_Solar_Transmittance,
            emissivity=ep_blind_mat.Front_Side_Slat_Infrared_Hemispherical_Emissivity,
            thickness=ep_blind_mat.Slat_Thickness,
            conductivity=ep_blind_mat.Slat_Conductivity,
            density=density,
            distance_to_glass=ep_blind_mat.Blind_to_Glass_Distance,
            slat_width=ep_blind_mat.Slat_Width,
            slat_separation=ep_blind_mat.Slat_Separation,
            slat_angle=ep_blind_mat.Slat_Angle,
            # transport_scenario=transport_scenario,
            # disposal_scenario=disposal_scenario,
            # cutting_waste=cutting_waste,
            # life_time=life_time,
        )
        # if ep_blind_mat.Name not in self.shading_materials:
        #     self.shading_materials[ep_blind_mat.Name] = material
        self.library.add(material)

        if ep_blind_mat.Name not in self.idf_values['blind_materials']:
            self.idf_values['blind_materials'][ep_blind_mat.Name] = {
                'Slat_Orientation': ep_blind_mat.Slat_Orientation,
                # # this is information included in the Shading.SlatWidth that uses this material:
                # 'Slat_Width': ep_blind_mat.Slat_Width,
                # # this is information included in the Shading.SlatSeparation that uses this material:
                # 'Slat_Separation': ep_blind_mat.Slat_Separation,
                # # this is information included in the Shading.SlatAngle that uses this material:
                # 'Slat_Angle': ep_blind_mat.Slat_Angle,
                'Back_Side_Slat_Beam_Solar_Reflectance': ep_blind_mat.Back_Side_Slat_Beam_Solar_Reflectance,
                'Slat_Diffuse_Solar_Transmittance': ep_blind_mat.Slat_Diffuse_Solar_Transmittance,
                'Front_Side_Slat_Diffuse_Solar_Reflectance': ep_blind_mat.Front_Side_Slat_Diffuse_Solar_Reflectance,
                'Back_Side_Slat_Diffuse_Solar_Reflectance': ep_blind_mat.Back_Side_Slat_Diffuse_Solar_Reflectance,
                'Slat_Beam_Visible_Transmittance': ep_blind_mat.Slat_Beam_Visible_Transmittance,
                'Front_Side_Slat_Beam_Visible_Reflectance': ep_blind_mat.Front_Side_Slat_Beam_Visible_Reflectance,
                'Back_Side_Slat_Beam_Visible_Reflectance': ep_blind_mat.Back_Side_Slat_Beam_Visible_Reflectance,
                'Slat_Diffuse_Visible_Transmittance': ep_blind_mat.Slat_Diffuse_Visible_Transmittance,
                'Front_Side_Slat_Diffuse_Visible_Reflectance': ep_blind_mat.Front_Side_Slat_Diffuse_Visible_Reflectance,
                'Back_Side_Slat_Diffuse_Visible_Reflectance': ep_blind_mat.Back_Side_Slat_Diffuse_Visible_Reflectance,
                'Slat_Infrared_Hemispherical_Transmittance': ep_blind_mat.Slat_Infrared_Hemispherical_Transmittance,
                'Back_Side_Slat_Infrared_Hemispherical_Emissivity': ep_blind_mat.Back_Side_Slat_Infrared_Hemispherical_Emissivity,
                # # this is information included in the Shading.DistanceToGlass that uses this material:
                # 'Blind_to_Glass_Distance': ep_blind_mat.Blind_to_Glass_Distance,
                'Blind_Top_Opening_Multiplier': ep_blind_mat.Blind_Top_Opening_Multiplier,
                'Blind_Bottom_Opening_Multiplier': ep_blind_mat.Blind_Bottom_Opening_Multiplier,
                'Blind_Left_Side_Opening_Multiplier': ep_blind_mat.Blind_Left_Side_Opening_Multiplier,
                'Blind_Right_Side_Opening_Multiplier': ep_blind_mat.Blind_Right_Side_Opening_Multiplier,
                'Minimum_Slat_Angle': ep_blind_mat.Minimum_Slat_Angle,
                'Maximum_Slat_Angle': ep_blind_mat.Maximum_Slat_Angle
            }

        return material

    def idf_shading(self, shading: Shading):  # , update_material: bool = True, shading_material: ShadingMaterial = None):
        """

        :param shading:
        :return: None
        """

        idf_values = self.idf_values['shadings'][shading.Name]
        idf_args = [
            'Construction_with_Shading_Name',
            'Shading_Device_Material_Name',
            'Shading_Control_Type',
            'Schedule_Name',
            'Setpoint',
            'Glare_Control_Is_Active',
            'Type_of_Slat_Angle_Control_for_Blinds',
            'Slat_Angle_Schedule_Name',
        ]
        idf_kwargs = {kw: idf_values[kw] for kw in idf_args}

        # override idf values with known data
        if shading.Material is not None:
            idf_kwargs['Shading_Device_Material_Name'] = shading.Material.RefName
        if shading.Construction is not None:
            idf_kwargs['Construction_with_Shading_Name'] = shading.Construction.RefName

        if shading.IsScheduled:
            is_scheduled = 'Yes'
        else:
            is_scheduled = 'No'

        self.idf.newidfobject(
            key='WindowProperty:ShadingControl'.upper(),
            Name=shading.Name,
            Shading_Type=shading.Type,
            Shading_Control_Is_Scheduled=is_scheduled,
            # Shading_Device_Material_Name=shading.Material.Name,
            **idf_kwargs
        )

        # if update_material:
        #
        #     if 'Blind' in shading.Type:
        #         if self.idf.getobject('WindowMaterial:Blind'.upper(), shading_material.Name) is not None:
        #             self.update_idf_blind_material(blind_mat=shading_material, shading=shading)
        #         else:
        #             self.idf_blind_material(blind_mat=shading_material, shading=shading)
        #
        #     elif 'Shade' in shading.Type:
        #         if self.idf.getobject('WindowMaterial:Shade'.upper(), shading_material.Name) is not None:
        #             self.update_idf_shade_material(shading_mat=shading_material, shading=shading)
        #         else:
        #             self.idf_shade_material(shading_mat=shading_material, shading=shading)
        #
        #     else:
        #         raise Exception('Shading Type needs to be either "*Shade" or "*Blind", not {st}'.format(
        #             st=shading.Type))

    def update_idf_shading(self, shading: Shading):  #, update_material: bool = False,
                           # shading_material: ShadingMaterial = None, update_construction: bool = False):
        """

        :param shading:
        :return: None
        """

        idf_shading = self.idf.getobject('WindowProperty:ShadingControl'.upper(), shading.Name)

        idf_shading.Shading_Type = shading.Type
        if shading.IsScheduled:
            idf_shading.Shading_Control_Is_Scheduled = 'Yes'
        else:
            idf_shading.Shading_Control_Is_Scheduled = 'No'

        if shading.Material is not None:
            idf_shading.Shading_Device_Material_Name = shading.Material.RefName
        if shading.Construction is not None:
            idf_shading.Construction_with_Shading_Name = shading.Construction.RefName

        # if update_material:
        #
        #     if 'Blind' in shading.Type:
        #         if self.idf.getobject('WindowMaterial:Blind'.upper(), shading_material.Name) is not None:
        #             self.update_idf_blind_material(blind_mat=shading_material, shading=shading)
        #         else:
        #             self.idf_blind_material(blind_mat=shading_material, shading=shading)
        #
        #     elif 'Shade' in shading.Type:
        #         if self.idf.getobject('WindowMaterial:Shade'.upper(), shading_material.Name) is not None:
        #             self.update_idf_shade_material(shading_mat=shading_material, shading=shading)
        #         else:
        #             self.idf_shade_material(shading_mat=shading_material, shading=shading)
        #
        #     else:
        #         raise Exception('Shading Type needs to be either "*Shade" or "*Blind", not {st}'.format(
        #             st=shading.Type))

    def fp_shading(self, ep_shading_control):  # WindowProperty:ShadingControl
        # WindowMaterial:Shade / WindowMaterial:Blind

        # todo this belongs to the simple energy calculation module
        shading_factors = {
            'shade': {
                'interior': 0.47,
                'exterior': 0.13,
            },
            'blind': {
                'interior': 0.55,
                'exterior': 0.12
            }
        }
        if 'Blind' in ep_shading_control.Shading_Type:
            placing = ep_shading_control.Shading_Type.split('Blind')[0]
            sh_factor = shading_factors['blind'][placing.lower()]
        elif 'Shade' in ep_shading_control.Shading_Type:
            placing = ep_shading_control.Shading_Type.split('Shade')[0]
            sh_factor = shading_factors['shade'][placing.lower()]
        else:
            raise Exception('Shading Type needs to be either "Shade" or "Blind", not {st}'.format(
                st=ep_shading_control.Shading_Type))

        material_name = ep_shading_control.Shading_Device_Material_Name
        # OR
        construction_name = ep_shading_control.Construction_with_Shading_Name

        if material_name:  # not empty string
            if 'Blind' in ep_shading_control.Shading_Type:
                ep_material = self.idf.getobject('WindowMaterial:Blind', material_name)


                # dist_to_glass = shading_material.Blind_to_Glass_Distance
                # slat_width = shading_material.Slat_Width
                # slat_separation = shading_material.Slat_Separation
                # slat_angle = shading_material.Slat_Angle
                fp_material = self.fp_blind_material(ep_material)

            elif 'Shade' in ep_shading_control.Shading_Type:
                ep_material = self.idf.getobject('WindowMaterial:Shade', material_name)

                # dist_to_glass = shading_material.Shade_to_Glass_Distance
                # slat_width = None
                # slat_separation = None
                # slat_angle = None
                fp_material = self.fp_shade_material(ep_material)

            else:
                raise Exception('Shading Type needs to be either "Shade" or "Blind", not {st}'.format(
                    st=ep_shading_control.Shading_Type))
            fp_material = fp_material.get_ref()
            fp_construction = None

        elif construction_name:  # not empty string
            # shading material is defined in a construction with glazing
            ep_construction = self.idf.getobject('Construction', construction_name)
            fp_construction = self.fp_construction(ep_construction)
            fp_construction = fp_construction.get_ref()
            fp_material = None
            #
            # # get the first layer from the outside, that defines a shading material
            # for mat_name in ep_construction.fieldvalues[2:]:
            #     if 'Blind' in ep_shading_control.Shading_Type:
            #         shading_material = self.idf.getobject('WindowMaterial:Blind', mat_name)
            #     elif 'Shade' in ep_shading_control.Shading_Type:
            #         shading_material = self.idf.getobject('WindowMaterial:Shade', mat_name)
            #     else:
            #         shading_material = None
            #
            #     if shading_material is not None:
            #         material_name = mat_name
            #         break
        else:
            raise Exception('Neither Shading_Device_Material_Name nor Construction_with_Shading_Name in {st}'.format(
                st=ep_shading_control.Name))

        is_scheduled_value = ep_shading_control.Shading_Control_Is_Scheduled
        if is_scheduled_value.lower() == 'yes':
            is_scheduled = True
        elif is_scheduled_value.lower() == 'no':
            is_scheduled = False
        else:
            raise Exception('Cannot parse IsScheduled value of: {v}'.format(v=is_scheduled_value))
        shading = Shading(
            name=ep_shading_control.Name,
            typ=ep_shading_control.Shading_Type,
            properties={},
            material=fp_material,
            construction=fp_construction,
            shading_factor=sh_factor,
            is_scheduled=is_scheduled,
            # distance_to_glass=dist_to_glass,
            # slat_width=slat_width,
            # slat_separation=slat_separation,
            # slat_angle=slat_angle

        )
        # if ep_shading_control.Name not in self.shadings:
        #     self.shadings[ep_shading_control.Name] = shading
        self.library.add(shading)

        if ep_shading_control.Name not in self.idf_values['shadings']:
            self.idf_values['shadings'][ep_shading_control.Name] = {
                'Construction_with_Shading_Name': ep_shading_control.Construction_with_Shading_Name,
                'Shading_Device_Material_Name': ep_shading_control.Shading_Device_Material_Name,
                'Shading_Control_Type': ep_shading_control.Shading_Control_Type,
                'Schedule_Name': ep_shading_control.Schedule_Name,
                'Setpoint': ep_shading_control. Setpoint,
                'Glare_Control_Is_Active': ep_shading_control.Glare_Control_Is_Active,
                'Type_of_Slat_Angle_Control_for_Blinds': ep_shading_control.Type_of_Slat_Angle_Control_for_Blinds,
                'Slat_Angle_Schedule_Name': ep_shading_control.Slat_Angle_Schedule_Name
            }

        return shading

    def idf_construction(self, construction: Construction):

        layer_kwrds = ['Layer_{}'.format(n+1) for n in range(len(construction.Layers))]
        layer_names = [layer.RefName for layer in construction.Layers]
        layer_kwargs = {kw: name for kw, name in zip(layer_kwrds, layer_names)}
        self.idf.newidfobject(
            key='Construction'.upper(),
            Name=construction.Name,
            **layer_kwargs
        )

    def update_idf_construction(self, construction: Construction):

        idf_construction = self.idf.getobject('Construction'.upper(), construction.Name)

        # delete previous layers from idf
        del idf_construction.fieldvalues[2:]

        # add new layers to the idf from construction
        for layer, material in zip(idf_construction.fieldnames[2:], construction.Layers):
            setattr(idf_construction, layer, material.RefName)

    def fp_construction(self, ep_construction):  # Construction
        construction = Construction(
            name=ep_construction.Name,
            layers=[]
        )
        for material_name in ep_construction.fieldvalues[2:]:
            material = self.idf.getobject('Material', material_name)
            glazing = self.idf.getobject('WindowMaterial:SimpleGlazingSystem', material_name)
            blind = self.idf.getobject('WindowMaterial:Blind', material_name)
            shade = self.idf.getobject('WindowMaterial:Shade', material_name)

            if material is not None:
                fp_mat = self.fp_opaque_material(material)
            elif glazing is not None:
                fp_mat = self.fp_window_material(glazing)
            elif blind is not None:
                fp_mat = self.fp_blind_material(blind)
            elif shade is not None:
                fp_mat = self.fp_shade_material(shade)
            else:
                 raise Exception('Material cannot be found in the idf: {m}'.format(m=material_name))

            construction.Layers.append(fp_mat.get_ref())

        # if ep_construction.Name not in self.constructions:
        #     self.constructions[ep_construction.Name] = construction
        self.library.add(construction)
        return construction

    def idf_fenestration_surface(self, window: FenestrationSurface, parent_surface: BuildingSurface = None):

        # eppy can only handle up to 4 corner points, not more

        vertex_kwargs = {}
        for i, vertex in enumerate(window.vertices):
            vertex_kwargs['Vertex_{c}_Xcoordinate'.format(c=i+1)] = vertex.x
            vertex_kwargs['Vertex_{c}_Ycoordinate'.format(c=i+1)] = vertex.y
            vertex_kwargs['Vertex_{c}_Zcoordinate'.format(c=i+1)] = vertex.z

        optional_kwargs = {}
        if parent_surface is not None:
            optional_kwargs['Building_Surface_Name'] = parent_surface.Name
        if window.FrameName is not None:
            optional_kwargs['Frame_and_Divider_Name'] = window.FrameName
        if window.Multiplier is not None:
            optional_kwargs['Multiplier'] = window.Multiplier
        if window.Shading is not None:
            optional_kwargs['Shading_Control_Name'] = window.Shading.RefName

        self.idf.newidfobject(
            key='FenestrationSurface:Detailed'.upper(),
            Name=window.Name,
            Surface_Type=window.SurfaceType,
            Construction_Name=window.Construction.RefName,
            # Building_Surface_Name=,
            # Outside_Boundary_Condition_Object=None,
            # View_Factor_to_Ground=,
            # Shading_Control_Name=window.Shading.RefName,
            # Frame_and_Divider_Name=,
            # Multiplier=window.Multiplier,
            Number_of_Vertices=len(window.vertices),
            **optional_kwargs,
            **vertex_kwargs
        )

    def update_idf_fenestration_surface(self, window: FenestrationSurface):

        idf_window = self.idf.getobject('FenestrationSurface:Detailed'.upper(), window.Name)

        idf_window.Surface_Type = window.SurfaceType
        idf_window.Construction_Name = window.Construction.RefName
        if window.Shading is not None:
            idf_window.Shading_Control_Name = window.Shading.RefName
        if window.FrameName is not None:
            idf_window.Frame_and_Divider_Name = window.FrameName
        idf_window.Multiplier = window.Multiplier
        idf_window.Number_of_Vertices = len(window.vertices)

        # update vertices
        for i, vertex in enumerate(window.vertices):
            setattr(idf_window, 'Vertex_{c}_Xcoordinate'.format(c=i + 1), vertex.x)
            setattr(idf_window, 'Vertex_{c}_Ycoordinate'.format(c=i + 1), vertex.y)
            setattr(idf_window, 'Vertex_{c}_Zcoordinate'.format(c=i + 1), vertex.z)

    def fp_fenestration_surface(self, ep_fenestration_surface):  # FenestrationSurface:Detailed

        vertices = []
        for coord in ep_fenestration_surface.coords:
            x, y, z = coord
            vertices.append(Point(x, y, z))

        ep_shading = self.idf.getobject('WindowProperty:ShadingControl', ep_fenestration_surface.Shading_Control_Name)
        if ep_shading is not None:
            shading = self.fp_shading(ep_shading).get_ref()
        else:
            shading = None

        ep_construction = self.idf.getobject('Construction', ep_fenestration_surface.Construction_Name)
        construction = self.fp_construction(ep_construction)

        if ep_fenestration_surface.Frame_and_Divider_Name != '':
            frame_name = ep_fenestration_surface.Frame_and_Divider_Name
        else:
            frame_name = None

        surf = FenestrationSurface(
            name=ep_fenestration_surface.Name,
            vertices=vertices,
            surface_type=ep_fenestration_surface.Surface_Type,
            shading=shading,
            construction=construction.get_ref(),
            shading_control_name=ep_fenestration_surface.Shading_Control_Name,
            frame_name=frame_name,
            multiplier=ep_fenestration_surface.Multiplier
            # geometry_rules= TODO handle original geometry rules
        )
        return surf

    def idf_building_surface(self, building_surface: BuildingSurface, parent_zone: Zone = None,
                             create_windows: bool = True):

        vertex_kwargs = {}
        for i, vertex in enumerate(building_surface.vertices):
            vertex_kwargs['Vertex_{c}_Xcoordinate'.format(c=i+1)] = vertex.x
            vertex_kwargs['Vertex_{c}_Ycoordinate'.format(c=i+1)] = vertex.y
            vertex_kwargs['Vertex_{c}_Zcoordinate'.format(c=i+1)] = vertex.z

        optional_kwargs = {}
        if parent_zone is not None:
            optional_kwargs['Zone_Name'] = parent_zone.Name

        if building_surface.OutsideBoundaryCondition.lower() == 'outdoors':
            optional_kwargs['Sun_Exposure'] = 'Sunexposed'
            optional_kwargs['Wind_Exposure'] = 'Windexposed'
        else:
            optional_kwargs['Sun_Exposure'] = 'Nosun'
            optional_kwargs['Wind_Exposure'] = 'Nowind'

        self.idf.newidfobject(
            key='BuildingSurface:Detailed'.upper(),
            Name=building_surface.Name,
            Surface_Type=building_surface.SurfaceType,
            Construction_Name=building_surface.Construction.RefName,
            # Zone_Name=,
            Outside_Boundary_Condition=building_surface.OutsideBoundaryCondition,  # TODO if 'surface'
            # Outside_Boundary_Condition_Object=None, # TODO fill with surface name
            # Sun_Exposure=,
            # Wind_Exposure=,
            # View_Factor_to_Ground=None,
            Number_of_Vertices=len(building_surface.vertices),
            **optional_kwargs,
            **vertex_kwargs
        )

        if create_windows:
            # make sure that no window belongs to this surface
            windows = self.idf.idfobjects['FenestrationSurface:Detailed'.upper()]
            surf_wins = [win for win in windows if win.Building_Surface_Name == building_surface.Name]

            # remove these windows form the idf
            for win in surf_wins:
                self.idf.removeidfobject(win)

            # create new windows
            for window in building_surface.Fenestration:
                self.idf_fenestration_surface(window, parent_surface=building_surface)

    def update_idf_building_surface(self, building_surface: BuildingSurface, fenestration_method: str = None):
        """

        :param building_surface:
        :param fenestration_method:
            'recreate' - delete existing windows and create new based on BuildingSurface.Fenestration
            'update' - leave existing fenestration and update the attributes only
                (e.g. if number of windows did not chagnge)
            None (default) - do not touch fenestration
        :return: None
        """

        idf_surf = self.idf.getobject('BuildingSurface:Detailed'.upper(), building_surface.Name)

        idf_surf.Surface_Type = building_surface.SurfaceType
        idf_surf.Construction_Name = building_surface.Construction.RefName
        # TODO we cannot handle this if it is adjacent to another surface:
        # building_surface.Outside_Boundary_Condition = building_surface.OutsideBoundaryCondition
        idf_surf.Number_of_Vertices = len(building_surface.vertices)

        # update vertices
        for i, vertex in enumerate(building_surface.vertices):
            setattr(idf_surf, 'Vertex_{c}_Xcoordinate'.format(c=i + 1), vertex.x)
            setattr(idf_surf, 'Vertex_{c}_Ycoordinate'.format(c=i + 1), vertex.y)
            setattr(idf_surf, 'Vertex_{c}_Zcoordinate'.format(c=i + 1), vertex.z)

        if fenestration_method == 'recreate':
            # get all windows for this surface
            windows = self.idf.idfobjects['FenestrationSurface:Detailed'.upper()]
            surf_wins = [win for win in windows if win.Building_Surface_Name == building_surface.Name]

            # remove these windows form the idf
            for win in surf_wins:
                self.idf.removeidfobject(win)

            # create new windows
            for window in building_surface.Fenestration:
                self.idf_fenestration_surface(window, parent_surface=building_surface)

        elif fenestration_method == 'update':
            for window in building_surface.Fenestration:
                self.update_idf_fenestration_surface(window)

    def fp_building_surface(self, ep_building_surface):  # BuildingSurface:Detailed

        vertices = []
        for coord in ep_building_surface.coords:
            x, y, z = coord
            vertices.append(Point(x, y, z))

        fenestration = []
        for ep_window in ep_building_surface.subsurfaces:
            window = self.fp_fenestration_surface(ep_window)
            fenestration.append(window)

        ep_construction = self.idf.getobject('Construction', ep_building_surface.Construction_Name)
        construction = self.fp_construction(ep_construction)

        return BuildingSurface(
            name=ep_building_surface.Name,
            vertices=vertices,
            fenestration=fenestration,
            surface_type=ep_building_surface.Surface_Type,
            construction=construction.get_ref(),
            # TODO put IuId of surface here if adjacent to another
            outside_boundary_condition=ep_building_surface.Outside_Boundary_Condition
            # geometry_rules= TODO handle original geometry rules
        )

    def idf_non_zone_surface(self, non_zone_surface: NonZoneSurface):

        vertex_kwargs = {}
        for i, vertex in enumerate(non_zone_surface.vertices):
            vertex_kwargs['Vertex_{c}_Xcoordinate'.format(c=i+1)] = vertex.x
            vertex_kwargs['Vertex_{c}_Ycoordinate'.format(c=i+1)] = vertex.y
            vertex_kwargs['Vertex_{c}_Zcoordinate'.format(c=i+1)] = vertex.z

        self.idf.newidfobject(
            key='Shading:Building:Detailed'.upper(),
            Name=non_zone_surface.Name,
            # Transmittance_Schedule_Name=,
            Number_of_Vertices=len(non_zone_surface.vertices),
            **vertex_kwargs
        )

    def update_idf_non_zone_surface(self, non_zone_surface: NonZoneSurface):

        idf_surf = self.idf.getobject('Shading:Building:Detailed'.upper(), non_zone_surface.Name)

        idf_surf.Number_of_Vertices = len(non_zone_surface.vertices)

        # update vertices
        for i, vertex in enumerate(non_zone_surface.vertices):
            setattr(idf_surf, 'Vertex_{c}_Xcoordinate'.format(c=i + 1), vertex.x)
            setattr(idf_surf, 'Vertex_{c}_Ycoordinate'.format(c=i + 1), vertex.y)
            setattr(idf_surf, 'Vertex_{c}_Zcoordinate'.format(c=i + 1), vertex.z)

    def fp_non_zone_surface(self, ep_shading_surface, construction: Construction = None):  # Shading:Building:Detailed

        if construction is None:
            construction = Construction('EmptyConstruction', [])
            self.library.add(construction)

        vertices = []
        for coord in ep_shading_surface.coords:
            x, y, z = coord
            vertices.append(Point(x, y, z))

        return NonZoneSurface(
            name=ep_shading_surface.Name,
            vertices=vertices,
            surface_type='undefined',
            construction=construction.get_ref()
            # geometry_rules= TODO handle original geometry rules
        )

    def idf_internal_mass(self, internal_mass: InternalMass, parent_zone: Zone = None):
        optional_kwargs = {}
        if parent_zone is not None:
            optional_kwargs['Zone_Name'] = parent_zone.Name

        self.idf.newidfobject(
            key='InternalMass'.upper(),
            Name=internal_mass.Name,
            Construction_Name=internal_mass.Construction.RefName,
            # Zone_Name=,
            Surface_Area=internal_mass.Area,
            **optional_kwargs
        )

    def update_idf_internal_mass(self, internal_mass: InternalMass):

        idf_mass = self.idf.getobject('InternalMass'.upper(), internal_mass.Name)

        idf_mass.Construction_Name = internal_mass.Construction.RefName
        idf_mass.Surface_Area = internal_mass.Area

    def fp_internal_mass(self, ep_internal_mass):  # InternalMass

        ep_construction = self.idf.getobject('Construction', ep_internal_mass.Construction_Name)
        construction = self.fp_construction(ep_construction)

        return InternalMass(
            name=ep_internal_mass.Name,
            construction=construction.get_ref(),
            area=ep_internal_mass.Surface_Area
        )

    def idf_zone(self, zone: Zone, create_surfaces: bool = True, create_internal_masses: bool = True,
                 create_fenestration: bool = True):

        optional_kwargs = {}
        if zone.DirectionOfRelativeNorth is not None:
            optional_kwargs['Direction_of_Relative_North'] = zone.DirectionOfRelativeNorth
        else:
            optional_kwargs['Direction_of_Relative_North'] = ''

        if zone.Origin is not None:
            optional_kwargs['X_Origin'] = zone.Origin.x
            optional_kwargs['Y_Origin'] = zone.Origin.y
            optional_kwargs['Z_Origin'] = zone.Origin.z
        else:
            optional_kwargs['X_Origin'] = ''
            optional_kwargs['Y_Origin'] = ''
            optional_kwargs['Z_Origin'] = ''

        self.idf.newidfobject(
            key='Zone'.upper(),
            Name=zone.Name,
            Type='',
            **optional_kwargs
            # Multiplier=,
            # Ceiling_Height=,
            # Volume=,
            # Floor_Area=,
            # Zone_Inside_Convection_Algorithm=,
            # Zone_Outside_Convection_Algorithm=,
        )

        if create_surfaces:
            # make sure that no surface belongs to this zone
            surfaces = self.idf.idfobjects['BuildingSurface:Detailed'.upper()]
            zone_surfs = [surf for surf in surfaces if surf.Zone_Name == zone.Name]

            for surf in zone_surfs:
                # windows are removed in the idf_building_surface method if create_windows is True

                # remove the surface form the idf
                self.idf.removeidfobject(surf)

            # create new surfaces
            for surface in zone.BuildingSurfaces:
                self.idf_building_surface(surface, parent_zone=zone, create_windows=create_fenestration)

        if create_internal_masses:
            # make sure that no internal mass belongs to this zone
            masses = self.idf.idfobjects['InternalMass'.upper()]
            zone_masses = [mass for mass in masses if mass.Zone_Name == zone.Name]

            for mass in zone_masses:
                # remove the mass form the idf
                self.idf.removeidfobject(mass)

            # create new masses
            for internal_mass in zone.InternalMasses:
                self.idf_internal_mass(internal_mass, parent_zone=zone)

    def update_idf_zone(self, zone: Zone, fenestration_method: str = None, surface_method: str = None,
                        internal_mass_method: str = None):
        """

        :param zone: Firepy Zone instance
        :param fenestration_method: 'recreate', 'update' or None; if None, fenestration are not updated
        :param surface_method: 'recreate', 'update' or None; if None, surfaces are not updated; if 'recreate'
            is used, than fenestration_method is ignored, and fenestration will be recreated
        :param internal_mass_method: 'recreate', 'update' or None; if None, internal masses are not updated
        :return: None
        """
        if surface_method == 'update':
            for surface in zone.BuildingSurfaces:
                self.update_idf_building_surface(surface, fenestration_method=fenestration_method)
        elif surface_method == 'recreate':
            # get all surfaces for this zone
            surfaces = self.idf.idfobjects['BuildingSurface:Detailed'.upper()]
            zone_surfs = [surf for surf in surfaces if surf.Zone_Name == zone.Name]

            for surf in zone_surfs:
                # windows are removed in the idf_building_surface method if create_windows is True

                # remove the surface form the idf
                self.idf.removeidfobject(surf)

            # create new surface
            for surface in zone.BuildingSurfaces:
                self.idf_building_surface(surface, parent_zone=zone, create_windows=True)
        elif surface_method is None:
            if fenestration_method == 'update':
                for surface in zone.BuildingSurfaces:
                    for window in surface.Fenestration:
                        self.update_idf_fenestration_surface(window)

            elif fenestration_method == 'recreate':
                for surface in zone.BuildingSurfaces:
                    # get all windows for this surface
                    windows = self.idf.idfobjects['FenestrationSurface:Detailed'.upper()]
                    surf_wins = [win for win in windows if win.Building_Surface_Name == surface.Name]

                    # remove these windows form the idf
                    for win in surf_wins:
                        self.idf.removeidfobject(win)

                    # create new windows
                    for window in surface.Fenestration:
                        self.idf_fenestration_surface(window, parent_surface=surface)

        if internal_mass_method == 'update':
            for mass in zone.InternalMasses:
                self.update_idf_internal_mass(mass)
        elif internal_mass_method == 'recreate':
            # get all masses for this zone
            masses = self.idf.idfobjects['InternalMass'.upper()]
            zone_masses = [mass for mass in masses if mass.Zone_Name == zone.Name]

            for mass in zone_masses:
                # remove the mass form the idf
                self.idf.removeidfobject(mass)

            # create new masses
            for internal_mass in zone.InternalMasses:
                self.idf_internal_mass(internal_mass, parent_zone=zone)

    def fp_zone(self, ep_zone):  # Zone

        building_surfaces = []
        internal_masses = []
        for surf_or_mass in ep_zone.zonesurfaces:
            if surf_or_mass.obj[0] == 'BuildingSurface:Detailed':
                surf = self.fp_building_surface(surf_or_mass)
                building_surfaces.append(surf)
            elif surf_or_mass.obj[0] == 'InternalMass':
                mass = self.fp_internal_mass(surf_or_mass)
                internal_masses.append(mass)

        if ep_zone.Direction_of_Relative_North != '':
            dorn = float(ep_zone.Direction_of_Relative_North)
        else:
            dorn = None

        if ep_zone.X_Origin != '':
            origin = Point(ep_zone.X_Origin, ep_zone.Y_Origin, ep_zone.Z_Origin)
        else:
            origin = None

        return Zone(
            name=ep_zone.Name,
            building_surfaces=building_surfaces,
            internal_masses=internal_masses,
            direction_of_relative_north=dorn,
            origin=origin
        )

    def idf_zone_list(self, building: Building):
        zone_kwrds = ['Zone_{}_Name'.format(n + 1) for n in range(len(building.Zones))]
        zone_names = [zone.Name for zone in building.Zones]
        zone_kwargs = {kw: name for kw, name in zip(zone_kwrds, zone_names)}
        self.idf.newidfobject(
            key='ZoneList'.upper(),
            Name=building.Name,
            **zone_kwargs
        )

    def update_idf_zone_list(self, building: Building):

        idf_zone_list = self.idf.getobject('ZoneList'.upper(), building.Name)

        # delete previous zones from idf zone list
        del idf_zone_list.fieldvalues[2:]

        # add new zones to the idf from building
        for fn, zone in zip(idf_zone_list.fieldnames[2:], building.Zones):
            setattr(idf_zone_list, fn, zone.Name)

    def to_model(self):

        # get building for general data
        ep_building = self.idf.idfobjects['Building']
        if len(ep_building) == 1:
            ep_building = ep_building[0]
        else:
            raise Exception('More than one or no Building found in idf: {bl}'.format(bl=[b.Name for b in ep_building]))

        # get zones
        zones = []
        ep_zone_list = self.idf.idfobjects['ZoneList']
        if not ep_zone_list:  # no ep_zone_list found in idf
            # all zones in the idf belong to the building
            for ep_zone in self.idf.idfobjects['Zone']:
                zone = self.fp_zone(ep_zone)
                zones.append(zone)
        elif len(ep_zone_list) == 1:
            # get zones based on ZoneList
            ep_zone_list = ep_zone_list[0]
            for zone_name in ep_zone_list.obj[2:]:
                ep_zone = self.idf.getobject('Zone', zone_name)
                zone = self.fp_zone(ep_zone)
                zones.append(zone)
        else:
            raise Exception('More than one ZoneList found in idf: {zl}'.format(zl=[z.Name for z in ep_zone_list]))

        # get all non-zone surfaces
        non_zone_surfaces = []
        for ep_shading_surf in self.idf.idfobjects['Shading:Building:Detailed']:
            nzs = self.fp_non_zone_surface(ep_shading_surf, construction=None)  # TODO assign construction somehow
            non_zone_surfaces.append(nzs)

        # constructions = {const.IuId: const for const in self.constructions.values()}
        # shadings = {shad.IuId: shad for shad in self.shadings.values()}
        # opaque_materials = {mat.IuId: mat for mat in self.opaque_materials.values()}
        # shading_materials = {mat.IuId: mat for mat in self.shading_materials.values()}
        # window_materials = {mat.IuId: mat for mat in self.window_materials.values()}

        # TODO we might need this
        # self.library.change_key(to='IuId')

        building_name = ep_building.Name
        building_function = 'residential'  # TODO assign building function somehow

        return Building(
            name=building_name,
            zones=zones,
            non_zone_surfaces=non_zone_surfaces,
            library=self.library,
            # constructions=constructions,
            # shadings=shadings,
            # opaque_materials=opaque_materials,
            # shading_materials=shading_materials,
            # window_materials=window_materials,
            building_function=building_function
            # global_geometry_rules= TODO handle original geometry rules
        )

    def update_idf(self, model: Building, update_collections: bool = True, zone_method: str = None,
                   non_zone_surf_method: str = None, fenestration_method: str = None,
                   surface_method: str = None, internal_mass_method: str = None):
        """

        :param model:
        :param update_collections: check Constructions, Shadings and Materials for update; existing items will be
            updated missing items will be created
        :param zone_method: 'recreate', 'update' or None; if None, Zones are not updated;
            if 'recreate' is used, than 'recreate' will be used for fenestration, internal mass and surfaces too
        :param non_zone_surf_method: 'recreate', 'update' or None; if None, NonZoneSurfaces are not updated;
        :param fenestration_method: 'recreate', 'update' or None; if None, Fenestration are not updated
        :param surface_method: 'recreate', 'update' or None; if None, Surfaces are not updated
        :param internal_mass_method: 'recreate', 'update' or None; if None, InternalMasses are not updated
        :return: None
        """

        # TODO purge IDF (e.g. unused materials)

        # collections
        if update_collections:
            # change default key because objects are referenced by name in idf
            if self.library.default_key != 'Name':
                self.library.change_key(to='Name')

            for construction in model.Library.constructions.values():
                if self.idf.getobject('Construction'.upper(), construction.Name) is not None:
                    self.update_idf_construction(construction)
                else:
                    self.idf_construction(construction)

            for shading in model.Library.shadings.values():
                if self.idf.getobject('WindowProperty:ShadingControl'.upper(), shading.Name) is not None:
                    self.update_idf_shading(shading)
                else:
                    self.idf_shading(shading)

            for material in model.Library.opaque_materials.values():
                if self.idf.getobject('MATERIAL'.upper(), material.Name) is not None:
                    self.update_idf_opaque_material(material)
                else:
                    self.idf_opaque_material(material)

            for material in model.Library.window_materials.values():
                if self.idf.getobject('WindowMaterial:SimpleGlazingSystem'.upper(), material.Name) is not None:
                    self.update_idf_window_material(material)
                else:
                    self.idf_window_material(material)

            for material in model.Library.shade_materials.values():
                if self.idf.getobject('WindowMaterial:Shade'.upper(), material.Name) is not None:
                    self.update_idf_shade_material(material)
                else:
                    self.idf_shade_material(material)

            for material in model.Library.blind_materials.values():
                if self.idf.getobject('WindowMaterial:Blind'.upper(), material.Name) is not None:
                    self.update_idf_blind_material(material)
                else:
                    self.idf_blind_material(material)

        # zones
        if zone_method == 'update':
            # update zone list
            idf_zone_list = self.idf.getobject('ZoneList'.upper(), model.Name)

            if idf_zone_list is not None:
                self.update_idf_zone_list(building=model)
            else:
                self.idf_zone_list(building=model)

            # update zones
            for zone in model.Zones:
                if self.idf.getobject('Zone'.upper(), zone.Name) is not None:
                    self.update_idf_zone(zone, fenestration_method=fenestration_method, surface_method=surface_method,
                                         internal_mass_method=internal_mass_method)
                else:
                    # create new zone if it is not in the idf
                    self.idf_zone(
                        zone,
                        create_surfaces=False if surface_method is None else True,
                        create_internal_masses=False if internal_mass_method is None else True,
                        create_fenestration=False if fenestration_method is None else True
                    )

        elif zone_method == 'recreate':
            # update zone list
            idf_zone_list = self.idf.getobject('ZoneList'.upper(), model.Name)

            if idf_zone_list is not None:
                self.update_idf_zone_list(building=model)
            else:
                self.idf_zone_list(building=model)

            for zone in model.Zones:
                # make sure to delete existing zone
                idf_zone = self.idf.getobject('Zone'.upper(), zone.Name)
                self.idf.removeidfobject(idf_zone)

                # create new zone
                self.idf_zone(zone, create_surfaces=True, create_internal_masses=True, create_fenestration=True)

        elif zone_method is None:
            # use the update_idf_zone method, because it does nothing with the zone itself,
            for zone in model.Zones:
                self.update_idf_zone(zone, fenestration_method=fenestration_method, surface_method=surface_method,
                                     internal_mass_method=internal_mass_method)

        # nonZoneSurfaces
        if non_zone_surf_method == 'update':
            for surf in model.NonZoneSurfaces:
                self.update_idf_non_zone_surface(surf)
        elif non_zone_surf_method == 'recreate':
            # remove all non_zone_surfaces from idf
            self.idf.idfobjects['Shading:Building:Detailed'.upper()] = []

            # create new surfaces
            for surface in model.NonZoneSurfaces:
                self.idf_non_zone_surface(surface)

import logging
from pathlib import Path
from json import JSONDecodeError
from typing import Union
import uuid
import re

import olca
import pandas as pd
import sqlalchemy
import requests

logger = logging.getLogger(__name__)


class OpenLCA:
    """
    Make openLCA calculations on a server through http requests
    """
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        if not host.startswith('http'):
            self.host = 'http://' + self.host
        self.url = '{host}:{port}'.format(host=self.host, port=self.port)

        # Table containing information about processes to update when localizing the product system
        self.energy_updates_data = None

    @property
    def energy_updates_data(self) -> pd.DataFrame:
        """
        DataFrame containing all processes to update in the product system
        :return:
        """
        return self._energy_updates_data

    @energy_updates_data.setter
    def energy_updates_data(self, source: Union[str, Path, pd.DataFrame]):
        if isinstance(source, str):
            # Read from path
            self._energy_updates_data = pd.read_csv(source, index_col='code')
        elif isinstance(source, Path):
            self._energy_updates_data = pd.read_csv(str(source), index_col='code')
        elif isinstance(source, pd.DataFrame):
            self._energy_updates_data = source

    def setup_server(self, target_country_id: str = None) -> str:
        """
        Setup data on server side
        :param target_country_id: openLCA id of the target country for localizations
        :return:
        """
        url = self.url + '/setup'

        logger.info('Setting OpenLCA server on server')
        data = self.energy_updates_data.to_json(orient='split')
        params = {}
        if target_country_id is not None:
            params['target_loc_id'] = target_country_id
        response = requests.post(url=url, params=params, data=data)
        return response.text

    def run(self, create: bool = False, localize: bool = False, calculate: bool = False, save: bool = False,
            process_id: str = None, system_name: str = None, system_id: str = None, method_id: str = None):
        """

        :param create:
        :param localize:
        :param calculate:
        :param save:
        :param process_id:
        :param system_name:
        :param system_id:
        :param method_id:
        :return:
        """
        url = self.url + '/run'

        mode = ''  # cre-loc-cal-save
        if create:
            if process_id is None or system_name is None:
                raise Exception('Process id and system name is required for system creation')
            mode += 'cre'
        if localize:
            if system_id is None and not create:
                raise Exception('System id or create option is required for localization')
            if not mode == '':
                mode += '-'
            mode += 'loc'
        if calculate:
            if system_id is None and not create or method_id is None:
                raise Exception('System id or create option and method id is required for calculation')
            if not mode == '':
                mode += '-'
            mode += 'cal'
        if save:
            if not mode == '':
                mode += '-'
            mode += 'save'

        params = {
            'mode': mode,
            'process_id': process_id,
            'system_name': system_name,
            'system_id': system_id,
            'method_id': method_id
        }
        logger.info('Sending request to OpenLCA server with mode: {}'.format(mode))
        response = requests.get(url=url, params=params)
        if calculate:
            try:
                result = response.json()
                return result
            except JSONDecodeError:
                return response.text
        else:
            return response.text


class OpenLCAIpc:
    """
    openLCA ipc implementation
    """
    def __init__(self, host: str, port: int,
                 db_host: str = None, db_port: int = None, db_user: str = None, db_pwd: str = None,
                 db_name: str = 'default'):
        self.client = olca.Client(port=10)  # a random number, will be updated in the next lines
        self._host = 'to_update'
        self._port = 10  # random
        self.host = host
        self.port = port
        if db_host is not None:
            connection_string = 'postgresql://{user}:{pw}@{host}:{port}'.format(
                user=db_user,
                pw=db_pwd,
                host=db_host,
                port=db_port
            )
            self.impact_db = sqlalchemy.create_engine(connection_string)
            self.db_name = db_name
        else:
            self.impact_db = None

        # Table containing information about processes to update when localizing the product system
        self.energy_updates_data = None

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, url: str):
        if url.startswith('http'):
            self._host = url
        else:
            self._host = 'http://{u}'.format(u=url)
        self.client.url = '{h}:{p}'.format(h=self.host, p=self.port)

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, number: int):
        self._port = number
        self.client.url = '{h}:{p}'.format(h=self.host, p=self.port)

    @property
    def energy_updates_data(self) -> pd.DataFrame:
        """
        DataFrame containing all processes to update in the product system
        :return:
        """
        return self._energy_updates_data

    @energy_updates_data.setter
    def energy_updates_data(self, source: Union[str, Path, pd.DataFrame]):
        if isinstance(source, str):
            # Read from path
            self._energy_updates_data = pd.read_csv(source, index_col='code')
        elif isinstance(source, Path):
            self._energy_updates_data = pd.read_csv(str(source), index_col='code')
        elif isinstance(source, pd.DataFrame):
            self._energy_updates_data = source

    def backup_process(self, process_id: str):
        """
        Backup an existing process in the openLCA database
        :param process_id: the id of the process in openLCA
        :return: None
        """
        process = self.client.get(olca.Process, process_id)
        backup = olca.Process()
        backup.from_json(process.to_json())
        backup.id = str(uuid.uuid4())
        backup.name = backup.name + ' backup'
        self.client.insert(backup)

    def localize_product_system(self, new_location: str, year: int = None,
                                system_ref: olca.Ref = None, system_name: str = None, system_id: str = None) -> str:
        """
        Update product system with the energy changes described in the energy_update_data

        :param new_location: OpenLCA country code of the new location - must be used in the
            Energy Updates Data ("Location new" column)
        :param year: adaptation year of the product system - must be used in the
            Energy Updates Data ("Year" column)
        :param system_ref: olca Ref to product system
        :param system_name: name of the product system in the database
        :param system_id: id of the product system in the database
        One of the above three is required - priority order is same as argument order
        :return: product system ref id
        """
        if system_ref is not None:
            system_id = system_ref.id
        elif system_name is not None:
            logger.debug('Finding product system: {sn}'.format(sn=system_name))
            system_ref = self.client.find(olca.ProductSystem, system_name)
            system_id = system_ref.id
        elif system_id is not None:
            system_id = system_id
        else:
            raise Exception('Please provide any of the following: system_ref / system_name / system_id')

        logger.debug('Getting product system from OpenLCA')
        system = self.client.get(olca.ProductSystem, system_id)

        original_location = system.reference_process.location

        if year is None:
            year_text = 'timeless'
        else:
            year_text = year

        # check if it needs localization
        logger.debug('Checking system localization')
        adaptation_text = ' - {l} {y}'.format(l=new_location, y=year_text)
        localized = False
        if original_location == new_location:
            logger.debug('System is a local system')
            if year is None:
                localized = True
            elif adaptation_text in system.name:
                logger.debug('System already adapted to: {l} {y}'.format(l=new_location, y=year_text))
                localized = True
        elif adaptation_text in system.name:
            logger.debug('System already adapted to: {loc} {y}'.format(loc=new_location, y=year))
            localized = True

        if localized:
            if adaptation_text not in system.name:
                system.name = system.name + adaptation_text
                logger.info('Updating product system name in OpenLCA')
                self.client.update(system)
            return system.id

        # continue if we really need adaptation
        message = 'Adapting product system: {sn} - {loc} -> {nloc} ({y})'.format(sn=system.name,
                                                                                 loc=original_location,
                                                                                 nloc=new_location,
                                                                                 y=year_text)
        logger.info(message)

        if self.energy_updates_data is None:
            raise Exception('Please set energy_updates_data attribute first before running localization')

        # select rows corresponding the new location and year
        location_filter = self.energy_updates_data['Location new'] == new_location
        if year is not None:
            year_filter = self.energy_updates_data['Year'].astype('float') == float(year)
        else:
            year_filter = self.energy_updates_data['Year'].isna()
        energy_updates = self.energy_updates_data[location_filter & year_filter]

        if original_location not in energy_updates.columns:
            message = 'Cannot update product system of original location: {loc} and year: {y}'.format(
                loc=original_location, y=year_text)
            message += ' Please provide data for the location in the energy_updates_data'
            raise Exception(message)

        updates = energy_updates[energy_updates[original_location] == 'update'].index.values
        logger.debug('Number of updates: {n}'.format(n=len(updates)))

        def add_process(process: olca.Process):
            """
            helper function to add a process and its process links to the product system
            # TODO this is very nice, but failed to produce a valid product system that can be calculated!
            """

            logger.debug('    Checking for new process in systems processes: {n}'.format(n=process.name))

            sys_process_ids = [proc.id for proc in system.processes]
            if process.id in sys_process_ids:
                logger.debug('    Process found')
            else:
                # we still don't know if the upstream processes of the newly added process are contained
                # in the product system or not, practically we'd need to rebuild the system
                # so instead we use recursion here below! :-)

                # add the process to the system
                logger.debug('    Adding process to systems processes')
                process_new_ref = olca.ProcessRef()
                process_new_ref.id = process.id
                process_new_ref.name = process.name
                process_new_ref.category_path = process.category.category_path + [process.category.name]
                process_new_ref.location = process.location.code
                process_new_ref.process_type = process.process_type
                system.processes.append(process_new_ref)

                # we need to add all the new process links from this new process
                logger.debug('    Adding exchanges to systems process_links')
                for exchange in process.exchanges:
                    if exchange.input and exchange.flow.flow_type == olca.FlowType.PRODUCT_FLOW:
                        # create new process_link
                        new_link = olca.ProcessLink()
                        new_link.provider = exchange.default_provider
                        new_link.flow = exchange.flow
                        new_link.process = process_new_ref
                        new_link.exchange = exchange
                        # add new link to system
                        system.process_links.append(new_link)

                        # check default provider and add to system:
                        prov = self.client.get(olca.Process, exchange.default_provider.id)
                        add_process(prov)

        for change_code in updates:

            process_old_id = energy_updates.loc[change_code, 'ID old']
            process_old_name = energy_updates.loc[change_code, 'Process Name old']
            process_old_loc = energy_updates.loc[change_code, 'Location old']
            process_new_id = energy_updates.loc[change_code, 'ID new']
            process_new_loc = energy_updates.loc[change_code, 'Location new']

            message = '    Updating: {name} - {loc} -> {nloc} ({y})'.format(name=process_old_name,
                                                                            loc=process_old_loc,
                                                                            nloc=process_new_loc,
                                                                            y=year_text)
            logger.debug(message)

            process_new = self.client.get(olca.Process, process_new_id)

            add_process(process_new)

            logger.debug('    Creating new provider from Process')
            new_provider = olca.Ref()
            new_provider.id = process_new.id
            new_provider.name = process_new.name
            new_provider.category_path = process_new.category.category_path + [process_new.category.name]

            logger.debug('    Finding process links (id: {id})'.format(id=process_old_id))
            for link in system.process_links:
                if link.provider.id == process_old_id:
                    logger.debug('        Updating process link: {ln}'.format(ln=link.process.name))
                    link.provider = new_provider

        # log adaptation to product systems description
        description_text = 'Adapted dataset: (location): {l} (year): {y} '.format(l=new_location, y=year_text)

        if system.description is not None:
            system.description = description_text + ' - ' + system.description
        else:
            system.description = description_text

        # update the name of the system too
        name_text = ' - {l} {y}'.format(l=new_location, y=year_text)

        system.name = system.name + name_text

        logger.info('Updating product system in OpenLCA')
        self.client.update(system)

        logger.debug('Finished updating')
        return system.id

    def get_impact_methods(self) -> pd.DataFrame:
        """
        List all Impact assessment methods from the database
        :return: DataFrame with columns Name, Id
        """
        methods = [[method.name, method.id] for method in self.client.get_descriptors(olca.ImpactMethod)]
        df = pd.DataFrame(data=methods, columns=['Name', 'Id'])
        return df

    def get_impact_method(self, method_name: str = None, method_id: str = None) -> olca.ImpactMethod:
        """
        Get impact method for id or name
        :param method_name: Name of the impact assessment method in the database
        :param method_id: Id of the impact assessment method in the database
        :return: olca ImpactMethod
        """

        if method_name is not None:
            method_ref = self.client.find(olca.ImpactMethod, method_name)
            method = self.client.get(olca.ImpactMethod, method_ref.id)
        elif method_id is not None:
            method = self.client.get(olca.ImpactMethod, method_id)
        else:
            raise Exception('Please provide either "method_name" or "method_id"')

        return method

    def calculate_product_system(self, method: olca.ImpactMethod,
                                 system_ref: olca.Ref = None, system_name: str = None, system_id: str = None,
                                 localization: str = None, year: int = None) -> pd.Series:
        """
        Calculate impacts of a product system with optional localization

        :param method: the impact assessment method to calculate with

        :param system_ref: olca Ref to product system
        :param system_name: name of the product system in the database
        :param system_id: id of the product system in the database
        One of the above three is required - priority order is same as argument order

        :param localization: OpenLCA country code of the localization; if None, no localizations are made
        :param year: year of adaptation, if None, "timeless" localization will be made
        :return: pandas Series with all impact categories
        """

        if system_ref is not None:
            system_id = system_ref.id
        elif system_name is not None:
            logger.debug('Finding product system: {sn}'.format(sn=system_name))
            system_ref = self.client.find(olca.ProductSystem, system_name)
            system_id = system_ref.id
        elif system_id is not None:
            system_id = system_id
        else:
            raise Exception('Please provide any of the following: system_ref / system_name / system_id')

        if year is None:
            year_text = 'timeless'
            year_num = '-1'
        else:
            year_text = year
            year_num = year

        # check existing result in database
        table_name = self.db_name + '_' + method.id.replace('-', '_') + '_res'
        query = 'SELECT * FROM {tbl}'.format(
            tbl=table_name,
        )
        # if table exists for the impact assessment method
        if self.impact_db.has_table(table_name):
            logger.debug('Checking for calculation results in database')
            calculated_results = pd.read_sql_query(query, self.impact_db)
            # if any of the calculated systems correspond to the requested
            sys_result = calculated_results[calculated_results['SystemId'] == system_id]
            if len(sys_result) > 0:
                if localization is not None:
                    location_text = localization
                else:
                    location_text = '-'
                location_filter = sys_result['Localization'] == location_text
                year_filter = sys_result['Year'] == year_num

                sys_result = sys_result[location_filter & year_filter]
                logger.debug('Number of systems found in database: {}'.format(len(sys_result)))

                if len(sys_result) > 0:
                    logger.info('Calculation result found in database')
                    # return results without metadata
                    meta = ['SystemId', 'SystemName', 'RefProcessId', 'RefProcessName', 'Localization', 'Year']
                    result_df = sys_result.drop(columns=meta)
                    # if only one result is present, simplify to series, else it will return a DataFrame
                    result_series = result_df.squeeze()
                    return result_series

        logger.debug('Getting product system')
        system = self.client.get(olca.ProductSystem, system_id)

        if localization is not None:
            adaptation_text = ' - {loc} {y}'.format(loc=localization, y=year_text)

            logger.debug('Checking system localization based on name')
            if adaptation_text in system.name:
                logger.debug('System already adapted to: {loc} {y}'.format(loc=localization, y=year))
            else:
                logger.debug('Updating system: {sn} -> {lo} {y}'.format(sn=system.name, lo=localization, y=year_text))
                system_id = self.localize_product_system(new_location=localization, system_id=system_id, year=year)
                logger.debug('Reloading product system')
                system = self.client.get(olca.ProductSystem, system_id)

        logger.info('Calculating system: {sn}'.format(sn=system.name))

        logger.debug('Setting up calculation')
        setup = olca.CalculationSetup()
        setup.calculation_type = olca.CalculationType.SIMPLE_CALCULATION
        setup.impact_method = method
        setup.product_system = system

        setup.amount = 1.0

        # calculate the result
        logger.debug('Calculating...')
        result = self.client.calculate(setup)

        logger.debug('Generating result table')
        # replace special characters in impact category names
        impacts = {re.sub('[^a-zA-Z0-9]', '_', ir.impact_category.name): ir.value
                   for ir in result.impact_results}

        result_series = pd.Series(data=impacts)

        logger.debug('Disposing result')
        self.client.dispose(result)

        if self.impact_db is not None:
            # Add entry to the database
            result_to_db = result_series.copy()
            result_to_db['SystemId'] = system_id
            result_to_db['SystemName'] = system.name
            result_to_db['RefProcessId'] = system.reference_process.id
            result_to_db['RefProcessName'] = system.reference_process.name
            if localization is None:
                result_to_db['Localization'] = '-'
            else:
                result_to_db['Localization'] = localization
            result_to_db['Year'] = year_num
            result_frame = result_to_db.to_frame().transpose()
            result_frame.to_sql(name=table_name, con=self.impact_db, if_exists='append', index=False)

        return result_series

    def create_product_system(self, process_ref: olca.ProcessRef = None, process_name: str = None,
                              process_id: str = None, localization: str = None, year: int = None) -> str:
        """
        Create product system from process
        :param process_ref: olca ProcessRef to process
        :param process_name: name of the process in the database
        :param process_id: id of the process in the database
        One of the above three is required - priority order is same as argument order
        :param localization: OpenLCA country code of target localization (if any)
        :param year: target year for adaptation (if any)
        :return: Product System Id
        """
        if process_ref is not None:
            process_id = process_ref.id
        elif process_name is not None:
            logger.debug('Finding product system: {sn}'.format(sn=process_name))
            process_ref = self.client.find(olca.Process, process_name)
            process_id = process_ref.id
        elif process_id is not None:
            process_id = process_id
        else:
            raise Exception('Please provide any of the following: process_ref / process_name / process_id')

        # check existing product systems
        if process_name is None:
            if process_ref is None:
                process_name = self.client.get(olca.Process, process_id).name
            else:
                process_name = process_ref.name

        if year is None:
            year_text = 'timeless'
        else:
            year_text = year

        p_systems = {ps.name: ps.id
                     for ps in self.client.get_descriptors(olca.ProductSystem)
                     if process_name in ps.name}
        if p_systems:  # non-empty dict
            # if localization is needed, only localized systems will be checked
            if localization is not None:
                adaptation_text = ' - {l} {y}'.format(l=localization, y=year_text)
                for ps_name, ps_id in p_systems.items():
                    if adaptation_text in ps_name:
                        # if year adaptation is needed, check existing ones for the year
                        message = 'Adapted ({c} {y}) product system already exists'.format(c=localization, y=year_text)
                        message += ' with id: {id}'.format(id=ps_id)
                        logger.info(message)
                        return ps_id
            else:
                # if no adapted system needed, check systems only with no localization
                for ps_name, ps_id in p_systems.items():
                    if ' - ' not in ps_name:  # delimiter sign before localization notation
                        logger.info('Product system already exists with id: {id}'.format(id=ps_id))
                        # return the first not localized system
                        return ps_id

        # if system does not exist, lets create it

        # This is not implemented yet in olca-ipc module, so generate the request manually
        def olca_post(client: olca.Client, method: str, params) -> dict:
            req = {
                'jsonrpc': '2.0',
                'id': client.next_id,
                'method': method,
                'params': params
            }
            client.next_id += 1
            resp = requests.post(client.url, json=req).json()  # type: dict
            err = resp.get('error')  # type: dict
            if err is not None:
                raise Exception('%i: %s' % (err.get('code'), err.get('message')))
            result = resp.get('result')
            if result is None:
                raise Exception(
                    'No error and no result: invalid JSON-RPC response')
            return result

        request_params = {
            "processId": process_id,
            # 'preferredType': "",  # UNIT_PROCESS (default), LCI_RESULT
            # 'providerLinking': ""  # PREFER (default), IGNORE, ONLY
        }
        logger.info('Creating product system for id: {id}'.format(id=process_id))
        response = olca_post(self.client, 'create/product_system', request_params)

        system_ref_id = response['@id']
        logger.info('Product system created with id: {id}'.format(id=system_ref_id))

        return system_ref_id

    def get_calculated_results(self, impact_method: olca.ImpactMethod = None, table_name=None):
        """
        Retrieve calculated impact results from database
        :param impact_method:
        :return:
        """

        if impact_method is None and table_name is None:
            table_names = self.impact_db.table_names()
            methods = [tn for tn in table_names if self.db_name in tn]
            return methods
        else:
            if table_name is None:
                table_name = self.db_name + '_' + impact_method.id.replace('-', '_') + '_res'
            query = 'SELECT * FROM {tbl}'.format(
                tbl=table_name,
            )

            if not self.impact_db.has_table(table_name):
                return 'No table found for name: {n}'.format(n=table_name)

            result = pd.read_sql_query(query, self.impact_db)
            return result

    def clear_table(self, table_name: str):
        """
        Delete existing table from impact database
        :param table_name: the name of the table in the database
        :return: message as string
        """
        if not self.impact_db.has_table(table_name):
            return 'No table found for name: {n}'.format(n=table_name)
        logger.info('Table {tn} will be deleted'.format(tn=table_name))
        if input('Are you sure? (y/n): ') == 'y':

            query = 'DROP TABLE "{n}"'.format(n=table_name)
            self.impact_db.execute(query)
            return 'Table {n} has been deleted'.format(n=table_name)
        else:
            return 'Cancelled'

    def clear_result(self, table_name: str, ref_process_id: str):
        """
        Delete existing result within table from impact database
        :param table_name: the name of the table in the database
        :param ref_process_id: the id of the reference process
        :return: message as string
        """
        if input('Are you sure? (y/n): ') == 'y':
            if not self.impact_db.has_table(table_name):
                return 'No table found for name: {n}'.format(n=table_name)

            query = 'DELETE FROM "{n}" WHERE "RefProcessId"=\'{pid}\''.format(n=table_name, pid=ref_process_id)
            self.impact_db.execute(query)
            return 'Result has been deleted'.format(n=table_name)
        else:
            return 'Cancelled'
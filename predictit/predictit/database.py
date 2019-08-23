import pandas as pd
import pyodbc
from sqlalchemy import create_engine
import urllib
from . import config

def database_load(server=config.server, database=config.database, freq='D', index_col = 'DateBK', data_limit=2000, last=1):
    """Load database into dataframe and create datetime index
    
    Keyword Arguments:
        server {string} -- Name of server (default: {config.server})
        database {string} -- Name of database (default: {config.database})
        freq {Datetime frequency} -- For example days 'D' or hours 'H' (default: {'D'})
        index_col {int} -- Index of predicted column (default: {'DateBK'})
        data_limit {int} -- Max lengt of data (default: {2000})
        last {bool} -- Include or not last value (default: {1})
    """
    server = 'SERVER={};'.format(server)
    database = 'DATABASE={};'.format(database)
    sql_params = r'DRIVER={ODBC Driver 13 for SQL Server};' + server + database + 'Trusted_Connection=yes;'

    sql_conn = pyodbc.connect(sql_params)

    columns = '''   D.[DateBK],
                    D.[IsoWeekYear]'''
    if freq == 'M':
        columns = columns + ''',
                    D.[MonthNumberOfYear]'''

    if freq == 'D':
        columns = columns + ''',
                    D.[MonthNumberOfYear],
                    D.[DayNumberOfMonth]'''

    if freq == 'H':
        columns = columns + ''',
                    D.[MonthNumberOfYear],
                    D.[DayNumberOfMonth],
                    D.[HourOfDay]'''

    columns_desc = '''   D.[DateBK] DESC,
                        D.[IsoWeekYear] DESC'''
    if freq == 'M':
        columns_desc = columns_desc + ''',
                    D.[MonthNumberOfYear] DESC'''

    if freq == 'D':
        columns_desc = columns_desc + ''',
                    D.[MonthNumberOfYear] DESC,
                    D.[DayNumberOfMonth] DESC'''

    if freq == 'H':
        columns_desc = columns_desc + ''',
                    D.[MonthNumberOfYear] DESC,
                    D.[DayNumberOfMonth] DESC,
                    D.[HourOfDay] DESC'''

    query = '''

        SELECT TOP ({})
            {col},
            sum([Number]) SumNumber,
            sum([Duration]) SumDuration


        FROM [dbo].[FactProduction] F
            INNER JOIN dbo.DimDateTime D
            ON F.DimDateTimeId = D.DimDateTimeId

        WHERE   DimScenarioId = 1
                and     DimProductionEventId = 1
                and     DimOperationOutId = 69

        GROUP BY
            {col}

        ORDER BY
            {col_desc}'''.format(data_limit,
             col=columns, col_desc=columns_desc)

    df = pd.read_sql(query, sql_conn)
    if freq == 'H':
        df['datetime'] = df['DateBK'].astype('str') + '  ' + df['HourOfDay'].astype('str') + ':00'
        df.drop('DateBK', 1, inplace=True)
        df.set_index('datetime', drop=True, inplace=True)
    else:
        df.set_index('DateBK', drop=True, inplace=True)
        
    print(query)

    dates = ['IsoWeekYear', 'MonthNumberOfYear', 'DayNumberOfMonth', 'HourOfDay']
    dates_columns = df.columns
    used_dates = [c for c in dates if c in dates_columns]
    df.drop(used_dates, axis=1, inplace=True)

    if last:
        df_not_last = df.iloc[::-1]

    else:
        df_not_last = df.iloc[1:-1, :]
        df_not_last = df_not_last.iloc[::-1]

    return df_not_last


def database_deploy(last_date, sum_number, sum_duration, freq='D'):
    """Deploy dataframe to SQL server. Differ on concrete database - necessary to setup
    
    Arguments:
        last_date {date} -- Last date of data
        sum_number {Values to deploy} -- One of predicted columns
        sum_duration {Values to deploy} -- One of predicted columns
        freq {str} -- Datetime frequency (default: {'D'})
    """

    lenght = len(sum_number)

    dataframe_to_sql = pd.DataFrame([])
    dataframe_to_sql['EventStart'] = pd.date_range(start=last_date, periods=lenght+1, freq=freq)
    dataframe_to_sql = dataframe_to_sql.iloc[1:]

    dataframe_to_sql['DimDateId'] = dataframe_to_sql['EventStart'].dt.date
    dataframe_to_sql['DimTimeId'] = dataframe_to_sql['EventStart'].dt.time
    dataframe_to_sql['DimShiftOrigId'] = [-1] * lenght
    dataframe_to_sql['DimOperationOutBk'] = ['K1'] * lenght
    dataframe_to_sql['DimProductionEventBk'] = [-1000] * lenght
    dataframe_to_sql['DimProductOutBk'] = [-1] * lenght
    dataframe_to_sql['DimEmployeeCode'] = ['D'] * lenght
    dataframe_to_sql['DimOrderBk'] = [-1] * lenght
    dataframe_to_sql['DimScenarioBk'] = ['Prediction {}'.format(freq)] * lenght

    dataframe_to_sql['Number'] = sum_number

    dataframe_to_sql['Duration'] = sum_duration

    dataframe_to_sql['MaxCycle'] = [0] * lenght
    dataframe_to_sql['DescriptionCze'] = [''] * lenght
    dataframe_to_sql['DescriptionEng'] = [''] * lenght
    dataframe_to_sql['DataFlowLogInsertId'] = [35] * lenght

    params = urllib.parse.quote_plus(r'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes'.format(driver=r'{SQL Server}', server=config.server, database=config.database))
    conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)

    engine = create_engine(conn_str)
    dataframe_to_sql.to_sql(name='FactProduction',con=engine, schema='Stage', if_exists='append', index=False)

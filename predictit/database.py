"""Module include two functions: database_load and database_deploy. First download data from database - it's necessary to set up connect credentials in Config and edit query!
The database_deploy than use predicted values and deploy it to database server.

"""

import pandas as pd


def database_load(server, database, freq='D', data_limit=2000):
    """Load database into dataframe and create datetime index. !!! This function have to be change for every particular database !!!

    Args:
        server (string, optional): Name of server.
        database (str, optional): Name of database.
        freq (str, optional): For example days 'D' or hours 'H'. Defaults to 'D'.
        data_limit (int, optional): Max lengt of data. Defaults to 2000.

    Returns:
        pd.DataFrame: Dataframe with data from database based on input SQL query.

    """

    import pyodbc

    server = 'SERVER={};'.format(server)
    database = 'DATABASE={};'.format(database)
    sql_params = r'DRIVER={ODBC Driver 13 for SQL Server};' + server + database + 'Trusted_Connection=yes;'

    sql_conn = pyodbc.connect(sql_params)

    columns = '''   D.[DateBK],
                    D.[IsoWeekYear]'''
    if freq == 'D':
        columns += ''',
                    D.[MonthNumberOfYear],
                    D.[DayNumberOfMonth]'''


    elif freq == 'H':
        columns += ''',
                    D.[MonthNumberOfYear],
                    D.[DayNumberOfMonth],
                    D.[HourOfDay]'''


    elif freq == 'M':
        columns += ''',
                    D.[MonthNumberOfYear]'''

    columns_desc = '''   D.[DateBK] DESC,
                        D.[IsoWeekYear] DESC'''
    if freq == 'D':
        columns_desc += ''',
                    D.[MonthNumberOfYear] DESC,
                    D.[DayNumberOfMonth] DESC'''


    elif freq == 'H':
        columns_desc += ''',
                    D.[MonthNumberOfYear] DESC,
                    D.[DayNumberOfMonth] DESC,
                    D.[HourOfDay] DESC'''


    elif freq == 'M':
        columns_desc += ''',
                    D.[MonthNumberOfYear] DESC'''

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
            {col_desc}'''.format(data_limit, col=columns, col_desc=columns_desc)

    df = pd.read_sql(query, sql_conn)
    if freq == 'H':
        df['datetime'] = df['DateBK'].astype('str') + '  ' + df['HourOfDay'].astype('str') + ':00'
        df.drop('DateBK', 1, inplace=True)
        df.set_index('datetime', drop=True, inplace=True)
    else:
        df.set_index('DateBK', drop=True, inplace=True)

    dates = ['IsoWeekYear', 'MonthNumberOfYear', 'DayNumberOfMonth', 'HourOfDay']
    dates_columns = df.columns
    used_dates = [c for c in dates if c in dates_columns]
    df.drop(used_dates, axis=1, inplace=True)

    df = df.iloc[1:, :]
    df = df.iloc[::-1]

    return df


def database_deploy(server, database, last_date, sum_number, sum_duration, freq='D'):
    """Deploy dataframe to SQL server. !!! Differ on concrete database - necessary to setup for each database.

    Args:
        last_date (date): Last date of data.
        sum_number (Values to deploy):  One of predicted columns.
        sum_duration (Values to deploy):  One of predicted columns.
        freq (str, optional):  Datetime frequency. Defaults to 'D'.

    """

    from sqlalchemy import create_engine
    import urllib

    length = len(sum_number)

    dataframe_to_sql = pd.DataFrame([])
    dataframe_to_sql['EventStart'] = pd.date_range(start=last_date, periods=length + 1, freq=freq)
    dataframe_to_sql = dataframe_to_sql.iloc[1:]

    dataframe_to_sql['DimDateId'] = dataframe_to_sql['EventStart'].dt.date
    dataframe_to_sql['DimTimeId'] = dataframe_to_sql['EventStart'].dt.time
    dataframe_to_sql['DimShiftOrigId'] = [-1] * length
    dataframe_to_sql['DimOperationOutBk'] = ['K1'] * length
    dataframe_to_sql['DimProductionEventBk'] = [-1000] * length
    dataframe_to_sql['DimProductOutBk'] = [-1] * length
    dataframe_to_sql['DimEmployeeCode'] = ['D'] * length
    dataframe_to_sql['DimOrderBk'] = [-1] * length
    dataframe_to_sql['DimScenarioBk'] = ['Prediction {}'.format(freq)] * length

    dataframe_to_sql['Number'] = sum_number

    dataframe_to_sql['Duration'] = sum_duration

    dataframe_to_sql['MaxCycle'] = [0] * length
    dataframe_to_sql['DescriptionCze'] = [''] * length
    dataframe_to_sql['DescriptionEng'] = [''] * length
    dataframe_to_sql['DataFlowLogInsertId'] = [35] * length

    params = urllib.parse.quote_plus(r'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes'.format(driver=r'{SQL Server}', server=server, database=database))
    conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)

    engine = create_engine(conn_str)
    dataframe_to_sql.to_sql(name='FactProduction', con=engine, schema='Stage', if_exists='append', index=False)

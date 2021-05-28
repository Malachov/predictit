"""Module include two functions: database_load and database_deploy. First download data from database - it's necessary to set up
connect credentials. The database_deploy than deploy data to the database server.

It is working only for mssql server so far.
"""

# Lazy imports
# import pandas as pd
# import pyodbc
# from sqlalchemy import create_engine
# import urllib


def database_load(server, database, query, use_last_line, reversit):
    """Load database into dataframe and create datetime index. !!! This function have to be change for every particular database !!!

    Note:

        This is how the query could look like in python.

        >>> query_example = f'''
        ...
        ...     SELECT TOP ({data_limit})
        ...         {col},
        ...         sum([Number]) SumNumber,
        ...         sum([Duration]) SumDuration
        ...
        ...     FROM [dbo].[Table] F
        ...         INNER JOIN dbo.DimDateTime D
        ...         ON F.DimDateTimeId = D.DimDateTimeId
        ...
        ...     WHERE      contidion = 1
        ...         and    condition2 = 1
        ...         and    DimOperationOutId = 69
        ...
        ...     GROUP BY
        ...         {col}
        ...
        ...     ORDER BY
        ...         {col_desc}
        ...
        ... '''

    Args:
        server (string): Name of server.
        database (str): Name of database.
        query (str, optional): Used query.
        use_last_line (bool, optional): Last value may have not been complete yet. If False, it's removed. Defaults to False.
        reversit (bool, optional): If want to limit number of loaded lines, you can use select top and use reversed order.
            If you need to reverse it back, set this to True. Defaults to False

    Returns:
        pd.DataFrame: Dataframe with data from database based on input SQL query.

    """

    import pandas as pd
    import pyodbc

    server = "SERVER={};".format(server)
    database = "DATABASE={};".format(database)
    sql_params = r"DRIVER={ODBC Driver 13 for SQL Server};" + server + database + "Trusted_Connection=yes;"

    sql_conn = pyodbc.connect(sql_params)

    df = pd.read_sql(query, sql_conn)

    if reversit:
        df = df.iloc[::-1]

    if not use_last_line:
        df = df.iloc[:-1, :]

    return df


def database_deploy(server, database, df):
    """Deploy dataframe to SQL server.

    Args:
        server (string): Name of server.
        database (str): Name of database.
        df (pd.DataFrame): Dataframe passed to database.
    """

    from sqlalchemy import create_engine
    import urllib

    driver = "{SQL Server}"
    params = urllib.parse.quote_plus(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes"
    )

    conn_str = f"mssql+pyodbc:///?odbc_connect={params}"

    engine = create_engine(conn_str)

    df.to_sql(
        name="FactProduction",
        con=engine,
        schema="Stage",
        if_exists="append",
        index=False,
    )




DDF_TYPES_MAP = {
    'X': 'string',
    'L': 'int',
    'D': 'float',
    'T': 'date',
    'C1': 'single',
    'S': 'delimited set',
    'B': 'boolean'
}


def read_ddf(path_ddf, auto_index_tables=True):
    ''' Returns a raw version of the DDF in the form of a dict of
    pandas DataFrames (one for each table in the DDF).
    
    Parameters
    ----------
    path_ddf : string, the full path to the target DDF
    
    auto_index_tables : boolean (optional)
        if True, will set the index for all returned DataFrames using the most
        meaningful candidate column available. Columns set into the index will
        not be dropped from the DataFrame.
    
    Returns    
    ----------
    dict of pandas DataFrames
    '''
    
    # Read in the DDF (which is a sqlite file) and retain all available
    # information in the form of pandas DataFrames.
    with sqlite3.connect(path_ddf) as conn:
        ddf = {}
        ddf['sqlite_master'] = pd.read_sql(
            'SELECT * FROM sqlite_master;', 
            conn
        )    
        ddf['tables'] = {
            table_name: 
            pd.read_sql('SELECT * FROM %s;' % (table_name), conn) 
            for table_name in ddf['sqlite_master']['tbl_name'].values
            if table_name.startswith('L')
        }
        ddf['table_info'] = {
            table_name:
            pd.read_sql("PRAGMA table_info('%s');" % (table_name), conn)
            for table_name in ddf['tables'].keys()
        }
    
    # If required, set the index for the expected Dataframes that should
    # result from the above operation.
    if auto_index_tables:        
        try:
            ddf['sqlite_master'].set_index(
                ['name'], 
                drop=False,
                inplace=True 
            )
        except:
            print (
                "Couldn't set 'name' into the index for 'sqlite_master'."
            )        
        for table_name in ddf['table_info'].keys():
            try:
                ddf['table_info'][table_name].set_index(
                    ['name'],
                    drop=False,
                    inplace=True 
                )
            except:
                print (
                    "Couldn't set 'name' into the index for '%s'."
                ) % (table_name)
 
        for table_name in ddf['tables'].keys():
            index_col = 'TableName' if table_name=='Levels' else ':P0' 
            try:
                ddf['table_info'][table_name].set_index(
                    ['name'],
                    drop=False,
                    inplace=True 
                )
            except:
                print (
                    "Couldn't set '%s' into the index for the '%s' "
                    "Dataframe."
                ) % (index_col, table_name)
   
    return ddf





def merge_text_meta(left_text, right_text, overwrite=False):
    """
    Merge known text keys from right to left, add unknown text_keys.
    """
    if overwrite:
        left_text.update(right_text)
    else:
        for text_key in right_text.keys():
            if not text_key in left_text:
                left_text[text_key] = right_text[text_key]

    return left_text

def merge_values_meta(left_values, right_values, overwrite=False):
    """
    Merge known left values from right to left, add unknown values.
    """
    for val_right in right_values:
        found = False
        for i, val_left in enumerate(left_values):
            if val_left['value']==val_right['value']:
                found = True
                left_values[i]['text'] = merge_text_meta(
                    val_left['text'],
                    val_right['text'],
                    overwrite=overwrite)
        if not found:
            left_values.append(val_right)

    return left_values

def merge_column_metadata(left_column, right_column, overwrite=False):
    """
    Merge the metadata from the right column into the left column.
    """
    _compatible_types(left_column, right_column)
    left_column['text'] = merge_text_meta(
            left_column['text'],
            right_column['text'],
            overwrite=overwrite)
    if 'values' in left_column and 'values' in right_column:
        left_column['values'] = merge_values_meta(
            left_column['values'],
            right_column['values'],
            overwrite=overwrite)
    return left_column

def _compatible_types(left_column, right_column):
    l_type = left_column['type']
    r_type = right_column['type']
    if l_type == r_type: return None
    all_types = ['array', 'int', 'float', 'single', 'delimited set', 'string',
                 'date', 'time', 'boolean']
    err = {
        'array': all_types,
        'int': [
            'float', 'delimited set', 'string', 'date', 'time', 'array'],
        'float': [
            'delimited set', 'string', 'date', 'time', 'array'],
        'single': all_types,
        'delimited set': [
            'string', 'date', 'time', 'array', 'int', 'float'],
        'string': [
            'int', 'float', 'single', 'delimited set', 'date', 'time', 'array'],
        'date': [
            'int', 'float', 'single', 'delimited set', 'string', 'time', 'array'],
        'time': [
            'int', 'float', 'single', 'delimited set', 'string', 'time', 'array'],
        }
    warn = {
        'int': [
            'single'],
        'float': [
            'int', 'single'],
        'delimited set': [
            'single'],
        'string': [
            'boolean']
    }
    if r_type in err.get(l_type, all_types):
        msg = "\n'{}': Trying to merge incompatibe types: Found '{}' in left "
        msg += "and '{}' in right dataset."
        raise TypeError(msg.format(left_column['name'], l_type, r_type))
    elif r_type in warn.get(l_type, all_types):
        msg = "\n'{}': Merge inconsistent types: Found '{}' in left "
        msg += "and '{}' in right dataset."
        warnings.warn(msg.format(left_column['name'], l_type, r_type))
    else:
        msg = "\n'{}': Found '{}' in left and '{}' in right dataset."
        raise TypeError(msg.format(left_column['name'], l_type, r_type))

def _update_mask_meta(left_meta, right_meta, masks, verbose, overwrite=False):
    """
    """
    # update mask
    if not isinstance(masks, list): masks = [masks]
    for mask in masks:
        old = left_meta['masks'][mask]
        new = right_meta['masks'][mask]
        for tk, t in new['text'].items():
            if not tk in old['text'] or overwrite:
               old['text'].update({tk: t})
        for item in new['items']:
            check_source = item['source']
            check = 0
            for old_item in old['items']:
                if old_item['source'] == check_source:
                    check = 1
                    try:
                        for tk, t in item['text'].items():
                            if not tk in old_item['text'] or overwrite:
                               old_item['text'].update({tk: t})
                    except:
                        if  verbose:
                            e = "'text' meta not valid for mask {}: item {}"
                            e = e.format(mask, item['source'].split('@')[-1])
                            print '{} - skipped!'.format(e)
                        else:
                            pass
            if check == 0:
                old['items'].append(item)
                # also add these items to ``meta['sets']``
                left_meta['sets'][mask]['items'].append(item['source'])


def merge_meta(meta_left, meta_right, from_set, overwrite_text=False,
               get_cols=False, get_updates=False, verbose=True):

    if verbose:
        print '\n', 'Merging meta...'

    if from_set is None:
        from_set = 'data file'

    # Find the columns to be merged
    if from_set in meta_right['sets']:
        if verbose:
            print ("New columns will be appended in the order found in"
                   " meta['sets']['{}'].".format(from_set))

        cols = []
        masks = []
        mask_items = {}
        for item in meta_right['sets'][from_set]['items']:
            source, name = item.split('@')
            if source == 'columns':
                cols.append(name)
            elif source == 'masks':
                masks.append(name)
                for item in meta_right['masks'][name]['items']:
                    s, n = item['source'].split('@')
                    if s == 'columns':
                        cols.append(n)
                        if meta_right['masks'][name].get('values'):
                            mask_items[n] = 'lib@values@{}'.format(name)
        cols = uniquify_list(cols)

        if masks:
            for mask in masks:
                if not mask in meta_left['masks']:
                    if verbose:
                        print "Adding meta['masks']['{}']".format(mask)
                    meta_left['masks'][mask] = meta_right['masks'][mask]
                else:
                    _update_mask_meta(meta_left, meta_right, mask, verbose,
                                      overwrite=overwrite_text)

        sets = [key for key in meta_right['sets']
                if not key in meta_left['sets']]
        if sets:
            for set_name in sorted(sets):
                if verbose:
                    print "Adding meta['sets']['{}']".format(set_name)
                meta_left['sets'][set_name] = meta_right['sets'][set_name]

        for val in meta_right['lib']['values'].keys():
            if not val in meta_left['lib']['values']:
                if verbose:
                    print "Adding meta['lib']['values']['{}']".format(val)
                meta_left['lib']['values'][val] = meta_right['lib']['values'][val]
            elif val == 'ddf' or (meta_left['lib']['values'][val] ==
                 meta_right['lib']['values'][val]):
                continue
            else:
                n_values = [v['value'] for v in meta_right['lib']['values'][val]]
                o_values = [v['value'] for v in meta_left['lib']['values'][val]]
                add_values = [v for v in n_values if v not in o_values]
                if add_values:
                    for value in meta_right['lib']['values'][val]:
                        if value['value'] in add_values:
                            meta_left['lib']['values'][val].append(value)

    else:
        if verbose:
            print (
                "No '{}' set was found, new columns will be appended"
                " alphanumerically.".format(from_set)
            )
        cols = meta_right['columns'].keys().sort(key=str.lower)

    col_updates = []
    for col_name in cols:
        if verbose:
            print '...', col_name
        # store properties
        props = copy.deepcopy(
            meta_right['columns'][col_name].get('properties', {}))
        # emulate the right meta
        right_column = emulate_meta(
            meta_right,
            meta_right['columns'][col_name])
        if col_name in meta_left['columns'] and col_name in cols:
            col_updates.append(col_name)
            # emulate the left meta
            left_column = emulate_meta(
                meta_left,
                meta_left['columns'][col_name])
            # merge the eumlated metadata
            meta_left['columns'][col_name] = merge_column_metadata(
                left_column,
                right_column,
                overwrite=overwrite_text)
        else:
            # add metadata
            if right_column.get('properties'):
                right_column['properties']['merged'] = True
            else:
                right_column['properties'] = {'merged': True}

            meta_left['columns'][col_name] = right_column
        if 'properties' in meta_left['columns'][col_name]:
            meta_left['columns'][col_name]['properties'].update(props)
        if col_name in mask_items:
            meta_left['columns'][col_name]['values'] = mask_items[col_name]

    for item in meta_right['sets'][from_set]['items']:
        if not item in meta_left['sets']['data file']['items']:
            meta_left['sets']['data file']['items'].append(item)

    if get_cols and get_updates:
        return meta_left, cols, col_updates
    elif get_cols:
        return meta_left, cols
    elif get_updates:
        return meta_left, col_updates
    else:
        return meta_left

def hmerge(dataset_left, dataset_right, on=None, left_on=None, right_on=None,
           overwrite_text=False, from_set=None, merge_existing=None, verbose=True):
    """
    Merge Quantipy datasets together using an index-wise identifer.

    This function merges two Quantipy datasets (meta and data) together,
    updating variables that exist in the left dataset and appending
    others. New variables will be appended in the order indicated by
    the 'data file' set if found, otherwise they will be appended in
    alphanumeric order. This merge happend horizontally (column-wise).
    Packed kwargs will be passed on to the pandas.DataFrame.merge()
    method call, but that merge will always happen using how='left'.

    Parameters
    ----------
    dataset_left : tuple
        A tuple of the left dataset in the form (meta, data).
    dataset_right : tuple
        A tuple of the right dataset in the form (meta, data).
    on : str, default=None
        The column to use as a join key for both datasets.
    left_on : str, default=None
        The column to use as a join key for the left dataset.
    right_on : str, default=None
        The column to use as a join key for the right dataset.
    overwrite_text : bool, default=False
        If True, text_keys in the left meta that also exist in right
        meta will be overwritten instead of ignored.
    from_set : str, default=None
        Use a set defined in the right meta to control which columns are
        merged from the right dataset.
    merge_existing : str/ list of str, default None, {'all', [var_names]}
        Specify if codes should be merged for delimited sets for defined
        variables.
    verbose : bool, default=True
        Echo progress feedback to the output pane.

    Returns
    -------
    meta, data : dict, pandas.DataFrame
       Updated Quantipy dataset.
    """
    def _merge_delimited_sets(x):
        codes = []
        x = str(x).replace('nan', '')
        for c in x.split(';'):
            if not c:
                continue
            if not c in codes:
                codes.append(c)
        if not codes:
            return np.NaN
        else:
            return ';'.join(sorted(codes)) + ';'

    if all([kwarg is None for kwarg in [on, left_on, right_on]]):
        raise TypeError("You must provide a column name for either 'on' or "
                        "both 'left_on' AND 'right_on'")
    elif not on is None and not (left_on is None and right_on is None):
        raise ValueError("You cannot provide a value for both 'on' and either/"
                         "both 'left_on'/'right_on'.")
    elif on is None and (left_on is None or right_on is None):
        raise TypeError("You must provide a column name for both 'left_on' "
                        "AND 'right_on'")
    elif not on is None:
        left_on = on
        right_on = on

    meta_left = copy.deepcopy(dataset_left[0])
    data_left = dataset_left[1].copy()

    if isinstance(dataset_right, tuple): dataset_right = [dataset_right]
    for ds_right in dataset_right:
        meta_right = copy.deepcopy(ds_right[0])
        data_right = ds_right[1].copy()
        slicer = data_right[right_on].isin(data_left[left_on].values.tolist())
        data_right = data_right.loc[slicer, :]

        if verbose:
            print '\n', 'Checking metadata...'

        if from_set is None:
            from_set = 'data file'

        # Merge the right meta into the left meta
        meta_left, cols, col_updates = merge_meta(meta_left, meta_right,
                                                  from_set, overwrite_text,
                                                  True, True, verbose)

        # col_updates exception when left_on==right_on
        if left_on==right_on:
            col_updates.remove(left_on)
        if not left_on==right_on and right_on in col_updates:
            update_right_on = True
        else:
            update_right_on = False

        if verbose:
            print '\n', 'Merging data...'

        # update columns which are in left and in right data
        if col_updates:
            updata_left = data_left.copy()
            updata_left['org_idx'] = updata_left.index.tolist()
            updata_left = updata_left.set_index([left_on])[col_updates+['org_idx']]
            updata_right = data_right.set_index(
                right_on, drop=not update_right_on)[col_updates].copy()
            sets = [c for c in col_updates
                    if meta_left['columns'][c]['type'] == 'delimited set']
            non_sets = [c for c in col_updates if not c in sets]

            if verbose:
                print '------ updating data for known columns'
            updata_left.update(updata_right[non_sets])
            if merge_existing:
                for col in sets:
                    if not (merge_existing == 'all' or col in merge_existing):
                        continue
                    if verbose:
                        print "..{}".format(col)
                    updata_left[col] = updata_left[col].combine(
                        updata_right[col],
                        lambda x, y: _merge_delimited_sets(str(x)+str(y)))
            updata_left.reset_index(inplace=True)
            for col in col_updates:
                data_left[col] = updata_left[col].astype(data_left[col].dtype)

        # append completely new columns
        if verbose:
            print '------ appending new columns'
        new_cols = [col for col in cols if not col in col_updates]
        if update_right_on:
            new_cols.append(right_on)

        kwargs = {'left_on': left_on,
                  'right_on': right_on,
                  'how': 'left'}

        data_left = data_left.merge(data_right[new_cols], **kwargs)

        if update_right_on:
            new_cols.remove(right_on)
            _x = "{}_x".format(right_on)
            _y = "{}_y".format(right_on)
            data_left.rename(columns={_x: right_on}, inplace=True)
            data_left.drop(_y, axis=1, inplace=True)

        if verbose:
            for col_name in new_cols:
                print '..{}'.format(col_name)
            print '\n'

    return meta_left, data_left

def vmerge(dataset_left=None, dataset_right=None, datasets=None,
           on=None, left_on=None, right_on=None,
           row_id_name=None, left_id=None, right_id=None, row_ids=None,
           overwrite_text=False, from_set=None, reset_index=True,
           verbose=True):
    """
    Merge Quantipy datasets together by appending rows.

    This function merges two Quantipy datasets (meta and data) together,
    updating variables that exist in the left dataset and appending
    others. New variables will be appended in the order indicated by
    the 'data file' set if found, otherwise they will be appended in
    alphanumeric order. This merge happens vertically (row-wise).

    Parameters
    ----------
    dataset_left : tuple, default=None
        A tuple of the left dataset in the form (meta, data).
    dataset_right : tuple, default=None
        A tuple of the right dataset in the form (meta, data).
    datasets : list, default=None
        A list of datasets that will be iteratively sent into vmerge
        in pairs.
    on : str, default=None
        The column to use to identify unique rows in both datasets.
    left_on : str, default=None
        The column to use to identify unique in the left dataset.
    right_on : str, default=None
        The column to use to identify unique in the right dataset.
    row_id_name : str, default=None
        The named column will be filled with the ids indicated for each
        dataset, as per left_id/right_id/row_ids. If meta for the named
        column doesn't already exist a new column definition will be
        added and assigned a reductive-appropriate type.
    left_id : str/int/float, default=None
        Where the row_id_name column is not already populated for the
        dataset_left, this value will be populated.
    right_id : str/int/float, default=None
        Where the row_id_name column is not already populated for the
        dataset_right, this value will be populated.
    row_ids : list of str/int/float, default=None
        When datasets has been used, this list provides the row ids
        that will be populated in the row_id_name column for each of
        those datasets, respectively.
    overwrite_text : bool, default=False
        If True, text_keys in the left meta that also exist in right
        meta will be overwritten instead of ignored.
    from_set : str, default=None
        Use a set defined in the right meta to control which columns are
        merged from the right dataset.
    reset_index : bool, default=True
        If True pandas.DataFrame.reindex() will be applied to the merged
        dataframe.
    verbose : bool, default=True
        Echo progress feedback to the output pane.

    Returns
    -------
    meta, data : dict, pandas.DataFrame
        Updated Quantipy dataset.
    """

    if from_set is None:
        from_set = 'data file'

    if not datasets is None:
        if not isinstance(datasets, list):
            raise TypeError(
                "'datasets' must be a list.")
        if not datasets:
            raise ValueError(
                "'datasets' must be a populated list.")
        for dataset in datasets:
            if not isinstance(dataset, tuple):
                raise TypeError(
                    "The datasets in 'datasets' must be tuples.")
            if not len(dataset)==2:
                raise ValueError(
                    "The datasets in 'datasets' must be tuples with a"
                    " size of 2 (meta, data).")

        dataset_left = datasets[0]
        if row_ids:
            left_id = row_ids[0]
        for i in range(1, len(datasets)):
            dataset_right = datasets[i]
            if row_ids:
                right_id = row_ids[i]
            meta_vm, data_vm = vmerge(
                dataset_left, dataset_right,
                on=on, left_on=left_on, right_on=right_on,
                row_id_name=row_id_name, left_id=left_id, right_id=right_id,
                overwrite_text=overwrite_text, from_set=from_set,
                reset_index=reset_index,
                verbose=verbose)
            dataset_left = (meta_vm, data_vm)

        return meta_vm, data_vm

    if on is None and left_on is None and right_on is None:
        blind_append = True
    else:
        blind_append = False
        if on is None:
            if left_on is None or right_on is None:
                raise ValueError(
                    "You may not provide a value for only one of"
                    "'left_on'/'right_on'.")
        else:
            if not left_on is None or not right_on is None:
                raise ValueError(
                    "You cannot provide a value for both 'on' and either/"
                    "both 'left_on'/'right_on'.")
            left_on = on
            right_on = on

    meta_left = cpickle_copy(dataset_left[0])
    data_left = dataset_left[1].copy()

    if not blind_append:
        if not left_on in data_left.columns:
            raise KeyError(
                "'{}' not found in the left data.".format(left_on))
        if not left_on in meta_left['columns']:
            raise KeyError(
                "'{}' not found in the left meta.".format(left_on))

    meta_right = cpickle_copy(dataset_right[0])
    data_right = dataset_right[1].copy()

    if not blind_append:
        if not right_on in data_left.columns:
            raise KeyError(
                "'{}' not found in the right data.".format(right_on))
        if not right_on in meta_left['columns']:
            raise KeyError(
                "'{}' not found in the right meta.".format(right_on))

    if not row_id_name is None:
        if left_id is None and right_id is None:
            raise TypeError(
                "When indicating a 'row_id_name' you must also"
                " provide either 'left_id' or 'right_id'.")

        if row_id_name in meta_left['columns']:
            pass
            # text_key_right = meta_right['lib']['default text']
            # meta_left['columns'][row_id_name]['text'].update({
            #     text_key_right: 'vmerge row id'})
        else:
            left_id_int = isinstance(left_id, (int, np.int64))
            right_id_int = isinstance(right_id, (int, np.int64))
            if left_id_int and right_id_int:
                id_type = 'int'
            else:
                left_id_float = isinstance(left_id, (float, np.float64))
                right_id_float = isinstance(right_id, (float, np.float64))
                if (left_id_int or left_id_float) and (right_id_int or right_id_float):
                    id_type = 'float'
                    left_id = float(left_id)
                    right_id = float(right_id)
                else:
                    id_type = 'str'
                    left_id = str(left_id)
                    right_id = str(right_id)
            if verbose:
                print (
                    "'{}' was not found in the left meta so a new"
                    " column definition will be created for it. Based"
                    " on the given 'left_id' and 'right_id' types this"
                    " new column will be given the type '{}'.".format(
                        row_id_name,
                        id_type))
            text_key_left = meta_left['lib']['default text']
            text_key_right = meta_right['lib']['default text']
            meta_left['columns'][row_id_name] = {
                'name': row_id_name,
                'type': id_type,
                'text': {
                    text_key_left: 'vmerge row id',
                    text_key_right: 'vmerge row id'}}
            id_mapper = "columns@{}".format(row_id_name)
            if not id_mapper in meta_left['sets']['data file']['items']:
                meta_left['sets']['data file']['items'].append(id_mapper)

        # Add the left and right id values
        if not left_id is None:
            if row_id_name in data_left.columns:
                left_id_rows = data_left[row_id_name].isnull()
                data_left.ix[left_id_rows, row_id_name] = left_id
            else:
                data_left[row_id_name] = left_id
        if not right_id is None:
            data_right[row_id_name] = right_id

    if verbose:
        print '\n', 'Checking metadata...'

    # Merge the right meta into the left meta
    meta_left, cols, col_updates = merge_meta(
        meta_left, meta_right,
        from_set=from_set,
        overwrite_text=overwrite_text,
        get_cols=True,
        get_updates=True,
        verbose=verbose)

    if not blind_append:
        vmerge_slicer = data_right[left_on].isin(data_left[right_on])
        data_right = data_right.loc[~vmerge_slicer]

    # convert right cols to delimited set if depending left col is delimited set
    for col in data_right.columns.tolist():
        if (meta_left['columns'].get(col, {}).get('type') == 'delimited set'
            and not meta_right['columns'][col]['type'] == 'delimited set'):
            data_right[col] = data_right[col].apply(
                lambda x: str(int(x)) + ';' if not np.isnan(x) else np.NaN)

    vdata = pd.concat([
        data_left,
        data_right
    ])

    # Determine columns that should remain in the merged data
    cols_left = data_left.columns.tolist()

    col_slicer = cols_left + [
        col for col in get_columns_from_set(meta_right, from_set)
        if not col in cols_left]

    vdata = vdata[col_slicer]

    if reset_index:
        vdata.reset_index(drop=True, inplace=True)

    if verbose:
        print '\n'

    return meta_left, vdata


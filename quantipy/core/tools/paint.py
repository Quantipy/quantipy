


def paint_index(meta, index, text_key, display_names=False,
                transform_names=None, grp_text_map=None):
    single_row = len(index.values) == 1
    levels = get_index_levels(index)
    col = levels[0]
    values = list(levels[1])
    if not col in meta['columns']:
        return index
    else:
        col_text = paint_col_text(
            meta, col, text_key, display_names, transform_names)
        values_text = paint_col_values_text(
            meta, col, values, text_key, grp_text_map)

        new_index = build_multiindex_from_tuples(
            col_text,
            values_text,
            ['Question', 'Values'],
            single_row)

        return new_index

def paint_view(meta, view, text_key=None, display_names=None,
               transform_names=False, axes=['x', 'y']):

    if text_key is None: text_key = finish_text_key(meta, {})
    if display_names is None: display_names = ['x', 'y']

    is_array = any(view.meta()[axis]['is_array'] for axis in ['x', 'y'])

    if is_array:
        df = paint_array(
            meta, view, text_key, display_names, transform_names, axes)
    else:
        df = view.dataframe.copy()
        grp_text_map = view.meta()['agg']['grp_text_map']
        df = paint_dataframe(
            meta, df, text_key, display_names, transform_names, axes,
            grp_text_map)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_dataframe(meta, df, text_key=None, display_names=None,
                    transform_names=False, axes=['x', 'y'],
                    grp_text_map=None):

    if text_key is None: text_key = finish_text_key(meta, {})
    if display_names is None: display_names = ['x', 'y']

    if 'x' in axes:
        display_x_names = 'x' in display_names

        if len(df.index.levels[0])>1:
            order = []
            for x in df.index.labels[0]:
                if x not in order:
                    order.append(x)
            levels = df.index.levels[0]
            it = sorted(zip(levels, order), key=lambda x: x[1])
            df.index = pd.concat([
                paint_dataframe(
                    meta, df.ix[[level], :], text_key, display_names,
                    transform_names, 'x', grp_text_map)
                for level, _ in it],
                axis=0).index
        else:
            df.index = paint_index(
                meta, df.index, text_key['x'],
                display_x_names, transform_names, grp_text_map)

    if 'y' in axes:
        display_y_names = 'y' in display_names

        if len(df.columns.levels[0])>1:
            df.columns = pd.concat([
                paint_dataframe(
                    meta, df.ix[:, [level]], text_key, display_names,
                    transform_names, 'y', grp_text_map)
                for level in df.columns.levels[0]],
                axis=1).columns
        else:
            df.columns = paint_index(
                meta, df.columns, text_key['y'],
                display_y_names, transform_names)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array(meta, view, text_key, display_names, transform_names, axes):

    df = view.dataframe.copy()
    grp_text_map = view.meta()['agg']['grp_text_map']
    columns_on_x = view.meta()['x']['is_array']
    axes_x = {True: 'x',
              False: 'y'}

    if 'x' in axes:
        display_x_names = axes_x.get(columns_on_x) in display_names
        index = paint_array_items_index(
            meta,
            df.index if columns_on_x else df.columns,
            text_key['x'],
            display_x_names)
    if 'y' in axes:
        display_y_names = axes_x.get(not columns_on_x) in display_names
        columns = paint_array_values_index(
            meta,
            df.columns if columns_on_x else df.index,
            text_key['y'],
            display_y_names,
            grp_text_map)

    df.index = index if columns_on_x else columns
    df.columns = columns if columns_on_x else index

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_index_levels(index):

    levels = []
    idx_values = index.values
    single_row = len(idx_values)==1
    if single_row:
        unzipped = [idx_values[0]]
        levels.append(unzipped[0][0])
        levels.append([unzipped[0][1]])
    else:
        unzipped = list(zip(*index.values))
        levels.append(unzipped[0][0])
        levels.append(unzipped[1])

    return levels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_col_text(meta, col, text_key, display_names, transform_names):

    col_meta = emulate_meta(meta, meta['columns'][col])
    if display_names:
        try:
            col_name = col
            if transform_names: col_name = transform_names.get(col, col)
            col_text = '{}. {}'.format(
                col_name, get_text(col_meta['text'], text_key))
        except UnicodeEncodeError:
            col_text = '{}. {}'.format(
                col_name, qp.core.tools.dp.io.unicoder(
                    get_text(col_meta['text'], text_key),
                    like_ascii=True))
    else:
        col_text = get_text(col_meta['text'], text_key)

    return col_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_add_text_map(meta, add_text_map, text_key):

    if add_text_map is None:
        add_text_map = {}
    else:
        try:
            add_text_map = {
                key: get_text(text, text_key)
                for key, text in add_text_map.items()}
        except UnicodeEncodeError:
            add_text_map = {
                key: qp.core.tools.dp.io.unicoder(
                    get_text(text, text_key, like_ascii=True))
                for key, text in add_text_map.items()}

    return add_text_map

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_col_values_text(meta, col, values, text_key, add_text_map=None):
    add_text_map = paint_add_text_map(meta, add_text_map, text_key)
    num_col = meta['columns'][col]['type'] in ['int', 'float']
    try:
        has_all = 'All' in values
        if has_all: values.remove('All')
        if not num_col:
            try:
                values_map = {
                    val['value']: get_text(val['text'], text_key)
                    for val in meta['columns'][col]['values']}
            except UnicodeEncodeError:
                values_map = {
                    val['value']: qp.core.tools.dp.io.unicoder(
                        get_text(val['text'], text_key, like_ascii=True))
                    for val in meta['columns'][col]['values']}
        else:
            values_map = {}
        values_map.update(add_text_map)
        values_text = [values_map[v] for v in values]
    except KeyError:
        values_text = values
    except ValueError:
        values_text = values
    if has_all:
        values_text = ['All'] + values_text

    return values_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_mask_text(meta, mask, text_key, display_names):

    mask_meta = meta['masks'][mask]
    if display_names:
        try:
            mask_text = '{}. {}'.format(
                mask, get_text(mask_meta['text'], text_key))
        except UnicodeEncodeError:
            mask_text = '{}. {}'.format(
                mask, qp.core.tools.dp.io.unicoder(
                    get_text(mask_meta['text'], text_key),
                    like_ascii=True))
    else:
        mask_text = get_text(mask_meta['text'], text_key)

    return mask_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_items_text(meta, mask, items, text_key):

    try:
        has_all = 'All' in items
        items = [i for i in items if not i=='All']
        items_map = {}
        try:
            for item in meta['masks'][mask]['items']:
                if isinstance(item['text'], dict):
                    text = get_text(item['text'], text_key)
                else:
                    source = item['source'].split('@')[-1]
                    text = get_text(meta['columns'][source]['text'], text_key)
                    text = text.replace(
                        '{} - '.format(
                            get_text(meta['masks'][mask]['text'],
                            text_key)),
                        '')
                items_map.update({item['source'].split('@')[-1]: text})
        except UnicodeEncodeError:
            for item in meta['masks'][mask]['items']:
                if isinstance(item['text'], dict):
                    text = qp.core.tools.dp.io.unicoder(
                        get_text(item['text'], text_key),
                        like_ascii=True)
                else:
                    source = item['source'].split('@')[-1]
                    text = qp.core.tools.dp.io.unicoder(
                        get_text(meta['columns'][source]['text'], text_key),
                        like_ascii=True)
                    text = qp.core.tools.dp.io.unicoder(
                        text.replace(
                            '{} - '.format(
                                get_text(meta['masks'][mask]['text'],
                                text_key)),
                            ''),
                        like_ascii=True)
                items_map.update({item['source'].split('@')[-1]: text})
        items_text = [items_map[i] for i in items]
        if has_all:
            items_text = ['All'] + items_text
    except KeyError:
        items_text = items
    except ValueError:
        items_text = items

    return items_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_values_text(meta, mask, values, text_key, add_text_map=None):

    add_text_map = paint_add_text_map(meta, add_text_map, text_key)

    # Values text
    values_meta = emulate_meta(meta, meta['masks'][mask]['values'])
    try:
        has_all = 'All' in values
        if has_all: values.remove('All')
        try:
            values_map = {
                val['value']: get_text(val['text'], text_key)
                for val in values_meta}
        except UnicodeEncodeError:
            values_map = {
                val['value']: qp.core.tools.dp.io.unicoder(
                    get_text(val['text'], text_key,
                    like_ascii=True))
                for val in values_meta}
        values_map.update(add_text_map)
        values_text = [values_map[v] for v in values]
        if has_all:
            values_text = ['All'] + values_text
    except KeyError:
        values_text = values
    except ValueError:
        values_text = values

    return values_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_multiindex_from_tuples(l0_text, l1_text, names, single_row):

    if single_row:
        new_index = pd.MultiIndex.from_tuples(
            [(l0_text, l1_text[0])], names=names)
    else:
        new_index = pd.MultiIndex.from_product(
            [[l0_text], l1_text], names=names)

    return new_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_items_index(meta, index, text_key, display_names):

    single_row = len(index.values)==1
    levels = get_index_levels(index)
    mask = levels[0]
    items = levels[1]

    mask_text = paint_mask_text(meta, mask, text_key, display_names)
    items_text = paint_array_items_text(meta, mask, items, text_key)

    new_index = build_multiindex_from_tuples(
        mask_text,
        items_text,
        ['Array', 'Questions'],
        single_row)

    return new_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_values_index(meta, index, text_key, display_names,
                             grp_text_map=None):

    single_row = len(index.values)==1
    levels = get_index_levels(index)
    mask = levels[0]
    values = levels[1]

    mask_text = paint_mask_text(meta, mask, text_key, display_names)

    values_text = paint_array_values_text(
        meta, mask, values, text_key, grp_text_map)

    new_index = build_multiindex_from_tuples(
        mask_text,
        values_text,
        ['Question', 'Values'],
        single_row)

    return new_index

def finish_text_key(meta, text_key):

    #add default to text_key
    default_text = meta['lib'].get('default text', 'None')
    if text_key is None:
        text_key = {}
    for key in ['x', 'y']:
        if key in list(text_key.keys()):
            if isinstance(text_key[key], (str, str)):
                text_key[key] = [text_key[key], default_text]
            elif isinstance(text_key[key], list):
                text_key[key].append(default_text)
            else:
                raise TypeError(
                    "text_key items must be <str> or <list>\n"
                    "Found: %s" % (type(text_key[key]))
                )
        else:
            text_key[key] = [default_text]

    return text_key
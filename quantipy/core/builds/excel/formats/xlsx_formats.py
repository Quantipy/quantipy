
import pickle
from itertools import chain, combinations

DESCRIPTIVES = [
    'DESCRIPTIVES',
    'bg-DESCRIPTIVES',
    'DESCRIPTIVES-XT',
    'bg-DESCRIPTIVES-XT',
    'brow-DESCRIPTIVES',
    'mrow-DESCRIPTIVES',
    'frow-DESCRIPTIVES']

DEFAULT = [
    'DEFAULT',
    'bg-DEFAULT',
    'frow-bg-DEFAULT']

BASE = ['BASE', 'frow-BASE', 'UBASE', 'frow-UBASE']

N_PCT_FORMAT = [
    '{}'.format,
    'bg-{}'.format,
    'frow-{}'.format,
    'frow-bg-{}'.format,
    '{}-NET'.format,
    'bg-{}-NET'.format,
    'brow-{}-NET'.format,
    'mrow-{}-NET'.format,
    'frow-{}-NET'.format]

N_PCT = list(chain(*[[fn(f) for fn in N_PCT_FORMAT] for f in ['N', 'PCT']]))

for i, name in enumerate(N_PCT[:]):
    N_PCT[i+1:i+1] = ['{}-UP'.format(name)]
    N_PCT[i+2:i+2] = ['{}-DOWN'.format(name)]

TESTS = [
    'TESTS', 'bg-TESTS', 'frow-TESTS', 'frow-bg-TESTS',
    'TESTS-UP', 'bg-TESTS-UP', 'frow-TESTS-UP', 'frow-bg-TESTS-UP',
    'TESTS-DOWN', 'bg-TESTS-DOWN', 'frow-TESTS-DOWN', 'frow-bg-TESTS-DOWN',
    'STR']

CELL_LIST = DESCRIPTIVES + DEFAULT + BASE + N_PCT + TESTS

class XlsxFormats(object):
    """
    A class for writing the quantipy.ExcelPainter format dictionary.
    """

    def __init__(self, properties={}):
            """
            Constructor.
            """

            super(XlsxFormats, self).__init__()

            # -------------------------- POSTIONAL
            self.start_row = 8
            self.start_column = 2
            self.row_height = 12.75
            self.row_wrap_trigger = 44
            self.y_header_height = 33.75
            self.y_row_height = 50
            # --------------------------

            # -------------------------- IMAGE
            self.no_logo = False
            self.img_name = 'qplogo_invert_lg.png'
            self.img_url = '\\'.join(['logo', self.img_name])
            self.img_size = [130, 130]
            self.img_insert_x = 0
            self.img_insert_y = 0
            self.img_x_offset = 0
            self.img_y_offset = 0
            # --------------------------

            #-------------------------- CELL DATA OPTIONS
            self.frequency_0_repr = '-'
            self.descriptives_0_repr = 0.00
            self.df_nan_repr = '__NA__'
            self.test_seperator = '.'
            #--------------------------

            #-------------------------- TEXT
            self.font_name = 'Arial'
            self.font_size = 9
            self.font_color = '#000000'
            self.font_color_label = '#000000'
            self.bold = False
            self.bold_y = False
            self.bold_x = False
            #--------------------------

            #-------------------------- TEXT (BASE)
            self.font_color_ubase = '#000000'
            self.font_color_ubase_text = '#000000'
            self.font_color_base = '#000000'
            self.font_color_base_text = '#000000'
            self.bold_ubase_text = False
            self.bold_ubase = False
            self.bold_base_text = False
            self.bold_base = False
            #--------------------------

            #-------------------------- TEXT (NETS)
            self.font_name_nets = 'Arial'
            self.font_size_nets = 9
            self.font_color_nets = '#000000'
            self.bold_nets = False
            self.italicise_nets = False
            #--------------------------

            #-------------------------- TEXT (DESCRIPTIVES)
            self.font_name_descriptives = 'Arial'
            self.font_size_descriptives = 9
            self.font_color_descriptives = '#000000'
            self.bold_descriptives = False
            #--------------------------

            #-------------------------- TEXT (TESTS)
            self.font_name_tests = 'Arial'
            self.font_size_tests = 9
            self.font_color_tests = '#000000'
            self.bold_tests = False
            self.font_super_tests = True
            self.display_test_level = True
            self.dummy_tests = False
            self.arrow_color_high = '#2EB08C'
            self.arrow_color_low = '#FC8EAC'
            #--------------------------

            #-------------------------- TEXT (STR)
            self.font_name_str = 'Arial'
            self.font_size_str = 9
            self.font_color_str = '#000000'
            self.column_width_str = 10
            #--------------------------

            #-------------------------- TEXT (ADDITIONAL)
            self.format_label_row = False
            #--------------------------

            #-------------------------- BORDERS
            self.border_color = '#D9D9D9'
            self.border_color_nets_top = '#D9D9D9'
            self.border_color_descriptives_top = '#D9D9D9'
            self.border_style_ext = 5
            self.border_style_int = 1
            #--------------------------

            #-------------------------- BACKGROUND
            self.bg_color = '#F2F2F2'
            self.bg_color_nets = '#FFFFFF'
            self.bg_color_tests = '#F2F2F2'
            self.bg_color_label = '#FFFFFF'
            #--------------------------

            #-------------------------- NUMBER
            self.num_format_n = '0'
            self.num_format_pct = '0%'
            self.num_format_descriptives = '0.00'
            self.num_format_default = '0.00'
            #--------------------------

            # Convert properties in the constructor to method calls.
            #--------------------------
            for key, value in list(properties.items()):
                command = 'set_{}'.format(key)
                getattr(self, command)(value)
            #--------------------------

            #-------------------------- POSITIONAL (INDEX)
            self._start_row_idx = self.start_row-1
            self._start_column_idx = self.start_column-1
            #--------------------------

    def get_start_row_idx(self):
        """
        Returns _start_row_idx attribute
        """
        return self._start_row_idx

    def get_start_column_idx(self):
        """
        Returns _start_column_idx attribute
        """
        return self._start_column_idx

    def set_start_row(self, start_row):
        """
        Set the starting row (first x key label).

        Parameters
        ----------
        start_row : int, default 1

        Returns
        -------
        None
        """
        self.start_row = start_row

    def set_start_column(self, start_column):
        """
        Set the starting column (first y axis column).

        Parameters
        ----------
        start_column : int, default 1

        Returns
        -------
        None
        """
        self.start_column = start_column

    def set_row_height(self, row_height):
        """
        Set the default row height.

        Parameters
        ----------
        row_height : int, default 12.75

        Returns
        -------
        None
        """
        self.row_height = row_height

    def set_row_wrap_trigger(self, row_wrap_trigger):
        """
        Set the label length to trigger wrap text on x key label /
        x key cateogry labels.

        Parameters
        ----------
        row_wrap_trigger : int, default 44

        Returns
        -------
        None
        """
        self.row_wrap_trigger = row_wrap_trigger

    def set_y_header_height(self, y_header_height):
        """
        Set the height for the y axis headers.

        Parameters
        ----------
        y_head_height : int, default 33.75

        Returns
        -------
        None
        """
        self.y_header_height = y_header_height

    def set_y_row_height(self, y_row_height):
        """
        Set the height for the y axis column headers.

        Parameters
        ----------
        y_row_height : int, default 50

        Returns
        -------
        None
        """
        self.y_row_height = y_row_height

    def set_no_logo(self, no_logo):
        """
        Set property to false if logo not required.

        Parameters
        ----------
        no_logo : bool, default False

        Returns
        -------
        None
        """
        self.no_logo = no_logo

    def set_img_name(self, img_name):
        """
        Set the name of the image to insert into non-toc sheets.

        Parameters
        ----------
        img_name : str, default qplogo_invert_lg.png

        Returns
        -------
        None
        """
        self.img_name = img_name

    def set_img_url(self, img_url):
        """
        Set the path to the image to insert into non-toc sheets.

        Parameters
        ----------
        img_url : str, default quantipy.core.builds.excel.formats.logo

        Returns
        -------
        None
        """
        self.img_url = img_url

    def set_img_size(self, img_size):
        """
        Set the image size.

        Parameters
        ----------
        img_size : list, default [130, 130]
            The width and height as items.

        Returns
        -------
        None
        """
        self.img_size = img_size

    def set_img_insert_x(self, img_insert_x):
        """
        Set the row to insert image.

        Parameters
        ----------
        img_insert_x : int, default 0

        Returns
        -------
        None
        """
        self.img_insert_x = img_insert_x

    def set_img_insert_y(self, img_insert_y):
        """
        Set the column to insert image.

        Parameters
        ----------
        img_insert_y : int, default 0

        Returns
        -------
        None
        """
        self.img_insert_y = img_insert_y

    def set_img_x_offset(self, img_x_offset):
        """
        Set the image offset for x axis.

        Parameters
        ----------
        img_x_offset : int, default 0

        Returns
        -------
        None
        """
        self.img_x_offset = img_x_offset

    def set_img_y_offset(self, img_y_offset):
        """
        Set the image offset for y axis.

        Parameters
        ----------
        img_y_offset : int, default 0

        Returns
        -------
        None
        """
        self.img_y_offset = img_y_offset

    def set_frequency_0_repr(self, frequency_0_repr):
        """
        Set the frequency view 0 represenattion.

        Parameters
        ----------
        frequency_0_repr : str, default '-'

        Returns
        -------
        None
        """
        self.frequency_0_repr = frequency_0_repr

    def set_descriptives_0_repr(self, descriptives_0_repr):
        """
        Set the descriptives view 0 represenattion.

        Parameters
        ----------
        descriptives_0_repr : int/ float, default 0.00

        Returns
        -------
        None
        """
        self.descriptives_0_repr = descriptives_0_repr

    def set_df_nan_repr(self, df_nan_repr):
        """
        Set the descriptives view 0 represenattion.

        Parameters
        ----------
        df_nan_repr : str, default '__NA__'

        Returns
        -------
        None
        """
        self.df_nan_repr = df_nan_repr

    def set_test_seperator(self, test_seperator):
        """
        Set the test seperator.

        Parameters
        ----------
        descriptives_0_repr : str, default 0.00

        Returns
        -------
        None
        """
        self.test_seperator = test_seperator


    def set_font_name(self, font_name):
        """
        Set the font name.

        Parameters
        ----------
        font_name : str, default 'Arial'

        Returns
        -------
        None
        """
        self.font_name = font_name

    def set_font_size(self, font_size):
        """
        Set the font size.

        Parameters
        ----------
        font_size : int, default 9

        Returns
        -------
        None
        """
        self.font_size = font_size

    def set_font_color(self, font_color):
        """
        Set the font color.

        Parameters
        ----------
        font_color : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color = font_color

    def set_font_color_label(self, font_color_label):
        """
        Set the font color for x-axis question labels.

        Parameters
        ----------
        font_color_label : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_label = font_color_label

    def set_bold(self, bold):
        """
        Set the bold property.

        Parameters
        ----------
        bold : bool, default False

        Returns
        -------
        None
        """
        self.bold = bold

    def set_bold_x(self, bold_x):
        """
        Set the bold property for x key labels.
        This does not include category labels

        Parameters
        ----------
        bold_x : bool, default False

        Returns
        -------
        None
        """
        self.bold_x = bold_x

    def set_bold_y(self, bold_y):
        """
        Set the bold property for y key labels.
        This includes headers and column labels.

        Parameters
        ----------
        bold_y : bool, default False

        Returns
        -------
        None
        """
        self.bold_y = bold_y

    def set_font_color_ubase(self, font_color_ubase):
        """
        Set the font color for the unweighted base views data.

        Parameters
        ----------
        font_color_ubase : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_ubase = font_color_ubase

    def set_font_color_ubase_text(self, font_color_ubase_text):
        """
        Set the font color for the unweighted base views text.

        Parameters
        ----------
        font_color_ubase_text : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_ubase_text = font_color_ubase_text

    def set_font_color_base(self, font_color_base):
        """
        Set the font color for the base views data.

        Parameters
        ----------
        font_color_base : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_base = font_color_base


    def set_font_color_base_text(self, font_color_base_text):
        """
        Set the font color for the base views text.

        Parameters
        ----------
        font_color_base_text : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_base_text = font_color_base_text

    def set_bold_ubase_text(self, bold_ubase_text):
        """
        Set the bold property for the unweighted base text.

        Parameters
        ----------
        bold_ubase_text : bool, default False

        Returns
        -------
        None
        """
        self.bold_ubase_text = bold_ubase_text

    def set_bold_ubase(self, bold_ubase):
        """
        Set the bold property for the unweighted base views.

        Parameters
        ----------
        bold_ubase : bool, default False

        Returns
        -------
        None
        """
        self.bold_ubase = bold_ubase

    def set_bold_base_text(self, bold_base_text):
        """
        Set the bold property for the base text.

        Parameters
        ----------
        bold_base_text : bool, default False

        Returns
        -------
        None
        """
        self.bold_base_text = bold_base_text

    def set_bold_base(self, bold_base):
        """
        Set the bold property for the base views.

        Parameters
        ----------
        bold_base : bool, default False

        Returns
        -------
        None
        """
        self.bold_base = bold_base

    def set_font_name_nets(self, font_name_nets):
        """
        Set the font name for nets views.

        Parameters
        ----------
        font_name_nets : str, default 'Arial'

        Returns
        -------
        None
        """
        self.font_name_nets = font_name_nets

    def set_font_size_nets(self, font_size_nets):
        """
        Set the font size for nets views.

        Parameters
        ----------
        font_size_nets : int, default 9

        Returns
        -------
        None
        """
        self.font_size_nets = font_size_nets


    def set_font_color_nets(self, font_color_nets):
        """
        Set the font color for nets views.

        Parameters
        ----------
        font_color_nets : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_nets = font_color_nets

    def set_bold_nets(self, bold_nets):
        """
        Set the bold property for nets views.

        Parameters
        ----------
        bold_nets : bool, default False

        Returns
        -------
        None
        """
        self.bold_nets = bold_nets

    def set_italicise_nets(self, italicise_nets):
        """
        Set the italic property for nets views.

        Parameters
        ----------
        italicise_nets : bool, default False

        Returns
        -------
        None
        """
        self.italicise_nets = italicise_nets

    def set_bg_color_nets(self, bg_color_nets):
        """
        Set the background color for nets views.

        Parameters
        ----------
        bg_color_nets : str, default #FFFFFF (white)

        Returns
        -------
        None
        """
        self.bg_color_nets = bg_color_nets

    def set_font_name_descriptives(self, font_name_descriptives):
        """
        Set the font name for descriptives views.

        Parameters
        ----------
        font_name_descriptives : str, default 'Arial'

        Returns
        -------
        None
        """
        self.font_name_descriptives = font_name_descriptives

    def set_font_size_descriptives(self, font_size_descriptives):
        """
        Set the font size for descriptives views.

        Parameters
        ----------
        font_size_descriptives : int, default 9

        Returns
        -------
        None
        """
        self.font_size_descriptives = font_size_descriptives


    def set_font_color_descriptives(self, font_color_descriptives):
        """
        Set the font color for descriptives views.

        Parameters
        ----------
        font_color_descriptives : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_descriptives = font_color_descriptives

    def set_bold_descriptives(self, bold_descriptives):
        """
        Set the bold property for descriptives views.

        Parameters
        ----------
        bold_descriptives : bool, default False

        Returns
        -------
        None
        """
        self.bold_descriptives = bold_descriptives

    def set_font_name_tests(self, font_name_tests):
        """
        Set the font name for test views.

        Parameters
        ----------
        font_name_tests : str, default 'Arial'

        Returns
        -------
        None
        """
        self.font_name_tests = font_name_tests

    def set_font_size_tests(self, font_size_tests):
        """
        Set the font size for test views.

        Parameters
        ----------
        font_size_tests : int, default 9

        Returns
        -------
        None
        """
        self.font_size_tests = font_size_tests

    def set_font_color_tests(self, font_color_tests):
        """
        Set the font color for test views.

        Parameters
        ----------
        font_color_tests : str, default '#000000'

        Returns
        -------
        None
        """
        self.font_color_tests = font_color_tests

    def set_bold_tests(self, bold_tests):
        """
        Set the bold property for test views.

        Parameters
        ----------
        bold_tests : bool, default False

        Returns
        -------
        None
        """
        self.bold_tests = bold_tests

    def set_font_super_tests(self, font_super_tests):
        """
        Set the superscript property for test views side annotations.

        Parameters
        ----------
        font_super_tests : bool, default True

        Returns
        -------
        None
        """
        self.font_super_tests = font_super_tests

    def set_display_test_level(self, display_test_level):
        """
        Set option to display test level.

        Parameters
        ----------
        display_test_level : bool, default True

        Returns
        -------
        None
        """
        self.display_test_level = display_test_level

    def set_dummy_tests(self, dummy_tests):
        """
        Set option to insert dummy rows for views that have no related tests.

        Parameters
        ----------
        dummy_tests : bool, default True

        Returns
        -------
        None
        """
        self.dummy_tests = dummy_tests

    def set_font_name_str(self, font_name_str):
        """
        Set the font name for profile tables.

        Parameters
        ----------
        font_name_str : str, default 'Arial'

        Returns
        -------
        None
        """
        self.font_name_str = font_name_str

    def set_font_size_str(self, font_size_str):
        """
        Set the font size for profile tables.

        Parameters
        ----------
        font_size_str : int, default 9

        Returns
        -------
        None
        """
        self.font_size_str = font_size_str

    def set_font_color_str(self, font_color_str):
        """
        Set the font color for profile tables.

        Parameters
        ----------
        font_color_str : int, default 9

        Returns
        -------
        None
        """
        self.font_color_str = font_color_str

    def set_column_width_str(self, column_width_str):
        """
        Set the column width for dataframe column sheets.

        Parameters
        ----------
        column_width_str : int, default 10

        Returns
        -------
        None
        """
        self.column_width_str = column_width_str

    def set_format_label_row(self, format_label_row):
        """
        Set the option to use label formatting in all row cells,
        up to last column of y-axis.

        Parameters
        ----------
        format_label_row : bool, default False

        Returns
        -------
        None
        """
        self.format_label_row = format_label_row

    def set_border_color(self, border_color):
        """
        Set the border color.

        Parameters
        ----------
        border_color : str, default '#D9D9D9'

        Returns
        -------
        None
        """
        self.border_color = border_color

    def set_border_color_nets_top(self, border_color_nets_top):
        """
        Set the top border color for nets.

        Parameters
        ----------
        border_color : str, default '#D9D9D9'

        Returns
        -------
        None
        """
        self.border_color_nets_top = border_color_nets_top

    def set_border_color_descriptives_top(self, border_color_descriptives_top):
        """
        Set the top border color for descriptives.

        Parameters
        ----------
        border_color : str, default '#D9D9D9'

        Returns
        -------
        None
        """
        self.border_color_descriptives_top = border_color_descriptives_top

    def set_border_style_ext(self, border_style_ext):
        """
        Set the exterior border style index.


        Parameters
        ----------
        border_style_ext : int, default 5
            The default corresponds to "Continous/ Weight = 3"
            http://xlsxwriter.readthedocs.org/format.html
        Returns
        -------
        None
        """
        self.border_style_ext = border_style_ext

    def set_border_style_int(self, border_style_int):
        """
        Set the exterior border style index.


        Parameters
        ----------
        border_style_int : int, default 1
            The default corresponds to "Continous/ Weight = 1"
            http://xlsxwriter.readthedocs.org/format.html
        Returns
        -------
        None
        """
        self.border_style_int = border_style_int

    def set_bg_color(self, bg_color):
        """
        Set the background color to apply to alternate rows of
        frequency views.

        Parameters
        ----------
        bg_color : str, default '#F2F2F2'

        Returns
        -------
        None
        """
        self.bg_color = bg_color

    def set_bg_color_tests(self, bg_color_tests):
        """
        Set the background color to apply to rows of
        column test views.

        Parameters
        ----------
        bg_color_tests : str, default '#FFFFFF'

        Returns
        -------
        None
        """
        self.bg_color_tests = bg_color_tests

    def set_bg_color_label(self, bg_color_label):
        """
        Set the background color to apply to label.

        Parameters
        ----------
        bg_color_label : str, default '#FFFFFF'

        Returns
        -------
        None
        """
        self.bg_color_label = bg_color_label

    def set_num_format_n(self, num_format_n):
        """
        Set the number format for counts based frequency views.

        Parameters
        ----------
        num_format_n : str, default '0'

        Returns
        -------
        None
        """
        self.num_format_n = num_format_n

    def set_num_format_pct(self, num_format_pct):
        """
        Set the number format for %% based frequency views.

        Parameters
        ----------
        num_format_pct : str, default '0%'

        Returns
        -------
        None
        """
        self.num_format_pct = num_format_pct

    def set_num_format_descriptives(self, num_format_descriptives):
        """
        Set the number format for descriptives views.

        Parameters
        ----------
        num_format_descriptives : str, default '0.00'

        Returns
        -------
        None
        """
        self.num_format_descriptives = num_format_descriptives

    def set_num_format_default(self, num_format_default):
        """
        Set the number format for default views.

        Parameters
        ----------
        num_format_default : str, default '0.00'

        Returns
        -------
        None
        """
        self.num_format_default = num_format_default

    def set_arrow_color_high(self, arrow_color_high):
        """
        Set the color for the up arrow in "test against total" views

        Parameters
        ----------
        arrow_color_high : str, default '#000000'

        Returns
        -------
        None
        """
        self.arrow_color_high = arrow_color_high

    def set_arrow_color_low(self, arrow_color_low):
        """
        Set the color for the down arrow in "test against total" views

        Parameters
        ----------
        arrow_color_low : str, default '#000000'

        Returns
        -------
        None
        """
        self.arrow_color_low = arrow_color_low

    def create_formats_dict(self):
        """
        Creates the dictionary of formatting options used to
        create xlsxwriter.Format objects used by quantipy.ExcelPainter().

        Parameters
        ----------


        Returns
        -------
        dict
        """
        self.format_dict = {}
        self._add_toc()
        self._add_y()
        self._add_tests()
        self._add_x()
        self._add_left()
        self._add_right()
        self._add_interior()
        self._add_italic()

# -----------------------------------------------------------------------------

    def _add_toc(self):
        """ Create the key-value pairs for self.format_dict(),
        for use in the toc.
        """
        self.format_dict.update(
            {
                'TOC-bold-14': {
                    'font_name': self.font_name,
                    'font_size': 14,
                    'bold': True},
                'TOC-bold-10': {
                    'font_name': self.font_name,
                    'font_size': 10,
                    'bold': True},
                'TOC-bold-center-10': {
                    'font_name': self.font_name,
                    'font_size': 10,
                    'bold': True,
                    'text_v_align': 2,
                    'text_h_align': 2},
                'TOC-10': {
                    'font_name': self.font_name,
                    'font_size': 10},
                'TOC-url': {
                    'font_name': self.font_name,
                    'font_size': 10,
                    'font_color': 'blue',
                    'underline':  1}})

    def _add_y(self):
        """ Create the key-value pairs for self.format_dict(),
        for use in the y-axis.
        """
        self.format_dict.update(
            {
                'y': {
                    'font_name': self.font_name,
                    'font_size': self.font_size,
                    'bold': self.bold_y,
                    'text_v_align': 2,
                    'text_h_align': 2,
                    'text_wrap': True,
                    'left': self.border_style_ext,
                    'top': self.border_style_ext,
                    'right': self.border_style_ext,
                    'bottom': self.border_style_ext,
                    'left_color': self.border_color,
                    'top_color': self.border_color,
                    'right_color': self.border_color,
                    'bottom_color': self.border_color}})

    def _add_tests(self):
        """ Create the key-value pairs for self.format_dict(),
        for use in the tests-letter row of y-axis.
        """
        self.format_dict.update(
            {
                'tests': {
                    # 'bg_color': self.bg_color_tests,
                    'bg_color': '#FFFFFF',
                    'font_name': self.font_name_tests,
                    'font_size': self.font_size_tests,
                    'font_color': self.font_color_tests,
                    'bold': self.bold_y,
                    'text_v_align': 2,
                    'text_h_align': 2,
                    'text_wrap': True,
                    'left': self.border_style_ext,
                    'top': self.border_style_ext,
                    'right': self.border_style_ext,
                    'bottom': self.border_style_ext,
                    'left_color': self.border_color,
                    'top_color': self.border_color,
                    'right_color': self.border_color,
                    'bottom_color': self.border_color}})

    def _add_x(self):
        """ Create the key-value pairs for self.format_dict(),
        for use in the x-axis.
        """
        self.format_dict.update(
            {
                'x_left_bold': {
                    'font_name': self.font_name,
                    'font_size': self.font_size,
                    'font_color': self.font_color_label,
                    'bold': self.bold_x,
                    'bg_color': self.bg_color_label,
                    'text_v_align': 2,
                    'text_h_align': 1,
                    'text_wrap': True},
                'x_right': {
                    'font_name': self.font_name,
                    'font_size': self.font_size,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True},
                'x_right_bold': {
                    'font_name': self.font_name,
                    'font_size': self.font_size,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True,
                    'bold': True},
                'x_right_italic': {
                    'font_name': self.font_name,
                    'font_size': self.font_size,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True,
                    'italic': True},
                'cell_details': {
                    'font_name': self.font_name_tests,
                    'font_size': self.font_size,
                    'text_v_align': 2,
                    'text_h_align': 1},
                'x_right_nets': {
                    'bold': self.bold_nets,
                    'bg_color': self.bg_color_nets,
                    'italic': self.italicise_nets,
                    'font_name': self.font_name_nets,
                    'font_size': self.font_size_nets,
                    'font_color': self.font_color_nets,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True},
                'x_right_descriptives': {
                    'font_name': self.font_name_descriptives,
                    'font_size': self.font_size_descriptives,
                    'font_color': self.font_color_descriptives,
                    'bold': self.bold_descriptives,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True},
                'x_right_tests': {
                    'num_format': '0.00',
                    'font_name': self.font_name_tests,
                    'font_size': self.font_size_tests,
                    'font_color': self.font_color_tests,
                    'font_script': self.font_super_tests,
                    'bold': self.bold_tests,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True},
                'x_right_ubase': {
                    'font_name': self.font_name,
                    'font_size': self.font_size,
                    'font_color': self.font_color_ubase_text,
                    'bold': self.bold_ubase_text,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True},
                'x_right_base': {
                    'font_name': self.font_name,
                    'font_size': self.font_size,
                    'font_color': self.font_color_base_text,
                    'bold': self.bold_base_text,
                    'text_v_align': 2,
                    'text_h_align': 3,
                    'text_wrap': True}})

    def _add_left(self):
        """ Create the key-value pairs for self.format_dict(),
        where key starts with 'left'
        """
        self._update_format_dict(
            border='left',
            additional_borders=['right', 'top', 'bottom'])

    def _add_right(self):
        """ Create the key-value pairs for self.format_dict(),
        where key starts with 'right'
        """
        self._update_format_dict(
            border='right',
            additional_borders=['top', 'bottom'])

    def _add_interior(self):
        """ Create the key-value pairs for self.format_dict(),
        where key starts with 'right'
        """
        self._update_format_dict(
            border='interior',
            additional_borders=['top', 'bottom'])

    def _update_format_dict(self, border, additional_borders):
        """ Update self.format_dict() given border and
        additional_borders
        """
        additional_borders.extend(
            list(
                chain(
                    *[
                        list(combinations(additional_borders, r))
                        for r in range(2, len(additional_borders)+1)])))
        borders_list = ['-']
        borders_list.extend(
            [
                '-{}-'.format('-'.join(x)) if isinstance(x, tuple)
                else '-{}-'.format(x) for x in additional_borders])
        for borders in borders_list:
            for cell in CELL_LIST:
                key = ''.join([border, borders, cell])
                self.format_dict.update({key: self._get_value(key)})

    def _get_value(self, key):
        """ Get the value for format key.
        """
        # Add alignment
        result = {}
        if not key.endswith('-STR'):
            result.update(self._get_alignments())
        # Add borders
        for border in ['left', 'right', 'top', 'bottom']:
            if '{}-'.format(border) in key:
                conditions = [
                    not key.endswith(('NET', 'NET-UP', 'NET-DOWN', 'DESCRIPTIVES')),
                    all(
                        [
                            key.endswith(('NET', 'NET-UP', 'NET-DOWN', 'DESCRIPTIVES')),
                            not border == 'top'])]
                if any(conditions):
                    result.update(
                        self._get_border(border, self.border_style_ext))
                elif key.endswith(('NET', 'NET-UP', 'NET-DOWN')):
                    result.update(
                        self._get_border(
                            border,
                            self.border_style_ext,
                            self.border_color_nets_top))
                elif key.endswith(('DESCRIPTIVES', 'DESCRIPTIVES-XT')):
                    result.update(
                        self._get_border(
                            border,
                            self.border_style_ext,
                            self.border_color_descriptives_top))
        if not 'left' in key:
            if not key.endswith('-STR'):
                result.update(
                    self._get_border('left', self.border_style_int))
        # Cell type
        if key.endswith(('-DESCRIPTIVES', '-DESCRIPTIVES-XT')):
            if not key.endswith('XT'):
                for border in ['top', 'bottom']:
                    if not border in list(result.keys()):
                        result.update(
                            self._get_border(
                                border,
                                self.border_style_int,
                                self.border_color_descriptives_top))
            result.update(self._get_num_format('DESCRIPTIVES'))
            result.update(self._get_font_format('DESCRIPTIVES'))
            result.update(self._get_bold_format('DESCRIPTIVES'))
            if 'bg' in key: result.update(
                self._get_bg_format('DESCRIPTIVES', True))
        elif key.endswith('-DEFAULT'):
            result.update(self._get_num_format('DEFAULT'))
            result.update(self._get_font_format('DEFAULT'))
            result.update(self._get_bg_format('DEFAULT', 'bg' in key))
        elif key.endswith('-BASE'):
            result.update(self._get_num_format('BASE'))
            result.update(self._get_font_format('BASE'))
        elif key.endswith('-UBASE'):
            result.update(self._get_num_format('UBASE'))
            result.update(self._get_font_format('UBASE'))
        elif key.endswith('-N'):
            result.update(self._get_num_format('N'))
            result.update(self._get_font_format('N'))
            result.update(self._get_bg_format('N', 'bg' in key))
        elif key.endswith(('-N-NET', '-N-NET-UP', '-N-NET-DOWN')):
            for border in ['top']:
                if not border in list(result.keys()):
                    result.update(
                        self._get_border(
                            border,
                            self.border_style_int,
                            self.border_color_nets_top))
            result.update(self._get_num_format('N'))
            result.update(self._get_font_format('NET'))
            result.update(self._get_bold_format('NET'))
            result.update(self._get_italic_format('NET'))
            if 'bg' in key:
                result.update(self._get_bg_format('N', True))
            else:
                result.update(self._get_bg_format('NET', True))
        elif key.endswith('-PCT'):
            result.update(self._get_num_format('PCT'))
            result.update(self._get_font_format('PCT'))
            result.update(self._get_bg_format('PCT', 'bg' in key))
        elif key.endswith(('-PCT-NET', '-PCT-NET-UP', '-PCT-NET-DOWN')):
            for border in ['top']:
                if not border in list(result.keys()):
                    result.update(
                        self._get_border(
                            border,
                            self.border_style_int,
                            self.border_color_nets_top))
            result.update(self._get_num_format('PCT'))
            result.update(self._get_font_format('NET'))
            result.update(self._get_bold_format('NET'))
            result.update(self._get_italic_format('NET'))
            if 'bg' in key:
                result.update(self._get_bg_format('PCT', True))
            else:
                result.update(self._get_bg_format('NET', True))
        elif key.endswith('-STR'):
            if not 'right' in list(result.keys()):
                result.update(self._get_border('right', self.border_style_int))
            result.update(self._get_font_format('STR'))
        elif '-TESTS' in key:
            result.update(self._get_font_format('TESTS'))
            result.update(self._get_bold_format('TESTS'))
            result.update(self._get_bg_format('TESTS', 'bg' in key))
        # Add top row if "frow"
        if 'frow' in key:
            if not 'top' in key:
                if key.endswith('DESCRIPTIVES'):
                    result.update(
                        self._get_border(
                            'top',
                            self.border_style_int,
                            self.border_color_descriptives_top))
                    result = {
                        k: v for k, v in list(result.items())
                              if 'bottom' not in k}
                elif key.endswith(('NET', 'NET-UP', 'NET-DOWN')):
                    result.update(
                        self._get_border(
                            'top',
                            self.border_style_int,
                            self.border_color_nets_top))
                else:
                    result.update(
                        self._get_border('top', self.border_style_int))
            # result = {
            #     k: v for k, v in result.items()
            #           if 'bottom' not in k}
        # Delete bottom row if "mrow"
        if 'mrow' in key:
            result = {
                k: v for k, v in list(result.items())
                if not k.startswith(('top', 'bottom'))}
        # Delete top row if "brow"
        if 'brow' in key:
            result = {
                k: v for k, v in list(result.items())
                if not k.startswith(('top'))}
        # Is this an arrow format? If so, modify the font color.
        if key.endswith('UP'):
            result['font_color'] = self.arrow_color_high
        elif key.endswith('DOWN'):
            result['font_color'] = self.arrow_color_low
        return result

    def _get_alignments(self):
        """ Returns standard alignments.
        """
        result = {'text_v_align': 2, 'text_h_align': 2}
        return result

    def _get_border(self, border, border_style, border_color=None):
        """ Returns left border with interior/ exterior style.
        """
        if not isinstance(border_color, (int, str)):
            border_color = self.border_color
        result = {
            border: border_style,
            '{}_color'.format(border): border_color}
        return result

    def _get_num_format(self, cell):
        """ Return number format based on cell type.
        """
        if cell == 'DESCRIPTIVES':
            result = {'num_format': self.num_format_descriptives}
        elif cell == 'DEFAULT':
            result = {'num_format': self.num_format_default}
        elif cell in ['BASE', 'UBASE', 'N']:
            result = {'num_format': self.num_format_n}
        elif cell == 'PCT':
            result = {'num_format': self.num_format_pct}
        return result

    def _get_font_format(self, cell):
        """ Return font format based on cell type.
        """
        if cell in ['DEFAULT', 'N', 'PCT']:
            result = {
                'font_name': self.font_name, 'font_size': self.font_size}
        elif cell == 'BASE':
            result = {
                'font_name': self.font_name,
                'font_size': self.font_size,
                'font_color': self.font_color_base,
                'bold': self.bold_base}
        elif cell == 'UBASE':
            result = {
                'font_name': self.font_name,
                'font_size': self.font_size,
                'font_color': self.font_color_ubase,
                'bold': self.bold_ubase}
        elif cell == 'DESCRIPTIVES':
            result = {
                'font_name': self.font_name_descriptives,
                'font_size': self.font_size_descriptives,
                'font_color': self.font_color_descriptives}
        elif cell == 'NET':
            result = {
                'font_name': self.font_name_nets,
                'font_size': self.font_size_nets,
                'font_color': self.font_color_nets}
        elif cell == 'STR':
            result = {
                'font_name': self.font_name_str,
                'font_size': self.font_size_str,
                'font_color': self.font_color_str}
        elif cell == 'TESTS':
            result = {
                'font_name': self.font_name_tests,
                'font_size': self.font_size_tests,
                'font_color': self.font_color_tests,
                'font_script': self.font_super_tests}
        return result

    def _get_bold_format(self, cell):
        """ Return bold format based on cell type.
        """
        if cell == 'DESCRIPTIVES':
            result = {'bold': self.bold_descriptives}
        elif cell == 'NET':
            result = {'bold': self.bold_nets}
        elif cell == 'TESTS':
            result = {'bold': self.bold_tests}
        return result

    def _get_bg_format(self, cell, required):
        """ Return bold format based on cell type.
        """
        if cell in ['DEFAULT', 'N', 'PCT', 'DESCRIPTIVES']:
            result = {'bg_color': self.bg_color if required else '#FFFFFF'}
        elif cell in ['NET']:
            result = {
                'bg_color': self.bg_color_nets if required else '#FFFFFF'}
        elif cell == 'TESTS':
            result = {
                'bg_color': self.bg_color_tests if required else '#FFFFFF'}
        return result

    def _get_italic_format(self, cell):
        """ Return bold format based on cell type.
        """
        if cell in ['NET']:
            result = {'italic': self.italicise_nets}
        return result

    def _add_italic(self):
        """ Add all format with italic set to True
        """
        for key, value in list(self.format_dict.items()):
            if not key.endswith(('STR', 'TESTS')):
                new_key = '-'.join([key, 'italic'])
                self.format_dict[new_key] = pickle.loads(
                    pickle.dumps(value, pickle.HIGHEST_PROTOCOL))
                self.format_dict[new_key]['italic'] = True
        return None


class XLSX_Formats(object):
    """
    A class for writing the quantipy.ExcelPainter format dictionary.
    """

    def __init__(self, properties={}):
            """
            Constructor.
            """

            super(XLSX_Formats, self).__init__()

            # -------------------------- IMAGE
            self.img_name = 'qplogo_invert_lg.png'
            self.img_url = '\\'.join(['logo', self.img_name])
            self.img_size = [130, 130]
            # --------------------------

            #-------------------------- CELL DATA OPTIONS
            self.frequency_0_repr = '-'
            self.descriptives_0_repr = 0.00
            self.test_seperator = '.'
            #-------------------------- 

            #-------------------------- TEXT
            self.font_name = 'Arial'
            self.font_size = 9
            self.font_color = 'black'
            self.font_bold = False
            self.font_bold_y = False
            self.font_bold_x = False
            #-------------------------- 

            #-------------------------- TEXT (STATS)
            self.font_name_descriptives = 'Arial'
            self.font_size_descriptives = 9
            self.font_color_descriptives = 'black'
            self.font_bold_descriptives = False
            #-------------------------- 

            #-------------------------- TEXT (TESTS)
            self.font_name_tests = 'Arial'
            self.font_size_tests = 9
            self.font_color_tests = 'black'
            self.font_bold_tests = False
            self.font_super_tests = True
            #--------------------------

            #-------------------------- TEXT (STR)
            self.font_name_str = 'Arial'
            self.font_size_str = 9
            self.font_color_str = 'black'
            #--------------------------

            #-------------------------- TEXT (ADDITIONAL)
            self.font_bold_base = False
            #--------------------------

            #-------------------------- BORDERS
            self.border_color = '#D9D9D9'
            self.border_style_ext = 5
            self.border_style_int = 1
            #-------------------------- 

            #-------------------------- BACKGROUND
            self.bg_color = '#F2F2F2'        
            # self.bg_color_tests = '#FFFFFF'   
            self.bg_color_tests = 0
            #-------------------------- 

            #-------------------------- NUMBER
            self.num_format_n = '0'
            self.num_format_pct = '0%'
            self.num_format_descriptives = '0.00'
            self.num_format_default = '0.00'
            #--------------------------
            
            # Convert properties in the constructor to method calls.
            for key, value in properties.items():
                getattr(self, 'set_' + key)(value)

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
        font_color : str, default 'black'

        Returns
        -------
        None
        """
        self.font_color = font_color

    def set_font_bold(self, font_bold):
        """
        Set the bold property. 
        
        Parameters
        ----------
        font_bold : bool, default False
        
        Returns
        -------
        None
        """
        self.font_bold = font_bold

    def set_font_bold_x(self, font_bold_x):
        """
        Set the bold property for x key labels. 
        This does not include category labels
        
        Parameters
        ----------
        font_bold_x : bool, default False
        
        Returns
        -------
        None
        """
        self.font_bold_x = font_bold_x

    def set_font_bold_y(self, font_bold_y):
        """
        Set the bold property for y key labels.
        This includes headers and column labels.
        
        Parameters
        ----------
        font_bold_y : bool, default False
        
        Returns
        -------
        None
        """
        self.font_bold_y = font_bold_y

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
        font_color_descriptives : str, default 'black'
        
        Returns
        -------
        None
        """
        self.font_color_descriptives = font_color_descriptives

    def set_font_bold_descriptives(self, font_bold_descriptives):
        """
        Set the bold property for descriptives views.

        Parameters
        ----------
        font_bold_descriptives : bool, default False
        
        Returns
        -------
        None
        """
        self.font_bold_descriptives = font_bold_descriptives

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
        font_color_tests : str, default 'black'
        
        Returns
        -------
        None
        """
        self.font_color_tests = font_color_tests

    def set_font_bold_tests(self, font_bold_tests):
        """
        Set the bold property for test views.

        Parameters
        ----------
        font_bold_tests : bool, default False
        
        Returns
        -------
        None
        """
        self.font_bold_tests = font_bold_tests

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

    def set_font_bold_base(self, font_bold_base):
        """
        Set the bold property for base views.

        Parameters
        ----------
        font_bold_base : bool, default False
        
        Returns
        -------
        None
        """
        self.font_bold_base = font_bold_base

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
        self._get_left()

# -----------------------------------------------------------------------------
    
    def _get_left(self):
        """ Create the key-value pairs for self.format_dict(), 
        where key starts with 'left'
        """
        sides_list = ['',
                      'right',
                      'top',
                      'bottom',
                      'right-top',
                      'right-bottom',
                      'right-top-bottom']
        
        types_list = ['DESCRIPTIVES', 'brow-DESCRIPTIVES',
                      'mrow-DESCRIPTIVES', 'frow-DESCRIPTIVES',
                      'DEFAULT', 
                      'N', 'bg-N', 'frow-N', 'frow-bg-N', 'N-NET',
                      'PCT', 'bg-PCT', 'frow-PCT', 'frow-bg-PCT', 'PCT-NET', 
                      'STR', 'TESTS',
                      'bg-DEFAULT', 'bg-TESTS']

        for side in sides_list:
            if len(side) > 0:
                sides = '-'.join(['left', side])
            else:
                sides= 'left'
            for cell_type in types_list:
                key = '-'.join([sides, cell_type]) 
                self.format_dict.update({key: self._get_value(key)})
        
    def _get_value(self, key):
        """
        """
        value = {'text_v_align': 2, 
                 'text_h_align': 2}
        
        # Starting side
        if key.startswith('left'):
            value.update({'left': self.border_style_ext,
                          'left_color': self.border_color})

        # Extra sides
        for side in ['right', 'top', 'bottom']:
            if '-{}-'.format(side) in key:
                value.update({side: self.border_style_ext,
                              '{}_color'.format(side): self.border_color})

        # Cell type
        if key.endswith('-DESCRIPTIVES'):
            for side in ['top', 'bottom']:
                if not side in value.keys():
                    value.update({side: self.border_style_int,
                                  '{}_color'.format(side): self.border_color})
            value.update({'num_format': self.num_format_descriptives,
                          'font_name': self.font_name_descriptives,
                          'font_size': self.font_size_descriptives,
                          'font_color': self.font_color_descriptives,
                          'bold': self.font_bold_descriptives})
        elif key.endswith('-DEFAULT'):
            value.update({'num_format': self.num_format_default,
                          'bg_color': self.bg_color if 'bg' in key else 0,
                          'font_name': self.font_name,
                          'font_size': self.font_size})
        elif key.endswith('-N'):
            if 'frow' in key:
                value.update({'top': self.border_style_int,
                              'top_color': self.border_color})
            value.update({'num_format': self.num_format_n,
                          'bg_color': self.bg_color if 'bg' in key else 0,
                          'font_name': self.font_name,
                          'font_size': self.font_size})
        elif key.endswith('-N-NET'):
            for side in ['top', 'bottom']:
                if not side in value.keys():
                    value.update({side: self.border_style_int,
                                  '{}_color'.format(side): self.border_color})
            value.update({'num_format': self.num_format_n,
                          'font_name': self.font_name,
                          'font_size': self.font_size})
        elif key.endswith('-PCT'):
            if 'frow' in key:
                value.update({'top': self.border_style_int,
                              'top_color': self.border_color})
            value.update({'num_format': self.num_format_pct,
                          'bg_color': self.bg_color if 'bg' in key else 0,
                          'font_name': self.font_name,
                          'font_size': self.font_size})
        elif key.endswith('-PCT-NET'):
            for side in ['top', 'bottom']:
                if not side in value.keys():
                    value.update({side: self.border_style_int,
                                  '{}_color'.format(side): self.border_color})
            value.update({'num_format': self.num_format_pct,
                          'font_name': self.font_name,
                          'font_size': self.font_size})
        elif key.endswith('-STR'):
            if not 'right' in value.keys():
                value.update({'right': self.border_style_int,
                              'right_color': self.border_color})
            value.update({'font_name': self.font_name_str,
                          'font_size': self.font_size_str,
                          'font_color': self.font_color_str})
        elif key.endswith('-TESTS'):
            value.update({'font_name': self.font_name_tests,
                          'font_size': self.font_size_tests,
                          'font_color': self.font_color_tests,
                          'font_script': self.font_super_tests,
                          'bg_color': self.bg_color if 'bg' in key \
                            else self.bg_color_tests,
                          'bold': self.font_bold_tests})

        return value




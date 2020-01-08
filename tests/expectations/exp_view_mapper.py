
ITERATIONS_DEFAULT = [
    "x|default|:|||default",
    "x|default|:||weight_a|default",
    "x|default|:||weight_b|default"]

ITERATIONS_CBASE = [
    "x|f|x:|||cbase",
    "x|f|x:||weight_a|cbase",
    "x|f|x:||weight_b|cbase"]

ITERATIONS_PCT = [
    "x|f|:|y||c%",
    "x|f|:|y|weight_a|c%",
    "x|f|:|y|weight_b|c%"]

ITERATIONS_COUNTS = [
    "x|f|:|||counts",
    "x|f|:||weight_a|counts",
    "x|f|:||weight_b|counts"]

ITERATIONS_BASIC = (
    ITERATIONS_DEFAULT + ITERATIONS_CBASE + ITERATIONS_PCT + ITERATIONS_COUNTS)

ITERATIONS_EVER = [
    "x|f|x[{1,2}]:|||ever",
    "x|f|x[{1,2}]:|y||ever",
    "x|f|x[{1,2}]:||weight_a|ever",
    "x|f|x[{1,2}]:|y|weight_a|ever"]

ITERATIONS_EVER_WGT = [
    "x|f|x[{1,2}]:||weight_b|ever",
    "x|f|x[{1,2}]:|y|weight_b|ever"]

ITERATIONS_EVER_NEVER = [
    "x|f|x[{1,2}]:|||ever (multi test)",
    "x|f|x[{1,2}]:|y||ever (multi test)",
    "x|f|x[{1,2}]:||weight_a|ever (multi test)",
    "x|f|x[{1,2}]:|y|weight_a|ever (multi test)",
    "x|f|x[{2,3}]:|||never (multi test)",
    "x|f|x[{2,3}]:|y||never (multi test)",
    "x|f|x[{2,3}]:||weight_a|never (multi test)",
    "x|f|x[{2,3}]:|y|weight_a|never (multi test)"]

# -----------------------------------------------------------------------------
# Quantipy views
# -----------------------------------------------------------------------------

DEFAULT_INT_UNWGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:|||default",
        "is_weighted": False,
        "weights": "",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "age",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (7, 1)
}

DEFAULT_INT_UNWGT = (
    212.848047, "x|default|:|||default", DEFAULT_INT_UNWGT_META)

DEFAULT_INT_WGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:||weight_a|default",
        "is_weighted": True,
        "weights": "weight_a",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "age",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (7, 1)
}

DEFAULT_INT_WGT = (
    212.821229, "x|default|:||weight_a|default", DEFAULT_INT_WGT_META)

DEFAULT_FLOAT_UNWGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:|||default",
        "is_weighted": False,
        "weights": "",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "weight_b",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (7, 1)
}

DEFAULT_FLOAT_UNWGT = (
    9.098562, "x|default|:|||default", DEFAULT_FLOAT_UNWGT_META)

DEFAULT_FLOAT_WGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:||weight_a|default",
        "is_weighted": True,
        "weights": "weight_a",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "weight_b",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (7, 1)
}

DEFAULT_FLOAT_WGT = (
    11.513689, "x|default|:||weight_a|default", DEFAULT_FLOAT_WGT_META)

DEFAULT_SINGLE_UNWGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:|||default",
        "is_weighted": False,
        "weights": "",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "gender",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (3, 1)
}

DEFAULT_SINGLE_UNWGT = (
    16510.0, "x|default|:|||default", DEFAULT_SINGLE_UNWGT_META)

DEFAULT_SINGLE_WGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:||weight_a|default",
        "is_weighted": True,
        "weights": "weight_a",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "gender",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (3, 1)
}

DEFAULT_SINGLE_WGT = (
    16510.0, "x|default|:||weight_a|default", DEFAULT_SINGLE_WGT_META)

DEFAULT_DELIMITED_SET_UNWGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:|||default",
        "is_weighted": False,
        "weights": "",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "q9",
        "is_multi": True,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (8, 1)
}

DEFAULT_DELIMITED_SET_UNWGT = (
    20504.0, "x|default|:|||default", DEFAULT_DELIMITED_SET_UNWGT_META)

DEFAULT_DELIMITED_SET_WGT_META = {
    "agg": {
        "name": "default",
        "fullname": "x|default|:||weight_a|default",
        "is_weighted": True,
        "weights": "weight_a",
        "method": "default",
        "text": "",
        "grp_text_map": None,
        "is_block": False
    },
    "x": {
        "name": "q9",
        "is_multi": True,
        "is_nested": False,
        "is_array": False
    },
    "y": {
        "name": "@",
        "is_multi": False,
        "is_nested": False,
        "is_array": False
    },
    "shape": (8, 1)
}

DEFAULT_DELIMITED_SET_WGT = (
    20622.967226, "x|default|:||weight_a|default", DEFAULT_DELIMITED_SET_WGT_META)

DEFAULT_SET_ON_SET = {
    "x_all": [[
        8255.000000000611, 3052.249139193268, 1723.1099695522712,
        896.6852717638529, 2280.3900812423453, 341.12706352421003,
        1304.518495129215, 2769.887205952252]],
    "y_all": [
        [8255.000000000611], [3052.2491391932563], [1723.1099695522678],
        [896.685271763852], [2280.390081242337], [341.1270635242103],
        [1304.518495129213], [2769.8872059522414]],
    "x_axis": ["All", 1, 2, 3, 4, 96, 98, 99]
}

BASES_FLOAT_ON_SINGLE = {
    "x|f|x:||weight_a|cbase": {
        "slice": ((None, None, None), (None, None, None)),
        "values": [[3854.951079386677, 4168.90892510124]],
        "meta": {
            "agg": {
                "name": "cbase",
                "fullname": "x|f|x:||weight_a|cbase",
                "is_weighted": True,
                "weights": "weight_a",
                "method": "frequency",
                "text": "Base",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "weight_b",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "gender",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "shape": (1, 2)
        }
    },
    "x|f|:y||weight_a|rbase": {
        "slice": ((0, 10, None), (None, None, None)),
        "values": [
            [0.606200775342], [0.228206355215], [0.229625249702],
            [0.233572172205], [4.410317090241001], [1.29664815927],
            [1.055148517124], [4.68784645512], [0.572517025908],
            [0.8659851290640002]],
        "meta": {
            "agg": {
                "name": "rbase",
                "fullname": "x|f|:y||weight_a|rbase",
                "is_weighted": True,
                "weights": "weight_a",
                "method": "frequency",
                "text": "Base",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "weight_b",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "gender",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "shape": (727, 1)
        }
    }
}

FREQ_SINGLE_ON_SET = {
    "x|f|:||weight_a|counts": {
        "slice": ((8, None, None), (5, None, None)),
        "values": [
            [0.0, 0.0, 3.44709414359, 0.0],
            [0.821382495122, 22.558078157094002, 0.0, 1.732273332958],
            [0.955723639461, 8.362635087097, 0.826341200496,
             24.219637692783998],
            [0.573290188245, 11.229184465293999, 1.8703201042210003,
             39.487432678395]],
        "meta": {
            "agg": {
                "name": "counts",
                "fullname": "x|f|:||weight_a|counts",
                "is_weighted": True,
                "weights": "weight_a",
                "method": "frequency",
                "text": "",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "q1",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "q3",
                "is_multi": True,
                "is_nested": False,
                "is_array": False
            },
            "shape": (12, 9)
        }
    },
    "x|f|:|y|weight_a|c%": {
        "slice": ((0, 4, None), (0, 4, None)),
        "values": [
            [3.4164893569353234, 3.417954270304682, 3.492018770073583,
             3.8586355507159444],
            [4.227870164831371, 4.578461082166348, 4.584324860290225,
             3.1928372306033808],
            [23.56828772697806, 22.987572077142847, 26.587049664959217,
             17.579103923262892],
            [41.25716641855391, 43.16460410688994, 38.3839681942993,
             50.59512968991407]],
        "meta": {
            "agg": {
                "name": "c%",
                "fullname": "x|f|:|y|weight_a|c%",
                "is_weighted": True,
                "weights": "weight_a",
                "method": "frequency",
                "text": "",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "q1",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "q3",
                "is_multi": True,
                "is_nested": False,
                "is_array": False
            },
            "shape": (12, 9)
        }
    },
    "x|f|:|x|weight_a|r%": {
        "slice": ((11, None, None), (None, None, None)),
        "values": [[
            64.12450762534469, 36.81913905767139, 80.14448727523266,
            2.8203344805561996, 0.3708073255480199, 0.1906190441644346,
            3.733704942819817, 0.6218816192189421, 13.129575262234159]],
        "meta": {
            "agg": {
                "name": "r%",
                "fullname": "x|f|:|x|weight_a|r%",
                "is_weighted": True,
                "weights": "weight_a",
                "method": "frequency",
                "text": "",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "q1",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "q3",
                "is_multi": True,
                "is_nested": False,
                "is_array": False
            },
            "shape": (12, 9)
        }
    },
}

FREQ_SINGLE_ON_SINGLE = {
    "x|f|:|||counts": {
        "slice": ((12, None, None), (8, None, None)),
        "values": [
            [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 3.0],
            [0.0, 2.0, 2.0, 20.0], [1.0, 1.0, 2.0, 8.0]],
        "meta": {
            "agg": {
                "name": "counts",
                "fullname": "x|f|:|||counts",
                "is_weighted": False,
                "weights": "",
                "method": "frequency",
                "text": "",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "religion",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "q1",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "shape": (16, 12)
        }
    },
    "x|f|:|y||c%": {
        "slice": ((0, 4, None), (0, 4, None)),
        "values": [
            [34.437086092715234, 31.41592920353982, 32.68416596104996,
             30.646193218170186],
            [37.086092715231786, 39.38053097345133, 38.01862828111769,
             39.027511196417144],
            [11.258278145695364, 6.637168141592921, 10.160880609652837,
             10.17274472168906],
            [7.28476821192053, 1.7699115044247788, 3.302286198137172,
             5.502239283429303]],
        "meta": {
            "agg": {
                "name": "c%",
                "fullname": "x|f|:|y||c%",
                "is_weighted": False,
                "weights": "",
                "method": "frequency",
                "text": "",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "religion",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "q1",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "shape": (16, 12)
        }
    },
    "x|f|:|x||r%": {
        "slice": ((15, None, None), (None, None, None)),
        "values": [[
            1.2658227848101267, 5.063291139240507, 35.44303797468354,
            30.37974683544304, 1.2658227848101267, 3.79746835443038,
            6.329113924050633, 1.2658227848101267, 1.2658227848101267,
            1.2658227848101267, 2.5316455696202533, 10.126582278481013]],
        "meta": {
            "agg": {
                "name": "r%",
                "fullname": "x|f|:|x||r%",
                "is_weighted": False,
                "weights": "",
                "method": "frequency",
                "text": "",
                "grp_text_map": None,
                "is_block": False
            },
            "x": {
                "name": "religion",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "y": {
                "name": "q1",
                "is_multi": False,
                "is_nested": False,
                "is_array": False
            },
            "shape": (16, 12)
        }
    },
}

DESC_MEANS_BASIC = {
    ("age", "q9"): [[
        34.034, 33.66126013724267, 33.94308943089431, 34.05971520440974,
        33.23160762942779, 33.92709867452135, 33.91605966007631]],
    ("age", "religion"): [[
        33.96017699115044, 33.86948297604035, 33.506637168141594,
        34.101063829787236, 34.888, 33.984375, 34.23529411764706, 39.0, 34.0,
        34.27272727272727, 34.421052631578945, 33.490196078431374,
        36.18181818181818, 36.458333333333336, 32.646511627906975]],
    ("religion", "q9"): [[
        3.2643908969210176, 3.3703703703703702, 3.360189573459716,
        3.3056603773584907, 3.4771573604060912, 3.314327485380117,
        3.3649588867805185]] ,
    ("religion", "religion"): [[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        14.0, 15.0]] ,
    ("weight_b", "q9"): [[
        1.0182296770611803, 1.0758356686570276, 1.0418269358274441,
        1.0456914765196268, 0.9312216701981225, 0.9610482592712275,
        0.9597691403990088]],
    ("weight_b", "religion"): [[
        0.9100697643462116, 1.0207477587602642, 1.0697000971715473,
        0.8942976339536192, 1.0811501419933254, 0.8132031171672097,
        0.9227091789183529, 0.731571892767, 1.2594213912240002,
        0.8741697110032545, 0.9767286348968889, 1.0179290882704313,
        1.0400800039273637, 0.911321417807913, 0.8929007681113169]],
}

DESC_MEANS_COMPLEX = {
    ('religion', 'religion'): {
        "values": [[
            1.0, 0.00, 0.00, 299.99999999999994, 5.0, 6.0, 900.00, 8.000,
            9.00, 10.00, 11.00, 12.00, 13.00, 14.0, 15.0]],
        "vk": "x|d.mean|x[{1,300,5,6,900,8,9,10,11,12,13,14,15,16}]:||weight_a|mean"  # noqa
        },
    ('religion', 'q9'): {
        "values": [[
            42.551728834710325, 39.24544561891599, 27.051829048758677,
            33.24349179295309, 37.32054819314336, 28.864524650046256,
            35.661188879722964]],
        "vk": "x|d.mean|x[{1,300,5,6,900,8,9,10,11,12,13,14,15,16}]:||weight_a|mean"  # noqa
    }
}

FREQUENCY_NET = [[1673.71012585387, 2019.67696380492]]

FREQUENCY_NPS = [
    [34.46666666666667, 34.6849656893325, 33.681765389082464,
     33.210840606339, 35.694822888283376, 37.187039764359355,
     38.60561914672216],
    [40.400000000000006, 39.30131004366812, 39.60511033681765,
     40.652273771244836, 34.05994550408719, 34.75699558173785,
     32.327436697884146],
    [7.766666666666666, 8.484092326887087, 8.362369337979095,
     7.808911345888838, 9.536784741144414, 8.541973490427099,
     8.047173083593478],
    [26.700000000000003, 26.200873362445414, 25.319396051103364,
     25.401929260450164, 26.158038147138964, 28.64506627393225,
     30.558446063128685]]

CUM_SUM_COUNTS = {
    "x|f.c:f|x++:|||counts_cumsum": [
        [473, 476], [585, 580], [878, 882], [1385, 1345], [2021, 1944],
        [2186, 2062], [2212, 2085]]
}

CUM_SUM_CPCTS = {
    "x|f.c:f|x++:|y||c%_cumsum": [
        [4.39733495, 12.87704422, 44.34887947, 45.85099939, 75.79648698,
         78.21926105, 100.],
        [8.29800121, 24.07026045, 58.21926105, 59.04300424, 79.80617807,
         82.14415506, 100.],
        [7.41368867, 22.4954573 , 55.22713507, 56.02665051, 76.74136887,
         80.48455482, 100.],
        [6.66262871, 16.93519079, 38.691702, 39.5517868 , 61.35675348,
         69.04906118, 100.],
        [33.28891581, 46.94124773, 73.50696548, 73.91883707, 76.6444579,
         81.07813446, 100.],
        [5.29376136, 13.50696548, 43.98546336, 45.24530588, 73.03452453,
         75.59055118, 100.]]
}

EFFECTIVE_BASE = {
    ("gender", "@"): [[5473.2]],
    ("gender", "q8"): [[641.0, 132.8, 417.5, 658.9, 823.5, 187.9, 35.1]],
}

DESC_OTHER_SOURCE = {
    "x|d.mean|weight_a:||weight_b|foreign_stats": [[
        1.4874813073223419, 1.4484446605625558, 1.57250526358392]],
    "x|d.stddev|weight_a:||weight_b|foreign_stats": [[
        0.922248166021751, 0.902477800807479, 0.9259579927078846]],
    "x|t.means.Dim.10|weight_a:||weight_b|source_sig_test":
        [['NONE', 'NONE', '[2]']]
}

# -----------------------------------------------------------------------------
# coltests means
# -----------------------------------------------------------------------------
MEAN_KWARGS1 = ("all", {
    'text': '(all codes))',
    'axis': 'x'})

SIG_KWARGS1 = ("DIM_means_test", {
    'metric': 'means',
    'text': 'SIG (means)',
    'iterators': {'level': [0.05]}
})

EXP_COLTEST1 = {
    "x|t.means.Dim.05|x:||weight_a|DIM_means_test": {
        "values": [['NONE', 'NONE', 'NONE', '[1, 2]', '[1, 2, 3]']],
        "level": 0.05
    }
}

MEAN_KWARGS2 = ("excl_9798", {
    'text': '(no missings))',
    'exclude': [97, 98],
    'axis': 'x'})

SIG_KWARGS2 = ("DIM_means_test", {
    'metric': 'means',
    'text': 'SIG (means)',
    'iterators': {'level': [0.20]}
})

EXP_COLTEST2 = {
    "x|t.means.Dim.20|x[{1,2,3,4,5}]:||weight_a|DIM_means_test": {
        "values": [['NONE', '[1, 3]', 'NONE', 'NONE', 'NONE']],
        "level": 0.20
    }
}

MEAN_KWARGS3 = ("excl_9798", {
    'text': '(no missings))',
    'exclude': [97, 98],
    'axis': 'x'})

SIG_KWARGS3 = ("DIM_means_test", {
    'metric': 'means',
    'text': 'SIG (means)',
    'iterators': {'level': ["low"]}
})

EXP_COLTEST3 = {
    "x|t.means.Dim.10|x[{1,2,3,4,5}]:|||DIM_means_test": {
        "values": [['[5]', '[5]', '[1, 2, 5, 6, 7, 97]', '[5]', 'NONE', '[5]',
                    '[5]', '[5]', 'NONE']],
        "level": 0.10
    }
}

MEAN_KWARGS4 = ("all", {
    'text': '(all codes))',
    'axis': 'x'})

SIG_KWARGS4 = ("askia_means_test", {
    'metric': 'means',
    'mimic': 'askia',
    'text': 'SIG (means)',
    'iterators': {'level': ["high"]}
})

EXP_COLTEST4 = {
    "x|t.means.askia.01|x:|||askia_means_test": {
        "values": [['NONE', '[1]', '[1]', '[1, 2, 3]', '[1, 2, 3, 4]']],
        "level": 0.01
    }
}

MEAN_KWARGS5 = ("excl. 6,8", {
    'stats': "mean",
    'exclude': [6, 8],
    'axis': 'x'})

SIG_KWARGS5 = ("total_tests", {
    'metric': 'means',
    'test_total': True
})

EXP_COLTEST5 = {
    "x|t.means.Dim.10+@|x[{1,2,3,4,5,7,9}]:|||total_tests": {
        "values": [
            ["['@L', 98]", '[98]', '[98]', "['@L', 98]", '[98]', '[98]',
             "['@H']"]],
        "level": 0.1
    }
}

MEAN_KWARGS6 = ("excl. 6,8", {
    'stats': "mean",
    'exclude': [6, 8],
    'axis': 'x'})

SIG_KWARGS6 = ("total_tests_flags", {
    'metric': 'means',
    'test_total': True,
    'flag_bases': [30, 100]
})

EXP_COLTEST6 = {
    "x|t.means.Dim.10+@|x[{1,2,3,4,5,7,9}]:|||total_tests_flags": {
        "values": [["['@L']", '*', 'NONE', "['@L']", 'NONE', '*', '**']],
        "level": 0.1
    }
}

COLTEST_MEAN_INPUT_EXPECT = [  # noqa
    ("q5_1", "locality", "weight_a", *MEAN_KWARGS1, *SIG_KWARGS1, EXP_COLTEST1),
    ("q5_1", "locality", "weight_a", *MEAN_KWARGS2, *SIG_KWARGS2, EXP_COLTEST2),
    ("q5_1", "q3", None, *MEAN_KWARGS3, *SIG_KWARGS3, EXP_COLTEST3),
    ("q5_1", "locality", None, *MEAN_KWARGS4, *SIG_KWARGS4, EXP_COLTEST4),
    ("q7_1", "q8", None, *MEAN_KWARGS5, *SIG_KWARGS5, EXP_COLTEST5),
    ("q7_1", "q8", None, *MEAN_KWARGS6, *SIG_KWARGS6, EXP_COLTEST6),
]

# -----------------------------------------------------------------------------
# coltests props
# -----------------------------------------------------------------------------
SIG_KWARGS7 = ("DIM_props_test", {
    'metric': 'props',
    'rel_to': 'y',
    'text': 'sig without overlap',
    'iterators': {'level': [0.20]}
})

EXP_COLTEST7 = {
    "x|t.props.Dim.20|:|y|weight_a|DIM_props_test": {
        "values": [
            ['[2, 3, 4, 5]', '[4]', 'NONE', 'NONE', 'NONE'],
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', '[1, 4]', 'NONE', 'NONE', '[4]'],
            ['[2, 5]', 'NONE', '[5]', '[2, 5]', 'NONE'],
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', '[1]', '[1]', 'NONE', 'NONE'],
            ['NONE', 'NONE', 'NONE', '[1]', '[1]'],
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['[2, 3]', 'NONE', 'NONE', 'NONE', '[2, 3]'],
            ['[3]', '[1, 3]', 'NONE', '[3]', '[3]']],
        "level": 0.2,
        "text": "sig without overlap"
    }
}

SIG_KWARGS8 = ("DIM_props_test", {
    'metric': 'props',
    'rel_to': 'y',
    'text': 'SIG (props)',
    'iterators': {'level': ["mid"]}
})

EXP_COLTEST8 = {
    "x|t.props.Dim.05|:|y||DIM_props_test": {
        "values": [
            ['[2, 3]', 'NONE', 'NONE', 'NONE', '[1, 2, 3, 4, 6, 7, 8, 97]',
             'NONE', '[2, 3]', '[2, 3, 97]', 'NONE'],
            ['[3, 97]', '[1, 3, 97]', '[97]', '[3, 97]', '[1, 2, 3, 4, 7, 97]',
             '[1, 2, 3, 4, 7, 97]', '[1, 3, 97]', '[3, 97]', 'NONE'],
            ['[2, 3, 4, 8, 97]', '[8, 97]', '[2, 4, 8, 97]', '[97]', '[97]',
             '[97]', '[2, 4, 8, 97]', '[97]', 'NONE'],
            ['[3]', '[3]', 'NONE', '[1, 2, 3, 5, 6, 7, 97]', 'NONE', 'NONE',
             'NONE', '[1, 2, 3, 4, 5, 6, 7, 97]', '[1, 2, 3, 7]'],
            ['[97]', '[97]', '[97]', '[97]', '[97]', '[97]', '[97]', '[97]',
             'NONE'],
            ['NONE', '[1, 5, 7]', '[1, 5, 7]', 'NONE', 'NONE', 'NONE', 'NONE',
             'NONE', '[1, 2, 3, 4, 5, 6, 7, 8]'],
            ['[5, 6]', '[1, 5, 6, 7, 8]', '[1, 2, 4, 5, 6, 7, 8]', '[6]',
             'NONE', 'NONE', '[6]', 'NONE', '[1, 2, 3, 4, 5, 6, 7, 8]']],
        "level": 0.05,
        "text": "SIG (props)"
    }
}

SIG_KWARGS9 = ("DIM_props_test", {
    'metric': 'props',
    'rel_to': 'y',
    'text': 'SIG (props, strict)',
    'iterators': {'level': [0.01]}
})

EXP_COLTEST9 = {
    "x|t.props.Dim.01|:|y|weight_a|DIM_props_test": {
        "values": [
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', 'NONE', '[1, 96]', '[96]', '[96]', 'NONE', 'NONE'],
            ['NONE', 'NONE', '[4]', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', 'NONE', '[98]', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', '[3]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
            ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE']],
        "level": 0.01,
        "text": "SIG (props, strict)"
    }
}

COLTEST_PROPS_INPUT_EXPECT = [  # noqa
    ("q1", "locality", "weight_a", *SIG_KWARGS7, EXP_COLTEST7),
    ("q5_1", "q3", None, *SIG_KWARGS8, EXP_COLTEST8),
    ("q9", "q8", "weight_a", *SIG_KWARGS9, EXP_COLTEST9),
]

COLTEST_MEANS_PROPS = {
    "x|t.props.Dim.10+@|:||weight_a|total_tests": [
        ['NONE', "['@L', 1, 3, 4, 5, 96]", 'NONE', '[5]', "['@H']", 'NONE',
         'NONE'],
        ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
        ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
        ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
        ['[4]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
        ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
        ['NONE', 'NONE', 'NONE', 'NONE', '[96]', 'NONE', 'NONE'],
        ['[5]', '[5]', 'NONE', '[5]', "['@H']", "['@L', 5]",
         "['@L', 1, 3, 4, 5]"],
        ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE']],
    "x|t.means.Dim.10+@|x:||weight_a|total_tests": [
        ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', "['@L', 2]", 'NONE']]
}
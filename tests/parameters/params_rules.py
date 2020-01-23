
def index_items(col, values):
    return [(col, i) for i in values]

SLICEX_PARAMS = [
    # {
    #     "xk": "religion",
    #     "yk": "ethnicity",
    #     "slice": [
    #         ("religion", [1, 3, 5, 7, 9, 11, 13, 15], "x"),
    #         ("ethnicity", [2, 4, 6, 8, 10, 12, 14, 16], "y")],
    #     "weight": None
    # },
    {
        "xk": "q5",
        "yk": "@",
        "sort": [("q5", "mean")],
        "weight": None
    }
]

SLICEX_EXP = [
    # {
    #     "x": index_items("religion", [1, 3, 5, 7, 9, 11, 13, 15]),
    #     "y": index_items("ethnicity", [2, 4, 6, 8, 10, 12, 14, 16]),
    # },
    {
        "x": index_items("q5", ['q5_4', 'q5_6', 'q5_1', 'q5_3', 'q5_5', 'q5_2']),
        "y": index_items("ethnicity", [2, 4, 6, 8, 10, 12, 14, 16]),
    }
]
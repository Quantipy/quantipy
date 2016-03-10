from quantipy.core.link import Link
from quantipy.core.view import View
from quantipy.core.chain import Chain
from quantipy.core.stack import Stack
from quantipy.core.cluster import Cluster
from quantipy.core.weights.rim import Rim
from quantipy.core.weights.weight_engine import WeightEngine
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews
import quantipy.core.helpers.functions as helpers
import quantipy.core.tools.dp as dp
import quantipy.core.tools.view as v

from quantipy.core.quantify.engine import Quantity, Test

from quantipy.core.builds.excel.excel_painter import ExcelPainter
from quantipy.sandbox import sandbox

try:
	from quantipy.core.builds.powerpoint.pptx_painter import PowerPointPainter
except:
	# print ("You do not have the required resources to use PowerPointPainter")
	pass

from quantipy.version import version as __version__
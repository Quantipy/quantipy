
from quantipy.core.view import View


class TestView:
    # The View object has undergone massive changes.
    # We will need to add tests for all new View attributes and methods
    # that are used make self-inspection possible.
    # Also: the constructor is changend completely.

    def test_view_fake(self):
        view = View()
        assert isinstance(view, View)

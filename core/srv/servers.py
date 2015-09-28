import json
import webbrowser

from .core import start_server, copy_html_template, open_tmp_file, cleanup_tmp_folder
from .handlers import RequestViewsHandler


def request_views_webeditor(request_views, host="localhost", port=8000):
    cleanup_tmp_folder()
    url = "http://{host}:{port}/core/srv/tmp/request_views.html".format(host=host, port=port)

    json_string = json.dumps(request_views)
    copy_html_template('request_views.html', json_string, "REPLACEJSON")
    tab = webbrowser.open_new_tab(url)

    # This runs forever and can only be shut down in the handler or by ctr+c
    start_server(host=host, port=port, handler=RequestViewsHandler)

    try:
        request_views = json.loads(open_tmp_file('request_views.json').readline())
    except:
        pass

    cleanup_tmp_folder()
    return request_views

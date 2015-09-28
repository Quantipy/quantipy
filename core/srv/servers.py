import json
import webbrowser
from collections import OrderedDict

from .core import start_server, copy_html_template, open_tmp_file, cleanup_tmp_folder
from .handlers import WebEditHandler


def webeditor(obj, host="localhost", port=8000):
    cleanup_tmp_folder()
    url = "http://{host}:{port}/core/srv/tmp/webedit.html".format(
    	host=host, 
    	port=port)

    json_string = json.dumps(obj, sort_keys=True)
    copy_html_template('webedit.html', json_string, "REPLACEJSON")
    tab = webbrowser.open_new_tab(url)

    # This runs forever and can only be shut down in the handler or by 
    # ctr+c
    start_server(host=host, port=port, handler=WebEditHandler)

    try:
        obj = json.loads(
        	open_tmp_file('obj.json').readline(),
        	object_pairs_hook=OrderedDict)
    except:
        pass

    cleanup_tmp_folder()
    return obj

import http.server
import urllib.parse
import cgi

from .core import shutdown_server, save_string_in_tmp_folder

class WebEditHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        """ Shut down if it's a get request """
        if 'shutdown' in self.path:
            shutdown_server(server_target=self.server)
        http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        """ Store the result in a file, shut down the serve and
        then continue the script """
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })
        for item in form.list:
            print(item.name)
            if item.name == "obj_json":
                save_string_in_tmp_folder(
                    data=item.value, 
                    filename="obj.json")
                break

        shutdown_server(server_target=self.server)
        http.server.SimpleHTTPRequestHandler.do_GET(self)

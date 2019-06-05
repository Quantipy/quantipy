import socketserver
import threading
import json
import time
import os
import shutil


from shutil import copyfile


def copy_html_template(name, new_string=None, old_string=None,
                       path="core/srv/html_templates", 
                       tmp_path="core/srv/tmp"):
    """ Copies a file from html_templates/ to tmp/ and replaces a string
        in the contents if it finds it.
    """
    filepath = "{}/{path}/{name}".format(
        os.getcwd(), path=path, name=name)
    
    tmp_filepath = "{}/{path}/{name}".format(
        os.getcwd(), path=tmp_path, name=name)

    copyfile(filepath, tmp_filepath)

    if all([new_string, old_string]):
        with open(tmp_filepath, "w+b") as fout:
            with open(filepath, "r+b") as fin:
                for line in fin:
                    fout.write(line.replace(old_string, new_string))

def save_string_in_tmp_folder(data, filename, path="core/srv/tmp"):
    filepath = "{}/{path}/{name}".format(
        os.getcwd(), path=path, name=filename)    
    with open(filepath, "w+b") as text_file:
        text_file.write(data)

def open_tmp_file(filename):
    filepath = "{}/core/srv/tmp/{name}".format(
        os.getcwd(), name=filename)
    return open(filepath, "r+b")

def cleanup_tmp_folder():
    folder = "{}/core/srv/tmp".format(os.getcwd())
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            pass
            # print e

def is_port_taken(host, port):
    """ Return True/False depending on if the port is taken or not"""
    socket = socketserver.socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        s.shutdown(1)
        time.sleep(2)
        return True
    except:
        return False

def shutdown_server(server_target):
    """ Spawns a thread that triggers the TCPServer.shutdown method """
    assassin = threading.Thread(target=server_target.shutdown)
    assassin.daemon = True
    assassin.start()

def print_server_message(host, port, handler):
    print("Quantipy http server version 1.0")
    print("Serving at: http://{host}:{port}".format(host=host, port=port))
    print("Handler : {name}".format(name=handler.__name__))

def start_server(host, port, handler):
    """ Starts a SimpleHTTPServer with a speciffic handler.

        The handler needs to trigger the TCPServer.shutdown method or 
        else the server runs until doomsday.
    """
    httpd = socketserver.TCPServer((host, port), handler)
    print_server_message(host, port, handler)
    httpd.serve_forever() # This is stopped by using the handler

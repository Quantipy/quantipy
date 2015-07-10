import os
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

"""
    The autotests.py runs the tests if any files change in the ./core/ folder.

    CODE COVERAGE ::
    Code coverage can be run by using `coverage run -m unittest discover`

    CODE COVERAGE REPORTS ::
    To create html coverage reports use `coverage html`, that creates a folder
    named /htmlcov/. Open /htmlcov/index.html using a browser to view the report

    note :: the reports must be created after running the coverage

    CODE COVERAGE INSTALL ::
    `pip install coverage`

"""

class TestHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print "Changes detected..."
        print "Running tests..."
        os.system('python -m unittest discover')
#        os.system('coverage run -m unittest discover')

if __name__ == "__main__":
    os.system('python -m unittest discover')
    path = sys.argv[1] if len(sys.argv) > 1 else './core/'
    event_handler = TestHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

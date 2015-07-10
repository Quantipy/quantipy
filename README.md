#Quantipy

#Required libraries before installation

    - Python 2.7
    - Pandas 0.14
    - pylzma

    Optional (recommended for development):
        - Watchdog
        - Coverage


##Installation for Windows:

Gohlke has precompiled all of the required libraries for windows:

_(Note: Python 2.7 must be installed before attempting to install these libraries)_

Allways choose the same: win32-py2.7 or amd64-py2.7

0. [setuptools](http://www.lfd.uci.edu/~gohlke/pythonlibs/#setuptools)
1. [NumPy (1.8)](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
2. [python-dateutil](http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-dateutil)
3. [pytz](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pytz)
4. [six](http://www.lfd.uci.edu/~gohlke/pythonlibs/#six)
5. [numexp](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numexpr)
6. [bottleneck](http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck)
7. [pandas](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pandas)
8. [pylzma](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pylzma)

[Git](http://www.sourcetreeapp.com/ "Download Link") is also required to install quantipy and all explanations in the following git guide are aimed at users that use this particular git client. It is recommended to install the SourceTree app during the installation process and follow the git guide below.

###Global installation
#### Using SourceTree

To make quantipy globally accessible by Python It needs to be inserted into the `site-packages` folder in the Python installation.

**Step 0:**

Open the SourceTree app :)

**Step 1:**

Click the 'clone repository' ![Clone Repository](https://www.dropbox.com/sh/n0rqv29618ehiw8/AAAhz2ffKSMjb3EBx3RjzKVUa/clone-repo.png?raw=1 "Clone Repo") icon.

**Step 2:**

Copy the Repository path from bitbucket (In the upper/right corner on bitbucket.com)

![Repository Path](https://www.dropbox.com/sh/n0rqv29618ehiw8/AABmq1wBpCzizN7r_nQ397Nca/Step1.a.png?raw=1 "Repository Path")

The path should look something like this:

    https://YOURUSERNAME@bitbucket.org/ygov/quantipy.git
    (`YOURUSERNAME` is the name of your bitbucket account and is case sensetive)

**Step 3.a:**

Paste the repository path (the link from bitbucket.com) into the _Source Path/URL_

**Step 3.b:**

The _Destination Path_ should contain the path to the `site-packages` folder for your Python installation. It should look something like this on Windows :

    C:/Python27/Lib/site-packages/quantipy

*(Note: The reason for this path is that all packages in `site-packages` are globally accessible by Python. Thus making quantipy a global installation)*

Image for clarification:

![Global Install](https://www.dropbox.com/sh/n0rqv29618ehiw8/AABiXGYX4tD4BZMPXsRqreqFa/Step3.global.png?raw=1 "Global Install")

**Step 3.c:**

Click 'Clone'.

You will be prompted for your password to your bitbucket account.

**FIN**

You should have successfully installed quantipy to your computer.
Close the SourceTree app.

Now you can import the quantipy library in your Python terminal.

    import quantipy

#Development

To develop quantipy you should clone the quantipy repository to your project directory.

##install
The method is the same as in the [Global Installation](#global-installation "Global Installation") part of this README except that the _Destination Path_ in **step 3.b** should be the path to your project directory.

####Example:
Assuming that there is a project folder named "c:\quantipy\_test\" then the _Destination Path_ in **step 3.b** should be "c:\quantipy\_test\quantipy\"

The resulting folder structure should be something like this:

    quantipy_test\
    quantipy_test\test.py
    quantipy_test\quantipy\        # This folder is the quantipy repository clone

_(Note: Only files in the quantipy folder are a part of the git repository.)_

Inside test.py it is now possible to import the library using:

    import quantipy as qp

    # Then you can do stuff like
    view = qp.View(...)
    stack = qp.Stack(...)
    # ... etc ...

##Workflow
The `Pull Request` workflow is the workflow that we (@Transmit) utilize the most.
It basically boils down to this :

1. A developer creates the feature in a dedicated branch in their local repo.
2. The developer pushes the branch to a public Bitbucket repository.
3. The developer files a pull request via Bitbucket.
4. The rest of the team reviews the code, discusses it, and alters it.
5. The project maintainer merges the feature into the official repository and closes the pull request.

[Here](https://www.atlassian.com/git/workflows#!pull-request "Pull Requests") is a good explenation of Git Workflows from Attlassian.

####Workflow Example:
This example shows how one would update the README.

_(Note: It is ok to swap step 1.a and step 1.b but this way has more order.)_

**Step 1.a**

Create a new branch from `master`
![Create New Branch](https://www.dropbox.com/sh/n0rqv29618ehiw8/AACO-ZX06quVU2y3AfqvgC9Fa/new-branch.png?raw=1 "Create New Branch")

![Create New Branch 2](https://www.dropbox.com/sh/n0rqv29618ehiw8/AAAEENGg-59lo5H72SFDpDfxa/new-branch2.png?raw=1 "Create New Branch")

**Step 1.b**

Make some changes to the README.md file (or any file).

**Step 1.c**

Commit those changes to the current branch by pressing ![Commit Icon](https://www.dropbox.com/sh/n0rqv29618ehiw8/AABZ4wkXKa6ePEeYXaLwN9pba/Commit-icon.png?raw=1 "Commit Icon")

The screen shows what files are staged for commit and what files will not be committed. (In this case there is only the README.md file to be changed.)

![Commit screen](https://www.dropbox.com/sh/n0rqv29618ehiw8/AAAOMCsCw206eSCBcDz5V9d9a/Screenshot%202014-08-19%2017.14.40.png?raw=1 "Commit Screen")

If you wish to push this branch to the bitbucket.com repository (ygov) then check `Push changes immidiately to ygov/< branch name >` and SourceTree pushes the local branch to ygov.

_(Note: In "Commit Options" it is possible to also create the `Pull Request` automatically, BUT I have not tested it.)_

Press commit

**Step 2**

*INFO: If you checked the `Push changes immidiately to ygov/< branch name >` in the previous step, then you can skip this step.

In SourceTree press ![Push](https://www.dropbox.com/sh/n0rqv29618ehiw8/AAAjdg4Hya1Pn7F6Q78uiERPa/Push.png?raw=1 "Push")

That opens a dialog.

In that dialog choose the branch(es) that you wish to push to ygov (the remote repository).

![Push](https://www.dropbox.com/sh/n0rqv29618ehiw8/AAAjBnjMNo7_bq4HGtOGTUSNa/Screenshot%202014-08-19%2015.38.49.png?raw=1 "Push dialog")

Click Ok.

The branch can now be found on the bitbucket.com/ygov.

**Step 3**

On bitbucket.org/ygov/quantipy press ![Pull Request](https://www.dropbox.com/sh/n0rqv29618ehiw8/AAAQJJpga3ctEO4vtMxHms2Qa/Screenshot%202014-08-19%2017.44.32.png?raw=1 "Pull Request") and then in the upper/right corner press `Create pull request`.

1. Choose the branch that is requested to be merged into the codebase.
2. Make shure that the pull request is into the correct remote branch.
3. Choose the reviewers of the pull-request.
4. Create the pull-request.

![Create Pull Request](https://www.dropbox.com/sh/n0rqv29618ehiw8/AAAesU7e_zC9slxhXwJCXHE2a/Pull-Request-page.png?raw=1 "Create Pull Request")

**Step 4**

The rest of the team reviews the code, discusses it, and alters it.

**Step 5**

The project maintainer merges the feature into the official repository and closes the pull request.

_(This can be done through the bitbucket.org web-interface, so the project maintainer doesn't actually have to have the project set up on his local computer.)_

##Tests
\#Add stuff

<!--
##Example of use:
Here is an example of quantipy usage.

    #-*- coding: utf-8 -*-
    from custom_view import CustomView, QuantipyViews
    from quantipy import Stack
    import random

    #These test files are included in the repository
    filename = "tests/example.csv"
    metadata = "tests/example.mdd"

    custom_view = CustomView('default','N','Column margins') # Cherry pick methods from the View
    stack = Stack(name='Stack 1', view=custom_view)

    # Link the data
    stack.link_data(data_key="Jan", filename=filename, metadata=metadata)
    stack.link_data(data_key="Feb", filename=filename, metadata=metadata)
    stack.link_data(data_key="Mar", filename=filename, metadata=metadata)

    # Create a custom view object

    # Create a View method to inject into custom_view (not defined in the custom_view class)
    def rand_view(link, name, kwargs):
        # A dummy function that takes a Random link from the stack
        # regardless of position in the stack and devides it by 2.
        tmp = link.stack
        # Drill through the stack until it's not a Stack/Pivot any more.
        while isinstance(tmp,type(link.stack)) or isinstance(tmp,type(link)):
            next_key = tmp.keys()[random.randrange(0,len(tmp.keys()))]
            next_key = next_key if next_key != "meta" else "data" # don't use 'meta'
            tmp = tmp[next_key]

        link[name] = tmp / 2

    # Add the new View method to the custom_view instance.
    custom_view.add_method('rand_view', rand_view)

    # Add the pivots with different views
    stack.add_link(data_keys="Jan") # This uses the default QuantipyViews()
    stack.add_link(data_keys="Feb", views=QuantipyViews('default wgt','N wgt'))  # Cherry Pick methods (This could also be done with CustomView)
    stack.add_link(data_keys="Mar", views=custom_view) # creates these views : 'default', 'N', 'Column margins' AND 'rand_view'

    # Print some examples from the custom_view links
    print "\nThe Stack data :"
    print "Stack['Mar']['data']['no_filter']['Gender'].keys() => %s " % stack['Mar']['data']['no_filter']['Gender'].keys()
    print "Stack['Mar']['data']['no_filter']['Gender']['AgeGroup'].keys() => %s " % stack['Mar']['data']['no_filter']['Gender']['AgeGroup'].keys()
    print "\n ##############\n # default         #\n ##############"
    print "Stack['Mar']['data']['no_filter']['Gender']['AgeGroup']['default'] => \n%s " % stack['Mar']['data']['no_filter']['Gender']['AgeGroup']['default']
    print "\n ##################\n # Column margins #\n ##################"
    print "Stack['Mar']['data']['no_filter']['Gender']['AgeGroup']['Column margins'] => \n%s " % stack['Mar']['data']['no_filter']['Gender']['AgeGroup']['Column margins']
    print "\n ###############\n # rand_view   #\n ###############"
    print "Stack['Mar']['data']['no_filter']['Gender']['AgeGroup']['rand_view'] => \n%s " % stack['Mar']['data']['no_filter']['Gender']['AgeGroup']['rand_view']
-->

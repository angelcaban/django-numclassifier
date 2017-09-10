=====
numclassifier
=====

A simple Django App that invokes an existing TensorFlow model in order to classify a handwritten number.

Quick start
-----------

1. Add "numclassifier" to your INSTALLED_APPS setting::

     INSTALLED_APPS = [
         ...
	 'numclassifier',
     ]

2. Include the app in URLconf of your project's urls.py::

     url(r'^numclassifier/', include('numclassifier.urls')),

3. Run `python manage.py migrate`

4. If running on a web server, make sure that your wsgi.py file contains the environment variable "VIRTUAL_ENV" pointing to the path to a directory that will have read/write access.



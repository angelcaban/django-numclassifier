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


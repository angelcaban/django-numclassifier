import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(name="django-numclassifier",
      version="0.1",
      packages=find_packages(),
      include_package_data=True,
      license="GPLv3",
      description="Numeric image classifier",
      long_description=README,
      url="http://www.angelcaban.net/",
      author="Angel Caban",
      author_email="cnnclassifier@mail.angelcaban.net",
      classifiers=[
          "Environment :: Web Environment",
          "Framework :: Django",
          "Framework :: Django :: 1.11",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: GNU General Public License v3",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
      ],
)


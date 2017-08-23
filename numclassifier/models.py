# -*- coding: utf-8 -*-

"""
    MNIST Classification with TensorFlow on Django
    Copyright (C) 2017  Angel Caban

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals

from django.db import models
from django.conf import settings


def user_directory_path(instance, filename):
    return '{0}user_{1}/%Y/%m/%d/{2}'.format(settings.MEDIA_ROOT,
                                             instance.user.id,
                                             filename)


class UserImage(models.Model):
    """
    Represents a drawn number, the result of CNN and
    user input about whether or not it's correct.
    """

    image = models.ImageField(upload_to=user_directory_path)
    guessed = models.IntegerField(default=-1)
    percent_guess = models.CharField(max_length=10, default="-000.000%")
    is_correct_guess = models.BooleanField(default=True)
    correct_guess = models.IntegerField(default=-1)

    def __str__(self):
        return "Image at {0}".format(self.image.url)


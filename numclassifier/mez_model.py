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

from django.db import models
from django.contrib import admin
from django.core.urlresolvers import reverse
from django.utils.translation import ugettext_lazy as _

from mezzanine.pages.models import Page
from mezzanine.pages.admin import PageAdmin

from models import UserImage


class MezPage(Page):
    images = UserImage()

    class Meta:
        verbose_name = _("MNIST CNN Example")

    def get_absolute_url(self):
        url_name = "numclassifier"
        kwargs = { "slug": self.slug }
        return reverse(url_name, kwargs=kwargs)


admin.site.register(MezPage, PageAdmin)


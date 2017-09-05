from django.db import models
from django.contrib import admin
from mezzanine.pages.models import Page
from mezzanine.pages.admin import PageAdmin
import models


class MezPage(Page):
    images = models.UserImages()


admin.site.register(MezPage, PageAdmin)


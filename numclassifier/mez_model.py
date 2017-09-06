from django.db import models
from django.contrib import admin
from mezzanine.core import models
from mezzanine.pages.models import Page
from mezzanine.pages.admin import PageAdmin
from models import UserImage


class MezPage(Page):
    images = UserImage()


admin.site.register(MezPage, PageAdmin)


# -*- coding: utf-8 -*-
# Generated by Django 1.9.13 on 2017-09-10 08:40
from __future__ import unicode_literals

from django.db import migrations, models
import numclassifier.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('numclassifier', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to=numclassifier.models.user_directory_path)),
                ('guessed', models.IntegerField(default=-1)),
                ('percent_guess', models.CharField(default='-000.000%', max_length=10)),
                ('is_correct_guess', models.BooleanField(default=True)),
                ('correct_guess', models.IntegerField(default=-1)),
            ],
        ),
    ]

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

import uuid
import base64
import os

from PIL import Image
import numpy as np

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponseServerError

from .num_cnn import predict, run_learn
from .models import UserImage


def index(request):
    """ Main template """
    context = {
    }
    return render(request, 'index.tmpl.html', context)


def imgup(request):
    """ Upload a user's number as a PNG file onto the filesystem """
    if request.method == 'POST':
        img_model = UserImage()
        pngbase64 = request.POST["imgBase64"]

        try:
            start_idx = pngbase64.index(',') + 1
            bin_png = base64.decodestring(pngbase64[start_idx:])
        except KeyError:
            return HttpResponseServerError("Malformed Image Data")
        except ValueError:
            return HttpResponseServerError("Malformed Image Data")

        img_model.image.name = str(uuid.uuid4()) + ".png"

        imgfile = open(str(img_model.image.path), 'wb')
        imgfile.write(bin_png)
        imgfile.close()

        img_model.save()

        return JsonResponse({'id': img_model.pk})

    return HttpResponseServerError("Only Accepting POSTs")


def guess(request):
    """ 1) Load Up image
        2) Pass Image to TF
        3) Score
        4) Return top scored class & loss
           JSON -- { 'guess' : 'NUMBER', 'estimate' : 'LOSS' }
    """
    if request.method != 'POST':
        return HttpResponseServerError("Only Accepting POSTs")

    img_id = request.POST["id"]
    img_model = get_object_or_404(UserImage, pk=img_id)

    img = Image.open(img_model.image.path)
    img.thumbnail((28, 28))
    im_array = np.array(img)
    im_grey_list = []
    for row in im_array:
        im_grey_list.append([np.amax(c) for c in row])

    img_data = np.array(im_grey_list, dtype=float)
    img_data = np.expand_dims(img_data, axis=0)

    # img_data is 28 by 28 image from 0 to 255 where 0 is
    #  white pixel and 255 is black pixel

    prediction = predict(img_data)
    num = np.argmax(prediction['probabilities'])
    percent = str(prediction['probabilities'][0][num] * 100.0) + "%"
    result = {
        'guess': num,
        'estimate': percent,
    }

    img_model.guessed = num
    img_model.percent_guess = prediction['probabilities'][0][num]
    img_model.is_correct_guess = True
    img_model.correct_guess = num
    img_model.save()

    return JsonResponse(result)


def learn(request):
    """ Run an individual number against the CNN along with
        the whole MNIST data.
        Return JSON With loss & id
    """
    if request.method != 'POST':
        return HttpResponseServerError("Only Accepting POSTs")

    os.chdir(os.environ['VIRTUAL_ENV'])

    img_id = request.POST["img_id"]
    img_model = get_object_or_404(UserImage, pk=img_id)
    img_model.is_correct_guess = False
    img_model.correct_guess = int(request.POST["truenum"])

    img = Image.open(img_model.image.path)
    img.thumbnail((28, 28))
    im_array = np.array(img)
    im_grey_list = []
    for row in im_array:
        im_grey_list.append([np.amax(c) for c in row])

    img_data = np.array(im_grey_list, dtype=float)
    img_data = np.expand_dims(img_data, axis=0)

    results = run_learn(pixels=img_data,
                        label=np.array([img_model.correct_guess],
                                       dtype=np.int32))
    img_model.save()
    return JsonResponse({
        'loss': str(results),
        'id': img_id,
    })

from tkinter import image_names
from django.db import models
import os
# Create your models here.
# models.py


class Caption(models.Model):

    Img = models.ImageField(upload_to='images/')


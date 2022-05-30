
from urllib.request import Request
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
BASE_DIR = 'C:/Users/vhari/Downloads/archive'
WORKING_DIR = 'C:/Users/vhari/Downloads/working'
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
  
# Create your views here.
def image_view(request):
  
    if request.method == 'POST':

       # request.FILES['Img'].name = "1.jpeg"
    

        form = ImageForm(request.POST, request.FILES)
        print(request.FILES['Img'].name)
        if form.is_valid():
            form.save()


            return redirect('success')
    else:
        form = ImageForm()
    return render(request, 'caption_generator/home.html', {'form' : form})
  
  
def success(request):
    
    
    
    image = Caption.objects.all()
    caption = generate_caption(str(image.order_by('-pk')[0].Img.name).split('/')[1])
    caption = caption.split()
    caption.remove("startseq")
    caption.remove("endseq")
    caption = " ".join(caption)
    return render(request, 'caption_generator/Result.html', {'images' : [image.order_by('-pk')[0]],'caption':caption})

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    features = pickle.load(open("C:/Users/vhari/Downloads/features (1).pkl","rb"))
    tokenizer = pickle.load(open("C:/Users/vhari/Downloads/working/tokenizer.pkl","rb"))
    model = load_model("C:/Users/vhari/Downloads/Vgg16/best_model (2).h5")
    max_length = 35

    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    return(y_pred)

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X=np.load("image.npz")
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=9, train_size=3500, test_size=500)
clf=LogisticRegression(solver='saga', multi_class='multinominal').fit(X_train, y_train)

def get_prediction(image):
    #opening the image
    im_pil=Image.open(image)
    #coverting to scalar quantity(converting the pixels into a value). L is the representation of the scalar quantity
    image_bw=im_pil.convert('L')
    #resizing the image into 28,28
    image_bw_resized=image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter=20
    #getting the minimum pixel of the image
    min_pixel=np.percentile(image_bw_resized, pixel_filter)
    #calculating the maximum pixel of the image
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel=np.max(image_bw_resized)
    #creating an array
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    #predicting the sample
    test_pred=clf.predict(test_sample)
    return test_pred[0]
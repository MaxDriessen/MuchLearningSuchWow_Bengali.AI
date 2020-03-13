from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import gc
import os
import psutil
import cv2


if not os.path.isdir('data'):
    os.mkdir('data')

def resize(df, size=64, need_progress_bar=True):
    resized = {}
    resize_size = 64
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image = df.loc[df.index[i]].values.reshape(137, 236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x, y, w, h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax, xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image = df.loc[df.index[i]].values.reshape(137, 236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x, y, w, h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax, xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized


train_df_ = pd.read_csv('bengaliai-cv19/train.csv')
train_df_ = train_df_.drop(['grapheme'], axis=1)

for i in tqdm(range(4)):
    train_df = pd.merge(pd.read_feather(f'train_image_data_{i}.feather'), train_df_, on='image_id', copy=False)
    print(train_df.drop(['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).columns)
    images = resize(train_df.drop(['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).astype(np.uint8))
    images = images.values.reshape(-1, 64, 64)
    # print(type(images))
    # print(images.dtype)
    gc.collect()

    for j in tqdm(range(len(images))):
        img = Image.fromarray(images[j])
        img.save(f'data/{train_df["image_id"].iloc[j]}.png')

    gc.collect()
    print(psutil.virtual_memory().percent)
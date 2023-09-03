import cv2
import numpy as np
from torchvision import transforms


'''
This function Load and preprocess a given  image
'''

def load_and_preprocess_image(img):
    img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    return img
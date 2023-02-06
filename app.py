import cv2
import torch 
import numpy as np
import streamlit as st
import torchvision.transforms.functional as F

from torchvision import transforms
from unet import Unet
from PIL import Image


IMAGE_HEIGHT = 320
IMAGE_WIDTH = 480

transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), 
            interpolation=transforms.InterpolationMode.NEAREST),
    ])

model = Unet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load("binary_segmentation_final.pt"))
model.eval()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw)

    image = Image.fromarray(final_img)
        
    if transform is not None:
        image = transform(image)

    image_np =  cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)

    image = transforms.ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()


    pred_mask = preds[0].detach()
    pred_mask = F.to_pil_image(pred_mask)
    pred_mask = np.asarray(pred_mask)
    pred_mask = np.expand_dims(pred_mask, 2)

    masked_img = np.where(pred_mask, np.array([255,0,0], dtype='uint8'), image_np)
    masked_img = cv2.addWeighted(image_np, 0.9, masked_img, 0.2,0)

    st.image(final_img, "Original Image")
    st.image(pred_mask, "Predicted mask")
    st.image(masked_img, "Masked image")


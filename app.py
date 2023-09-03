import cv2
import numpy as np
from model import *
from data import load_and_preprocess_image

import json
import streamlit as st
import requests
from io import BytesIO


st.set_page_config(
    page_title="Explainability AI App",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded"
)


# st.title('Explanabilty AI App')
st.title('Model Interpreter 🧐')
st.markdown(" ")


with st.sidebar:
    st.markdown("# Description")
    st.markdown("---")
    st.markdown("This app take a user-provided images and try to make predictions about the contents of the image.\n \
                  In addition, it generate visual representation highlighting the regions within the input image that \
                  influence the model's classification decision for a specific class.")
    st.markdown("---")
    st.markdown("# Settings")
    pre_model = st.sidebar.checkbox("Deep learning model", value=True, help="Pretrained model used to make the predictions.")
    click = st.sidebar.button("Resnet-50", disabled=bool(pre_model))
    with st.form("Settings"):
        method = str(st.selectbox("Techiniques used", ["GradCAM", "CAM"],help=" (Gradient-weighted Class Activation Map vs Class Activation Map) used to visually interpret and explain the decisions made by deep neural networks."))
        use_rgb = str(st.selectbox("Image format", ["RGB", "not RGB"],help=" Whether to use an RGB or BGR heatmap, it should be set to True if 'img' is in RGB format."))
       
        submitted = st.form_submit_button("Submit")

@st.cache
def get_ressources():
    return models.resnet50(pretrained=True)
resnet = get_ressources()


img = Image.open("images/cat_dog.jpeg")

tab_names = ["Upload", "Image URL"]
selected_tab = st.selectbox("Select a tab", tab_names)

if selected_tab == "Upload":
    file = st.file_uploader("Upload image", key="file_uploader")
    if file is not None:
        try:
            img = Image.open(file)
        except:
            st.error("The image you uploaded does not seem to be in a valid format.")

elif selected_tab == "Image URL":
    url_text = st.empty()
    url = url_text.text_input("Image URL", key="image_url") #image adress
    
    if url!="":
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")

# show the image
with st.expander("image", expanded=True):
    st.image(img, use_column_width=True)

#Load and preprocess  the image
img_tensor = load_and_preprocess_image(img)

#load the imagenet class labels
with open("imagenet_class_labels.txt") as f:
    imagenet_class_labels = eval(f.read())

#make prediction
Button = st.button("Make prediction")

def prediction(model=resnet,img_tensor=img_tensor):
    model.eval()
    pred = model(img_tensor.unsqueeze(0))

    # Get the corresponding class label from ImageNet class labels
    k = 2
    percents = torch.nn.functional.softmax(pred.squeeze(), dim=0) * 100
    top_values, top_indices = percents.topk(k)

    st.markdown("### The top 2 predicted labels are:")
    for i in range(k):
        label = imagenet_class_labels[top_indices[i].item()]
        percent = top_values[i].item()
        st.write(f" {label}: {percent:.2f}%")

if Button:
    prediction()

st.markdown("---")



def plot_heatmap_plus_image(img,heatmap):
    
    # Resize and preprocess the heatmap
    img = np.array(img)
    if use_rgb=="RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend the heatmap with the original image
    alpha = 0.4
    superimposed_img = heatmap = np.float32(heatmap) / 255
    superimposed_img = (1 - alpha) * heatmap + alpha * img

    # Display the blended image
    with st.expander("image",expanded=True):
        st.image(superimposed_img)
        

if submitted and method =="CAM":
    st.markdown("### CAM has been selected")
    model = CAM(resnet)
    model.eval()

    # Perform the forward pass
    img_tensor = img_tensor.unsqueeze(0)
    pred = model(img_tensor)

    # Get the weight (parameters) of the final FC layer
    class_weights = model.classifier.weight.data

    # Get the feature maps from the last convolutional layer
    feature_maps = model.features_conv(img_tensor)

    # Calculate the class activation map (CAM)
    cam = torch.matmul(class_weights, feature_maps.view(feature_maps.size(0), feature_maps.size(1), -1))
    cam = cam.view(cam.size(0), -1, feature_maps.size(2), feature_maps.size(3))
    cam = torch.sum(cam, dim=1, keepdim=True)

    # Normalize the CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cam.squeeze().detach().numpy()
    plot_heatmap_plus_image(img,cam)


elif submitted and method == "GradCAM":
    st.markdown("### GradCAM has been selected ")
    model = GradCAM(resnet)
    model.eval()
    pred = model(img_tensor.unsqueeze(0))
    
    # get the gradient of the output with respect to the parameters of the model
    idx = pred.argmax(dim=0)
    pred[idx].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(img_tensor.unsqueeze(0)).detach()


    # weight the channels by corresponding gradients
    for i in range(activations.size(1)):
        activations[:,i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    grad_cam = torch.mean(activations, dim=1).squeeze()

    # draw the heatmap
    # grad_cam = np.maximum(grad_cam, 0)
    grad_cam /= torch.max(grad_cam)
    grad_cam = grad_cam.detach().numpy()
    plot_heatmap_plus_image(img,grad_cam)




















    # # provide options to either select upload an image or fetch from URL
# upload_tab, url_tab = st.tabs(["Upload", "Image URL"])

# with upload_tab:
#     file = st.file_uploader("Upload image", key="file_uploader")
#     if file is not None:
#         try:
#             img = Image.open(file)
#         except:
#             st.error("The image you uploaded does not seem to be in a valid format. Try uploading a png or jpg file.")

# with url_tab:
#     url_text = st.empty()
#     url = url_text.text_input("Image URL", key="image_url")
    
#     if url!="":
#         try:
#             response = requests.get(url)
#             img = Image.open(BytesIO(response.content))
#         except:
#             st.error("The URL does not seem to be valid.")









# Model Interpreter 🧐

This app take a user-provided images and try to make predictions about the contents of the image.
In addition, it either **CAM** (Class Activation Map) or **[GradCAM](https://arxiv.org/pdf/1610.02391.pdf)** to generate visual representation highlighting the regions within the input image that influence the model's classification decision for a specific class.

## Description of the methods
The goal is to visualize which parts of the input image contribute most to the model's decision for a specific class. These two approaches differ in how they attribute importance to different regions of the input image.

- **CAM**: uses the learned weight parameters of the final fully connected layer (classifier) to generate the activation map. The activation maps is obtained by doing a weighted combinations of the feature maps (Global Poolin Average).
- **GradCAM**: takes advantage of the gradients of the predicted class with respect to the feature maps of a convolutional layer. The gradients provide information about how small changes in each pixel of the feature maps affect the final class score.

These two techniques are implemented from scratch using Pytorch.

## Setup

1. Clone the repo 
2. create an activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
   ```
3. Install the requirements
```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
   ```
4. Run the app locally
```bash
streamlit run app.py
```
5. Test the app directly [here](https://explainability-ai-vu74z27mh3h3v9dbs5mo69.streamlit.app/)

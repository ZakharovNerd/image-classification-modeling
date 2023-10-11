import argparse
import pickle

import torch
from PIL import Image

from src.data.transforms import data_transforms


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='test_image.jpg', help='path to image')
    return parser.parse_args()


def predict_image(image_path):
    with open('encoder.pkl', 'rb') as file:
        loaded_encoder = pickle.load(file)
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = data_transforms['val'](image).unsqueeze(0)  # Add batch dimension

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Check if a GPU is available and move the input tensor to the same device as the model
    model = torch.load('weights/vgg16_feature_extractor.pth', map_location=device)
    model.eval()

    model.to(device)  # Move the model to the same device as the input
    image = image.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(image)

    # Return the predicted classes
    outputs = (outputs > .2)

    return loaded_encoder.inverse_transform(outputs.cpu())


if __name__ == '__main__':
    args = arg_parse()

    predictions = predict_image(args.image_path)

    print("Predicted Classes:", predictions)

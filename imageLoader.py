from split_image import split_image
from PIL import Image
from CNN import ModelLogger, MNIST_ResNet, show_sample_images
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import sys
from gamehub import stepper
import torchvision.transforms.v2 as transforms
from functions import print_sudoku_style
def modelLoader():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelLogger  = ModelLogger("model_1", device=device)
    model = MNIST_ResNet(classes=11)
    model = modelLogger.load_model(model)
    return model

def prepper(image,n=9, split_dir = "./splits/"):
    split_image(image,n,n,should_cleanup=False,should_square=True, output_dir=split_dir, should_quiet=True)
    imageName = os.path.basename(image)
    imageName, file_extension = os.path.splitext(imageName)
    for i in range(n**2):
        dirr  = split_dir+imageName+f"_{i}"+file_extension
        image = Image.open(dirr)
        # Convert the image to grayscale. The `"L"` argument in Pillow represents grayscale mode.
        grayscale_image = image.convert("L")
        # Save the grayscale image
        grayscale_image.save(dirr)
    return get_numbers(imageName+file_extension,split_dir=split_dir, n=n)
def get_numbers(imageName,split_dir="./splits/", n=9):
    numbers = []
    transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((35,35)), transforms.CenterCrop((28,28))])
    model = modelLoader()
    imageName, file_extension = os.path.splitext(imageName)
    for i in range(n**2):
        dirr  = split_dir+imageName+f"_{i}"+file_extension
        X=1-transform(Image.open(dirr))
        X = X.unsqueeze(0)/255.0
        with torch.no_grad():
            model.eval()
            yd = model(X)
            # print(yd)
            num = np.argmax(yd).item()
            num = num if num != 10 else 0
        numbers.append(num)
    return numbers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="Solve Sudoku based on input image", description="add Imagepath, --dimension default=9, --splits_dir  default=./splits/ ")

    if len(sys.argv) <2:
        print("No arguments provided.\n")
        parser.print_help()
        exit(0)
    
    parser = argparse.ArgumentParser(usage="Solve Sudoku based on input image", description="add Imagepath, --dimension default=9, --splits_dir  default=./splits/ ")
    parser.add_argument("ImagePath", help="name of the image, or file path, with extension e.g. [.png] [.jpg]")
    parser.add_argument("--dimension",default=9,type=int, help="Dimension of the Sudoku, eg 9x9 -> 9, DEFAULT=9")
    parser.add_argument("--splits_dir",default="./splits/",type=str, help="Directory for use of temporary image, ie reading the numbers, DEFAULT=./splits/")
    args = parser.parse_args()
    numbers  = np.array(prepper(args.ImagePath,n=args.dimension, split_dir=args.splits_dir))
    base = numbers.reshape((args.dimension,args.dimension))
    print_sudoku_style(base)
    stepper(base)

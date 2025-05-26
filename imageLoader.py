CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

print(f"{CYAN}--------------{RESET} {MAGENTA}Sudoku Solver{RESET} {CYAN}--------------{RESET}")

print("loading CNN model and CNN related tools...")
print("I promise this will not take long(er than doing it manually)")

from split_image import split_image
from PIL import Image
from CNN import ModelLogger, MNIST_ResNet, show_sample_images
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import time
import sys
from gamehub import solver
import torchvision.transforms.v2 as transforms
from functions import Sudoku
import numpy as np
from PIL import Image

class TrimLowActivity:
    def __init__(self, threshold=0.5, percent=5):
        self.threshold = threshold      # e.g., pixel value threshold (0.5)
        self.percent = percent / 100.0  # e.g., 5% = 0.05

    def __call__(self, img):
        # Convert PIL to NumPy
        img_np = np.array(img).astype(np.float32)

        # Normalize if in 0-255 range
        if img_np.max() > 1:
            img_np /= 255.0

        # Row mask: keep rows where enough pixels > threshold
        row_activity = np.mean(img_np > self.threshold, axis=1)
        valid_rows = row_activity > self.percent

        # Column mask: same for columns
        col_activity = np.mean(img_np > self.threshold, axis=0)
        valid_cols = col_activity > self.percent

        # Trim the image
        trimmed = img_np[np.ix_(valid_rows, valid_cols)]

        # Resize to 28x28 again (or keep as is if preferred)
        trimmed_img = Image.fromarray((trimmed * 255).astype(np.uint8)).resize((28, 28))

        return trimmed_img

def modelLoader():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelLogger  = ModelLogger("model_1", device=device, output_dir="./cnn/")
    model = MNIST_ResNet(classes=11)
    model = modelLogger.load_model(model)
    return model

def prepper(image,n=9, split_dir = "./splits/", cropy=35, low_activity_threshold=0.5, low_activity_percent=10):
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
    return get_numbers(imageName+file_extension,split_dir=split_dir, n=n, crop=cropy, low_activity_threshold=low_activity_threshold, low_activity_percent=low_activity_percent)
def get_numbers(imageName,split_dir="./splits/", n=9, crop=35, low_activity_threshold=0.5, low_activity_percent=10):
    numbers = []
    transform = transforms.Compose([TrimLowActivity(threshold=low_activity_threshold, percent=low_activity_percent), transforms.PILToTensor(), transforms.Resize((crop,crop)), transforms.CenterCrop((28,28))])
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
    print("\033[96m -- model loaded --\033[0m ")
    parser = argparse.ArgumentParser(usage="Solve Sudoku based on input image", description="add Imagepath, --dimension default=9, --splits_dir  default=./splits/ ")

    if len(sys.argv) <2:
        print("No arguments provided.\n")
        parser.print_help()
        exit(0)
    parser = argparse.ArgumentParser(usage="Solve Sudoku based on input image", description="add Imagepath, --dimension default=9, --splits_dir  default=./splits/ ")
    parser.add_argument("ImagePath", help="name of the image, or file path, with extension e.g. [.png] [.jpg]")
    parser.add_argument("--dimension",default=9,type=int, help="Dimension of the Sudoku, eg 9x9 -> 9, DEFAULT=9")
    parser.add_argument("--splits_dir",default="./splits/",type=str, help="Directory for use of temporary image, ie reading the numbers, DEFAULT=./splits/")
    parser.add_argument("--center_crop",default=35,type=int, help="Images of numbers get resized to this value and then center cropped to 28x28 before CNN evalution, DEFAULT=35")
    parser.add_argument("--low_activity_threshold",default=0.5,type=float, help="At what treshold is are rows and columns of an image removed, DEFAULT=0.5")
    parser.add_argument("--low_activity_percent",default=10,type=int, help="At what percentage of low activity in the image is a row or column removed, DEFAULT=10")
    args = parser.parse_args()

    numbers  = np.array(prepper(args.ImagePath,n=args.dimension, split_dir=args.splits_dir, cropy=args.center_crop))
    base = numbers.reshape((args.dimension,args.dimension))
    SudokuBase = Sudoku(base)
    print("\033[92m Sudoku Base:\033[0m")
    SudokuBase.print_sudoku()
    start = time.time()
    print("Solving Sudoku started at ", time.strftime("%H:%M:%S"))
    solver(SudokuBase)
    print("Solving Sudoku finished at ", time.strftime("%H:%M:%S"))
    print("Time taken: ", time.time()-start)
    print("\033[92m Sudoku Solved:\033[0m")
    SudokuBase.print_sudoku_progress()


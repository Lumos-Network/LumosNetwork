from PIL import Image
import os
import math

h = 32
w = 32
c = 3

def mean_std(directory):
    mean = [0,0,0];
    std = [0,0,0];
    scale = 0
    with open(directory, "r") as fp:
        scale = h*w*len(fp.readlines())
        fp.close()
    with open(directory, "r") as fp:
        for path in fp.readlines():
            img = Image.open(path[:len(path)-1])
            img = img.convert("RGBA")
            for pixel in img.getdata():
                mean[0] += pixel[0] / 255
                mean[1] += pixel[1] / 255
                mean[2] += pixel[2] / 255
        fp.close()
    for i in range(len(mean)):
        mean[i] /= scale
    with open(directory, "r") as fp:
        for path in fp.readlines():
            img = Image.open(path[:len(path)-1])
            img = img.convert("RGBA")
            for pixel in img.getdata():
                std[0] += ((pixel[0] / 255) - mean[0])**2
                std[1] += ((pixel[1] / 255) - mean[1])**2
                std[2] += ((pixel[2] / 255) - mean[2])**2
        fp.close()
    for i in range(len(std)):
        std[i] /= scale
        std[i] = math.sqrt(std[i])
    return mean, std

mean, std = mean_std("./data/cifar10/train.txt")
print(mean)
print(std)

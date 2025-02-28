from PIL import Image
import pickle
import os
import numpy as np
import math
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "knnFileAlg2")

model = pickle.load(open(file_path, 'rb'))

FACTOR = 256**(1/255)

def scaleUp(value: int, flag = True):
    if round(value) > 255 and flag:
        raise ValueError("Input is greater than 255. Current input is:"+str(value))
    return math.log((value+1), FACTOR)

def scaleDown(value: int, flag = True):
    if round(value) > 255 and flag:
        raise ValueError("Input is greater than 255. Current input is:"+str(value))
    return (FACTOR**value)-1

def create_image_from_rgb(rgb_values, image):
    width, height = image.size
    # Rearrange from column-major to row-major
    row_major_rgb = [[0, 0, 0]] * (width * height)
    
    for x in range(width):
        for y in range(height):
            # Map column-major index to row-major index
            column_major_index = x * height + y
            row_major_index = y * width + x
            row_major_rgb[row_major_index] = rgb_values[column_major_index]
    
    # Convert to a NumPy array and reshape
    rgb_array = np.array(row_major_rgb, dtype=np.uint8).reshape((height, width, 3))
    
    # Create and display the image
    image = Image.fromarray(rgb_array)
    image.show()

def retrieveRows(image):
    hex_colors = list()
    width, height = image.size
    for x in range(width):
        for y in range (height):
            r, g, b = image.getpixel((x, y))
            # hex_color = [scaleUp(r), scaleDown(g), scaleDown(b)]
            hex_color = [r,g,b]
            hex_colors.append(hex_color)
    
    return hex_colors

def predict(hex_colors):
    pixels = list(model.predict(hex_colors))
    
    return pixels

def filter(n, pixels, image):
    width, height = image.size
    filtered_pixels = []

    if n > 0:
        for i in range(len(pixels)):
            flagU = []
            flagD = []
            flagR = []
            flagL = []
            flagNU = []
            flagND = []
            flagNR = []
            flagNL = []
            if pixels[i] == 1:
                xLoc = (i//height)+1
                yLoc = (i%height)+1
                #Checking above point
                if (yLoc-(n)) > 0:
                    for x in range(1, n+1):
                        if pixels[i-x] == 1:
                            flagU.append(True)
                #Checking below point
                if (yLoc+(n)) <= height:
                    for x in range(1, n+1):
                        if pixels[i+x] == 1:
                            flagD.append(True)
                #Checking right of point
                if (xLoc+(n)) <= width:
                    for x in range(1, n+1):
                        if pixels[i+(x*height)] == 1:
                            flagR.append(True)
                #Checking left of point
                if (xLoc-(n)) > 0:
                    for x in range(1, n+1):
                        if pixels[i-(x*height)] == 1:
                            flagL.append(True)

            if pixels[i] == 0:
                xLoc = (i//height)+1
                yLoc = (i%height)+1
                #Checking above point
                if (yLoc-(n)) > 0:
                    for x in range(1, n+1):
                        if pixels[i-x] == 0:
                            flagNU.append(True)
                #Checking below point
                if (yLoc+(n)) <= height:
                    for x in range(1, n+1):
                        if pixels[i+x] == 0:
                            flagND.append(True)
                #Checking right of point
                if (xLoc+(n)) <= width:
                    for x in range(1, n+1):
                        if pixels[i+(x*height)] == 0:
                            flagNR.append(True)
                #Checking left of point
                if (xLoc-(n)) > 0:
                    for x in range(1, n+1):
                        if pixels[i-(x*height)] == 0:
                            flagNL.append(True)

            if len(flagU) == n or len(flagD) == n or len(flagR) == n or len(flagL) == n:
                filtered_pixels.append(1)
            elif len(flagNU) != n and len(flagND) != n and len(flagNR) != n and len(flagNL) != n:
                filtered_pixels.append(1)
            else:
                filtered_pixels.append(0)
    else:
        filtered_pixels = pixels
    
    return filtered_pixels

if __name__ == "__main__":
    #Start Timer
    start = time.time()

    #Open Image
    image = Image.open(r"C:\Users\Jekam\Documents\03-Projects\Project Awareness\test.png")
    image = image.convert('RGB')

    #Get and predict water pixels
    hex_colors = retrieveRows(image=image)
    pixels = predict(hex_colors=hex_colors)

    #Filter out noise
    n = int(round(float(input("Enter noise reduction value: "))))
    pixels = filter(n=n, pixels=pixels, image=image)

    #Display image
    for i in range(len(pixels)):
        if pixels[i] == 1:
            hex_colors[i] = [0, 0, 255]

    create_image_from_rgb(hex_colors, image=image)

    #End timer
    end = time.time()
    execution = end - start
    minutes = int(execution // 60)
    seconds = int(execution % 60)
    print(f"Execution time: {minutes}m {seconds}s")
import numpy as np
from PIL import Image
import time
from main import retrieveRows, predict, create_image_from_rgb, filter
import matplotlib.pyplot as plt

data = []
upperRange = 10000

for j in range(2, upperRange):
    print(j)
    i = j*2
    width = 2
    height = j
    hex_colors = [[0,0,0]]*i
    
    # Convert to a NumPy array and reshape
    rgb_array = np.array(hex_colors, dtype=np.uint8).reshape((height, width, 3))
    
    # Create and display the image
    image = Image.fromarray(rgb_array)

    #Start Timer
    start = time.time()

    #Get and predict water pixels
    hex_colors = retrieveRows(image=image)
    pixels = predict(hex_colors=hex_colors)

    #Filter out noise
    pixels = filter(n=1, pixels=pixels, image=image)

    #Display image
    for i in range(len(pixels)):
        if pixels[i] == 1:
            hex_colors[i] = [0, 0, 255]

    # create_image_from_rgb(hex_colors, image=image)

    #End timer
    end = time.time()
    execution = end - start
    data.append(execution)

xpoints = np.arange(2,upperRange)
for i in range(len(xpoints)):
    xpoints[i] *= 2

plt.plot(xpoints, data)
plt.title("Execution time vs Image size")
plt.xlabel("Image Size / Number of Pixels")
plt.ylabel("Execution Time / s")
plt.show()
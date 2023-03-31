import os, sys
sys.path.append(os.getcwd())

import numpy as np
# from PIL import Image
import cv2

folder_path = "simulator/forest_coverage/perlin"

imgs = []
for i in range(100):
    path = folder_path + f"/{i}.npy"
    m = np.load(path)
    # normalize 0.2 - 0.8 to be between 0 and 1
    m = (m - 0.2) / (0.8 - 0.2)

    m_resized = cv2.resize(m, (256, 256))
    # img = Image.fromarray(np.uint8(m * 255) , 'L')
    # img = img.resize((256, 256))
    # img = np.array(img)
    # imgs.append(img)

    imgs.append(m_resized)

stacked_imgs = np.stack(imgs, axis=0)
print(stacked_imgs.shape)

np.save("simulator/forest_coverage/perlins.npy", stacked_imgs)
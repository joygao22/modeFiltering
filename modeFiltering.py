# Replace all file paths with your own
# standard packages
import numpy as np
import statistics
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2

# tensorflow / keras
from tensorflow import keras

#for making pictures
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import segmentation_models as sm

#chopping up images
from patchify import patchify, unpatchify


def calculate_pixel_error(img1, img2):
    # Ensure both images have the same dimensions
    assert img1.shape == img2.shape, "Images must have the same dimensions"

    # Calculate the absolute difference between the images
    diff = cv2.absdiff(img1, img2)

    # Calculate the pixel error (sum of differences)
    total_error = np.sum(diff)

    # Mean squared error
    mse = np.mean(diff ** 2)

    # Root mean squared error
    rmse = np.sqrt(mse)

    # Percentage of differing pixels
    differing_pixels = np.sum(diff != 0)
    total_pixels = img1.shape[0] * img1.shape[1]

    # return total_error, mse, rmse, percentage_differing
    return total_error, mse, rmse

# necessary scaler
scaler = MinMaxScaler()

# for compiling the model
weights = [0.5, 0.5]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics=['accuracy']

# load trained model
model_path = r'C:\Users\nezamoddin\Documents\Glaciers\Saved_Results\binaryClass_128_savedModel_1.keras'
model = keras.models.load_model(model_path, compile=False)
model.compile(loss=total_loss, optimizer='adam', metrics=metrics)


# read in image and patchify
image_dataset = []
patch_size = 128

root_directory = r'C:\Users\nezamoddin\Documents\Glaciers\Semantic_segmentation_dataset - Copy'
path = r'C:\Users\nezamoddin\Documents\Glaciers\Semantic_segmentation_dataset - Copy'

w = 0
# the code for the patchify should be the same as before
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':
        images = os.listdir(path)
        for k, image_name in enumerate(images):
            if image_name.endswith(".tif"):
                image_dataset = []
                image = cv2.imread(path+"/"+image_name, 1)
                SIZE_X = (image.shape[1]//patch_size)*patch_size # Nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size # Nearest size divisible by our patch size
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  # Crop from top left corner
                plt.imshow(image)
                image = np.array(image)

                # Patchify each image
                print("Now patchifying image:", path+"/"+image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

                for s in range(patches_img.shape[0]):
                    for t in range(patches_img.shape[1]):

                        single_patch_img = patches_img[s,t,:,:]
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        single_patch_img = single_patch_img[0] # Drop the extra unnecessary dimension that patchify adds.
                        image_dataset.append(single_patch_img)

                image_dataset = np.array(image_dataset)
                test_input = np.array(single_patch_img)

                ### Now we unpatchify
                prediction = (model.predict(image_dataset)) # obtain the predictions
                predicted_img=np.argmax(prediction, axis=3) # Returns the indices of the maximum values along the third axis
                predicted_img = np.repeat(predicted_img[:, :, :, np.newaxis], 3, axis=3) # Repeat the values in the third axis to an added axis
                predicted_img = np.reshape(predicted_img, patches_img.shape) # reshape the predicted image into the proper patch dimension
                unpatchedImg = unpatchify(predicted_img, image.shape) * 63 # unpatchify
                unpatchImgName = "beforeFilter" + str(w) + ".png"
                cv2.imwrite(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\unpatchified\\' + unpatchImgName, unpatchedImg)

                img_noisy1 = cv2.imread(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\unpatchified\\' + unpatchImgName, cv2.IMREAD_GRAYSCALE)

                # Obtain the number of rows and columns of the image
                m, n = img_noisy1.shape

                # Initialize a new image to hold the result
                img_new1 = img_noisy1.copy()


                # Define the interval for median blurring
                interval = 128

                # left
                for i in range(2, m - 3):
                    for j in range(interval, n - interval, interval):
                        # Extract the 6x5 neighborhood
                        neighborhood = img_noisy1[i - 2:i + 4, j - 2:j + 3]
                        neighborhood = neighborhood.flatten()
                        mode_value = statistics.mode(neighborhood)
                        # Set the mode value to the center pixel of the neighborhood
                        img_new1[i, j] = mode_value
                        img_new1[i, j+1] = mode_value

                # up
                for i in range(interval, m - interval, interval):
                    for j in range(2, n - 3):
                        # Extract the 6x5 neighborhood
                        neighborhood = img_noisy1[i - 2:i + 3, j - 2:j + 4]
                        neighborhood = neighborhood.flatten()
                        mode_value = statistics.mode(neighborhood)
                        # Set the mode value to the center pixel of the neighborhood
                        img_new1[i, j] = mode_value
                        img_new1[i+1, j] = mode_value

                # Save the image with the selective mode filtering applied
                cv2.imwrite(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\filtered\\'+'new_filtered' + str(w) + '.png', img_new1)
                # Display the image
                plt.figure(figsize=(24, 12), num=1, clear=True)
                plt.subplot(121)
                plt.imshow(image) # show original image
                plt.subplot(122)
                plt.imshow(unpatchedImg) # show unpatchified image
                plt.savefig(r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\pictures\\' + image_name[0:-4:1] + '.png')
                plt.show()


                beforeFiltering = cv2.imread(
                     r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\unpatchified\\' + 'beforeFilter' + str(w) + '.png')

                filtered = cv2.imread(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\filtered\\' + 'new_filtered' + str(w) + '.png')

                mask = cv2.imread(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\masks\\'+'mask' + str(w) + '.tif')

                beforeFiltering = cv2.cvtColor(beforeFiltering, cv2.COLOR_BGR2GRAY)
                filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                if beforeFiltering.shape != filtered.shape:
                    print("Resizing the images to match dimensions.")
                    beforeFiltering = cv2.resize(beforeFiltering, (filtered.shape[1], filtered.shape[0]))

                if filtered.shape != mask.shape:
                    print("Resizing the images to match dimensions.")
                    mask = cv2.resize(mask, (filtered.shape[1], filtered.shape[0]))

                rows, cols = mask.shape

                for i in range(rows):
                    for j in range(cols):
                        if mask[i, j] == 223:
                            mask[i, j] = 63
                        elif mask[i, j] != 223:
                            mask[i, j] = 0


                cv2.imwrite(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\binaryMasks\\' + 'binaryMask' + str(w) + '.png', mask)

                print("*******************************************************************")
                # total_error, mse, rmse, percentage_differing = calculate_pixel_error(beforeFiltering, mask)
                total_error, mse, rmse = calculate_pixel_error(beforeFiltering, mask)
                print(f"Total Pixel Error: {total_error}")
                print(f"Mean Squared Error (MSE): {mse}")
                print(f"Root Mean Squared Error (RMSE): {rmse}")
                plt.figure(figsize=(24, 12), num=1, clear=True)

                # Show the Binary Mask image
                plt.subplot(121)
                plt.imshow(mask)
                plt.title("Binary Mask")

                # Show the unpatchified image
                plt.subplot(122)
                plt.imshow(beforeFiltering)
                plt.title("Before Filtering")

                # Add the text with the error metrics
                plt.figtext(0.5, 0.01, f"Total Pixel Error: {total_error}\n"
                                       f"Mean Squared Error (MSE): {mse}\n"
                                       f"Root Mean Squared Error (RMSE): {rmse}\n",
                            ha="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
                plt.savefig(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\pictures\\' + 'maskUnpatchify' + str(
                        w) + '.png')
                plt.show()
                errorBefore = total_error


                # total_error, mse, rmse, percentage_differing = calculate_pixel_error(filtered, mask)
                total_error, mse, rmse = calculate_pixel_error(filtered, mask)
                print(f"Total Pixel Error: {total_error}")
                print(f"Mean Squared Error (MSE): {mse}")
                print(f"Root Mean Squared Error (RMSE): {rmse}")
                plt.figure(figsize=(24, 12), num=1, clear=True)

                # Show the Binary Mask image
                plt.subplot(121)
                plt.imshow(mask)
                plt.title("Binary Mask")

                # Show the after filtered image
                plt.subplot(122)
                plt.imshow(filtered)
                plt.title("After Filtering")

                # Add the text with the error metrics
                plt.figtext(0.5, 0.01, f"Total Pixel Error: {total_error}\n"
                                       f"Mean Squared Error (MSE): {mse}\n"
                                       f"Root Mean Squared Error (RMSE): {rmse}\n",
                            ha="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
                plt.savefig(
                    r'C:\\Users\\nezamoddin\\Documents\\Glaciers\\patchify\\pictures\\' + 'maskFilter' + str(w) + '.png')
                plt.show()
                errorAfter = total_error


                numDifferent = int(errorBefore) - int(errorAfter)

                #May need to change bc need to multiply 103,424 by two bc border is two lines
                percentChange = (int(numDifferent)/205468)*100

                # Write data to the file
                with open(r'C:\Users\nezamoddin\Documents\Glaciers\results.txt', 'a') as file:
                    file.write(
                            f"**6x5 Fixed Double Pixel Mode Filter:** \n"
                            f"Number of Changed Border Pixels:  {numDifferent}\n"
                            f"Percent of Border Pixels Changed:  {percentChange}\n\n")
                w += 1



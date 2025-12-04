import numpy as np
from PIL import Image, ImageOps
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

#claire

def gaussian_blur(image_path, sigma=1.0):

    # larger sigma: less noisy but less detailed, also slower

    #loads image and makes grayscale
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image_array = np.asarray(image).astype(np.float32)

    # Kernel size based on sigma
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2

    # Initialize Gaussian kernel
    kernel = np.empty((kernel_size, kernel_size), dtype=np.float32)

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            kernel[i + half, j + half] = np.exp(-((i**2 + j**2) / (2 * sigma**2))) * (
                    1.0 / (2 * np.pi * sigma**2)
            )

    # Normalize kernel so it sums to 1-helps brightness
    kernel /= kernel.sum()

    # 2D convolution (Gaussian blur)
    blurred = convolve2d(image_array, kernel, mode='same', boundary='symm')

    return image_array, blurred.astype(np.float32), kernel_size, half


 # mizna

def compute_gradients(smoothed, sigma, kernel_size, half):

    Gx = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    Gy = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            base = np.exp(-((i * i + j * j) / (2 * sigma * sigma)))
            Gx[i + half, j + half] = -(i / (2 * np.pi * sigma**4)) * base
            Gy[i + half, j + half] = -(j / (2 * np.pi * sigma**4)) * base

    Ix = convolve2d(smoothed, Gx, mode='same', boundary='symm')
    Iy = convolve2d(smoothed, Gy, mode='same', boundary='symm')

    magnitude = np.sqrt(Ix * Ix + Iy * Iy)
    direction = np.arctan2(Iy, Ix)  # radians

    return magnitude, direction


#Anna

def non_maximum_suppression(gradient_magnitude, gradient_direction):

    M, N = gradient_magnitude.shape
    thinned = np.zeros((M, N), dtype=np.float32)

    # Convert direction to degrees in [0, 180)
    angle = np.degrees(gradient_direction) % 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 0.0
            r = 0.0

            # Angle 0°
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            # Angle 45°
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            # Angle 90°
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            # Angle 135°
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            # Keep only if it is a local maximum
            eps = 0.05 * gradient_magnitude[i, j]   # allow 5% variation

            if (gradient_magnitude[i, j] + eps >= q) and (gradient_magnitude[i, j] + eps >= r):
                thinned[i, j] = gradient_magnitude[i, j]



    return thinned


def visualize_all(original_gray, smoothed, magnitude, direction, thinned):

    # Normalize magnitude for display
    mag_max = magnitude.max()
    if mag_max > 0:
        mag_disp = magnitude / mag_max * 255.0
    else:
        mag_disp = magnitude


    thin_max = thinned.max()
    if thin_max > 0:
        # Work only with nonzero values
        nonzero_vals = thinned[thinned > 0]

        if nonzero_vals.size > 0:

            thresh_val = np.percentile(nonzero_vals, 80)
            thin_binary = (thinned >= thresh_val).astype(np.float32) * 255.0
        else:
            thin_binary = thinned
    else:
        thin_binary = thinned

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Grayscale Image")
    plt.imshow(original_gray, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Smoothed (Gaussian Blur)")
    plt.imshow(smoothed, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Gradient Magnitude (Normalized)")
    plt.imshow(mag_disp, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Thinned Edges (Strongest Pixels)")
    plt.imshow(thin_binary, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.show()



def main():
    # file name
    image_path = r"C:\Users\mizna\OneDrive\Documents\image.png"
    sigma = 1.0                # can be adjusted


    original_gray, smoothed, kernel_size, half = gaussian_blur(image_path, sigma)


    magnitude, direction = compute_gradients(smoothed, sigma, kernel_size, half)

    print("magnitude max:", magnitude.max())
    print("magnitude min:", magnitude.min())


    thinned = non_maximum_suppression(magnitude, direction)

    print("thinned max:", thinned.max())
    print("thinned nonzero pixel count:", np.count_nonzero(thinned))

    # Visualize all stages
    visualize_all(original_gray, smoothed, magnitude, direction, thinned)


if __name__ == "__main__":
    main()


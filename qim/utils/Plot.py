import matplotlib.pyplot as plt

def plot_images(images_dict):
    """ General plotting function for displaying multiple images.

    Parameter:
        @param images_dict (dict): A dictionary where the keys are the titles of the images (str) and the values are the image arrays (numpy.ndarray).
    
    Usage example:
        images_to_plot = {
            title1 : image1,
            title2 : image2,
            ...
        }
        plot_images(images_to_plot)

    """
    num_images = len(images_dict)
    plt.figure(figsize=(3 * num_images, 3))
    
    for i, (title, image) in enumerate(images_dict.items(), start=1):
        plt.subplot(1, num_images, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
import os
import numpy as np
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

def plot_metrics(x_label, metrics_dict, output_path=None):
    """Plot the changes of different metrics with x using line charts of different colors, and optionally save the images.
        
    Parameters:
        @param x_label (str): The label for the x-axis.
        @param metrics_dict (dict): 
            Dictionary containing multiple sets of data, where the key is the metric name (str) and the value is a tuple containing (x_values, metric_values).
        @param output_path (str): Optional, the file name (includes relative folder path) to save the images. If None, the images will not be saved.
    
    Usage example:
        metrics_dict = {
            'Accuracy': ([0.1, 0.2, 0.3], [0.8, 0.85, 0.9]),
            'Loss': ([0.1, 0.2, 0.3], [0.4, 0.35, 0.3])
        }
        plot_metrics(metrics_dict, output_dir='/path/to/save/metrics_plot.png')
    """
    colors = ['b', 'g', 'r', 'm']

    num_metrics = len(metrics_dict)
    num_cols = min(2, num_metrics)
    num_rows = (num_metrics + num_cols - 1) // num_cols
    plt.figure(figsize=(6 * num_cols, 4 * num_rows))
    
    for i, (metric_name, (x_values, metric_values)) in enumerate(metrics_dict.items(), start=1):
        plt.subplot(num_rows, num_cols, i)
        plt.plot(x_values, metric_values, color=colors[i % len(colors)], label=metric_name)
        plt.title(f'{metric_name} vs. {x_label}')
        plt.xlabel(f'{x_label}')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Image saved to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    # Test case for plot_images
    def test_plot_images():
        images_to_plot = {
            'Image 1': np.random.rand(10, 10),
            'Image 2': np.random.rand(10, 10),
        }
        plot_images(images_to_plot)
        print("plot_images test passed.")

    # Test case for plot_metrics
    def test_plot_metrics():
        metrics_dict = {
            'Accuracy': ([0.1, 0.2, 0.3], [0.8, 0.85, 0.9]),
            'Loss': ([0.1, 0.2, 0.3], [0.4, 0.35, 0.3])
        }
        plot_metrics('Epoch', metrics_dict, './test_plot_metrics.png')
        print("plot_metrics test passed.")

    # Run tests
    test_plot_images()
    test_plot_metrics()
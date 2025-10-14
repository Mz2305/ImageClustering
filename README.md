# ImageClustering

A small Python project that performs color-based image segmentation using KMeans clustering. The script clusters the pixels of an input image to extract dominant colors and produces a segmented image and a saved matplotlib figure comparing the original and the segmented result.

## Repository structure

- `code/`
	- `main.py` — main script that reads an image from `resources/`, applies KMeans clustering, displays the figure and saves outputs to `output/`.
- `resources/` — input sample images (e.g. `colors.jpg`, `person.jpg`).
- `output/` — generated images (segmented images and saved figures).

## Code highlights

- The script reads the image with OpenCV (`cv2.imread`) and converts it to RGB for display with matplotlib.
- The image is reshaped to a 2D array of pixels and KMeans is used to cluster colors into `number_of_colors` clusters.
- The cluster centers are used to create a segmented image where each pixel is replaced by its cluster center color.
- The script saves both the matplotlib figure (`fig.savefig`) and the segmented image (converted back to BGR before `cv2.imwrite`).

## Example

Run the script to produce the outputs, then include the generated images in the README using relative paths. Replace `colors` with the image name you used.

| Original | Segmented |
|---|---|
| ![Original image](resources/colors.jpg) | ![Segmented image](output/output_colors_5.jpg) |
| *resources/colors.jpg* | *output/output_colors_5.jpg* |

## Requirements

- Python 3.8+ (tested with a modern CPython)
- Packages:
	- opencv-python (cv2)
	- numpy
	- matplotlib
	- scikit-learn

## Usage

1. Place an input image in `resources/`, for example `resources/colors.jpg`.
2. Edit `code/main.py` to change `name_image` or other parameters (number of clusters).
3. Run the script from the repository root

Expected outputs (written to `output/`):
- `figure_<name>.png` — saved matplotlib figure showing the original image and the segmented result.
- `output_<name>.jpg` — segmented image saved with OpenCV.

## Troubleshooting

- If `cv2.imread` returns `None`, check the path to `resources/<name>.jpg` and that the file exists.
- If matplotlib windows do not appear when running remotely (e.g., over SSH), consider saving the figure (`fig.savefig`) and running headless.

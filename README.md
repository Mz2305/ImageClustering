# ImageClustering

Python project for image segmentation and color clustering using KMeans. The repository contains scripts for processing static images and live webcam components that perform clustering using KMeans and MiniBatchKMeans.

## Repository structure

- `images` - Static images scripts
	- `code/` — main scripts:
		- `main_images.py` — offline image clustering scripts (read from `resources/`, save to `output/`).
	- `resources/` — sample input images (e.g. `colors.jpg`, `person.jpg`).
	- `output/` — resulting segmented images and saved figures.
- `webcam/` — Webcam scripts
	- `main_webcam.py` — interactive MiniBatchKMeans version with trackbars for `K` and `batch_size` and keyboard controls.

## Image clustering (offline)

1. Put an input image in `images/resources/` (e.g. `images/resources/colors.jpg`).
2. Set the image name in `images/code/main_images.py` (variable `name_image`).
3. Run
4. In `images/output/` you will see `output_<name>_X.jpg`, that represent the segmented image (X may indicate the number of clusters).


### Example — Before & After (with slider)

| Original | Segmented |
|---|---|
| ![Original](images/resources/colors.jpg) | ![Segmented](images/output/output_colors_10.jpg) |
| *images/resources/colors.jpg* | *images/output/output_colors_10.jpg* |

## Webcam (live) — new project piece

Run `webcam/main_webcam.py` — interactive MiniBatchKMeans version with UI controls:
  - "Controls" window with trackbars:
    - `Clusters K` — number of clusters (K)
    - `Batch Size` — mini-batch size used by MiniBatchKMeans
  - Keyboard controls:
    - `q` — quit
    - `d` / `a` — increase / decrease K
    - `w` / `s` — increase / decrease batch
  - Shows FPS in the console.
  - Displays the live webcam feed alongside the segmented output

### Tips for `main_webcam.py` (MiniBatchKMeans)

- `Batch Size` (trackbar) controls how many pixels are used to update centroids at each `partial_fit`. Larger values give more stable updates but are slower.
- `K` (Clusters) affects the granularity of detected colors.



### Requirements

- Python 3.8+
- Python packages:
  - opencv-python
  - numpy
  - scikit-learn
  - matplotlib (optional)
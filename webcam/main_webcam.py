import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time

K = 5
batch_size = 2000
kmeans = None
max_K = 100
max_batch = 50000

def nothing(x):
    pass

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 240, 75)
cv2.createTrackbar("Clusters K", "Controls", K, max_K, nothing)
cv2.createTrackbar("Batch Size", "Controls", batch_size, max_batch, nothing)

def incremental_kmeans_segment(frame, kmeans):
    # Scale for speed
    frame_small = cv2.resize(frame, (240, 180))
    Z = frame_small.reshape((-1, 3))

    # Update existing centroids instead of refitting from scratch
    kmeans.partial_fit(Z)

    labels = kmeans.predict(Z)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented = centers[labels].reshape(frame_small.shape)

    # Create the palette of dominant colors
    palette = create_palette(centers, labels)

    return segmented, palette

def create_palette(centers, labels, width=300, height=50):
    # Count how many pixels belong to each cluster
    counts = np.bincount(labels.flatten())

    # Sort colors by frequency
    sorted_idx = np.argsort(-counts)
    centers = centers[sorted_idx]
    counts = counts[sorted_idx]

    # Calculate proportions
    total = np.sum(counts)
    ratios = counts / total

    # Create the bar (palette)
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    start = 0
    for color, ratio in zip(centers, ratios):
        end = start + int(ratio * width)
        cv2.rectangle(palette, (start, 0), (end, height), color.tolist(), -1)
        start = end
    return palette

def main():
    global K, batch_size, kmeans

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return

    print("Press 'q' to exit.")
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Key handling BEFORE reading the trackbar values
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):  # increase K
            new_K = min(K + 1, max_K)
            cv2.setTrackbarPos("Clusters K", "Controls", new_K)
        elif key == ord('a'):  # decrease K
            new_K = max(K - 1, 1)
            cv2.setTrackbarPos("Clusters K", "Controls", new_K)
        elif key == ord('w'):  # increase batch
            new_batch = min(batch_size + 1000, max_batch)
            cv2.setTrackbarPos("Batch Size", "Controls", new_batch)
        elif key == ord('s'):  # decrease batch
            new_batch = max(batch_size - 1000, 1)
            cv2.setTrackbarPos("Batch Size", "Controls", new_batch)

        # Then read the updated values from the trackbars
        new_K = cv2.getTrackbarPos("Clusters K", "Controls")
        new_batch = cv2.getTrackbarPos("Batch Size", "Controls")

        if new_K < 1: new_K = 1
        if new_batch < 1: new_batch = 1

        # If they change, reinitialize the model
        if kmeans is None or new_K != K or new_batch != batch_size:
            K = new_K
            batch_size = new_batch
            kmeans = MiniBatchKMeans(
                n_clusters=K,
                batch_size=batch_size,
                init="k-means++",
                n_init=1,
                max_iter=10
            )
            print(f"[INFO] Reinitialized: K={K}, batch_size={batch_size}")
            

        # Segmentation
        clustered, palette = incremental_kmeans_segment(frame, kmeans)
        clustered = cv2.resize(clustered, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Show
        cv2.imshow("Original", frame)
        cv2.imshow("Incremental Mini-Batch KMeans", clustered)
        cv2.imshow("Palette", palette)

        # FPS
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.1f}", end="\r")


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

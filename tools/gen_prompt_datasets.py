import argparse
import cv2
import os
import random
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def parse_args():
    parser = argparse.ArgumentParser(description='Generate prompts into train and test sets')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path of the dataset')
    return parser.parse_args()


def save_empty_prompts(save_path, gt_file_name, img):
    for prompt_type in ["random_points", "center_points", "box", "unfilled_mask", "filled_mask"]:
        cv2.imwrite(os.path.join(save_path, f"{prompt_type}_{gt_file_name}"), img)


if __name__ == '__main__':
    args = parse_args()

    # Set up paths for dataset and prompt output
    dataset_path = args.dataset_path
    gt_dataset_path = os.path.join(dataset_path, "binary_annotations/training")
    save_path = os.path.join(dataset_path, "prompts_test/training")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load dataset
    gt_file_names = os.listdir(gt_dataset_path)
    for gt_file_name in tqdm(gt_file_names):
        gt_file_path = os.path.join(gt_dataset_path, gt_file_name)
        img = cv2.imread(gt_file_path, 0)

        # Find coordinates of all 1-pixel points
        points = np.argwhere(img == 1)

        num_points = len(points)
        if num_points == 0:
            # Save original image if no 1-pixel points found
            save_empty_prompts(save_path, gt_file_name, np.zeros((256, 256)))
        else:
            # Use DBSCAN algorithm to cluster the data into different groups
            dbscan = DBSCAN(eps=8, min_samples=9).fit(points)

            # Find the centroids of each cluster.
            centers = []
            for label in set(dbscan.labels_):
                if label == -1:
                    continue
                cluster_indices = points[dbscan.labels_ == label]
                center = np.mean(cluster_indices, axis=0, dtype=int)
                centers.append(center)

            if len(centers) == 0:
                save_empty_prompts(save_path, gt_file_name, np.zeros((256, 256)))

            # For each cluster, find the 9 points closest to the centroid and mark them as points of that cluster
            center_points = []
            for center in centers:
                distances = cdist([center], points)
                nearest_indices = np.argsort(distances)[0][:9]
                center_points.append(points[nearest_indices])
            # Choose the closest 9 points from each cluster
            center_points = np.concatenate(center_points)[:9]

            # Randomly select 9 points
            random_points = points[np.random.choice(points.shape[0], 9, replace=False)]

            # Create image with point prompt
            point_img = np.zeros((2, 256, 256))
            random_x = np.random.randint(-20, 20, len(center_points))
            random_y = np.random.randint(-20, 20, len(center_points))
            center_y, center_x = np.array(center_points).T
            x = np.clip(center_x + random_x, 0, 255)
            y = np.clip(center_y + random_y, 0, 255)
            point_img[0][random_points[:, 0], random_points[:, 1]] = 255
            for i in range(len(center_points)):
                while point_img[1][y[i]][x[i]] == 255:
                    random_x[i], random_y[i] = np.random.randint(-20, 20, 2)
                    x[i] = np.clip(center_x[i] + random_x[i], 0, 255)
                    y[i] = np.clip(center_y[i] + random_y[i], 0, 255)
                point_img[1][y[i]][x[i]] = 255

            # Find bounding box for 1 data block
            min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
            max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

            # Create image with bounding box prompt
            bbox_img = np.zeros((256, 256))
            bbox_img[min_x][min_y] = bbox_img[max_x][max_y] = 255

            # Create image with mask prompt(multiple random points)
            pre_mask_img = np.zeros((256, 256))
            num_selected = int(num_points * 0.008)
            selected_points = random.sample(points.tolist(), num_selected)
            for point in selected_points:
                random_x = random.randint(-5, 5)
                random_y = random.randint(-5, 5)
                x = np.clip(point[1] + random_x, 0, 255)
                y = np.clip(point[0] + random_y, 0, 255)
                cv2.circle(pre_mask_img, (x, y), 1, 1, -1)

            # Dilate and close morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            pre_mask_img = cv2.morphologyEx(cv2.dilate(pre_mask_img, kernel, iterations=5), cv2.MORPH_CLOSE, kernel)
            pre_mask_img = pre_mask_img.astype(np.uint8)

            # Draw contours and fill pixels inside contours on a black image to create a lower-precision ground truth
            contours, hierarchy = cv2.findContours(pre_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pre_mask_img = np.where(pre_mask_img == 1, 255, 0)
            mask_img = np.zeros((256, 256), dtype=np.uint8)
            cv2.drawContours(mask_img, contours, -1, 255, thickness=-1)

            # Using Mask Images to Reduce Accuracy
            mask = np.ones((32, 32), dtype=np.float32)
            mask = cv2.resize(mask, (256, 256))
            mask = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_img = mask_img * mask

            # Expanding and corroding the contour, causing the boundary to become tortuous
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_img = cv2.morphologyEx(cv2.dilate(mask_img, kernel, iterations=2), cv2.MORPH_CLOSE, kernel)

            # Save new prompt images
            cv2.imwrite(os.path.join(save_path, f"random_points_{gt_file_name}"), point_img[0])
            cv2.imwrite(os.path.join(save_path, f"center_points_{gt_file_name}"), point_img[1])
            cv2.imwrite(os.path.join(save_path, f"box_{gt_file_name}"), bbox_img)
            cv2.imwrite(os.path.join(save_path, f"unfilled_mask_{gt_file_name}"), pre_mask_img)
            cv2.imwrite(os.path.join(save_path, f"filled_mask_{gt_file_name}"), mask_img)

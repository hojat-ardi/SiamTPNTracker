import numpy as np
import re
import os

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def read_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Use regular expression to extract floating-point numbers
        boxes = [re.findall(r"[-+]?\d*\.\d+|\d+", line) for line in lines]
        # Convert extracted strings to floats
        boxes = [list(map(float, box)) for box in boxes]
    return np.array(boxes)

def calculate_ao_for_sequence(gt_file_path, pred_file_path):
    gt_boxes = read_bounding_boxes(gt_file_path)
    pred_boxes = read_bounding_boxes(pred_file_path)
    ious = np.array([calculate_iou(gt_box, pred_box) for gt_box, pred_box in zip(gt_boxes, pred_boxes)])
    return np.mean(ious)


# Directories containing the ground truth and estimation files
gt_dir = '/home/ardi/Desktop/Dataset/GOT-10k/got-10k/ground_truth'
pred_dir = '/home/ardi/Desktop/project/SiamTPNTracker/results/got10k'

# List all txt files in each directory
gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]

# Assume that the file names are the same in both directories and can be used to match files
sequences_ao = []

for gt_file in gt_files:
    # Construct the full file path for ground truth and prediction
    gt_file_path = os.path.join(gt_dir, gt_file)
    pred_file_path = os.path.join(pred_dir, gt_file)  # Assumes the same filename for the prediction

    # Calculate the AO for this sequence and add it to the list
    ao = calculate_ao_for_sequence(gt_file_path, pred_file_path)
    sequences_ao.append(ao)

# Calculate the overall AO across all sequences
overall_ao = np.mean(sequences_ao)

print(f'Overall Average Overlap (AO) across all sequences: {overall_ao}')
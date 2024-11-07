import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VOCDetection
import torch.nn.functional as F
from numpy.linalg import norm

net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")

with open("yolo-coco/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


class DIPCNN(nn.Module):
    def __init__(self):
        super(DIPCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 104 * 104, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def apply_DIP(image, dip_params):
    brightness, contrast, sharpness = dip_params

    avg_brightness = np.average(norm(image, axis=2)) / np.sqrt(3)

    brightness = max(1/avg_brightness * 3000, brightness)  # Brightness from avg brightness of pixels or dip params
    contrast = max(1.15, contrast)    
    sharpness = max(1.15, sharpness)

    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    if sharpness > 1:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Simple sharpening filter
        image = cv2.filter2D(image, -1, kernel)
    
    return image

# Object detection for YOLO with image enhancement
def detect_objects(image_path, cnn_model):
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((416, 416))])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict DIP parameters using CNN
    dip_params = cnn_model(image_tensor).detach().numpy()[0]

    filtered_image = apply_DIP(image, dip_params)

    blob = cv2.dnn.blobFromImage(filtered_image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass to get detections
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Eliminate redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green
            cv2.rectangle(filtered_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(filtered_image, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return filtered_image

def train(model: nn.Module,
          loss_fn: nn.modules.loss._Loss,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          epoch: int=0):
    # ----------- <Your code> ---------------

    model.train()

    train_losses = []
    train_counter = []

    for batch_idx, (images, targets) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(images)
        # print(targets)
        loss = loss_fn(output, targets)
        # loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0: # We record our output every 10 batches
            train_losses.append(loss.item()) # item() is to get the value of the tensor directly
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        if batch_idx % 100 == 0: # We visulize our output every 10 batches
            print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item():.3f}')

    return train_losses

def prediction_accuracy(image_tensor, annotation, cnn_model, show_images, enhanced_model, score_threshold):
    # Convert the image tensor to a numpy array and reshape for OpenCV
    numpy_image = image_tensor.mul(255).byte().numpy()  # Scale to [0, 255]
    image = np.transpose(numpy_image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if(show_images):
        cv2.imshow("Original Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    height, width = image.shape[:2]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((416, 416))])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict DIP parameters using CNN
    dip_params = cnn_model(image_tensor).detach().numpy()[0]

    filtered_image = apply_DIP(image, dip_params) if enhanced_model else image

    if(enhanced_model):
        # Filter enhanced YOLO Model
        blob = cv2.dnn.blobFromImage(filtered_image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > score_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=0.4)

        pred_boxes, pred_labels = [], []

        # Draw predicted bounding boxes and labels from filtered image
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(class_names[class_ids[i]])

                pred_boxes.append([x, y, w, h])
                pred_labels.append(label)
                confidence = confidences[i]

                color = (0, 255, 0)  # Green
                cv2.rectangle(filtered_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(filtered_image, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    if(not enhanced_model):
        # Baseline YOLOv3 Model
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        base_boxes, base_confidences, base_class_ids = [], [], []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > score_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    base_boxes.append([x, y, w, h])
                    base_confidences.append(float(confidence))
                    base_class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(base_boxes, base_confidences, score_threshold=score_threshold, nms_threshold=0.4)

        pred_boxes, pred_labels = [], []

        # Draw predicted bounding boxes and labels from original image
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = base_boxes[i]
                label = str(class_names[base_class_ids[i]])

                pred_boxes.append([x, y, w, h])
                pred_labels.append(label)
                confidence = base_confidences[i]

                color = (0, 255, 0)  # Green
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    actual_boxes = []
    actual_labels= []

    # Draw ground truth bounding boxes
    for obj in annotation['annotation']['object']:
        label = obj['name'][0] 
        box = obj['bndbox']
        xmin = int(box['xmin'][0])
        ymin = int(box['ymin'][0])
        xmax = int(box['xmax'][0])
        ymax = int(box['ymax'][0])

        actual_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
        actual_labels.append(label)

        color = (255, 0, 0)  # Blue for ground truth
        cv2.rectangle(filtered_image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(filtered_image, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    iou_threshold = 0.5
    true_positives = 0

    # print(pred_boxes, actual_boxes, pred_labels, actual_labels)
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        for gt_box, gt_label in zip(actual_boxes, actual_labels):
            if gt_label == pred_label and calculate_iou(pred_box, gt_box) >= iou_threshold:
                true_positives += 1
                break
    
    # Calculate precision and recall
    precision = true_positives / len(pred_boxes) if pred_boxes else 0
    recall = true_positives / len(actual_boxes) if actual_boxes else 0

    if(show_images):
        print(f"Image Average Precision: {precision:.2f}, Image Recall: {recall:.2f}")
        cv2.imshow("Enhanced YOLOv3 Object Detection", filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filtered_image, precision, recall

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset (VOCDetection function wants images in jpg but dataset was provided in png, so either convert images or change it in function)
dataset = VOCDetection(root='Dataset', year='2007', image_set='test', transform=transform) 
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split into train and test
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize CNN model
cnn_model = DIPCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# for epoch in range(1, 3):
#     train(cnn_model, criterion, optimizer, train_loader, epoch)

# Set to true to run model on a specific image selected
specific_image_demo = False

if(specific_image_demo):
    # If trying to test on a specific image
    image_path = "kite.jpg"
    # annotation_path = "Dataset/VOCdevkit/VOC2007/Annotations/YT_Google_043.png" 
    result_image = detect_objects(image_path, cnn_model)

    # Display the result
    cv2.imshow("Enhanced YOLOv3 Object Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    mAP = 0
    mRecall = 0

    # Set demo to true to see results only on one image of dataset and display it
    demo = True

    # Iterate over each image and target (annotation)
    for images, targets in test_loader:
        single_image = images.squeeze(0)
        single_target = targets

        # print(single_image, single_target)

        # Set enhanced_model to false to test yolo model without image filtering
        pred_image, curr_precision, curr_recall = prediction_accuracy(single_image, single_target, cnn_model, show_images=demo, enhanced_model=True, score_threshold=.5)
        mAP += curr_precision
        mRecall += curr_recall

        if(demo):
            break # Stop after first image
    
    mAP = mAP / test_size
    mRecall = mRecall / test_size

    if(not demo):
        print(f"Mean Average Precision: {mAP:.2f}, Mean Recall: {mRecall:.2f}")

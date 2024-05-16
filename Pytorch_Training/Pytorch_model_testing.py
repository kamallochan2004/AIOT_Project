from PIL import Image, ImageDraw
import numpy as np
import cv2
from mtcnn import MTCNN
import torch
import torch.nn as nn
from torchvision import transforms, models



# Loading ready model
state_dict = torch.load('D:/another_try_mask/best.pt', map_location=torch.device('cpu'))


class FaceMaskModel(nn.Module):
    def __init__(self):
        super(FaceMaskModel, self).__init__()
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)




# Face mask detection model
model = FaceMaskModel()
model.load_state_dict(state_dict)
model.eval()


# Transformation for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# labels (with mask and without mask)
labels = ['Without Mask', 'With Mask']



detector = MTCNN()



# Initialize the video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open the camera")
    exit()
    
    

# Loop over frames from the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    # face detection using MTCNN
    boxes = detector.detect_faces(rgb_frame)

    
    
    # detected faces
    for box in boxes:
        x, y, width, height = box['box']

    
    
    
    
     #face region from the frame
        face = frame[y:y+height, x:x+width]

        
        
        #  face to PIL Image format
        pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        
        transformed_face = transform(pil_face)

        
        
        # a batch dimension to the face image
        transformed_face = transformed_face.unsqueeze(0)

        
        with torch.no_grad():
            outputs = model(transformed_face)
            _, predicted = torch.max(outputs, 1)

        
        
        label = labels[predicted.item()]

        # bounding box and lebel
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        

cap.release()
cv2.destroyAllWindows()

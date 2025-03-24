import mlflow
import torchvision.transforms.v2 as v2
import torch
from itertools import compress
from utils.constant_utils import celeba_columns
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from utils.transform_utils import ShuffleNet_V2_X0_5_FaceTransforms
import time
import cv2





def make_predictions(model, img, transforms):

    ## Apply the transformations
    img = transforms(img)

    ## Check the shape of the transformed image
    if img.shape != torch.Size([1, 3, 224, 224]):
        img = img.unsqueeze(0)

    res = None

    with torch.no_grad():
        res = model(img)
        res = torch.nn.Sigmoid()(res)
        res = res > 0.5
        res = list(compress(celeba_columns, res[0].tolist()))
    
    return res
    




def detect_faces(img, detector):
    preds = detector.predict(img)

    num_of_images = preds[0].boxes.xyxy.shape[0]
    pad = 15
    for idx in range(num_of_images):

        x1, y1, x2, y2 = preds[0].boxes.xyxy[idx].tolist()
        cropped_img = img.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad))
        
        yield cropped_img





def detect_frame(img, detector):
    print('Detecting frame...')
    img = v2.ToPILImage()(img)
    preds = detector.predict(img)
    
    num_of_images = preds[0].boxes.xyxy.shape[0]
    pad = 25
    for idx in range(num_of_images):

        x1, y1, x2, y2 = preds[0].boxes.xyxy[idx].tolist()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        
        cropped_img = img.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad))
        bbox_conf = round(preds[0].boxes.conf[idx].item(), 3)
        
        yield cropped_img, x, y, w, h, bbox_conf





def detect_image(img, model, detector, transforms):
    #img = Image.open('./static/face_image.jpeg')

    for idx, cropped_img in enumerate(detect_faces(img, detector)):


        start = time.time()

        res = make_predictions(
            model,
            cropped_img,
            transforms
        )

        
        if 'Male' not in res:
            res.append('Female')


        inference_time = (time.time() - start) * 1000
        print(f"Inference time for image {idx + 1}: {inference_time} ms")
        print(res)
     
        _, ax = plt.subplots(figsize=(8, 8))

        ## Display the image

        ax.imshow(cropped_img)
        ax.axis('off')

        ## Add predictions as text
        result_text = "\n".join(res)
        ax.text(
            1.05, 0.5, result_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

        plt.show()
    




def detect_video(img, model, detector, transforms):

    ## Detect faces

    for idx, (cropped_img, x, y, w, h, conf) in enumerate(detect_frame(img, detector)):

        if conf < 0.7:
            continue

        start = time.time()

        res = make_predictions(
            model,
            cropped_img,
            transforms
        )

        
        if 'Male' not in res:
            res.append('Female')


        inference_time = (time.time() - start) * 1000
        print(f"Inference time for image {idx + 1}: {inference_time} ms")
        print(res)
        cv2.putText(img, str(conf), (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        for j, label in enumerate(res):
            cv2.putText(img, f'{label}', (x + w + 5, y + (j * 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 255, 255), 1)
        




def process_video(video_path):

    mlflow.set_tracking_uri('http://localhost:8080')

    model = mlflow.pytorch.load_model('mlflow-artifacts:/365604127024099804/feef9253f36b45148a2513966cf7f0ab/artifacts/model/')
    detector = YOLO('./static/yolov11n-face.pt')

    ## Detect faces
    #img = Image.open('./static/face_image.jpeg')
    transforms = ShuffleNet_V2_X0_5_FaceTransforms(inference=True, pad=25)

    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False):
        print('Error opening the video file!')

    frames = []

    recognition_times = []

    


    with torch.inference_mode():
        while (cap.isOpened()):
            
            ret, frame = cap.read()
            
            if not ret:
                break
            
            print('==================================================================')
            start = time.time()
            detect_video(frame, model, detector, transforms)

            end = time.time()

            rec_time = (end - start) * 1000
            recognition_times.append(rec_time)
            print('Overall time: ', rec_time)

            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
            

    cap.release()
    cv2.destroyAllWindows()
    print('Avg time: ', sum(recognition_times) / len(recognition_times))
    print('Max. time: ', max(recognition_times))




if __name__ == '__main__':
    process_video('./static/REACHER Season 3 - Official Trailer _ Prime Video.mp4')
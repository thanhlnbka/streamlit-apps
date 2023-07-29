import requests
import cv2
import time

def predict_image(api_url, image_path):
    url = api_url
    files = {"image": open(image_path, "rb")}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get predictions. Status code: {response.status_code}")

if __name__ == "__main__":
    api_url = "http://localhost:5000/predict"
    image_path = "images/1.jpg"
    orig_image  = cv2.imread(image_path)
    try:
        t1 = time.time()
        predictions = predict_image(api_url, image_path)
        print(predictions.keys())
        boxes = predictions["boxes"]
        labels = predictions["labels"]
        scores = predictions["probs"]
        for i in range(len(boxes)): 
            box =boxes[i]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        # cv2.imwrite("response.jpg", orig_image)
        print("estimate time : ", time.time() - t1, "senconds")
    except Exception as e:
        print(f"Error: {e}")
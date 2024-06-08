import PIL.Image
from fastapi import FastAPI, File, UploadFile
from collections import Counter

# STEP 1: Import the necessary modules. 모듈가져오기
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: 추론기 만들기
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)
# 계속 로드하지 않도록함, 추론 객체는 한 번만 만들어두기
# 결론: 모델 사용을 확실히 해두고, 서버 띄우기

app = FastAPI()

import io
import PIL
import numpy as np

## 이미지 파일 처리할때는 이 방법 추천!
## 이미지 타입 확인하기
@app.post("/uploadfile/") #메타 정보만(ex)파일 이름, 확장자) 만 os에 보냄
async def create_upload_file(file: UploadFile):

    byte_file = await file.read()

    # STEP 3: Load the input image. 추론할 데이터 가져오고
    ## 이미지 파일 읽을 수 있게 변경해주기 (외우기!)
    # 1. convert char array to binary array
    image_bin = io.BytesIO(byte_file)

    # 2. create PIL image from binary array
    pil_img = PIL.Image.open(image_bin)

    # 3. Convert MP image from PIL IMAGE
    image = mp.Image(
    image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))


    # STEP 4:  추론된 결과
    detection_result = detector.detect(image)
    print(detection_result)
            

    # STEP 5: Process the classification result. In this case, visualize it. 어떻게 보여줄지 
    answer_list = []
    for detection in detection_result.detections:
        answer_list.append(detection.categories[0].category_name)

    num = Counter(answer_list)    

    return {"result": num}

#uvicorn cls_api:app
IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

import cv2
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
#   cv2.imshow("test", img)
#   cv2.waitKey(0)


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)





# STEP 1: Import the necessary modules. 모델(패키지) 가져오기(임포트하기)
import mediapipe as mp      # 필요 모델 임포트
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.        가져온 모델을 활용해서 추론기를 만들기
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite0.tflite')  # 모델 경로
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)        # 경로를 따라 만들어진 추론기


# STEP 3: Load the input image.        이미지 로드시키기
image = mp.Image.create_from_file(IMAGE_FILENAMES[1])

# STEP 4: Classify the input image.     데이터 추론시키기(결과값)
classification_result = classifier.classify(image)

# STEP 5: Process the classification result. In this case, visualize it.        결과값 보여주기
top_category = classification_result.classifications[0].categories[0]
result = (f"{top_category.category_name} ({top_category.score:.2f})")

print(result)
# display_batch_of_images(images, predictions)
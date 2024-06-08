# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2         name = "antelopev2"는 antelopev2으로 설정을 바꾼거(안하면 기본:buffalo_l)
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP 3
# img = ins_get_image('t1')
img = cv2.imread('k1.jpeg', cv2.IMREAD_COLOR)

# STEP 4
faces = app.get(img)

# STEP 5
# then print all-to-all face similarity
feats = []
feats.append(faces[0].normed_embedding)
feats.append(faces[0].normed_embedding)

# for face in faces:
# feats.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats[0], feats[1].T)
print(sims)

# rimg = app.draw_on(img, faces)
# cv2.imwrite("./k1_output.jpeg", rimg)


# print(len(faces))
# print(faces[0].embedding)
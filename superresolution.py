## pacotes
import matplotlib.pyplot as plt
import argparse
import imutils
import time
import cv2
import os


## converter as imagens do BGR para o RGB

def plt_imshow(title, image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)

	plt.grid(False)
	plt.show()


args = {
    "model": "export/LapSRN_x8.pb", #### nome do modelo
}


# extract the model name and model scale from the file path
modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

print("[INFO] loading super resolution model: {}".format(args["model"]))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
#sr.readModel("/content/TF-ESPCN/export/ESPCN_x4.pb")
sr.readModel(args["model"])
### infromaca o caminho do modelo - arquivo com pb no nome.
sr.setModel(modelName, modelScale)

### caminho da imagem!
uploaded_image_path = 'img.jpg'
image = cv2.imread(uploaded_image_path, cv2.IMREAD_UNCHANGED)
image = cv2.convertScaleAbs(image)

print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))

start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print("[INFO] super resolution took {:.6f} seconds".format(
	end - start))

print("[INFO] w: {}, h: {}".format(upscaled.shape[1],
	upscaled.shape[0]))

#interpolacaoa bicubica
start = time.time()
bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]),
	interpolation=cv2.INTER_CUBIC)
end = time.time()
print("[INFO] bicubic interpolation took {:.6f} seconds".format(
	end - start))

## plotar as 3 iamgens, a original, a interpolaçao bicubica e deep learning com super resolution.
plt_imshow("Original", image)
plt_imshow("Bicubic", bicubic)
plt_imshow("Super Resolution", upscaled)

### salvar a imagem.

%cd /content
!mkdir output
%cd /content/output
#cv2.imwrite("output_original.png", image)
#cv2.imwrite("esporos_b_bicubic.png", bicubic)
cv2.imwrite("esporos_b_sr.png", upscaled)


import numpy as np
np.bool = np.bool_

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
# pip install --upgrade tensorflow==2.9.1

import tensorflow_probability as tfp
# pip install --upgrade tensorflow_probability==0.11

from dju.utils import display

IMAGE_SIZE = 32
N_COMPONENTS = 5
EPOCHS = 10
BATCH_SIZE = 128

# 데이터 로드
(x_train, _), (_, _) = datasets.fashion_mnist.load_data()

# 데이터 전처리
def preprocess(imgs):
    imgs = np.expand_dims(imgs, -1)
    imgs = tf.image.resize(imgs, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    return imgs


input_data = preprocess(x_train)

# 훈련 세트에 있는 샘플 출력하기
display(input_data)

# PixelCNN 모델 정의
dist = tfp.distributions.PixelCNN(
    image_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=N_COMPONENTS,
    dropout_p=0.3,
)

# 모델 입력을 정의합니다.
image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

# 로그 가능도를 손실 함수로 정의합니다.
log_prob = dist.log_prob(image_input)

# 모델을 정의합니다.
pixelcnn = models.Model(inputs=image_input, outputs=log_prob)
pixelcnn.add_loss(-tf.reduce_mean(log_prob))

# 모델 컴파일 및 훈련
pixelcnn.compile(optimizer=optimizers.Adam(0.001),)

tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def generate(self):
        return dist.sample(self.num_img).numpy()

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generate()
        display(
            generated_images,
            n=self.num_img,
            save_to="./output/generated_img_%03d.png" % (epoch),
        )


img_generator_callback = ImageGenerator(num_img=2)

pixelcnn.fit(
    input_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=True,
    callbacks=[tensorboard_callback, img_generator_callback],
)

generated_images = img_generator_callback.generate()

display(generated_images, n=img_generator_callback.num_img)
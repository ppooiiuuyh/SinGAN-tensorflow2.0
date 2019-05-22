import sys

sys.path.append('../') #root path
from models.generator import *
from models.discriminator import *
from utils.data_utils import *
from utils.utils import *
from utils.xdog import *
from functools import partial

class Model_Tester():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):# Build Generator and Discriminator
        """ model """
        self.generatorF = GeneratorUnet(self.config.channels)


        """ saver """
        self.step = tf.Variable(0,dtype=tf.int64)
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                       generatorF=self.generatorF,
                                        )


    @tf.function
    def inference_runner(self, image):
        return self.generatorF(image)


    def inference(self, image, scale):
        s = scale
        ori_h, ori_w = image.shape[0],image.shape[1] #original shape
        h_ = ori_h * s - ori_h * s % 8
        w_ = ori_w * s - ori_w * s % 8


        image = cv2.resize(image, (int(w_), int(h_)))
        image_xdog = xdog(image)

        image = normalize(image)
        image = image.reshape(1, image.shape[0], image.shape[1], self.config.channels)
        result = self.generatorF(image).numpy()
        result = denormalize(np.squeeze(result))

        kernel = np.ones((3, 3), np.uint8);   factor = 1 ; intensity = 0.3
        image_xdog = cv2.erode(image_xdog, kernel, iterations=factor - 1)  # // make dilation image
        if result.shape.__len__() > 2:  image_xdog = image_xdog.reshape(image_xdog.shape[0], image_xdog.shape[1], 1)
        imageC = (np.copy(result) * ((image_xdog > 20) * (1 - intensity) + intensity)).astype(np.uint8)

        print(result.shape, imageC.shape)
        result = np.concatenate([result,imageC], axis=1)
        return result


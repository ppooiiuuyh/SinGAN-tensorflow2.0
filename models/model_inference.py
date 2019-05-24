import sys
sys.path.append('../') #root path
from models.generator import *
from models.discriminator import *
from utils.utils import *
from functools import partial
import numpy as np
import math

partial_resize = partial(tf.image.resize, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
class Model_Inference():
    def __init__(self, config, target_image):
        self.config = config
        self.num_scale = config.num_scale
        self.target_image = target_image
        self.build_model()
        self.step = tf.Variable(0,dtype=tf.int64)


    def build_model(self):# Build Generator and Discriminator
        """ model """
        self.target_images = [partial_resize(self.target_image, [int(self.target_image.shape[1] * (3 / 4) ** i), int(self.target_image.shape[2] * (3 / 4) ** i)]) for i in range(self.num_scale+1)]
        for i in self.target_images: print(i.shape)
        self.generators = [Generator(self.config.channels, N = i) for i in range(self.num_scale+1)]
        self.discriminators = [Discriminator(self.config.channels) for i in range(self.num_scale+1)]


    def restore(self):
        self.generators = [tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir,"generator_scale_{}.h5".format(i)),  custom_objects={'InstanceNorm':InstanceNorm}) for i in range(self.num_scale+1)]


    @tf.function
    def __inference(self, z_fixed, N=0, start_N=None):
        if start_N is None: start_N = self.num_scale


        priors = [None for i in range(0, self.num_scale + 1)]
        #priors are replaced with reconX upto start_N
        for i in range(start_N+1,self.num_scale + 1)[::-1]:
            if i == self.num_scale:
                priors[i] = tf.zeros_like(self.target_images[i])
            else:
                priors = self.generators[i + 1]([z_fixed if i + 1 == self.num_scale else tf.zeros_like(self.target_images[i + 1]), priors[i + 1]])
                priors = partial_resize(priors[i], [self.target_images[i].shape[1], self.target_images[i].shape[2]])

        #from start_N
        for i in range(N, start_N+1)[::-1]:
            if i == self.num_scale:
                priors[i] = tf.zeros_like(self.target_images[i])

            else:
                z = tf.random.normal(self.target_images[i + 1].shape)
                priors[i] = self.generators[i + 1]([z, priors[i + 1]])
                priors[i] = partial_resize(priors[i], [self.target_images[i].shape[1], self.target_images[i].shape[2]])

        z = tf.random.normal(self.target_images[N].shape)
        gen_output = self.generators[N]([z, priors[N]], training=True)
        return gen_output


    def inference(self,start_N = None):
        if start_N is None : start_N = self.num_scale

        np.random.seed(0)
        z_fixed = np.random.normal(size = self.target_images[-1].shape).astype(np.float32)
        gen_output = self.__inference(z_fixed=z_fixed, start_N=start_N)
        return  denormalize(gen_output.numpy()[0]), denormalize(self.target_images[0].numpy()[0])




    @tf.function
    def __inference_with_inject(self,N=0, inject_N=None,inject_image=None):
        priors = [None for i in range(0, self.num_scale + 1)]

        for i in range(N, inject_N + 1)[::-1]:
            if  i == inject_N:
                priors[i] = inject_image

            else:
                z = tf.random.normal(self.target_images[i + 1].shape)
                #z = tf.zeros_like(self.target_images[i + 1])
                priors[i] = self.generators[i + 1]([z, priors[i + 1]])
                priors[i] = partial_resize(priors[i], [self.target_images[i].shape[1], self.target_images[i].shape[2]])

        z = tf.random.normal(inject_image.shape) if inject_N == 0 else tf.random.normal(self.target_images[N].shape)
        #z = tf.zeros_like(self.target_images[0])
        gen_output = self.generators[N]([z, priors[N]], training=True)
        return gen_output

    def inference_paint_to_image(self, inject_N=None):
        inject_N=self.num_scale-1
        gen_output = self.__inference_with_inject(inject_N=inject_N, inject_image=self.target_images[inject_N])
        return  denormalize(gen_output.numpy()[0]), denormalize(self.target_images[0].numpy()[0])

    def inference_harmonization(self, inject_N=None):
        inject_N= self.num_scale-1
        gen_output = self.__inference_with_inject(inject_N=inject_N, inject_image=self.target_images[inject_N])
        return  denormalize(gen_output.numpy()[0]), denormalize(self.target_images[0].numpy()[0])

    def inference_editing(self, inject_N=None):
        inject_N= self.num_scale-3
        gen_output = self.__inference_with_inject(inject_N=inject_N, inject_image=self.target_images[inject_N])
        return  denormalize(gen_output.numpy()[0]), denormalize(self.target_images[0].numpy()[0])





    @tf.function
    def __inference_for_sr(self,noise, inject_image):
        gen_output = self.generators[0]([noise, inject_image])
        return gen_output


    def inference_sr(self, scale_factor=2, repeat = 4):
        Hs = [int(self.target_images[0].shape[1] + self.target_images[0].shape[1]* scale_factor **(i/repeat) ) for i in range(repeat)]
        Ws = [int(self.target_images[0].shape[2] + self.target_images[0].shape[2]* scale_factor **(i/repeat) ) for i in range(repeat)]

        gen_output = self.target_images[0]
        for H,W in zip(Hs,Ws):
            inject_img_temp = partial_resize(gen_output,[H,W])
            z = tf.zeros_like(inject_img_temp)#z = tf.random.normal(inject_img_temp.shape)
            gen_output = self.__inference_for_sr(noise = z, inject_image=inject_img_temp)

        return  denormalize(gen_output.numpy()[0]), denormalize(partial_resize(self.target_images[0],[Hs[-1],Ws[-1]]).numpy()[0])


import sys
sys.path.append('../') #root path
from models.generator import *
from models.discriminator import *
from utils.utils import *
from functools import partial
import numpy as np

partial_resize = partial(tf.image.resize, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
class Model_Train():
    def __init__(self, config, target_image):
        self.config = config
        self.num_scale = config.num_scale
        self.target_image = target_image
        self.step = tf.Variable(0,dtype=tf.int64)
        self.build_model()
        log_dir = os.path.join(config.summary_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)


    def build_model(self):# Build Generator and Discriminator
        """ model """
        self.target_images = [partial_resize(self.target_image, [int(self.target_image.shape[1] * (3 / 4) ** i), int(self.target_image.shape[2] * (3 / 4) ** i)]) for i in range(self.num_scale+1)]
        for i in self.target_images:
            print(i.shape)

        self.generators = [Generator(self.config.channels, N = i, num_scale = self.num_scale) for i in range(self.num_scale+1)]
        self.discriminators = [Discriminator(self.config.channels, N= i, num_scale = self.num_scale) for i in range(self.num_scale+1)]
        learning_rate = 5e-4 * (0.1 ** tf.cast(self.step //1600, dtype=tf.float32))
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        #self.generator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0. , beta_2=0.9)
        #self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0., beta_2=0.9)


    def save(self):
        for i in range(self.num_scale+1):
            self.generators[i].save(os.path.join(self.config.checkpoint_dir,"generator_scale_{}.h5".format(i)))
            self.discriminators[i].save(os.path.join(self.config.checkpoint_dir,"discriminator_scale_{}.h5".format(i)))

    def restore(self, N = None):
        for i in range(self.num_scale+1):
            self.generators[i] = tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir,"generator_scale_{}.h5".format(i)),  custom_objects={'InstanceNorm':InstanceNorm})
            self.discriminators[i] = tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir,"discriminator_scale_{}.h5".format(i)), custom_objects={'SpecConv2DLayer': SpecConv2DLayer})

        if N is not None :
            self.generators[N] = tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir, "generator_scale_{}.h5".format(N+1)), custom_objects={'InstanceNorm': InstanceNorm})
            self.discriminators[N] = tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir, "discriminator_scale_{}.h5".format(N+1)), custom_objects={'SpecConv2DLayer': SpecConv2DLayer})

    @tf.function
    def training(self, z_fixed, N):
        priors = [None for i in range(0, self.num_scale + 1)]
        for i in range(N, self.num_scale + 1)[::-1]:
            if i == self.num_scale:
                priors[i] = tf.zeros_like(self.target_images[i])
            else:
                z = tf.random.normal(self.target_images[i + 1].shape)
                priors[i] = self.generators[i + 1]([z, priors[i + 1]])
                priors[i] = partial_resize(priors[i], [self.target_images[i].shape[1], self.target_images[i].shape[2]])

        prior_recons = [None for i in range(0, self.num_scale + 1)]
        for i in range(N, self.num_scale + 1)[::-1]:
            if i == self.num_scale:
                prior_recons[i] = tf.zeros_like(self.target_images[i])
            else:
                prior_recons[i] = self.generators[i + 1]([z_fixed if i + 1 == self.num_scale else tf.zeros_like(self.target_images[i + 1]), prior_recons[i + 1]])
                prior_recons[i] = partial_resize(prior_recons[i], [self.target_images[i].shape[1], self.target_images[i].shape[2]])

        for i in range(6):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                z = tf.random.normal(self.target_images[N].shape)
                gen_output = self.generators[N]([z, priors[N]], training=True)
                gen_recon_output = self.generators[N]([z_fixed if N == self.num_scale else tf.zeros_like(self.target_images[N]), prior_recons[N]])
                disc_real_output = self.discriminators[N]([self.target_images[N]], training=True)
                disc_generated_output = self.discriminators[N]([gen_output], training=True)

                """ loss for discriminator """
                disc_loss = discriminator_adv_loss(disc_real_output, disc_generated_output)
                # disc_loss = getHingeDLoss(disc_real_output, disc_generated_output)
                # disc_loss = dicriminator_wgan_loss(self.discriminators[N],target_image=target_image,fake_image=gen_output, batch_size=self.config.batch_size)

                """ loss for generator """
                gen_adv_loss = generator_adv_loss(disc_generated_output)
                # gen_adv_loss = getHingeGLoss(disc_generated_output)
                # gen_adv_loss = generator_wgan_loss(disc_generated_output)
                gen_recon_loss = tf.reduce_mean(tf.square(gen_recon_output - self.target_images[N]))
                gen_loss = gen_adv_loss + 10 * gen_recon_loss

            """ optimize """
            G_vars = self.generators[N].trainable_variables
            D_vars = self.discriminators[N].trainable_variables
            if i < 3:
                discriminator_gradients = disc_tape.gradient(disc_loss, D_vars)
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D_vars))
            else :
                generator_gradients = gen_tape.gradient(gen_loss, G_vars)
                self.generator_optimizer.apply_gradients(zip(generator_gradients, G_vars))



        inputs_concat = tf.concat([z, priors[N], prior_recons[N], self.target_images[N]], axis=2)
        return_dicts = {"inputs_concat": inputs_concat}
        return_dicts.update({'disc_loss': disc_loss})
        return_dicts.update({'gen_loss': gen_adv_loss})
        return_dicts.update({'gan_loss': disc_loss + gen_adv_loss})
        return_dicts.update({'rec_loss': gen_recon_loss})
        return_dicts.update({'gen_output': tf.concat([z, priors[N], gen_output, prior_recons[N], gen_recon_output, self.target_images[N]], axis=2)})
        return return_dicts



    def train_step(self, N=None, summary_name = "train", log_interval = 100):
        np.random.seed(0)
        z_fixed = np.random.normal(size = self.target_images[-1].shape).astype(np.float32)
        #z_fixed = tf.random.normal(self.target_images[-1].shape, seed=0)

        """ training """
        if N == None : N = self.num_scale
        result_logs_dict = self.training(z_fixed=z_fixed, N = N)


        #cv2.imshow('image',denormalize(np.concatenate([gen_outputs[N].numpy(),recon_outputs[N].numpy(),input_resized ],axis=2)[0]))
        #cv2.waitKey(10)


        """ log summary """
        if summary_name and self.step.numpy() % log_interval == 0:
            with self.train_summary_writer.as_default():
                for key, value in result_logs_dict.items():
                    value = value.numpy()
                    if len(value.shape) == 0:
                        tf.summary.scalar("{}_{}_{}".format(N, summary_name,key), value, step=self.step)
                    elif len(value.shape) in [3,4]:
                        tf.summary.image("{}_{}_{}".format(N, summary_name, key), denormalize(value), step=self.step)

        log = "disc loss:{} gen loss:{} rec loss:{}".format(result_logs_dict["disc_loss"], result_logs_dict["gen_loss"], result_logs_dict["rec_loss"])
        return log

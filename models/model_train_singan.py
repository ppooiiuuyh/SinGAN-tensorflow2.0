import sys
sys.path.append('../') #root path
from models.generator import *
from models.discriminator import *
from utils.utils import *
from functools import partial


class Model_Train():
    def __init__(self, config):
        self.config = config
        self.num_scale = 8
        self.build_model()
        log_dir = os.path.join(config.summary_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def build_model(self):# Build Generator and Discriminator
        """ model """
        self.generators = [Generator(self.config.channels, N = i) for i in range(self.num_scale+1)]
        self.discriminators = [Discriminator(self.config.channels) for i in range(self.num_scale+1)]
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


        """ saver """
        self.step = tf.Variable(0,dtype=tf.int64)
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                   generator_optimizer=self.generator_optimizer,
                                   discriminator_optimizer=self.discriminator_optimizer,
                                )

        self.save_manager = tf.train.CheckpointManager(self.ckpt, self.config.checkpoint_dir, max_to_keep=3)
        self.save  = lambda : self.save_manager.save(checkpoint_number=self.step) #exaple : model.save()




    @tf.function
    def training(self,prior_recon, prior, target_image, N = 0):
        if N == self.num_scale:
            prior = tf.zeros_like(target_image)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            z = tf.random.normal(target_image.shape)
            gen_output = self.generators[N]([z,prior], training=True)
            gen_recon_output = self.generators[N]([tf.zeros_like(prior),prior_recon]) if N == self.num_scale else self.generators[N]([prior_recon, tf.zeros_like(prior)])
            disc_real_output = self.discriminators[N]([target_image], training=True)
            disc_generated_output = self.discriminators[N]([gen_output], training=True)

            """ loss for discriminator """
            disc_loss = discriminator_adv_loss(disc_real_output, disc_generated_output)

            """ loss for generator """
            gen_adv_loss = generator_adv_loss(disc_generated_output)
            gen_recon_loss = tf.reduce_mean(tf.square(gen_recon_output - target_image))
            gen_loss = gen_adv_loss + 10 * gen_recon_loss

        G_vars = self.generators[N].trainable_variables
        D_vars = self.discriminators[N].trainable_variables
        generator_gradients = gen_tape.gradient(gen_loss, G_vars)
        discriminator_gradients = disc_tape.gradient(disc_loss, D_vars)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D_vars))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, G_vars))


        inputs_concat = tf.concat([z, prior, target_image], axis=2)
        return_dicts = {"inputs_concat" :inputs_concat}
        return_dicts.update({'disc_loss' : disc_loss})
        return_dicts.update({'gen_loss' : gen_loss})
        return_dicts.update({'gan_loss': disc_loss+ gen_loss})
        return_dicts.update({'rec_loss' : gen_recon_loss})
        return_dicts.update({'gen_output': tf.concat([z, prior,gen_output,gen_recon_output, target_image], axis=2) })
        return return_dicts, gen_output, gen_recon_output



    def train_step(self,input, summary_name = "train", log_interval = 100):
        partial_resize = partial(tf.image.resize, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        recon_outputs = [None for i in range(self.num_scale+1)]
        gen_outputs = [None for i in range(self.num_scale+1)]
        z_fixed = tf.random.normal(partial_resize(input, [int(input.shape[1]* (3/4) **self.num_scale), int(input.shape[2]* (3/4) **self.num_scale)]).shape)

        """ training """
        while True:
            N = self.num_scale
            input_resized = partial_resize(input, [int(input.shape[1] * (3 / 4) ** N), int(input.shape[2] * (3 / 4) ** N)])
            if N == self.num_scale:
                result_logs_dict, gen_outputs[N], recon_outputs[N] = self.training(prior_recon = z_fixed, prior= None, target_image = input_resized, N = N)
            else :
                output_prior = partial_resize(recon_outputs[N+1], [input_resized.shape[1], input_resized.shape[2]])
                recon_prior = partial_resize(recon_outputs[N+1], [input_resized.shape[1], input_resized.shape[2]])
                result_logs_dict, gen_outputs[N], recon_outputs[N] = self.training(prior_recon = recon_prior, prior= output_prior, target_image = input_resized, N = N)

            print("[train] N:{} step:{} gan loss:{} rec loss:{}".format(N, self.step.numpy(), result_logs_dict["gan_loss"], result_logs_dict["rec_loss"]))

            #cv2.imshow('image',denormalize(np.concatenate([gen_outputs[N].numpy(),recon_outputs[N].numpy(),input_resized ],axis=2)[0]))
            cv2.waitKey(10)


            """ log summary """
            if summary_name and self.step.numpy() % log_interval == 0:
                with self.train_summary_writer.as_default():
                    for key, value in result_logs_dict.items():
                        value = value.numpy()
                        if len(value.shape) == 0:
                            tf.summary.scalar("{}_{}".format(summary_name,key), value, step=self.step)
                        elif len(value.shape) in [3,4]:
                            tf.summary.image("{}_{}".format(summary_name, key), denormalize(value), step=self.step)



            self.step.assign_add(1)
        return ""

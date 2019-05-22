import sys

sys.path.append('../') #root path
from models.generator import *
from models.discriminator import *
from utils.data_utils import *
from utils.utils import *
from functools import partial

class Model_Train():
    def __init__(self, config):
        self.config = config
        self.build_model()
        log_dir = os.path.join(config.summary_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def build_model(self):# Build Generator and Discriminator
        """ model """
        self.generatorF = Generator(self.config.channels)
        self.generatorG = Generator(self.config.channels)
        self.discriminatorA = Discriminator(self.config.channels)
        self.discriminatorB = Discriminator(self.config.channels)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        #self.generator.summary()


        """ saver """
        self.step = tf.Variable(0,dtype=tf.int64)
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                   generator_optimizer=self.generator_optimizer,
                                   discriminator_optimizer=self.discriminator_optimizer,
                                   generatorF=self.generatorF,
                                   generatorG=self.generatorG,
                                   discriminatorA=self.discriminatorA,
                                   discriminatorB = self.discriminatorB)

        self.save_manager = tf.train.CheckpointManager(self.ckpt, self.config.checkpoint_dir, max_to_keep=3)
        self.save  = lambda : self.save_manager.save(checkpoint_number=self.step) #exaple : model.save()




    @tf.function
    def training(self, inputs):
        unpaired_input, unpaired_target = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            A_from_B = self.generatorF(unpaired_target, training=True)
            B_from_A = self.generatorG(unpaired_input, training=True)

            identity_A = self.generatorF(unpaired_input, training=True)
            identity_B = self.generatorG(unpaired_target, training=True)
            recon_A = self.generatorG(B_from_A, training=True)
            recon_B = self.generatorF(A_from_B, training=True)

            """ loss for discriminator """
            # dicriminator A
            discA_real_output = self.discriminatorA(unpaired_input, training=True)
            discA_generated_output = self.discriminatorA(A_from_B, training=True)
            discA_loss = discriminator_adv_loss(discA_real_output, discA_generated_output)

            # dicriminator A
            discB_real_output = self.discriminatorB(unpaired_target, training=True)
            discB_generated_output = self.discriminatorB(B_from_A, training=True)
            discB_loss = discriminator_adv_loss(discB_real_output, discB_generated_output)

            disc_loss = discA_loss + discB_loss

            """ loss for generator """
            gen_adv_lossF = generator_adv_loss(discA_generated_output)
            gen_adv_lossG = generator_adv_loss(discB_generated_output)
            gen_L1_loss = L1loss(recon_A, unpaired_input) + L1loss(recon_B, unpaired_target) + L1loss(identity_A, unpaired_input) + L1loss(identity_B, unpaired_target)
            gen_loss = gen_adv_lossF + gen_adv_lossG + gen_L1_loss * 10

        G_vars = self.generatorF.trainable_variables + self.generatorG.trainable_variables
        D_vars = self.discriminatorA.trainable_variables + self.discriminatorB.trainable_variables
        generator_gradients = gen_tape.gradient(gen_loss, G_vars)
        discriminator_gradients = disc_tape.gradient(disc_loss, D_vars)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D_vars))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, G_vars))

        inputs_concat = tf.concat([unpaired_input, unpaired_target], axis=2)
        return_dicts = {"inputs_concat" :inputs_concat}
        return_dicts.update({'disc_loss' : disc_loss})
        return_dicts.update({'gen_loss' : gen_loss})
        return_dicts.update({'B_from_A': tf.concat([unpaired_input,B_from_A,unpaired_target],axis=2)})
        return_dicts.update({'A_from_B': tf.concat([unpaired_target,A_from_B,unpaired_input],axis=2)})
        return return_dicts


    def train_step(self,iterator, summary_name = "train", log_interval = 100):
        """ training """
        result_logs_dict = self.training(iterator.__next__())

        """ log summary """
        if summary_name and self.step.numpy() % log_interval == 0:
            with self.train_summary_writer.as_default():
                for key, value in result_logs_dict.items():
                    value = value.numpy()
                    if len(value.shape) == 0:
                        tf.summary.scalar("{}_{}".format(summary_name,key), value, step=self.step)
                    elif len(value.shape) in [3,4]:
                        tf.summary.image("{}_{}".format(summary_name, key), denormalize(value), step=self.step)


        """ return log str """
        return "g_loss : {}".format(result_logs_dict["gen_loss"])


    # Typically, the test datasets is not large
    def test_step(self, iterator, summary_name = "test"):
        '''
        self.images_val = []
        for input_image_test in iterator:
            gen_output_test = self.generatorG(input_image_test, training=False)
            self.images_val.append(tf.concat([input_image_test,gen_output_test],axis=2))

        """ log summary """
        if summary_name and self.step.numpy() %100 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.image("{}_image".format(summary_name), denormalize(tf.concat(self.images_val, axis=0).numpy()), step=self.step)

        """ return log str """
        '''
        return ""




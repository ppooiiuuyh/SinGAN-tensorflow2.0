import sys
sys.path.append('../')
from models.ops import *
import numpy as np





initializer = tf.initializers.VarianceScaling()
def Generator(channels = 3, N = 0, num_scale = 8):
    def conv_block(x, filters, size, strides, initializer=initializer, activation = tf.keras.layers.LeakyReLU(alpha=0.2)):
        x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='SAME',kernel_initializer=initializer, use_bias=True)(x)
        x = InstanceNorm()(x)
        if activation is not None:
            x = activation(x)
        return x

    inputs_noise = tf.keras.layers.Input(shape=[None, None, channels])
    inputs = tf.keras.layers.Input(shape=[None, None, channels])

    fsize = 32 * 2**((num_scale - N)//4)
    x = tf.keras.layers.Add()([inputs, inputs_noise])
    x = conv_block(x, fsize,3,1)
    x = conv_block(x, fsize, 3, 1)
    x = conv_block(x, fsize, 3, 1)
    x = conv_block(x, fsize, 3, 1)
    x = conv_block(x, 3, 3, 1, activation=None)
    x = tf.keras.layers.Add()([x, inputs])
    output = x

    return tf.keras.Model(inputs=[inputs_noise,inputs], outputs=[output])










if __name__ == "__main__":
    dummy_input = np.random.random([4,512,512,1]).astype(np.float32)
    dummy_noise = np.random.random([4,8]).astype(np.float32)
    """
    generator = Augcycle_Latentencoder(1)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        output = generator([dummy_input,dummy_input])
        loss = L1loss(1, output)

    generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,  generator.trainable_variables))
    print(generator_gradients)

    """
    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        output = generator([dummy_input,dummy_noise])
        #print(output)
        loss = L1loss(1, output)

    #generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    #generator_optimizer.apply_gradients(zip(generator_gradients,  generator.trainable_variables))

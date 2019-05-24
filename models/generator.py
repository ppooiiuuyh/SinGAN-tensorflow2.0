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








def GeneratorUnet(channels):
    #initializer = tf.random_normal_initializer(0., 0.02)
    initializer = tf.initializers.VarianceScaling()

    def conv_block (x, filters, size, strides, initializer=initializer):
        #x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='SAME',kernel_initializer=initializer, use_bias=True)(x)
        x =SpecConv2DLayer(filters, size, strides=strides, padding='SAME',kernel_initializer=initializer, use_bias=True)(x)
        #x = InstanceNorm()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x


    inputs = tf.keras.layers.Input(shape=[None, None, channels])

    #encoding
    x0 = conv_block(inputs,32,5,2)
    x0 = conv_block(x0,64,3,1)
    x1 = conv_block(x0, 128, 3, 2)
    x1 = conv_block(x1, 128, 3, 1)
    x = conv_block(x1, 256, 3, 2)
    x = conv_block(x, 256, 3, 1)


    #decoding
    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,256,3,1)
    x = tf.keras.layers.concatenate([x, x1], axis = -1)
    x = conv_block(x,256,3,1)
    x = conv_block(x,128,3,1)

    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,128,3,1)
    x = tf.keras.layers.concatenate([x, x0], axis = -1)
    x = conv_block(x,128,3,1)
    x = conv_block(x,64,3,1)

    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,64,3,1)
    x = conv_block(x,32,3,1)

    #output = tf.keras.layers.Conv2D(channels, 3, strides=1, padding='same',kernel_initializer=initializer, use_bias=True)(x)
    output = SpecConv2DLayer(channels, 3, strides=1, padding='SAME',kernel_initializer=initializer, use_bias=True)(x)
    #output = InstanceNorm()(output)
    output = tf.keras.layers.Activation('sigmoid')(output)

    return tf.keras.Model(inputs=inputs, outputs=output)






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
    generator = Augcycle_Generator(1)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        output = generator([dummy_input,dummy_noise])
        #print(output)
        loss = L1loss(1, output)

    #generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    #generator_optimizer.apply_gradients(zip(generator_gradients,  generator.trainable_variables))

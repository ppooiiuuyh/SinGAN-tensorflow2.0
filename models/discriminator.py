import sys
sys.path.append('../')
from models.ops import *
import numpy as np



initializer = tf.initializers.VarianceScaling()
def Discriminator(channels = 3, N = 0, num_scale = 8):
    def conv_block (x, filters, size, strides, initializer=initializer):
        x = SpecConv2DLayer(filters, size, strides=strides, padding='SAME', kernel_initializer=initializer, use_bias=True)(x)
        #x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='VALID',kernel_initializer=initializer, use_bias=True)(x)
        #x = InstanceNorm()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x


    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    inputs_padded = tf.keras.layers.ZeroPadding2D(padding=(5,5))(inputs)
    #encoding
    fsize = 32 * 2**((num_scale - N)//4)
    x = conv_block(inputs_padded, fsize,3,1)
    x = conv_block(x, fsize, 3, 1)
    x = conv_block(x, fsize, 3, 1)
    x = conv_block(x, fsize, 3, 1)
    x = conv_block(x, fsize, 3, 1)
    output = x
    #output = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1,2,3]))(x)

    return tf.keras.Model(inputs=[inputs], outputs=[output])


"""
def Discriminator(channels = 3):
    def conv_block(x, filters, size, strides, initializer=initializer):
        x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='SAME',kernel_initializer=initializer, use_bias=True)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = InstanceNorm()(x)
        return x

    inputs = tf.keras.layers.Input(shape=[None, None, channels])

    #encoding
    x0 = conv_block(inputs,32,4,2)
    x1 = conv_block(x0, 64, 4, 2)
    x2 = conv_block(x1, 128, 4, 2)
    x = conv_block(x2, 256, 4, 2)
    x = tf.keras.layers.Conv2D(1, 4, strides=1, padding='SAME', kernel_initializer=initializer, use_bias=True)(x)

    output = x
    print(x)
    return tf.keras.Model(inputs=[inputs], outputs=[output])
"""





def Discriminator_Patch(channels, rf = 189):
    initializer = tf.random_normal_initializer(0., 0.02)
    #initializer = tf.initializers.VarianceScaling()

    def conv_block (x, filters, size, strides, initializer=initializer):
        #x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',kernel_initializer=initializer, use_bias=True)(x)
        x = SpecConv2DLayer( filters, size, strides)(x)
        #x = InstanceNorm()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x

    def res_block (x, filters, size, strides, initializer=initializer):
        input = x
        x = conv_block(x,filters, size, strides=strides, initializer=initializer)
        x = conv_block(x,filters, size, strides=strides, initializer=initializer)
        x = tf.keras.layers.Add()([x, input])
        return x

    inputs = tf.keras.layers.Input(shape=[None, None, channels], name='input_image')
    x = conv_block(inputs, 32, 5, 2)

    x = conv_block(x, 64, 5, 2)
    x = res_block(x, 64,3,1)

    x = conv_block(x, 128, 3, 2)
    x = res_block(x, 128,3,1)

    if rf > 45:
        x = conv_block(x, 256, 3, 2)
        x = res_block(x, 256, 3, 1)

    if rf > 93:
        x = conv_block(x, 256, 3, 2)
        x = res_block(x, 256, 3, 1)

    output = conv_block(x, 1, 1, 1)

    return tf.keras.Model(inputs=inputs, outputs=output)



if __name__ == "__main__":
    dummy_input = np.random.random([4,512,512,1]).astype(np.float32)
    dummy_noise = np.random.random([4,8]).astype(np.float32)

    generator = Augcycle_Discriminator_latent(8)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        output = generator(dummy_noise)
        print(output)
        loss = L1loss(1, output)

    generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,  generator.trainable_variables))
    print(generator_gradients)
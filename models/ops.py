import tensorflow as tf
import numpy as np

def L1loss(input,target):
    #return tf.reduce_sum(tf.reduce_mean(tf.abs(input - target),axis=0))
    return tf.reduce_mean(tf.abs(input - target))

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(
            tf.ones_like(disc_generated_output),
            disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + 100 * l1_loss

    return total_gen_loss


def discriminator_adv_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    gen_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + gen_loss

    return total_disc_loss

def generator_adv_loss(disc_generated_output):
    gan_loss = loss_object(
            tf.ones_like(disc_generated_output),
            disc_generated_output)

    return gan_loss



def getHingeDLoss(disc_real_output, disc_generated_output):
    return tf.reduce_mean(tf.nn.relu(1.0 - disc_real_output)) + tf.reduce_mean(tf.nn.relu(1.0 + disc_generated_output))


def getHingeGLoss(disc_generated_output):
    return -tf.reduce_mean(disc_generated_output)




def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

"""
def tfresize(x,scale):
    return tf.image.resize(
        x, [tf.cast(scale * tf.cast(tf.shape(x)[1], tf.float32), tf.int32),
            tf.cast(scale * tf.cast(tf.shape(x)[2], tf.float32), tf.int32)],
        antialias = False,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
"""


class Tfresize(tf.keras.layers.Layer):
    def __init__(self, h_factor,w_factor,**kwargs):
        super(Tfresize, self).__init__(**kwargs)
        self.h_factor = h_factor
        self.w_factor  =w_factor

    def build(self, input_shape):
        self.shape = input_shape

    def call(self, input):
        return  tf.keras.backend.resize_images(input,self.h_factor,self.w_factor,data_format = "channels_last")


class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        depth = input_shape[-1]
        self.scale = self.add_weight("scale", shape=[depth], dtype = tf.float32, initializer = initializer)
        self.offset = self.add_weight("offset", shape=[depth], dtype = tf.float32, initializer = initializer )

    def call(self, input):
        mean, variance = tf.nn.moments(input, axes=[1, 2], keepdims=True)
        epsilon = 1e-8
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return self.scale*normalized + self.offset



class AdaptiveInstanceNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdaptiveInstanceNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, input):
        x, mu, sigma = input
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        epsilon = 1e-8
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        output = sigma*normalized + mu
        return output


class SpecConv2DLayer(tf.keras.layers.Layer):
    def __init__(self, filters, ksize, strides, padding="SAME",
            kernel_initializer=None, use_bias=True):
        super(SpecConv2DLayer, self).__init__()
        self.ksize = ksize
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.kernel_initializer =  kernel_initializer
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                shape=[self.ksize, self.ksize, int(input_shape[-1]), self.filters],
                initializer=self.kernel_initializer,
                trainable=True)
        if self.use_bias:
            self.bias = self.add_variable("bias",
                    shape=[self.filters],
                    initializer=tf.constant_initializer(0.0),
                    trainable=True)
        w_shape = self.kernel.shape.as_list()
        w_reshaped = tf.reshape(self.kernel, [-1, w_shape[-1]])
        self.u = self.add_variable("u",
                shape=[1, w_shape[-1]],
                initializer=tf.random_normal_initializer(0, 1),
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

    def call(self, input, training=False):
        w_shape = self.kernel.shape.as_list()
        w_reshaped = tf.reshape(self.kernel, [-1, w_shape[-1]])
        u_hat = self.u
        v_hat = tf.math.l2_normalize(tf.matmul(u_hat, tf.transpose(w_reshaped)))
        u_hat = tf.math.l2_normalize(tf.matmul(v_hat, w_reshaped))
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))

        w_norm = w_reshaped / sigma
        w_norm = tf.reshape(w_norm, self.kernel.shape.as_list())

        self.u.assign(u_hat)

        y = tf.nn.conv2d(input, w_norm,
                strides=[1, self.strides, self.strides, 1],
                padding=self.padding)

        if self.use_bias:
            y = y + self.bias

        return y



class Reduce_____(tf.keras.layers.Layer):
    def __init__(self,func, axis, **kwargs):
        super(Reduce_____, self).__init__(**kwargs)
        self.axis = axis
        self.func = func

    def build(self, input_shape):
        pass

    def call(self, input):
        output = self.func(input,axis=self.axis)
        return output




if __name__ == "__main__":
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = np.random.random([1,512,512,3])
    #print(InstanceNorm()(inputs))
    #print(Tfresize(2.0,2.0)(inputs))
    #print(AdaptiveInstanceNorm()([inputs, inputs, inputs]))
    #print(Reduce_____(tf.reduce_mean,[1,2]))
    x = SpecConv2DLayer(64,3,1)(inputs)
    #model = tf.keras.Model(inputs=[inputs], outputs=x)
    print(x)
import tensorflow as tf
import numpy as np

from .normalization import GlobalLayerNorm as gLN
from .normalization import CausalLayerNorm as cLN


class Encoder(tf.keras.layers.Layer):

    def __init__(self, L, N):
        super(Encoder, self).__init__(name='Encoder')

        self.L, self.N = L, N

        self.conv1d = tf.keras.layers.Conv1D(N, kernel_size=L, strides=L//2, padding='same', use_bias=False)
        self.relu = tf.keras.layers.ReLU()
        
    def call(self, input):
        
        """
        Args:
            mixture: [M, W], M is batch size, W is original wave length
        Returns:
            mixture_w: [M, T, N], where W = T * (L/2)
        """
        mixture = tf.expand_dims(input, axis=2) # [M, W, 1]
        mixture = self.conv1d(mixture) # [M, T, N]
        mixture = self.relu(mixture)  # [M, T, N]
        
        return mixture


class Decoder(tf.keras.layers.Layer):

    def __init__(self, T, N, L, C):
        super(Decoder, self).__init__(name='Decoder')

        self.N, self.L, self.C = N, L, C
        
        self.apply_mask = tf.keras.layers.Multiply()

        self.d0 = tf.keras.layers.Lambda(lambda x : x[:,:,0,:]) #[M, T, N] C=0
        self.d1 = tf.keras.layers.Lambda(lambda x : x[:,:,1,:]) #[M, T, N] C=1
        self.d2 = tf.keras.layers.Lambda(lambda x : x[:,:,2,:]) #[M, T, N] C=2 
        self.d3 = tf.keras.layers.Lambda(lambda x : x[:,:,3,:]) #[M, T, N] C=3

        self.conv1dt = tf.keras.layers.Conv1DTranspose(1, kernel_size=16, strides=8, padding='same', 
                                                       use_bias=False)
        self.reorder = tf.keras.layers.Permute([2,1])

    def call(self, mixture, masks):
        """
        Args:
            mixture: [M, T, N]
            mask: [M, T, C, N]
        Returns:
            source: [M, C, T]
        """
        output = tf.expand_dims(mixture, axis=2) #[M, T, 1, N]
        output = self.apply_mask([output, masks]) #[M, T, 1, N] * [M, T, C, N] -> [M, T, C, N]
        
        results = []
        
        #[M, T, C, N] -> [M, T, N] -> [M, W, 1]
        d0 = self.conv1dt(self.d0(output)) 
        d1 = self.conv1dt(self.d1(output))
        d2 = self.conv1dt(self.d2(output)) 
        d3 = self.conv1dt(self.d3(output)) 
        
        output = tf.concat([d0,d1,d2,d3], axis=2) # C*[M, W, 1] -> [M, W, C]
        output = self.reorder(output) #[M, W, C] -> [M, C, W]
        return output #[M, C, W]

class Separator(tf.keras.layers.Layer):

    def __init__(self, N, B, Sc, H, P, X, R, C, T, causal=False):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck and residual paths` 1x1-conv blocks
            Sc: Number of channels in skip-connection paths` 1x1-conv blocks
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            causal: causal or non-causal
        """
        super(Separator, self).__init__(name='Separator')

        self.C = C
        # [M, T, N] -> [M, T, N]
        self.layer_norm = tf.keras.layers.LayerNormalization()
        # [M, T, N] -> [M, T, B]
        self.bottleneck_conv1x1 = tf.keras.layers.Conv1D(B, kernel_size=1, use_bias=False)
        # [M, T, B] -> [M, T, B]
        self.TCN = []
        for r in range(R):
            for x in range(X):
                if r==R-1 and x == X-1:
                    self.TCN.append(Conv1DBlock(B=B, Sc=Sc, H=H, P=P, r=r, x=x, causal=False, is_last=True))
                else:
                    self.TCN.append(Conv1DBlock(B=B, Sc=Sc, H=H, P=P, r=r, x=x, causal=False))
        self.TCN[-1].is_last = True

        self.add = tf.keras.layers.Add()
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        # [M, T, B] -> [M, T, C*N]
        self.conv1d = tf.keras.layers.Conv1D(filters=C*N, kernel_size=1)
        #[M, T, C*N] -> [M, T, C, N]
        self.reshape = tf.keras.layers.Reshape(target_shape=(T, C, N))
        
    def call(self, input):
        """
        Args:
            mixture_w: [M, K, N], M is batch size
        returns:
            est_mask: [M, K, C, N]
        """
        M, K, N = input.shape

        output = self.layer_norm(input) #[M, T, N]
        output = self.bottleneck_conv1x1(output) #[M, T, N]

        skip_connections = []
        for i in range(len(self.TCN)):
            residual, skip = self.TCN[i](output)
            output = residual
            skip_connections.append(skip)
        output = self.add(skip_connections) #[M, T, Sc]
        output = self.prelu(output) #[M, T ,Sc]
        output = self.conv1d(output) #[M, T, C*N]
        output = self.reshape(output) #[M, T, C, N]
        return output #[M, T, C, N]

class Conv1DBlock(tf.keras.layers.Layer):

    def __init__(self, B, Sc, H, P, r, x, causal=False, is_last=False):
        super(Conv1DBlock, self).__init__(name=f'conv1d_block_r{r}_x{x}')

        self.is_last = is_last

        self.causal = causal
        # [M, T, B] -> [M, T, H]
        self.conv1d = tf.keras.layers.Conv1D(filters=H, kernel_size=1, use_bias=False)
        self.dconv1d = tf.keras.layers.Conv1D(filters=H, kernel_size=P, 
                                              dilation_rate=2**x, padding='same', 
                                              groups=H)
        
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        
        self.skip_out = tf.keras.layers.Conv1D(filters=Sc, kernel_size=1) #[M, T, H] -> [M, T ,Sc]
        if not is_last:
            self.res_out = tf.keras.layers.Conv1D(filters=B, kernel_size=1) #[M, T, H] -> [M, T, B]
            self.add = tf.keras.layers.Add()

        if causal:
            self.norm1 = cLN()
            self.norm2 = cLN()
        else:
            self.norm1 = gLN()
            self.norm2 = gLN()
        
    def call(self, input):
        """
        Args:
            x: [M, K, B]
        Returns:
            [M, K, B], [M, K, Sc]
        """
        # [M, T, B] -> [M, T, H]
        output = self.norm1(self.prelu1(self.conv1d(input)))
        output = self.norm2(self.prelu2(self.dconv1d(output)))
        #[M, T, H] -> [M, T, Sc]
        skip = self.skip_out(output)
        #[M, T, H] -> [M, T, B]
        residual = input
        if not self.is_last:
            residual = self.res_out(output)
            residual = self.add([input, residual])

        return residual, skip #[M, K, B], [M, K, Sc]

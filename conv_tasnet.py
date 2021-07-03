import tensorflow as tf

from utility import loss, models


class ConvTasNet(tf.keras.Model):

    def __init__(self, N, L, B, Sc, H, P, X, R, C, T, causal=False):
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
            casual: causal or non-causal
        """
        super(ConvTasNet, self).__init__()
        #Hyper-parameters
        self.N, self.L, self.B, self.Sc ,self.H, self.P, self.X, self.R, self.C, self.T = N, L, B, Sc, H, P, X, R, C, T
        self.causal = causal

        #encoder
        self.encoder = models.Encoder(L=L, N=N)
        #separator
        self.separator = models.Separator(N=N, B=B, Sc=Sc, H=H, P=P, X=X, R=R, C=C, T=T, causal=causal)
        #decoder
        self.decoder = models.Decoder(T=T, N=N, L=L, C=C)

    def call(self, input):
        
        mixture = self.encoder(input)
        masks = self.separator(mixture)
        output = self.decoder(mixture, masks)

        return output


def test_conv_tasnet():
    x = torch.rand(2, 32000)
    model = ConvTasNet()
    x = model.predict(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
import tensorflow as tf
import numpy as np

from os import path, listdir, getcwd

from utility import loss, metrics
import conv_tasnet
import dataset

from absl import app
from absl import flags

STEMS = "vocals", "drums", "bass", "other"

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", f"{getcwd()}/checkpoint", "Directory to save weights")
flags.DEFINE_string("dataset_path", f"{getcwd()}/musdb18", "Dataset Path")

flags.DEFINE_integer("epochs", 500, "Number of epochs to repeat")
flags.DEFINE_integer("E", 400, "Number of Batch in one epoch")
flags.DEFINE_integer("V", 5, "Number of Batch in valid data")
flags.DEFINE_integer("M", 16, "Batch Size")
flags.DEFINE_integer("C", 4, "Number of STEMS")

flags.DEFINE_integer("N", 512, "Number of filters in autoencoder")
flags.DEFINE_integer("L", 16, "Length of the filters in samples")
flags.DEFINE_integer("B", 128, "Number of channels in bottleneck and the residual paths' 1x1-conv blocks")
flags.DEFINE_integer("H", 512, "Number of channels in convolutional blocks")
flags.DEFINE_integer("Sc", 128, "Number of channels in skip-connection paths' 1x1-conv blocks")
flags.DEFINE_integer("P", 3, "Kernel size in convolultional blocks")
flags.DEFINE_integer("X", 8, "Number of convolutional blocks in each repeat")
flags.DEFINE_integer("R", 3, "Number of repeats")

flags.DEFINE_integer("T", 128, "length of one segments")

flags.DEFINE_integer("memory_limit", 10, "Memory Limit")
flags.DEFINE_bool("causal", False, "Flag for the system's causality")

def main(argv):
    W = int((FLAGS.T) * (FLAGS.L / 2))

    data = dataset.Dataset(E=FLAGS.E, M=FLAGS.M, V=FLAGS.V, C=FLAGS.C, W=W, dataset_path=FLAGS.dataset_path,
                           STEMS=STEMS, subsets="test", memory_limit=FLAGS.memory_limit)

    model = conv_tasnet.ConvTasNet(N=FLAGS.N, L=FLAGS.L, B=FLAGS.B, Sc=FLAGS.Sc, H=FLAGS.H,
                                 P=FLAGS.P, X=FLAGS.X, R=FLAGS.R, C=FLAGS.C, T=FLAGS.T, causal=FLAGS.causal)            
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=5, learning_rate=1e-3), loss=loss.SDR(), metrics=[metrics.SNR])
    model.build(input_shape=(FLAGS.M, W))
    model.summary()

    epoch = 0
    if path.exists(FLAGS.checkpoint):
        checkpoints = [name for name in listdir(FLAGS.checkpoint) if "ckpt" in name]
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split(".")[0]
        print("checkpoints: ", checkpoint_name)
        epoch = int(checkpoint_name) + 1
        model.load_weights(f"{FLAGS.checkpoint}/{checkpoint_name}.ckpt")

    print("Evaluate Model")

    x, y = data.get_dataset()
    results = model.evaluate(x, y, batch_size=FLAGS.M)
    print(results)

if __name__ == '__main__':
    #with tf.device('/cpu:0'):
    app.run(main)

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import youtube_dl

from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir, getcwd

from utility import loss, metrics
import conv_tasnet
import dataset


FLAGS = flags.FLAGS
STEMS = "vocals", "drums", "bass", "other"
video_id = "NpngifA8iRM"
flags.DEFINE_bool("interpolate", False,
                  "Interpolate overlapping part of each rows")

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", f"{getcwd()}/checkpoint", "Directory to save weights")
flags.DEFINE_string("dataset_path", f"{getcwd()}/musdb18", "Dataset Path")

flags.DEFINE_integer("epochs", 100, "Number of epochs to repeat")
flags.DEFINE_integer("E", 400, "Number of Batch in one epoch")
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

flags.DEFINE_bool("causal", False, "Flag for the system's casuality")

def youtube_dl_hook(d):
    if d["status"] == "finished":
        print("Done downloading...")

def main(argv):
    checkpoint_dir = FLAGS.checkpoint
    if not path.exists(checkpoint_dir):
        raise ValueError(f"'{checkpoint_dir}' does not exist")

    checkpoints = [name for name in listdir(checkpoint_dir) if "ckpt" in name]
    if not checkpoints:
        raise ValueError(f"No checkpoint exists")
    checkpoints.sort()
    checkpoint_name = checkpoints[-1].split(".")[0]

    W = int(FLAGS.T * FLAGS.L / 2)

    model = conv_tasnet.ConvTasNet(N=FLAGS.N, L=FLAGS.L, B=FLAGS.B, Sc=FLAGS.Sc, H=FLAGS.H,
                                 P=FLAGS.P, X=FLAGS.X, R=FLAGS.R, C=FLAGS.C, T=FLAGS.T, causal=FLAGS.causal)            
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=5), loss=loss.SDR())
    model.build(input_shape=(FLAGS.M, W))
    model.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "44100",
        }],
        "outtmpl": "%(title)s.wav",
        "progress_hooks": [youtube_dl_hook],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_id, download=False)
        status = ydl.download([video_id])

    title = info.get("title", None)
    filename = title + ".wav"
    audio, sr = librosa.load(filename, sr=44100, mono=True)

    audio_length = audio.shape[0]

    num_fragments = audio_length // W

    audio_length_output = num_fragments * W

    print("predicting...")

    audio = audio[:audio_length]

    model_input = np.zeros((num_fragments, W))

    for i in range(num_fragments):
        begin = i * W
        end = begin + W
        model_input[i] = audio[begin:end]

    separated = model.predict(model_input) #[M, C, W]
    separated = np.transpose(separated, (1, 0, 2)) #[C, M, W]

    separated = np.reshape(separated, (FLAGS.C, audio_length_output)) #[C, M*W]

    print("saving...")

    for idx, stem in enumerate(STEMS):
        sf.write(f"{title}_{stem}.wav", separated[idx], sr)


if __name__ == '__main__':
    app.run(main)

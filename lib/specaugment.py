import tensorflow as tf
from tensorflow import keras


#ref https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_tensorflow.py
def frequency_masking(mel_spectrogram, val, frequency_masking_para=8, frequency_mask_num=1):    
    fbank_size = mel_spectrogram.shape
    n, v = fbank_size[1], fbank_size[2]

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        val = tf.cast(val, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=val-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0        
        mask = tf.concat((tf.ones(shape=(1, n- f0 - f, v , 1)),
                          tf.zeros(shape=(1, f, v, 1)),
                          tf.ones(shape=(1, f0, v, 1)),
                          ), 1)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def time_masking(mel_spectrogram, tau, time_masking_para=20, time_mask_num=1):
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)        
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0  
        mask = tf.concat((tf.ones(shape=(1, n, v-t0-t, 1)),
                          tf.zeros(shape=(1, n, t, 1)),
                          tf.ones(shape=(1, n, t0, 1)),
                          ), 2)        
        mel_spectrogram = mel_spectrogram * mask         
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def spec_augment(mel_spectrogram, label="null"):
    #print(mel_spectrogram.shape)
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    frequency_spectrogram = frequency_masking(mel_spectrogram, val=v)
    frequency_time_spectrogram = time_masking(frequency_spectrogram, tau=tau)
    tf.reshape(mel_spectrogram, [40,101,1])
    #visualization_spectrogram(mel_spectrogram, label)
    #visualization_spectrogram(frequency_spectrogram, label)
    #visualization_spectrogram(frequency_time_spectrogram, label)    

    return frequency_time_spectrogram


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing first one result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :, 0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.title(title)
    plt.tight_layout()
    plt.show()
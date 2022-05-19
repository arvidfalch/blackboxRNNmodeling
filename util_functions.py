import numpy as np
import pandas as pd
import scipy
import librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
import IPython.display as Ipd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import os
import sklearn
import math

import scipy 
#import seaborn as sns
import soundfile as sf
import time 


# Evaluation metrics

def energy_normalized_mae(true, predicted):
    y_true = true / np.max(true)
    y_pred = predicted / np.max(predicted)
    return sklearn.metrics.mean_absolute_error(y_true, y_pred)

def ESR(true, predicted):
    return np.sum(np.square(np.abs(true-predicted))) / np.sum(np.square(np.abs(true)))


def normalized_ESR(true, predicted):
    true = true / np.max(true)
    predicted = predicted/np.max(predicted)
    return np.sum(np.square(np.abs(true-predicted))) / np.sum(np.square(np.abs(true)))

def model_description(model):
    print(model.summary())
    
# Function to return metrics on testset
def avg_metrics_on_predictions(dry, wet, model, sr, frame, k_fold, randomized=False):
    '''Function to compute metrics on multiple segments of the test set. k_fold determines how many
    segments are analysed. If random is set to false, it will compute metrics for index number 0 to k_fold,
    enabling the user to compute metrics for the whole test set if k_fold is set to dry_shape[0].
    Else if random is True then the function will randomly pick k_fold number of segments from the test set.'''
    import random
    
    predicted = np.zeros((dry.shape[0],dry.shape[1]))
    original_wet = np.zeros((dry.shape[0],dry.shape[1]))
    r2 = np.array([])
    mae = np.array([])
    timer = 0
    for i in range(k_fold):
        if randomized==False:
            start = time.time()
            to_predict = prepare_audio_seq(dry, i, frame)
            prediction = model.predict(to_predict)
            prediction = prediction.flatten()
            stop = time.time()
            timer += (stop-start)

            predicted[i]= prediction
            original_wet[i] = wet[i]

            r2 = np.append(r2, sklearn.metrics.r2_score(original_wet[i], predicted[i]))
            mae = np.append(mae, sklearn.metrics.mean_absolute_error(original_wet[i], predicted[i]))

        elif randomized==True:
            random_choice = random.randint(0,k_fold)
            start = time.time()
            to_predict = prepare_audio_seq(dry, random_choice, frame)
            prediction = model.predict(to_predict)
            prediction = prediction.flatten()
            stop = time.time()
            timer += (stop-start)

            predicted[i]= prediction
            original_wet[i] = wet[random_choice]

            r2 = np.append(r2, sklearn.metrics.r2_score(original_wet[i], predicted[i]))
            mae = np.append(mae, sklearn.metrics.mean_absolute_error(original_wet[i], predicted[i]))

    print('The model: ')
    model_description(model)

    print('R2 individual scores for segments is {}'.format(r2))
    print('Mae individual scores for segments is {}'.format(mae))

    print('Overall average metrics for original wet audio vs predicted on test set:' )
    #mean absolute error (lower the better)
    MAE_ = sklearn.metrics.mean_absolute_error(original_wet, predicted)
    print('Mae: {}'.format(MAE_))
    R2_ = sklearn.metrics.r2_score(original_wet, predicted)
    EN_MAE_ = energy_normalized_mae(original_wet, predicted)
    ESR_ = ESR(original_wet, predicted)
    print('Energy Normalized Mae: {}'.format(EN_MAE_))
    print('R2: {}'.format(R2_) )
    print('ERS: {}'.format(ESR_))

    print('Inference time for {} seconds of audio was {} seconds'.format((dry.size/sr),(timer)))
    inference_time = timer / (dry.size/sr)

    return MAE_, R2_, EN_MAE_, ESR_, inference_time


# Functions to create the dataset, bot the testset and trainingset

def create_dataset(input_file, output_file, size_training, size_test, frame, sr):
    '''Function to create and split the training and testing data. 
    First both dry and wet file are cut into segments of 5 seconds, which are randomly
    shuffled and split into a training set and a test set. Then features are pulled out of these segments 
    at random according to the size given of the training and dataset. The function returns features_train, 
    features_test, targets_train, targets_test.'''

    import random
    
    signal = input_file
    wet = output_file
    
    # Creating the foundation of the dataset by splitting the whole audio into 5 seconds segments
    segment_size = sr * 5
    dry_segments = np.zeros((int(signal.size/segment_size), segment_size))
    wet_segments = np.zeros((int(signal.size/segment_size), segment_size))
    counter = 0

    for i in range(0, signal.size-segment_size-1, segment_size):
                
        dry_segment = signal[i:i+segment_size]
        
        wet_segment = wet[i:i+segment_size]

        dry_segments[counter,:], wet_segments[counter,:] = dry_segment, wet_segment
        counter+= 1
    
    from sklearn.model_selection import train_test_split

    # Splitting the segments into the training and testing set
    dry_train, dry_test, wet_train, wet_test = train_test_split(dry_segments, wet_segments, test_size=0.2, random_state=5)


    # Creating the training set (randomly pulling frames and target value from all segments in train set)

    features_train = np.zeros((size_training, frame))
    targets_train = np.zeros((size_training))
    counter = 0

    for i in range(size_training):
        random_index = random.randint(0,dry_train.shape[0]-1)
        dry_slice = dry_train[random_index]
        wet_slice = wet_train[random_index]
        random_start = random.randint(0,segment_size-frame-2)
        features = dry_slice[random_start:random_start+frame]
        target = wet_slice[random_start+frame-1]
        features_train[counter,:], targets_train[counter] = features, target
        counter += 1

    # Creating the testing set (randomly puling frames and target value from segments in test set)

    features_test = np.zeros((size_test, frame))
    targets_test = np.zeros((size_test))
    counter = 0
    for i in range(size_test):
        random_index = random.randint(0,dry_test.shape[0]-1)
        dry_slice = dry_test[random_index]
        wet_slice = wet_test[random_index]
        random_start = random.randint(0,segment_size-frame-2)
        features = dry_slice[random_start:random_start+frame]
        target = wet_slice[random_start+frame-1]
        features_test[counter,:], targets_test[counter] = features, target
        counter += 1

    

    return dry_test, wet_test, features_train, features_test, targets_train, targets_test

def create_sequential_dataset(input_file, output_file, size_training, size_test, frame, sr):
    '''Function to create and split the training and testing data. 
    First both dry and wet file are cut into segments of 5 seconds, which are randomly
    shuffled and split into a training set and a test set. Then features are pulled out of these segments 
    sequentially to the size given of the training (size/segments) and test dataset. The function returns 
    dry_test, wet_test (which are used for inference later) and features_train, 
    features_test, targets_train, targets_test.'''

    import random
    
    signal = input_file
    wet = output_file
    
    # Creating the foundation of the dataset by splitting the whole audio into 5 seconds segments
    segment_size = sr * 5
    dry_segments = np.zeros((int(signal.size/segment_size), segment_size))
    wet_segments = np.zeros((int(signal.size/segment_size), segment_size))
    counter = 0

    for i in range(0, signal.size-segment_size-1, segment_size):
                
        dry_segment = signal[i:i+segment_size]
        
        wet_segment = wet[i:i+segment_size]

        dry_segments[counter,:], wet_segments[counter,:] = dry_segment, wet_segment
        counter+= 1
    
    from sklearn.model_selection import train_test_split

    # Splitting the segments into the training and testing set
    dry_train, dry_test, wet_train, wet_test = train_test_split(dry_segments, wet_segments, test_size=0.2, random_state=5)

    
    # Creating the training set (pulling frames in sequence from training and testing segments)
    

    train_frames_pr_segment = int(size_training / dry_train.shape[0])
    test_frames_pr_segment = int(size_test / dry_test.shape[0])

   

    features_train = np.zeros((size_training, frame))
    targets_train = np.zeros((size_training))
    counter = 0

    for i in range(0,dry_train.shape[0]):
        dry_segment = dry_train[i]
        wet_segment = wet_train[i]
        random_start = random.randint(0,dry_segment.size-train_frames_pr_segment-frame-1)
        for x in range(0+frame, train_frames_pr_segment+frame-2):
            start_point = x + random_start      
            features = dry_segment[start_point:start_point+frame]
            target = wet_segment[start_point+frame-1]
            
            features_train[counter,:], targets_train[counter] = features, target
            counter+= 1
    

    # Creating the testing set (sequentiually pulling frames and target value from segments in test set)

    features_test = np.zeros((size_test, frame))
    targets_test = np.zeros((size_test))
    counter = 0
    for i in range(0,dry_test.shape[0]):
        dry_segment = dry_test[i]
        wet_segment = wet_test[i]
        random_start = random.randint(0,dry_segment.size-test_frames_pr_segment-frame-1)
        
        for x in range(0+frame, test_frames_pr_segment+frame-1):
            start_point = x + random_start      
            features = dry_segment[start_point:start_point+frame]
            target = wet_segment[start_point+frame-1]
            
            features_test[counter,:], targets_test[counter] = features, target
            counter+= 1 

 

    return dry_test, wet_test, features_train, features_test, targets_train, targets_test

# Function to check the filter responses of optional filters

def myFIRFiltResponse(b,title, sr, a=1): 
    #sns.set_theme()
    w, h = scipy.signal.freqz(b,a)
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.plot((w/math.pi)*sr, np.log10(np.abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Normalized Frequency')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w/math.pi, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')
    plt.show()

# Unused function to prepare a wav file in order to predict it (Making frames the model can understand)

def prepare_audio(filename, sr, dur, offset, frame):
    audio, dummy = librosa.load(filename, sr=sr, mono=True, duration=dur, offset=offset )
    counter = 0
    
    audio = np.pad(audio, (frame-1,0))
    
    results = np.zeros((audio.size, frame))
    for i in range(0+frame,audio.size-frame-1):
        
            segment = audio[i-frame:i]
            

            results[counter,:] = segment
            counter+= 1
    
    return results

# Function to prepare audio from the testset so the models can predict the audio. 
# Index is an integer from 0 to 18, the different 5 seconds segment from the testset.

def prepare_audio_seq(dry_test, index, frame):
    audio = dry_test[index]
    
    counter = 0
    
    audio = np.pad(audio, (frame-1,0))
    audio = audio[0:audio.size-frame+1]
    
    results = np.zeros((audio.size, frame))
    for i in range(0+frame,audio.size-frame-1):
        
            segment = audio[i-frame:i]
            

            results[counter,:] = segment
            counter+= 1
    
    return results

# Functions for plotting and visual inspection

def mySpectrogram(s,sr,title):
    #sns.set_theme()
    D = librosa.stft(s)
    DdB = librosa.amplitude_to_db(abs(D))
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(DdB, sr=sr, x_axis='time', y_axis='hz')
    plt.title(title)
    plt.show()

def myWaveform(s,title):
    plt.figure(figsize=(14, 3))
    plt.plot(s)
    plt.title(title)

    plt.show()

def compare_waveforms(original, predicted, true_output, title, start,stop):
    plt.figure(figsize=(16,6))
    plt.plot(original[start:stop])
    plt.plot(predicted[start:stop])
    plt.plot(true_output[start:stop])
    plt.legend(['original', 'predicted', 'true output'])
    plt.title(title)
    plt.show()

def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual, alpha=0.3, color='blue')
    plt.plot(range(rows), predictions, alpha=0.3, color='green')
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Time in samples')
    plt.ylabel('')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')



    
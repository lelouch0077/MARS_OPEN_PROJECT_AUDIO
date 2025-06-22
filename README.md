# MARS_OPEN_PROJECT_AUDIO4

## PROJECT DESCRIPTION
We use the dataset provided which contains audio recordings of actors speaking in 8 emotional tones (like happy, sad, angry, etc.).Feature extraction from raw audio included extraction of mel+del+del square spectrograms along with pitch and energy. These features were then used for emotion classification using a CNN BiLSTM + Attention Model.The dataset included 2452 files.80% of the data was used for training 20% for validation. Metrics were reported on the validation data.

## PRE PROCESSING METHODOLOGY
Each audio file was converted into a tensor of size[32,5,64,200]. This means that we have 5 channels of input log mel spectrograms,its first order ad second order derivative(del and del2),pitch(mean pitch per frame),energy(RMS per frame).To ensure uniform time dimesnions for all videos are trimmed to 4 seconds and 200 frames. Normalize features to zero mean and variance 1.

## MODEL  PIPELINE
The model consists of a single convolution layer which scans the spectrogram to detect local patterns and captures spatial features across frequency and time. I have used only 1 convolutional layer with kernel_size=3 and padding=1. After convolution we perform a 2x2 maxpooling.Now this tensor is passed to the LSTM layer. LSTM processes the CNN output and since i have applied bidirectional LSTM it will use both forward and backward hidden states. The bidirectional LSTM helps to capture context from both past and future. This is important as it helps to understand tone and transition in the voice.After this i have added attention weights which map each vector output from lsm to a certain scalar score indicating how important each time step is in determining emotion. After this i use a linear classifier to classify the final output

### TRAINING
An 80/20 split was performed on the original dataset.Cross entropy loss funcion with label smoothing of 0.1 was used. Optimizer used was AdamW with learning rate of 1e-3 and weight decay of 1e-4. I also used a cosine annealing lr scheduler which decays the learning rate according to a cosine function. The model was trained for 50 epochs. Validation metrics include accuracy score,f1 score and confusion matrix.

## VALIDATION METRICS
Accuracy=86.55%

F1 Score=86.68%

PER CLASS ACCURACY

Class Neutral Accuracy: 0.9429

Class Calm Accuracy: 0.9036

Class Happy Accuracy: 0.8611

Class Sad Accuracy: 0.8235

Class Angry Accuracy: 0.8395

Class Fearful Accuracy: 0.8158

Class Disgusted Accuracy: 0.8810

Class Surprised Accuracy: 0.9412


![image](https://github.com/user-attachments/assets/0046937f-c2f8-4e1b-868c-3f0994eabc4a)
![image](https://github.com/user-attachments/assets/61ff523d-d940-45f4-8158-e538c90552df)
![image](https://github.com/user-attachments/assets/7560713d-70e8-47e9-b300-d08fe1c31dfe)





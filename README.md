# Facial_Emotion_Detection
1.clone the repository
2.cd Facial_Emotion_Detection
3.create your own databse and give username and password
4.Download the FER-2013 dataset from "https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view " and unzip it inside the src folder. This will create the folder data.
5.If you want to train this model, use: cd src then python emotions.py --mode train
6.If you want to view the predictions without training again, 
you can download the pre-trained model from "https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view" and then run:cd src then python emotions.py --mode display
                                                                                                                                                                                                                       python emotions.py --mode display
7.The folder structure is of the form:
src:

data (folder)
emotions.py (file)
haarcascade_frontalface_default.xml (file)
model.h5 (file)

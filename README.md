## Cat and Dog classification
Using VGG19 and Data Augmentation

Source code in ````Model/source.ipynb````

[Deploy Model by using streamlit cloud](https://catanddogclassification.streamlit.app/)

![Deploy](https://github.com/KhoaHCMUS/Cat_and_Dog_classification/blob/master/image/deploy.png)

## Guide run local
Install all library to run app.py file:

```
pip install -r requirements.txt
```
Then you can run the below syntax to run the application
```Python
streamlit run app.py
```
*Ortherwise, You can test your picture by ipynb file in ````Model/test.ipynb````*
![test](https://github.com/KhoaHCMUS/Cat_and_Dog_classification/blob/master/image/test.png)


Data:
```
----------Train-------------
                                             imgpath labels  encoded_labels
0  /content/drive/MyDrive/Golden/Dataset/training...   dogs               1
1  /content/drive/MyDrive/Golden/Dataset/training...   dogs               1
2  /content/drive/MyDrive/Golden/Dataset/training...   dogs               1
3  /content/drive/MyDrive/Golden/Dataset/training...   dogs               1
4  /content/drive/MyDrive/Golden/Dataset/training...   dogs               1
(8007, 3)
--------Validation----------
                                             imgpath labels  encoded_labels
0  /content/drive/MyDrive/Golden/Dataset/test_set...   dogs               1
1  /content/drive/MyDrive/Golden/Dataset/test_set...   cats               0
2  /content/drive/MyDrive/Golden/Dataset/test_set...   dogs               1
3  /content/drive/MyDrive/Golden/Dataset/test_set...   cats               0
4  /content/drive/MyDrive/Golden/Dataset/test_set...   cats               0
(1417, 3)
----------Test--------------
                                             imgpath labels  encoded_labels
0  /content/drive/MyDrive/Golden/Dataset/test_set...   cats               0
1  /content/drive/MyDrive/Golden/Dataset/test_set...   cats               0
2  /content/drive/MyDrive/Golden/Dataset/test_set...   dogs               1
3  /content/drive/MyDrive/Golden/Dataset/test_set...   cats               0
4  /content/drive/MyDrive/Golden/Dataset/test_set...   cats               0
(608, 3)
```


![View data](https://github.com/KhoaHCMUS/Cat_and_Dog_classification/blob/master/image/data.png)


## Model
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 inputLayer (InputLayer)     [(None, 224, 224, 3)]     0         
                                                                 
 AugmentationLayer (Sequenti  (None, 224, 224, 3)      0         
 al)                                                             
                                                                 
 vgg19 (Functional)          (None, 512)               20024384  
                                                                 
 dense_18 (Dense)            (None, 256)               131328    
                                                                 
 activation_103 (Activation)  (None, 256)              0         
                                                                 
 batch_normalization_103 (Ba  (None, 256)              1024      
 tchNormalization)                                               
                                                                 
 dropout_9 (Dropout)         (None, 256)               0         
                                                                 
 dense_19 (Dense)            (None, 2)                 514       
                                                                 
 activationLayer (Activation  (None, 2)                0         
 )                                                               
                                                                 
=================================================================
Total params: 20,157,250V
Trainable params: 132,354
Non-trainable params: 20,024,896
_________________________________________________________________
None
```
After that Tuning to get best parameters
## Score after Tuning:
```
F1 Score: 0.9768497330282226
              precision    recall  f1-score   support

        cats       0.97      0.98      0.98       283
        dogs       0.98      0.97      0.98       324

    accuracy                           0.98       607
   macro avg       0.98      0.98      0.98       607
weighted avg       0.98      0.98      0.98       607
```
## Performance
![View data](https://github.com/KhoaHCMUS/Cat_and_Dog_classification/blob/master/image/performance.png)







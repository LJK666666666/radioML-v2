# overall

| model | accuracy |
|-------|----------|
| resnet | 0.5537 |
| cnn1d | 0.5494 |
| cnn2d | 0.4731 |
| complexnn | 0.5365 |
| transformer | 0.4786 |
| resnet + augment | 0.5934 |




# resnet

Overall accuracy: 0.5537
SNR -20.0 dB: Accuracy = 0.0924
SNR -18.0 dB: Accuracy = 0.1027
SNR -16.0 dB: Accuracy = 0.0887
SNR -14.0 dB: Accuracy = 0.1244
SNR -12.0 dB: Accuracy = 0.1560
SNR -10.0 dB: Accuracy = 0.1817
SNR -8.0 dB: Accuracy = 0.3182
SNR -6.0 dB: Accuracy = 0.4110
SNR -4.0 dB: Accuracy = 0.5698
SNR -2.0 dB: Accuracy = 0.7006
SNR 0.0 dB: Accuracy = 0.7702
SNR 2.0 dB: Accuracy = 0.8174
SNR 4.0 dB: Accuracy = 0.8241
SNR 6.0 dB: Accuracy = 0.8379
SNR 8.0 dB: Accuracy = 0.8543
SNR 10.0 dB: Accuracy = 0.8514
SNR 12.0 dB: Accuracy = 0.8452
SNR 14.0 dB: Accuracy = 0.8464
SNR 16.0 dB: Accuracy = 0.8416
SNR 18.0 dB: Accuracy = 0.8179

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.54      0.57      0.55      4000
      AM-DSB       0.52      0.53      0.52      4000
      AM-SSB       0.28      0.24      0.26      4000
        BPSK       0.58      0.62      0.60      4000
       CPFSK       0.60      0.64      0.62      4000
        GFSK       0.68      0.64      0.66      4000
        PAM4       0.71      0.69      0.70      4000
       QAM16       0.55      0.55      0.55      4000
       QAM64       0.66      0.60      0.63      4000
        QPSK       0.54      0.58      0.56      4000
        WBFM       0.42      0.44      0.43      4000

    accuracy                           0.55     44000
   macro avg       0.55      0.55      0.55     44000
weighted avg       0.55      0.55      0.55     44000


# cnn1d

Overall accuracy: 0.5494
SNR -20.0 dB: Accuracy = 0.0951
SNR -18.0 dB: Accuracy = 0.0878
SNR -16.0 dB: Accuracy = 0.1087
SNR -14.0 dB: Accuracy = 0.1131
SNR -12.0 dB: Accuracy = 0.1341
SNR -10.0 dB: Accuracy = 0.2015
SNR -8.0 dB: Accuracy = 0.3191
SNR -6.0 dB: Accuracy = 0.4958
SNR -4.0 dB: Accuracy = 0.5926
SNR -2.0 dB: Accuracy = 0.6907
SNR 0.0 dB: Accuracy = 0.7593
SNR 2.0 dB: Accuracy = 0.7901
SNR 4.0 dB: Accuracy = 0.8245
SNR 6.0 dB: Accuracy = 0.8233
SNR 8.0 dB: Accuracy = 0.8170
SNR 10.0 dB: Accuracy = 0.8364
SNR 12.0 dB: Accuracy = 0.8382
SNR 14.0 dB: Accuracy = 0.8235
SNR 16.0 dB: Accuracy = 0.8035
SNR 18.0 dB: Accuracy = 0.8130

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.66      0.53      0.59      4000
      AM-DSB       0.51      0.64      0.57      4000
      AM-SSB       0.28      0.74      0.41      4000
        BPSK       0.71      0.60      0.65      4000
       CPFSK       0.80      0.60      0.69      4000
        GFSK       0.79      0.63      0.70      4000
        PAM4       0.87      0.66      0.75      4000
       QAM16       0.40      0.18      0.25      4000
       QAM64       0.53      0.59      0.56      4000
        QPSK       0.54      0.56      0.55      4000
        WBFM       0.60      0.30      0.40      4000

    accuracy                           0.55     44000
   macro avg       0.61      0.55      0.56     44000
weighted avg       0.61      0.55      0.56     44000

# cnn2d 

Overall accuracy: 0.4731
SNR -20.0 dB: Accuracy = 0.1001
SNR -18.0 dB: Accuracy = 0.0878
SNR -16.0 dB: Accuracy = 0.0859
SNR -14.0 dB: Accuracy = 0.1000
SNR -12.0 dB: Accuracy = 0.1176
SNR -10.0 dB: Accuracy = 0.1693
SNR -8.0 dB: Accuracy = 0.2962
SNR -6.0 dB: Accuracy = 0.4203
SNR -4.0 dB: Accuracy = 0.4648
SNR -2.0 dB: Accuracy = 0.5476
SNR 0.0 dB: Accuracy = 0.6104
SNR 2.0 dB: Accuracy = 0.6801
SNR 4.0 dB: Accuracy = 0.7155
SNR 6.0 dB: Accuracy = 0.7161
SNR 8.0 dB: Accuracy = 0.7186
SNR 10.0 dB: Accuracy = 0.7232
SNR 12.0 dB: Accuracy = 0.7612
SNR 14.0 dB: Accuracy = 0.7197
SNR 16.0 dB: Accuracy = 0.6894
SNR 18.0 dB: Accuracy = 0.7179

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.28      0.50      0.36      4000
      AM-DSB       0.53      0.60      0.57      4000
      AM-SSB       0.27      0.44      0.34      4000
        BPSK       0.55      0.57      0.56      4000
       CPFSK       0.56      0.57      0.57      4000
        GFSK       0.56      0.65      0.61      4000
        PAM4       0.82      0.60      0.69      4000
       QAM16       0.35      0.23      0.28      4000
       QAM64       0.55      0.51      0.53      4000
        QPSK       0.57      0.25      0.34      4000
        WBFM       0.56      0.27      0.36      4000

    accuracy                           0.47     44000
   macro avg       0.51      0.47      0.47     44000
weighted avg       0.51      0.47      0.47     44000


# complexnn

Overall accuracy: 0.5365
SNR -20.0 dB: Accuracy = 0.0866
SNR -18.0 dB: Accuracy = 0.0882
SNR -16.0 dB: Accuracy = 0.0971
SNR -14.0 dB: Accuracy = 0.1058
SNR -12.0 dB: Accuracy = 0.1270
SNR -10.0 dB: Accuracy = 0.1849
SNR -8.0 dB: Accuracy = 0.3196
SNR -6.0 dB: Accuracy = 0.4780
SNR -4.0 dB: Accuracy = 0.5756
SNR -2.0 dB: Accuracy = 0.6911
SNR 0.0 dB: Accuracy = 0.7470
SNR 2.0 dB: Accuracy = 0.7666
SNR 4.0 dB: Accuracy = 0.8007
SNR 6.0 dB: Accuracy = 0.8146
SNR 8.0 dB: Accuracy = 0.8009
SNR 10.0 dB: Accuracy = 0.8179
SNR 12.0 dB: Accuracy = 0.8099
SNR 14.0 dB: Accuracy = 0.8176
SNR 16.0 dB: Accuracy = 0.7955
SNR 18.0 dB: Accuracy = 0.7868

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.71      0.48      0.57      4000
      AM-DSB       0.53      0.48      0.51      4000
      AM-SSB       0.27      0.80      0.40      4000
        BPSK       0.71      0.61      0.65      4000
       CPFSK       0.87      0.59      0.70      4000
        GFSK       0.75      0.62      0.68      4000
        PAM4       0.88      0.65      0.75      4000
       QAM16       0.45      0.24      0.31      4000
       QAM64       0.54      0.53      0.53      4000
        QPSK       0.54      0.54      0.54      4000
        WBFM       0.45      0.37      0.41      4000

    accuracy                           0.54     44000
   macro avg       0.61      0.54      0.55     44000
weighted avg       0.61      0.54      0.55     44000


# transformer

Overall accuracy: 0.4786
SNR -20.0 dB: Accuracy = 0.0902
SNR -18.0 dB: Accuracy = 0.0840
SNR -16.0 dB: Accuracy = 0.0980
SNR -14.0 dB: Accuracy = 0.1018
SNR -12.0 dB: Accuracy = 0.1068
SNR -10.0 dB: Accuracy = 0.1398
SNR -8.0 dB: Accuracy = 0.2201
SNR -6.0 dB: Accuracy = 0.2969
SNR -4.0 dB: Accuracy = 0.3894
SNR -2.0 dB: Accuracy = 0.4732
SNR 0.0 dB: Accuracy = 0.5995
SNR 2.0 dB: Accuracy = 0.6809
SNR 4.0 dB: Accuracy = 0.7511
SNR 6.0 dB: Accuracy = 0.7743
SNR 8.0 dB: Accuracy = 0.7706
SNR 10.0 dB: Accuracy = 0.8034
SNR 12.0 dB: Accuracy = 0.8076
SNR 14.0 dB: Accuracy = 0.7916
SNR 16.0 dB: Accuracy = 0.7888
SNR 18.0 dB: Accuracy = 0.7828

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.58      0.41      0.48      4000
      AM-DSB       0.49      0.65      0.56      4000
      AM-SSB       0.22      0.94      0.36      4000
        BPSK       0.84      0.52      0.64      4000
       CPFSK       0.57      0.52      0.55      4000
        GFSK       0.70      0.41      0.52      4000
        PAM4       0.90      0.60      0.72      4000
       QAM16       0.53      0.12      0.20      4000
       QAM64       0.51      0.47      0.49      4000
        QPSK       0.89      0.41      0.56      4000
        WBFM       0.61      0.20      0.30      4000

    accuracy                           0.48     44000
   macro avg       0.62      0.48      0.49     44000
weighted avg       0.62      0.48      0.49     44000

# resnet + augment

## 1

Overall accuracy: 0.5934
SNR -20.0 dB: Accuracy = 0.0947
SNR -18.0 dB: Accuracy = 0.0882
SNR -16.0 dB: Accuracy = 0.1008
SNR -14.0 dB: Accuracy = 0.1117
SNR -12.0 dB: Accuracy = 0.1395
SNR -10.0 dB: Accuracy = 0.2042
SNR -8.0 dB: Accuracy = 0.3260
SNR -6.0 dB: Accuracy = 0.4945
SNR -4.0 dB: Accuracy = 0.6357
SNR -2.0 dB: Accuracy = 0.7688
SNR 0.0 dB: Accuracy = 0.8383
SNR 2.0 dB: Accuracy = 0.8736
SNR 4.0 dB: Accuracy = 0.8849
SNR 6.0 dB: Accuracy = 0.8965
SNR 8.0 dB: Accuracy = 0.8924
SNR 10.0 dB: Accuracy = 0.9026
SNR 12.0 dB: Accuracy = 0.9105
SNR 14.0 dB: Accuracy = 0.9061
SNR 16.0 dB: Accuracy = 0.8935
SNR 18.0 dB: Accuracy = 0.8827

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.63      0.56      0.60      4000
      AM-DSB       0.54      0.58      0.56      4000
      AM-SSB       0.29      0.60      0.39      4000
        BPSK       0.75      0.60      0.67      4000
       CPFSK       0.72      0.63      0.67      4000
        GFSK       0.70      0.65      0.68      4000
        PAM4       0.80      0.68      0.74      4000
       QAM16       0.66      0.61      0.63      4000
       QAM64       0.70      0.66      0.68      4000
        QPSK       0.62      0.58      0.60      4000
        WBFM       0.54      0.36      0.43      4000

    accuracy                           0.59     44000
   macro avg       0.63      0.59      0.60     44000
weighted avg       0.63      0.59      0.60     44000

## 2


Overall accuracy: 0.5997
SNR -20.0 dB: Accuracy = 0.0915
SNR -18.0 dB: Accuracy = 0.0798
SNR -16.0 dB: Accuracy = 0.1022
SNR -14.0 dB: Accuracy = 0.1266
SNR -12.0 dB: Accuracy = 0.1408
SNR -10.0 dB: Accuracy = 0.2162
SNR -8.0 dB: Accuracy = 0.3494
SNR -6.0 dB: Accuracy = 0.4971
SNR -4.0 dB: Accuracy = 0.6101
SNR -2.0 dB: Accuracy = 0.7533
SNR 0.0 dB: Accuracy = 0.8374
SNR 2.0 dB: Accuracy = 0.8921
SNR 4.0 dB: Accuracy = 0.9069
SNR 6.0 dB: Accuracy = 0.9121
SNR 8.0 dB: Accuracy = 0.9076
SNR 10.0 dB: Accuracy = 0.9017
SNR 12.0 dB: Accuracy = 0.9235
SNR 14.0 dB: Accuracy = 0.9151
SNR 16.0 dB: Accuracy = 0.9033
SNR 18.0 dB: Accuracy = 0.9027

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.72      0.54      0.62      4000
      AM-DSB       0.56      0.60      0.58      4000
      AM-SSB       0.28      0.78      0.41      4000
        BPSK       0.79      0.61      0.69      4000
       CPFSK       0.79      0.61      0.69      4000
        GFSK       0.72      0.64      0.68      4000
        PAM4       0.84      0.68      0.75      4000
       QAM16       0.66      0.59      0.62      4000
       QAM64       0.72      0.67      0.69      4000
        QPSK       0.78      0.54      0.64      4000
        WBFM       0.58      0.34      0.43      4000

    accuracy                           0.60     44000
   macro avg       0.68      0.60      0.62     44000
weighted avg       0.68      0.60      0.62     44000


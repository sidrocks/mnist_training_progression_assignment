***MNIST Neural Network***

The goal is to achieve 99.4% test accuracy consistently, with less than or equal to 15 epochs and 8k parameters

***Model : <a href="https://github.com/sidrocks/mnist_training_progression_assignment/blob/main/Model_1.py"> Model_1.py </a>***

  ***Target***
  <li> Do the basic setup - transforms, data loader, basic network architecture, basic training and test loop </li>
  <li> Make a lighter model </li>
  <li> Include Batch Normalization to increase model efficicency. Stabilize training, faster convergence, better generalization </li>
  
  ***Results***
  <li> Parameters: 10970 </li>
  <li> Best Training Accuracy: 99.80 </li>
  <li> Best Test Accuracy: 99.20 </li>

  ***Analysis***
  1. Exceptional Performance - The model achieves outstanding accuracy on the MNIST dataset, consistently performing above 99% on the test set as early as from 2nd Epoch
  2. Efficient Learning - The model learns very quickly, achieving high accuracy within just a few epochs.
  3. Lightweight and Efficient: With a total of only 10,970 parameters, the model is highly efficient in terms of memory and computational resources.
  4. Stable Training: The training process appears stable, with smooth improvements in accuracy and consistent reduction in loss, likely aided by the use of Batch Normalization layers throughout the network
  5. Slight Overfitting: Slight overfitting is observed by the 4th epoch and the difference between Training and Test accuracy has started to rise by the 15th Epoch. Further, while the overall test accuracy remains very high, slight oscillations in test accuracy after epoch 8 could be an early indicator of the model beginning to overfit.This suggests the model may not generalize well for unseen data

  Since our target is to reach 99.4% test accuracy with fewer parameters <8k, to achieve generalization other techniques such as regularization, data augmentation, GAP need to be explored

  *** Model Architecture and Training Logs***

        --- Starting training for Model_1 ---
      ----------------------------------------------------------------
              Layer (type)               Output Shape         Param #
      ================================================================
                  Conv2d-1           [-1, 10, 26, 26]              90
             BatchNorm2d-2           [-1, 10, 26, 26]              20
                    ReLU-3           [-1, 10, 26, 26]               0
                  Conv2d-4           [-1, 10, 24, 24]             900
             BatchNorm2d-5           [-1, 10, 24, 24]              20
                    ReLU-6           [-1, 10, 24, 24]               0
                  Conv2d-7           [-1, 20, 22, 22]           1,800
             BatchNorm2d-8           [-1, 20, 22, 22]              40
                    ReLU-9           [-1, 20, 22, 22]               0
              MaxPool2d-10           [-1, 20, 11, 11]               0
                 Conv2d-11           [-1, 10, 11, 11]             200
            BatchNorm2d-12           [-1, 10, 11, 11]              20
                   ReLU-13           [-1, 10, 11, 11]               0
                 Conv2d-14             [-1, 10, 9, 9]             900
            BatchNorm2d-15             [-1, 10, 9, 9]              20
                   ReLU-16             [-1, 10, 9, 9]               0
                 Conv2d-17             [-1, 20, 7, 7]           1,800
            BatchNorm2d-18             [-1, 20, 7, 7]              40
                   ReLU-19             [-1, 20, 7, 7]               0
                 Conv2d-20             [-1, 10, 7, 7]             200
            BatchNorm2d-21             [-1, 10, 7, 7]              20
                   ReLU-22             [-1, 10, 7, 7]               0
                 Conv2d-23             [-1, 10, 1, 1]           4,900
      ================================================================
      Total params: 10,970
      Trainable params: 10,970
      Non-trainable params: 0
      ----------------------------------------------------------------
      Input size (MB): 0.00
      Forward/backward pass size (MB): 0.61
      Params size (MB): 0.04
      Estimated Total Size (MB): 0.65
      ----------------------------------------------------------------
      EPOCH: 0
      Loss=0.0117 Batch_id=937 Accuracy=95.85: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 48.53it/s] 
      
      Test set: Average loss: 0.0536, Accuracy: 9829/10000 (98.29%)
      
      EPOCH: 1
      Loss=0.0739 Batch_id=937 Accuracy=98.55: 100%|██████████████████████████████████████████████| 938/938 [00:20<00:00, 45.57it/s] 
      
      Test set: Average loss: 0.0397, Accuracy: 9864/10000 (98.64%)
      
      EPOCH: 2
      Loss=0.0035 Batch_id=937 Accuracy=98.89: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 49.76it/s] 
      
      Test set: Average loss: 0.0383, Accuracy: 9874/10000 (98.74%)
      
      EPOCH: 3
      Loss=0.0522 Batch_id=937 Accuracy=99.06: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 48.30it/s] 
      
      Test set: Average loss: 0.0321, Accuracy: 9889/10000 (98.89%)
      
      EPOCH: 4
      Loss=0.0017 Batch_id=937 Accuracy=99.22: 100%|██████████████████████████████████████████████| 938/938 [02:39<00:00,  5.87it/s] 
      
      Test set: Average loss: 0.0265, Accuracy: 9907/10000 (99.07%)
      
      EPOCH: 5
      Loss=0.0004 Batch_id=937 Accuracy=99.28: 100%|██████████████████████████████████████████████| 938/938 [03:19<00:00,  4.70it/s] 
      
      Test set: Average loss: 0.0331, Accuracy: 9896/10000 (98.96%)
      
      EPOCH: 6
      Loss=0.0195 Batch_id=937 Accuracy=99.38: 100%|██████████████████████████████████████████████| 938/938 [01:44<00:00,  9.01it/s] 
      
      Test set: Average loss: 0.0327, Accuracy: 9893/10000 (98.93%)
      
      EPOCH: 7
      Loss=0.0052 Batch_id=937 Accuracy=99.41: 100%|██████████████████████████████████████████████| 938/938 [00:21<00:00, 44.14it/s] 
      
      Test set: Average loss: 0.0247, Accuracy: 9902/10000 (99.02%)
      
      EPOCH: 8
      Loss=0.0008 Batch_id=937 Accuracy=99.46: 100%|██████████████████████████████████████████████| 938/938 [00:21<00:00, 44.23it/s] 
      
      Test set: Average loss: 0.0285, Accuracy: 9894/10000 (98.94%)
      
      EPOCH: 9
      Loss=0.0070 Batch_id=937 Accuracy=99.55: 100%|██████████████████████████████████████████████| 938/938 [01:04<00:00, 14.50it/s] 
      
      Test set: Average loss: 0.0279, Accuracy: 9910/10000 (99.10%)
      
      EPOCH: 10
      Loss=0.0002 Batch_id=937 Accuracy=99.64: 100%|██████████████████████████████████████████████| 938/938 [00:22<00:00, 41.02it/s] 
      
      Test set: Average loss: 0.0268, Accuracy: 9912/10000 (99.12%)
      
      EPOCH: 11
      Loss=0.0329 Batch_id=937 Accuracy=99.67: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 49.73it/s] 
      
      Test set: Average loss: 0.0257, Accuracy: 9913/10000 (99.13%)
      
      EPOCH: 12
      Loss=0.0074 Batch_id=937 Accuracy=99.67: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 49.21it/s] 
      
      Test set: Average loss: 0.0310, Accuracy: 9905/10000 (99.05%)
      
      EPOCH: 13
      Loss=0.0046 Batch_id=937 Accuracy=99.76: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 48.21it/s] 
      
      Test set: Average loss: 0.0257, Accuracy: 9920/10000 (99.20%)
      
      EPOCH: 14
      Loss=0.0000 Batch_id=937 Accuracy=99.80: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 48.08it/s] 
      
      Test set: Average loss: 0.0343, Accuracy: 9898/10000 (98.98%)

***Model : <a href="https://github.com/sidrocks/mnist_training_progression_assignment/blob/main/Model_2.py"> Model_2.py </a>***

  ***Target***
  <li> Make the model lighter < 8k </li>
  <li> Use Global Average Pooling, remove the large dense layer </li>
  <li> Include dropouts</li>
  
  ***Results***
  <li> Parameters: 6664 </li>
  <li> Best Training Accuracy: 99.09 </li>
  <li> Best Test Accuracy: 99.36 </li>

  ***Analysis***
  1. Lighter Model and Regularization using GAP is showing better stability, 99%+ accuracy observed from 6th epoch onwards
  2. The consistent positive gap between Training and Test accuracy indicates the model not over-fitting and seems to be generalizing well
  3. Iterations are more efficient and seems to be inching more towards our goal, still short of our 99.4% target
  
  *** Model Architecture and Training Logs***

      ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 26, 26]              72
           BatchNorm2d-2            [-1, 8, 26, 26]              16
                  ReLU-3            [-1, 8, 26, 26]               0
                Conv2d-4           [-1, 10, 24, 24]             720
           BatchNorm2d-5           [-1, 10, 24, 24]              20
                  ReLU-6           [-1, 10, 24, 24]               0
             MaxPool2d-7           [-1, 10, 12, 12]               0
                Conv2d-8            [-1, 8, 12, 12]              80
           BatchNorm2d-9            [-1, 8, 12, 12]              16
                 ReLU-10            [-1, 8, 12, 12]               0
              Dropout-11            [-1, 8, 12, 12]               0
               Conv2d-12           [-1, 16, 10, 10]           1,152
          BatchNorm2d-13           [-1, 16, 10, 10]              32
                 ReLU-14           [-1, 16, 10, 10]               0
               Conv2d-15             [-1, 16, 8, 8]           2,304
          BatchNorm2d-16             [-1, 16, 8, 8]              32
                 ReLU-17             [-1, 16, 8, 8]               0
               Conv2d-18             [-1, 10, 8, 8]             160
          BatchNorm2d-19             [-1, 10, 8, 8]              20
                 ReLU-20             [-1, 10, 8, 8]               0
              Dropout-21             [-1, 10, 8, 8]               0
               Conv2d-22             [-1, 20, 6, 6]           1,800
          BatchNorm2d-23             [-1, 20, 6, 6]              40
                 ReLU-24             [-1, 20, 6, 6]               0
               Conv2d-25             [-1, 10, 6, 6]             200
            AvgPool2d-26             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 6,664
    Trainable params: 6,664
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.40
    Params size (MB): 0.03
    Estimated Total Size (MB): 0.43
    ----------------------------------------------------------------
    EPOCH: 0
    Loss=0.1169 Batch_id=937 Accuracy=86.46: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 51.25it/s] 
    
    Test set: Average loss: 0.0790, Accuracy: 9795/10000 (97.95%)
    
    EPOCH: 1
    Loss=0.0106 Batch_id=937 Accuracy=97.66: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 50.01it/s] 
    
    Test set: Average loss: 0.0517, Accuracy: 9837/10000 (98.37%)
    
    EPOCH: 2
    Loss=0.0187 Batch_id=937 Accuracy=98.15: 100%|██████████████████████████████████████████████| 938/938 [00:37<00:00, 24.75it/s] 
    
    Test set: Average loss: 0.0473, Accuracy: 9852/10000 (98.52%)
    
    EPOCH: 3
    Loss=0.1532 Batch_id=937 Accuracy=98.38: 100%|██████████████████████████████████████████████| 938/938 [00:20<00:00, 45.96it/s] 
    
    Test set: Average loss: 0.0329, Accuracy: 9901/10000 (99.01%)
    
    EPOCH: 4
    Loss=0.0467 Batch_id=937 Accuracy=98.54: 100%|██████████████████████████████████████████████| 938/938 [00:22<00:00, 41.56it/s] 
    
    Test set: Average loss: 0.0300, Accuracy: 9906/10000 (99.06%)
    
    EPOCH: 5
    Loss=0.0120 Batch_id=937 Accuracy=98.59: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 50.82it/s] 
    
    Test set: Average loss: 0.0292, Accuracy: 9907/10000 (99.07%)
    
    EPOCH: 6
    Loss=0.0105 Batch_id=937 Accuracy=98.76: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 50.79it/s] 
    
    Test set: Average loss: 0.0278, Accuracy: 9912/10000 (99.12%)
    
    EPOCH: 7
    Loss=0.0096 Batch_id=937 Accuracy=98.88: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 51.47it/s] 
    
    Test set: Average loss: 0.0249, Accuracy: 9918/10000 (99.18%)
    
    EPOCH: 8
    Loss=0.0058 Batch_id=937 Accuracy=98.80: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 49.53it/s] 
    
    Test set: Average loss: 0.0234, Accuracy: 9919/10000 (99.19%)
    
    EPOCH: 9
    Loss=0.0176 Batch_id=937 Accuracy=98.86: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 49.49it/s] 
    
    Test set: Average loss: 0.0226, Accuracy: 9924/10000 (99.24%)
    
    EPOCH: 10
    Loss=0.0197 Batch_id=937 Accuracy=98.92: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 49.35it/s] 
    
    Test set: Average loss: 0.0271, Accuracy: 9920/10000 (99.20%)
    
    EPOCH: 11
    Loss=0.0448 Batch_id=937 Accuracy=99.04: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 49.13it/s] 
    
    Test set: Average loss: 0.0235, Accuracy: 9928/10000 (99.28%)
    
    EPOCH: 12
    Loss=0.0106 Batch_id=937 Accuracy=99.09: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 49.67it/s] 
    
    Test set: Average loss: 0.0227, Accuracy: 9918/10000 (99.18%)
    
    EPOCH: 13
    Loss=0.1074 Batch_id=937 Accuracy=99.07: 100%|██████████████████████████████████████████████| 938/938 [00:19<00:00, 48.97it/s] 
    
    Test set: Average loss: 0.0210, Accuracy: 9929/10000 (99.29%)
    
    EPOCH: 14
    Loss=0.0008 Batch_id=937 Accuracy=99.08: 100%|██████████████████████████████████████████████| 938/938 [00:18<00:00, 49.56it/s] 
    
    Test set: Average loss: 0.0206, Accuracy: 9936/10000 (99.36%)
    
    --- Finished training for Model_2 ---
    
    
    

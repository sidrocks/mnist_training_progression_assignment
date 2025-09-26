***MNIST Neural Network***

The goal is to achieve 99.4% test accuracy consistently, with less than or equal to 15 epochs and 8k parameters

***Model : <a href="https://github.com/sidrocks/mnist_training_progression_assignment/blob/main/Model_1.py"> Model_1.py </a>***

  ***Target***
  <li> Do the basic setup - transforms, data loader, basic network architecture, basic training and test loop </li>
  <li> Make a lighter model </li>
  <li> Include Batch Normalization to increase model efficicency. Stabilize training, faster convergence, better generalization </li>
  
  ***Results***
  <li> Parameters: 10970 </li>
  <li> Best Training Accuracy: 99.83 </li>
  <li> Best Test Accuracy: 99.26 </li>

  ***Analysis***
  1. Exceptional Performance - The model achieves outstanding accuracy on the MNIST dataset, consistently performing above 99% on the test set as early as from 2nd Epoch
  2. Efficient Learning - The model learns very quickly, achieving high accuracy within just a few epochs.
  3. Lightweight and Efficient: With a total of only 10,970 parameters, the model is highly efficient in terms of memory and computational resources.
  4. Stable Training: The training process appears stable, with smooth improvements in accuracy and consistent reduction in loss, likely aided by the use of Batch Normalization layers throughout the network
  5. Slight Overfitting: Slight overfitting is observed by the 4th epoch and the difference between Training and Test accuracy has started to rise by the 15th Epoch. Further, while the overall test accuracy remains very high, slight oscillations in test accuracy after epoch 8 could be an early indicator of the model beginning to overfit.This suggests the model may not generalize well for unseen data

  Since our target is to reach 99.4% test accuracy with fewer parameters <8k, to achieve generalization other techniques such as regularization, data augmentation, GAP need to be explored

***Model : <a href="https://github.com/sidrocks/mnist_training_progression_assignment/blob/main/Model_2.py"> Model_2.py </a>***

  ***Target***
  <li> Make the model lighter < 8k </li>
  <li> Use Global Average Pooling, remove the large dense layer </li>
  <li> Include dropouts</li>
  
  ***Results***
  <li> Parameters: 6664 </li>
  <li> Best Training Accuracy: 99.02 </li>
  <li> Best Test Accuracy: 99.36 </li>

  ***Analysis***
  1. Lighter Model and Regularization using GAP is showing better stability, 99%+ accuracy observed from 6th epoch onwards
  2. The consistent positive gap between Training and Test accuracy indicates the model not over-fitting and seems to be generalizing well
  3. Iterations are more efficient and seems to be inching more towards our goal, still short of our 99.4% target
  

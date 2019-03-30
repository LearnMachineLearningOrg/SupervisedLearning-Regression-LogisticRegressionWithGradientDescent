# SupervisedLearning-Regression-LogisticRegressionWithGradientDescent

1. A Logistic regression takes input and returns an output of probability, a value between 0 and 1. Logistic regression uses a function called Sigmoid function to do this

2. Sigmoid function: This sigmoid function is responsible for predicting or classifying a given input
 
3. Our objective is to find out the weights (Theta) values, such that the logistic regression function is optimized. Whether the algorithm is performing well or not is defined by a cost function. In this task we will use the loss minimizing approach with the use of gradient descent
 
4. Gradient descent is an optimization algorithm that can be used to find the local minimum of a function. So, in our scenario we will use the Gradient descent algorithm to find the minimum of the cost function. The procedure is to iterate for ‘n’ number of times, in each iteration calculating the new values for weights, and checking whether the cost function reached the minimum

5. Math behind gradient descent: 
We will be applying partial derivatives with respect to weights to the cost function to point us to the lowest point. A derivative of   
zero means you are at either a local minima or maxima. Which means that the closer we get to zero, the better. When we  reach close to,  if not, zero with our derivatives, we also inevitably get the lowest value for our cost function.
 
6. We use a hyper-parameter called learning rate that defines how fast gradient descent finds the optimal weights (‘thetas’). The weights are updated by subtracting the derivative (gradient descent) times the learning rate. A larger value for the learning rate than which is required means that the steps we take are too big and that you might miss the lowest point entirely. A smaller value for the learning rate than which is required means that the steps we take are too small and it might take long time to converge

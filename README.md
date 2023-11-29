# Sentiment-Prediction-on-Movie-reviews
 A regression/classification machine-learning problem.
 
Competition link - https://www.kaggle.com/competitions/sentiment-prediction-on-movie-reviews

In this dataset each record represents a movie-review pair with movie title, description, genres, duration, director, actors, users' ratings, review text, reviewer name, etc. Your task is to build an ML model to predict sentiment of the review text.

**MODELS**

**1. Logistic Regression:**

   - **Data:** 
     - **Features:** The features include information such as review text, audience score, and whether the reviewer is a frequent reviewer.
     - **Target Variable:** The target variable is the sentiment of the movie review (0 for "NEGATIVE" and 1 for "POSITIVE").

   - **Model:** 
     - **Explanation:** Logistic Regression is a linear classification model that estimates the probability of a binary outcome (sentiment) based on a linear combination of input features.
     - **Role:** Its role is to learn a set of weights (coefficients) for each feature and combine them linearly to make predictions about the probability of a review being positive or negative.

   - **Loss Function:** 
     - **Explanation:** The loss function used is binary cross-entropy (log loss). It measures the dissimilarity between predicted probabilities and actual labels.
     - **Role:** It guides the model during training to minimize the difference between predictions and true labels.

   - **Optimization Procedure/Training:** 
     - **Explanation:** Solver = 'saga,' Penalty = 'None,' Class-Weight = 'balanced.'
     - **Role:** The optimization process adjusts model weights iteratively to minimize the loss. Solver 'saga' is a variant of Stochastic Average Gradient Descent (SAG).

   - **Evaluation:** 
     - **Explanation:** The model's performance is evaluated using accuracy and F1-score.
     - **Role:** These metrics assess the correctness of the model's predictions and its ability to balance precision and recall.

**2. XGBoost (Extreme Gradient Boosting) Classifier:**

   - **Data:** Same as Logistic Regression.

   - **Model:** 
     - **Explanation:** XGBoost is an ensemble model that combines multiple decision trees.
     - **Role:** It combines the predictions of individual trees to make a final prediction.

   - **Loss Function:** 
     - **Explanation:** XGBoost uses gradient boosting with various loss functions, including logistic loss for classification.
     - **Role:** It guides the boosting process to minimize the chosen loss function.

   - **Optimization Procedure/Training:** 
     - **Explanation:** Parameters like the number of estimators, learning rate, and max depth are tuned to optimize model performance.
     - **Role:** These parameters affect how the ensemble of trees is constructed and how they contribute to the final prediction.

   - **Evaluation:** 
     - **Explanation:** The model's performance is evaluated using accuracy.
     - **Role:** It measures the overall correctness of the model's predictions.

**3. Stochastic Gradient Descent (SGD) Classifier:**

   - **Data:** Same as Logistic Regression.

   - **Model:** 
     - **Explanation:** SGD Classifier is a linear classification model.
     - **Role:** It learns linear decision boundaries to separate classes.

   - **Loss Function:** 
     - **Explanation:** Various loss functions are available, such as hinge loss for SVM-like behavior and log loss for logistic regression-like behavior.
     - **Role:** The loss function determines the type of classification behavior.

   - **Optimization Procedure/Training:** 
     - **Explanation:** The model uses stochastic gradient descent with various parameters (e.g., learning rate, regularization).
     - **Role:** These parameters control the gradient descent process during training.

   - **Evaluation:** 
     - **Explanation:** The model's performance is evaluated using accuracy.
     - **Role:** It measures the overall correctness of the model's predictions.

**4. K-Nearest Neighbors (KNN) Classifier:**

   - **Data:** Same as Logistic Regression.

   - **Model:** 
     - **Explanation:** KNN is a non-parametric classification model that classifies data points based on their nearest neighbors.
     - **Role:** It assigns labels based on the majority class among the k-nearest data points.

   - **Loss Function:** 
     - **Explanation:** KNN doesn't have a loss function in the traditional sense; it relies on distance metrics.
     - **Role:** The choice of distance metric (e.g., Euclidean) affects how nearest neighbors are determined.

   - **Optimization Procedure/Training:** 
     - **Explanation:** No explicit training; the model stores the training data and classifies new data points based on their neighbors.
     - **Role:** It classifies new data points based on their similarity to training examples.

   - **Evaluation:** 
     - **Explanation:** The model's performance is evaluated using accuracy.
     - **Role:** It measures the overall correctness of the model's predictions.

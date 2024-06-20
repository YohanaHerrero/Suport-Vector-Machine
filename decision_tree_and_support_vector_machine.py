import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from snapml import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from snapml import SupportVectorMachine
from sklearn.metrics import hinge_loss
import time

# download the dataset
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
raw_data=pd.read_csv(url)
print("There are ",len(raw_data)," observations in the credit card fraud dataset.")
print("There are ",len(raw_data.columns), " variables in the dataset.")
#display the first items in the table to know how the table looks
raw_data.head()

#to make this case a more realistic example, we inflate the original data set by 10 times because finantial datasets are usually
#much larger than ours
n_replicas = 10
# inflate the original dataset
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns) #i repeat each row 10 times
print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")
# display first rows in the new dataset
big_raw_data.head()

##############################################  Data Processing  ##########################################################

#data preprocessing such as scaling/normalization is useful for linear models to accelerate the training convergence
#standardize features by removing the mean and scaling to unit variance
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30]) #I exclude the Time variable from the dataset
data_matrix = big_raw_data.values
#X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30] #same as X = big_raw_data.drop(columns=['Time','Class']).values 
#y: labels vector
y = data_matrix[:, 30] #same as doing y = big_raw_data['Class']
# data normalization
X = normalize(X, norm="l1")

#create train and test sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

##############################################  Decision Tree with sklearn  ##########################################################

#set up the decision tree with sklearn. But first, is the dataset a balanced set? Let's check
#get the set of distinct classes
labels = big_raw_data.Class.unique()  
#get the count of each class: how many are and are not fraudulent transactions
sizes = big_raw_data.Class.value_counts().values
# plot the class value counts in a pie plot
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')  
ax.set_title('Target Variable Value Counts')
plt.show()

#the dataset is highly unbalanced (see pie plot above). This case requires special attention when training or when evaluating the quality of a model.
#One way of handing this case at train time is to bias the model to pay more attention to the samples in the minority class. 
#The models under the current study will be configured to take into account the class weights of the samples at train/fit time.

#compute the sample weights to be used as input to the train routine so that it takes into account the class imbalance of the dataset
weights_train = compute_sample_weight('balanced', y_trainset)  

##############################################  Decision Tree with sklearn  ##########################################################

#Decision Tree Classifier Model from scikit-learn 
sklearn_fraudTree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=35)
#train (fit) the model with weights using the training set
t0 = time.time() #i will time how long it takes to fit the training set
sklearn_fraudTree.fit(X_trainset,y_trainset, sample_weight=weights_train)
sklearn_time = time.time()-t0
print("Scikit-Learn Training time (s):  {0:.5f}".format(sklearn_time))

##############################################  Decision Tree with Snap ML  ##########################################################

#Decision Tree Classifier Model from Snap ML (offers multi-threaded CPU/GPU training of decision trees, it is around 10x faster than sklearn)
# to use the GPU, set the use_gpu parameter to True: snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=True)
#to set the number of CPU threads used at training time, set the n_jobs parameter
snapml_fraudTree = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)
#train (fit) the model with weights using the training set
t0 = time.time()
snapml_fraudTree.fit(X_trainset, y_trainset, sample_weight=weights_train)
snapml_time = time.time()-t0
print("Snap ML Training time (s):  {0:.5f}".format(snapml_time))

##############################################  Models evaluation  ##########################################################
#the sklearn model
#run inference and compute the probabilities of the test samples to belong to the class of fraudulent transactions (class 1), according to sklearn
sklearn_pred = sklearn_fraudTree.predict_proba(X_testset)[:,1]
#evaluate the Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score from the predictions
sklearn_roc_auc = roc_auc_score(y_testset, sklearn_pred) #check how many positive and negative predictions the model gets
print('Scikit-Learn ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

#the snap ml model
#run inference and compute the probabilities of the test samples to belong to the class of fraudulent transactions(class 1), according to Snap ML
snapml_pred = snapml_fraudTree.predict_proba(X_testset)[:,1]
#evaluate the Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score from the prediction scores
snapml_roc_auc = roc_auc_score(y_testset, snapml_pred)   
print('Snap ML ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))

#OF COURSE, both decision tree models provide the same score on the test dataset but the Snap ML is around 12x faster than sklearn

##############################################  Support Vector Machine model with sklearn  ##########################################################

#instatiate a scikit-learn SVM model to indicate the class imbalance at fit time, set class_weight='balanced'
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False) #loss: Specifies the loss function. fit_intercept: Whether or not to fit an intercept
# train a linear Support Vector Machine model using Scikit-Learn
t0 = time.time()
sklearn_svm.fit(X_trainset, y_trainset)
sklearn_time = time.time() - t0
print("Scikit-Learn Training time (s):  {0:.2f}".format(sklearn_time))

##############################################  Support Vector Machine model with Snap ML  ##########################################################

snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
#train an SVM model using Snap ML
t0 = time.time()
model = snapml_svm.fit(X_trainset, y_trainset)
snapml_time = time.time() - t0
print("Snap ML Training time (s):  {0:.2f}".format(snapml_time))

##############################################  Models evaluation  ##########################################################

#run inference using the Scikit-Learn model
#get the confidence scores for the test samples
sklearn_pred = sklearn_svm.decision_function(X_testset)
#evaluate accuracy on test set
acc_sklearn  = roc_auc_score(y_testset, sklearn_pred)
print("Scikit-Learn ROC-AUC score:   {0:.3f}".format(acc_sklearn))

# run inference using the Snap ML model
# get the confidence scores for the test samples
snapml_pred = snapml_svm.decision_function(X_testset)
# evaluate accuracy on test set
acc_snapml  = roc_auc_score(y_testset, snapml_pred)
print("[Snap ML] ROC-AUC score:   {0:.3f}".format(acc_snapml))

#the two models provide the same score on the test dataset. However, as in the case of decision trees, Snap ML runs the training routine faster than Scikit-Learn.

##############################################  Another models evaluation  ##########################################################

#evaluate the hinge loss from the predictions
loss_snapml = hinge_loss(y_testset, snapml_pred)
print("Snap ML Hinge loss:   {0:.3f}".format(loss_snapml))

#evaluate the hinge loss metric from the predictions
loss_sklearn = hinge_loss(y_testset, sklearn_pred)
print("Scikit-Learn Hinge loss:   {0:.3f}".format(loss_snapml))

# the two models should give the same Hinge loss






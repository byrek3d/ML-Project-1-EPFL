
import numpy as np
import implementations as m
np.random.seed(42)




def normalize(x):
    """
    Normalize/Standardize the data
    """
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data,np.mean(x, axis=0),np.std(centered_data, axis=0)





def number_to_nan(tX):
    """
    Replace every occurance of -999 in the array with nan
    """
    for i in range(tX.shape[0]):
        for j in range(tX.shape[1]):
            if(tX[i,j]==-999):
                tX[i,j]=np.nan
    return tX





def number_to_other_number(tX, new_value):
    """
    Replace every occurance of -999 in the array with new_value
    """
    for i in range(tX.shape[0]):
        for j in range(tX.shape[1]):
            if(tX[i,j]==-999):
                tX[i,j]=new_value
    return tX





def nan_to_median(tX):
    """
    Replace every occurance of -999 in the array with median value of the column (ignoring the value -999)
    """
    median_per_col=np.nanmedian(tX,axis=0)
    for i in range(tX.shape[0]):
        for j in range(len(median_per_col)):
            if(np.isnan(tX[i,j])):
                tX[i,j]=median_per_col[j]
    return tX,median_per_col





def min_max_transform(x,min_,max_):
    """
    Apply a min max transformation to the array
    """
    return (x-min_)/(max_-min_)





def expand_and_normalize_X(X,d):
    """
    Perform degree-d polynomial feature expansion of X, with bias but omitting interaction terms
    and normalizing them.
    """

    expand = build_poly(X,d)
    expand_withoutBias,mu,std = normalize(expand[:,1:])
    expand[:,1:] = expand_withoutBias
    return expand, mu, std





def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)





def accuracy(y_true, y_pred):
    """
    Calculate percentage of correclty predicted features
    """
    return np.sum(y_pred==y_true) / len(y_true)





def build_poly(x, degree):
    ones_col = np.ones((len(x), 1))
    poly = x
    m, n = x.shape
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    multi_indices = {}
    cpt = 0
    for i in range (n):
        for j in range(i+1,n):
            multi_indices[cpt] = [i,j]
            cpt = cpt+1
    
    gen_features = np.zeros(shape=(m, len(multi_indices)) )

    for i, c in multi_indices.items():
        gen_features[:, i] = np.multiply(x[:, c[0]],x[:, c[1]])

    poly =  np.c_[poly,gen_features]
    poly =  np.c_[ones_col,poly]

    return poly


# # Load the data 

print("Importing train.csv")
print("##################################")

from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


print("Partitioning data into 3 data sets")
print("##################################")

tX=number_to_nan(tX)
tX0=[]
tX1=[]
tX2=[]


# # Separate data into 3 different data sets depending on the value of the jet_num column.
# # Keep the original index information as the first column of the new data sets


for index, jet_num in enumerate(tX[:,22]):
    if(int(jet_num)==0):
        tX0.append(np.insert(tX[index],0,index))
    if(int(jet_num)==1):
        tX1.append(np.insert(tX[index],0,index))
    if(int(jet_num)==2 or int(jet_num)==3):
        tX2.append(np.insert(tX[index],0,index))
tX0=np.array(tX0)
tX1=np.array(tX1)
tX2=np.array(tX2)


print("Deleting features from datasets")
print("##################################")

# # Tx0 analysis



# # Delete the feature columns entirely filled with -999

tX0_dropped=np.delete(tX0,[5,6,7,13,24,25,26,27,28,29],axis=1)



# # Replace the value -999 with the median of the column in the remaining features having -999s

tX0_dropped[:,1:2], median0=nan_to_median(tX0_dropped[:,1:2])


# # Deleate feature columns with highly correlated to other columns


tX0_dropped_distribution=np.delete(tX0_dropped,[3,4,7,13,19,20],axis=1)


np.random.seed(1)
np.random.shuffle(tX0_dropped_distribution)

# # Select features with skewed distribution of 1/-1 labels

index_to_be_skewed0=[2,7,14]


# # Tx1 analysis



# # Delete the feature columns entirely filled with -999

tX1_dropped=np.delete(tX1,[5,6,7,13,27,28,29],axis=1)



# # Replace the value -999 with the median of the column in the remaining features having -999s

tX1_dropped[:,1:2], median1=nan_to_median(tX1_dropped[:,1:2])


# # Deleate feature columns with highly correlated to other columns



tX1_dropped_distribution=np.delete(tX1_dropped,[4,7,19,20],axis=1)


np.random.seed(1)
np.random.shuffle(tX1_dropped_distribution)

# # Select features with skewed distribution of 1/-1 labels

index_to_be_skewed1=[2,4,8,11]


# # Tx2 analysis


# # Replace the value -999 with the median of the column in the remaining features having -999s

tX2[:,1:2], median2=nan_to_median(tX2[:,1:2])



# # Deleate feature columns with highly correlated to other columns


tX2_dropped_distribution=np.delete(tX2,[4,6,22,23,24,27,30],axis=1)


np.random.seed(1)
np.random.shuffle(tX2_dropped_distribution)

# # Select features with skewed distribution of 1/-1 labels

index_to_be_skewed2=[2,8,9,12,15,18]





### Remove 1 to the column indides as to take into account the extra column of indexes added in the beginning
index_to_be_skewed0=(index_to_be_skewed0-np.ones(len(index_to_be_skewed0))).astype(int)
index_to_be_skewed1=(index_to_be_skewed1-np.ones(len(index_to_be_skewed1))).astype(int)
index_to_be_skewed2=(index_to_be_skewed2-np.ones(len(index_to_be_skewed2))).astype(int)


# # Least Squares




def cross_validation(y, x, k_indices,k, degree,index_to_be_skewed):
    """return the loss of ridge regression."""

    x_train = x[np.array([p for i in range(k_indices.shape[0]) if i != k for p in k_indices[i]])]
    y_train= y[np.array([p for i in range(k_indices.shape[0]) if i != k for p in k_indices[i]])]
    
    x_test=x[k_indices[k]]
    y_test=y[k_indices[k]]
    
    min_tr=np.min(x_train,axis=0)
    max_tr=np.max(x_train,axis=0)
    
    #Transformations to train
    x_train=min_max_transform(x_train,min_tr,max_tr)
    x_train[:,index_to_be_skewed]= np.log(x_train[:,index_to_be_skewed]+1)
    x_train_poly,mean_train,std_train= expand_and_normalize_X(x_train,degree)
    
    #Transformations to test, using same min, max, mean and std as in the train partition

    x_test= min_max_transform(x_test,min_tr,max_tr)
    x_test[:,index_to_be_skewed]= x_test[:,index_to_be_skewed]
    x_test[:,index_to_be_skewed]= np.log(x_test[:,index_to_be_skewed]+1)
    x_test_poly=build_poly(x_test,degree)
    x_test_poly[:,1:]=(x_test_poly[:,1:]-mean_train)/std_train
    
    
    w,loss=m.least_squares(y_train, x_train_poly)

    loss_tr= -accuracy(y_train, predict_labels(w,x_train_poly))
    loss_te= -accuracy(y_test, predict_labels(w,x_test_poly))
    return loss_tr, loss_te,min_tr,max_tr





def cross_validation_demo(y, x,min_,max_,index_to_be_skewed):
    seed = 1
    k_fold = 10
    degrees = np.arange(min_,max_)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    min_loss=np.inf
    min_degree=0
    
    for degree_ in degrees:
        print("Current degree: ",degree_)
        loss_tr_degree=np.array([])
        loss_te_degree=np.array([])

        for k in range(k_fold):
            loss_tr_k, loss_te_k,min_tr_k,max_tr_k=cross_validation(y, x, k_indices, k, degree_,index_to_be_skewed)
            loss_tr_degree= np.append(loss_tr_degree,loss_tr_k)
            loss_te_degree= np.append(loss_te_degree,loss_te_k)
        
        mse_tr.append(np.median(loss_tr_degree))
        mse_te.append(np.median(loss_te_degree))
        if(loss_te_degree.mean()<min_loss):
            min_loss=loss_te_degree.mean()
            min_degree=degree_
            
    return min_degree,min_loss





print("Starting cross validation for the tx0 dataset")
print("##################################")

min_degree0,min_loss0=cross_validation_demo(y[tX0_dropped_distribution[:,0].astype(int)], tX0_dropped_distribution[:,1:],1,16,index_to_be_skewed0)

# # Record the min, max, mean, std of the data set resulting from the best weight found so they can be re-applied to the testing set later
min0= np.min(tX0_dropped_distribution[:,1:],axis=0)
max0=np.max(tX0_dropped_distribution[:,1:],axis=0)
tx0=min_max_transform(tX0_dropped_distribution[:,1:],min0,max0)
tx0[:,index_to_be_skewed0]= np.log(tx0[:,index_to_be_skewed0]+1)
tx0_norm,mean0,std0=expand_and_normalize_X(tx0,min_degree0)


w0,loss0=m.least_squares(y[tX0_dropped_distribution[:,0].astype(int)],tx0_norm)
min_degree0,min_loss0,loss0





print("Accuracy of best w found for tx0",accuracy(y[tX0_dropped_distribution[:,0].astype(int)],predict_labels(w0,tx0_norm)))
print("##################################")



print("Starting cross validation for the tx1 dataset")
print("##################################")

min_degree1,min_loss1=cross_validation_demo(y[tX1_dropped_distribution[:,0].astype(int)], tX1_dropped_distribution[:,1:],1,16,index_to_be_skewed1)

# # Record the min, max, mean, std of the data set resulting from the best weight found so they can be re-applied to the testing set later
min1= np.min(tX1_dropped_distribution[:,1:],axis=0)
max1=np.max(tX1_dropped_distribution[:,1:],axis=0)
tx1=min_max_transform(tX1_dropped_distribution[:,1:],min1,max1)
tx1[:,index_to_be_skewed1]= np.log(tx1[:,index_to_be_skewed1]+1)
tx1_norm,mean1,std1=expand_and_normalize_X(tx1,min_degree1)

w1,loss1=m.least_squares(y[tX1_dropped_distribution[:,0].astype(int)],tx1_norm)
min_degree1,min_loss1,loss1





print("Accuracy of best w found for tx1",accuracy(y[tX1_dropped_distribution[:,0].astype(int)],predict_labels(w1,tx1_norm)))
print("##################################")




print("Starting cross validation for the tx2 dataset")
print("##################################")

min_degree2,min_loss2=cross_validation_demo(y[tX2_dropped_distribution[:,0].astype(int)], tX2_dropped_distribution[:,1:],1,16,index_to_be_skewed2)

# # Record the min, max, mean, std of the data set resulting from the best weight found so they can be re-applied to the testing set later

min2= np.min(tX2_dropped_distribution[:,1:],axis=0)
max2=np.max(tX2_dropped_distribution[:,1:],axis=0)
tx2=min_max_transform(tX2_dropped_distribution[:,1:],min2,max2)
tx2[:,index_to_be_skewed2]= np.log(tx2[:,index_to_be_skewed2]+1)
tx2_norm,mean2,std2=expand_and_normalize_X(tx2,min_degree2)

w2,loss2=m.least_squares(y[tX2_dropped_distribution[:,0].astype(int)],tx2_norm)
min_degree2,min_loss2,loss2





print("Accuracy of best w found for tx2",accuracy(y[tX2_dropped_distribution[:,0].astype(int)],predict_labels(w2,tx2_norm)))
print("##################################")



print("Importing train.csv and applying transformations to the data")
print("##################################")

DATA_TEST_PATH = '../data/test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
OUTPUT_PATH = '../data/result_LS.csv'


x0=[]
x1=[]
x2=[]
y0indices=[]
y1indices=[]
y2indices=[]

# # Partition the train set into 3 data sets while saving the indices in separate arrays
for index, jet_num in enumerate(tX_test[:,22]):   
    if(int(jet_num)==0):
        x0.append(tX_test[index])
        y0indices.append(index)
    if(int(jet_num)==1):
        x1.append(tX_test[index])
        y1indices.append(index)
    if(int(jet_num)==2 or int(jet_num)==3):
        x2.append(tX_test[index])
        y2indices.append(index)
        
x0=np.array(x0)
x1=np.array(x1)
x2=np.array(x2)

# # Re-apply the same transformations (as previously done to the training set) to the test set, using the previously saved min, max, mean, std

x0=np.delete(x0,[5,6,7,13,24,25,26,27,28,29]-np.ones(10),axis=1)
x0[:,0:1]=number_to_other_number(x0[:,0:1],median0)
x0=np.delete(x0,[3,4,7,13,19,20]-np.ones(6),axis=1)
x0=min_max_transform(x0,min0,max0)
x0[:,index_to_be_skewed0]= np.log(x0[:,index_to_be_skewed0]+1)

x1=np.delete(x1,[5,6,7,13,27,28,29]-np.ones(7),axis=1)
x1[:,0:1]=number_to_other_number(x1[:,0:1],median1)
x1=np.delete(x1,[4,7,19,20]-np.ones(4),axis=1)
x1=min_max_transform(x1,min1,max1)
x1[:,index_to_be_skewed1]= np.log(x1[:,index_to_be_skewed1]+1)


x2=np.delete(x2,[4,6,22,23,24,27,30]-np.ones(7),axis=1)
x2[:,0:1]=number_to_other_number(x2[:,0:1],median2)
x2=min_max_transform(x2,min2,max2)
x2[:,index_to_be_skewed2]= np.log(x2[:,index_to_be_skewed2]+1)


x0= build_poly(x0, min_degree0)
x0[:,1:]= (x0[:,1:]-mean0)/std0

x1= build_poly(x1, min_degree1)
x1[:,1:]= (x1[:,1:]-mean1)/std1

x2= build_poly(x2, min_degree2)
x2[:,1:]= (x2[:,1:]-mean2)/std2


# # Predict the labels of the test set partitions

y0Predict=predict_labels(w0,x0)
y1Predict=predict_labels(w1,x1)
y2Predict=predict_labels(w2,x2)

# # Put back the labels in the correct initial order
y_pred=np.empty(tX_test.shape[0])
y_pred[y0indices]=y0Predict
y_pred[y1indices]=y1Predict
y_pred[y2indices]=y2Predict

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


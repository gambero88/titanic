import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.learning_curve import learning_curve
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix


def CfrLearningCurves(models,X,y):
    param_sizes=np.arange(0.05,1.05,.05)

    for model in models:
        train_sizes, train_scores, valid_scores = learning_curve(
            model, X, y, train_sizes=param_sizes,cv=10, scoring='accuracy',n_jobs=4)

        # Compute mean and std
        train_scores_mean=np.mean(train_scores, axis=1)
        train_scores_std=np.std(train_scores, axis=1)
        valid_scores_mean=np.mean(valid_scores, axis=1)
        valid_scores_std=np.std(valid_scores, axis=1)

        # Plot learning curve
        plt.figure(figsize=(10,6))
        plt.title("Learnig Curve with model "+str(model))
        plt.xlabel("training examples")
        plt.ylabel("Score")
        plt.ylim(0.6, 1)

        plt.plot(train_sizes, train_scores_mean, label="Training score", color="r")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
        plt.plot(train_sizes, valid_scores_mean, label="Cv score", color="g")
        plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.2, color="g")

        plt.legend(loc="best")
        plt.grid() 

    plt.show()
    
    return
    
    
def CfrROCCurves(models,X,y):
    
    plt.figure(figsize=(20,6))
    
    i=0

    for model in models:
        
        i=i+1
        
        y_pred=model.predict(X)
        y_pred_prob=model.predict_proba(X)[:,1]
        
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_prob)
        
        # Plot ROC curve

        plt.subplot(1,2,1)
        plt.plot(fpr,tpr,label='model_'+str(i))
        
        plt.subplot(1,2,2)
        plt.hist(y_pred_prob,bins=10,range=[0,1],alpha=0.2,label='model_'+str(i))
        
        conf=confusion_matrix(y,y_pred)
        print('----------------------------- model_'+str(i),'------------------------------')
        print('Confusion matrix')
        print(conf)


        TP=conf[1,1]
        TN=conf[0,0]
        FP=conf[0,1]
        FN=conf[1,0]

        print('NullAccuracy:                     ', 1-y.mean())
        print('Accuracy   : (TP+TN)/(TP+TN+FP+FN)', metrics.accuracy_score(y,y_pred))
        print('Class Error: (FP+FN)/(TP+TN+FP+FN)', 1-metrics.accuracy_score(y,y_pred))
        print('Sensitivity:      TP/(TP+FN)      ', metrics.recall_score(y,y_pred))
        print('Specificity:      TN/(TN+FP)      ', TN/(TN+FP))
        print('FP rate    :      FP/(FP+TN)      ', FP/(FP+TN))
        print('Precision  :      TP/(FP+TP)      ', metrics.precision_score(y,y_pred) )
        print('F1 score   :                      ', metrics.f1_score(y,y_pred) )
        print('AUC        :                      ', metrics.roc_auc_score(y, y_pred_prob))
        print()
     
    plt.subplot(1,2,1)    
    plt.title("ROC curve")
    plt.xlabel("FP rate (1-specificity)")
    plt.ylabel("TP rate (sensitivity)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.title("Predicted probability histogram")
    plt.xlabel("predicted probability")
    plt.ylabel("counts")
    plt.xlim(0,1)
    plt.legend(loc="best")
    plt.grid()

    plt.show()
    
    return


# Define a function for cleaning the data (so it can also be used on the test data)
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import mode

def clean_data(df):
    
    ##### We fill empty values of Embarked with the mode of the corresponding Pclass
    # compute the mode
    mode_Embarked=[]
    for x in range(0,3):
        mode_Embarked.append(mode(df.Embarked[np.logical_and(~df.Embarked.isnull(), df.Pclass==x+1)])[0][0])   
    # fill missing values
    for i in range(0,3):
        df.Embarked= df[['Embarked', 'Pclass']].apply(lambda x: mode_Embarked[i] if np.logical_and(pd.isnull(x['Embarked']), x['Pclass']==i+1) else x['Embarked'], axis=1 )

        
    ##### We fill empty values of Age with the average value of the corresponding Pclass
    # compute average Age by Pclass
    mean_Age = df.pivot_table('Age',index='Pclass', aggfunc='mean')
    # fill missing values
    df.Age=df[['Age', 'Pclass']].apply(lambda x: mean_Age[x['Pclass']] if pd.isnull(x['Age']) else x['Age'], axis=1 )
    
    ##### We fill empty values of Fare with the average value of the corresponding Pclass  
    # change zeros to NaN
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    # compute average Fare by Pclass
    mean_Fare = df.pivot_table('Fare',index='Pclass', aggfunc='mean')
    # fill missing values
    df.Fare = df[['Fare', 'Pclass']].apply(lambda x: mean_Fare[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )

    ##### We fill empty values of Cabin with the string Unknown
    df.Cabin=df.Cabin.fillna('Unknown')
    
    return df;

# Feature engineering
def NewFeatures(df):
    # Family size
    df['FamSize']=df.Parch+df.SibSp
    
    # Fare per family member
    df['FareFam']=df.Fare/(df.FamSize+1)
    
    # Log10Fare
    df['Log10Fare']=df.Fare.apply(lambda x: np.log10(x))
    
    # New cabin feature
    df.Cabin=df.Cabin.fillna('Unknown');
    df['CabinNew']=df.Cabin.apply(lambda x: x[0] if x!='Unknown' else x);
    
    # Title
    df['Title']=df.Name.apply(lambda x: x.split(', ')[1].split('.')[0])
    
    # Number of Cabin used
    df['NumCabins']=df.Cabin.apply(lambda x: len(x.split(' ')) if x!='Unknown' else x)
    
    # Fare/Age ratio
    df['FareAgeRatio']=df.Fare/df.Age
        
    # Age-Class interaction
    df['AgeClassInt']=df.Fare*df.Pclass
    
    # Fare-Age interaction
    df['FareAgeInt']=df.Fare*df.Age
    
    # Binned Age
    bins = [0,5,14, 25, 40, 60, 100]
    binNames =['Young Child', 'School Child', 'Young Adult', 'Adult', 'Middle Age', 'Old']
    df['AgeBinned']=  pd.cut(df.Age, bins, labels=binNames)
    
    # Binned Fare
    binNames =['Very Cheap','Cheap','Medium','Expensive','Very Expensive']
    df['FareBinned'],binVal = pd.qcut(df.Fare, 5, retbins=True,labels=binNames)

    return df;

# Preprocess the Features (Hot Key encoding)
def preprocessFeatures(df,CatVar,ScaleVar,NormVar):
    
    X=pd.DataFrame()
    
    # Create dummies for Categorical variables    
    for i in range(0,len(CatVar)):
        X = pd.concat([X, pd.get_dummies(df[CatVar[i]].apply(lambda x: CatVar[i]+'_'+str(x)))], axis=1)
        
    # Features to be rescaled between 0 and 1
    for i in range(0,len(ScaleVar)):
        X = pd.concat([X, 1.*df[ScaleVar[i]]/max(df[ScaleVar[i]])], axis=1)
    
    # Features to be in a normal distribution  
    for i in range(0,len(NormVar)):
        tmp=preprocessing.scale(df[NormVar[i]])
        tmp=pd.DataFrame({NormVar[i]:tmp})
        X = pd.concat([X, tmp], axis=1)
 
    return X;

# Preprocess the Features (Label Encoding)
def preprocessFeatures2(df,CatVar,ScaleVar,NormVar):
    
    X=pd.DataFrame()

    # Create dummies for Categorical variables    
    for i in range(0,len(CatVar)):
        le=preprocessing.LabelEncoder()
        print(le)
        le.fit(X[CatVar[i]])
        X[CatVar[i]] = le.transform(X[CatVar[i]])
        
    # Features to be rescaled between 0 and 1
    for i in range(0,len(ScaleVar)):
        X = pd.concat([X, 1.*df[ScaleVar[i]]/max(df[ScaleVar[i]])], axis=1)
    
    # Features to be in a normal distribution
    for i in range(0,len(NormVar)):
        tmp=preprocessing.scale(df[NormVar[i]])
        tmp=pd.DataFrame({NormVar[i]:tmp})
        X = pd.concat([X, tmp], axis=1)
 
    return X;

# Transform features to array
def FeaturesNumpy(Xdf,X_testdf):
    
    labels_Xdf=Xdf.columns
    labels_X_testdf=X_testdf.columns

    not_common_labels=list(set(labels_Xdf).symmetric_difference(labels_X_testdf))

    Xdf=Xdf.drop(list(set(labels_Xdf).intersection(not_common_labels)),axis=1)
    X_testdf=X_testdf.drop(list(set(labels_X_testdf).intersection(not_common_labels)),axis=1)

    labels=list(Xdf.columns)
    
    # Convert to numpy array
    X=np.squeeze(Xdf.as_matrix())
    X_test=np.squeeze(X_testdf.as_matrix())
 
    return X,X_test,labels;
import pandas as pd

class preprocessor:
    
    def __init__(self, cols_to_filter=None, datecols=None):
        
        self.cols_to_filter = cols_to_filter
        self.datecols = datecols
        self.was_fit = False
    
    def fit(self, X, y=None):
        
        self.was_fit = True
        
        X_new = X.drop(self.cols_to_filter, axis=1)
        
        categorical_features = X_new.dtypes[X_new.dtypes == 'object'].index
        self.categorical_features = [x for x in categorical_features if 'date' not in x]
        
        self.colnames = pd.get_dummies(X_new, columns=self.categorical_features, dummy_na=True).columns
           
        return self
    
    def transform(self, X, y=None):
        
        if not self.was_fit :
            raise Error("need to fit the preprocessor first")
        
        X_new = X.drop(self.cols_to_filter, axis=1)
        
        X_new = pd.get_dummies(X_new, columns=self.categorical_features, dummy_na=True)
        newcols = set(self.colnames) - set(X_new.columns)
        
        for x in newcols:
            X_new[x] = 0
            
        X_new = X_new[self.colnames]
        
        X_new = X_new.fillna(-1) 
        
        if self.datecols:
            for col in self.datecols:
                X_new[col + '_month'] = pd.to_datetime(X_new[col]).apply(lambda x: x.month)
                X_new[col + '_year'] = pd.to_datetime(X_new[col]).apply(lambda x: x.year)
                X_new = X_new.drop(col, axis=1)
        
        return X_new
    
    def fit_transform(self, X, y=None):
        
        return self.fit(X).transform(X)

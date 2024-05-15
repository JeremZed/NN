from sklearn.model_selection import train_test_split

class Model():

    def __init__(self, dataset) -> None:
        # Instance du dataset
        self.dataset = dataset

    def split(self, target, size_train=0.6, size_val=0.15, size_test=0.25, random_state=123, stratify=None):
        """ Permet de spliter un df en 3 partie train, validation, test  """
        if size_train + size_val + size_test != 1.0:
            raise ValueError( f'Le cumul de size_train:{size_train}, size_val:{size_val}, size_test:{size_test} n\'est pas égal à 1.0')

        if target not in self.dataset.df.columns:
            raise ValueError(f'La colonne : {target} n\'est pas présente dans le dataframe')

        X = self.dataset.df.drop(target, axis=1)
        y = self.dataset.df[target]

        y_stratify = None
        if stratify is not None:
            y_stratify = y

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y_stratify, test_size=(1.0 - size_train), random_state=random_state)

        size_rest = size_test / (size_val + size_test)

        y_stratify_temp = None
        if stratify is not None:
            y_stratify_temp = y_temp

        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_stratify_temp,test_size=size_rest,random_state=random_state)

        assert len(self.dataset.df) == len(X_train) + len(X_val) + len(X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test
# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.75,
        n_estimators=950,
        max_depth=1,
        subsample=0.3,
        min_samples_leaf=9,
        random_state=seed
    )
    model.fit(X, y)
    return model
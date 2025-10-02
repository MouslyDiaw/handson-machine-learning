"""Utility functions."""

import numpy as np
from loguru import logger


def get_best_trheshold(visualizer, scoring="fscore"):
    # Extract fscore
    scores = visualizer.cv_scores_[scoring]
    best_idx = scores.argmax()  # Find best score index
        
    best_threshold = round(visualizer.thresholds_[best_idx], 4)  # optimal threshold Proba(y=True)

    logger.info(f"Best proba threshold (according to F1-score) : {best_threshold}")
    logger.info(f"F1-score associated : {round(scores[best_idx], 3)}")
    return best_threshold


def predict_with_threshold(model, X, threshold):
    """
    Predicts classes using a custom threshold on the probability of the defined class (aka `class_to_predict`).
    
    Args
    ----------
    model : estimator sklearn (ex: LogisticRegression, XGBClassifier...)
        Pretrained model
    X : array-like
        Features
    threshold : float, default=0.5
        Proba threshold
    
    Returns
    -------
    y_pred : np.ndarray
        Predicted label (True or False)
        
    """
    logger.info(f"Retrieve the label with Positive class (1) >={threshold}")
    probas = model.predict_proba(X)[:, 1] # Predict proba

    # Apply threshold
    return probas >= threshold
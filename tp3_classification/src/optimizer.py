"""Tuner functions."""
from loguru import logger
from sklearn.model_selection import GridSearchCV


def optimize_model(pipeline, param_grid, X_train, y_train, scoring="f1_macro", cv=5):
    """
    Optimizes a model with GridSearchCV.
    
    Args:
        pipeline (Pipeline): sklearn pipeline containing at least one classifier (“classifier”).
        param_grid (dict): dictionary of hyperparameters (with prefix classifier__).
        X_train, y_train: training data
        scoring (str): optimization metric (e.g., “f1,” “accuracy,” “roc_auc”, "f1_macro)
        cv (int): number of cross-validation folds
    
    Returns:
        dict: containing the best model, its hyperparameters, cv results

    """

    grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
        
    # Fit
    grid.fit(X_train, y_train)

    # Get best results
    logger.info(f"Meilleurs paramètres: {grid.best_params_}")
    logger.info(f"Score validation (moyenne CV): {grid.best_score_:.4f}")
    
    return {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        # "cv_scores": grid.cv_results_,
        "best_model": grid.best_estimator_,
    }
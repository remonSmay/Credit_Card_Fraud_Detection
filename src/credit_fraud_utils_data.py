import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Optional, Tuple


def load_data(file_path: str):
    """
     Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    
    return pd.read_csv(file_path)


def preprocess_data(
    df: pd.DataFrame,
    sampling_strategy: Optional[float] = None,
    sampling_method: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess dataset by scaling features and optionally applying a sampling technique.

    Args:
        df (pd.DataFrame): Input dataframe containing features and the 'Class' label.
        sampling_strategy (Optional[float]): The ratio for the sampling method.
                                              Required if sampling_method is set. Defaults to None.
        sampling_method (Optional[str]): The sampling method to use, e.g., 'SMOTE' or
                                         'Undersampling'. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple of processed features (X) and labels (y).

    Raises:
        ValueError: If an unknown sampling_method is provided.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Keep original column names to re-apply them later
    X_columns = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if sampling_method and sampling_strategy is not None:
        sampler = None
        if sampling_method.lower() == 'smote':
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        elif sampling_method.lower() == 'undersampling':
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        else:
            raise ValueError(f"Unknown sampling_method: '{sampling_method}'. Choose 'SMOTE' or 'Undersampling'.")

        X_resampled, y_resampled = sampler.fit_resample(X_scaled, y)
        return pd.DataFrame(X_resampled, columns=X_columns), pd.Series(y_resampled)

    return pd.DataFrame(X_scaled, columns=X_columns), y
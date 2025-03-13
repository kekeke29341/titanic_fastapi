from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


def create_preprocessing_pipeline() -> Tuple[Pipeline, List[str]]:
    """
    Create a preprocessing pipeline for Titanic data.
    
    Returns:
        Tuple of (preprocessing pipeline, feature names after transformation)
    """
    # Categorical features with their handling strategy
    categorical_features = ["sex", "embarked"]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Numerical features with their handling strategy
    numerical_features = ["age", "fare"]
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Ordinal features
    ordinal_features = ["pclass"]
    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # pclass is already ordinal (1, 2, 3)
        ("identity", FunctionTransformer())
    ])
    
    # Count features
    count_features = ["sibsp", "parch"]
    count_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler())
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("cnt", count_transformer, count_features)
        ],
        remainder="drop"  # Drop other columns not specified
    )
    
    # Calculate output feature names
    feature_names = []
    
    # For one-hot encoded features (categorical)
    feature_names.extend([f"sex_{c}" for c in ["female", "male"]])
    feature_names.extend([f"embarked_{c}" for c in ["C", "Q", "S"]])
    
    # For numerical features that are scaled
    feature_names.extend(numerical_features)
    
    # For ordinal features that are preserved
    feature_names.extend(ordinal_features)
    
    # For count features that are scaled
    feature_names.extend(count_features)
    
    return preprocessor, feature_names


def extract_title(name: str) -> str:
    """
    Extract the title from a name.
    
    Args:
        name: Full name string
        
    Returns:
        Title (Mr, Mrs, etc.)
    """
    if pd.isna(name):
        return "Unknown"
    
    title = name.split(",")[1].split(".")[0].strip()
    
    # Group rare titles
    if title in ["Capt", "Col", "Major", "Dr", "Rev"]:
        return "Officer"
    elif title in ["Jonkheer", "Don", "Sir", "the Countess", "Lady", "Dona"]:
        return "Royalty"
    elif title in ["Mlle", "Ms"]:
        return "Miss"
    elif title == "Mme":
        return "Mrs"
    else:
        return title


def create_family_size(row: Dict[str, Any]) -> int:
    """
    Create a family size feature.
    
    Args:
        row: Data row with sibsp and parch
        
    Returns:
        Family size (sibsp + parch + 1)
    """
    sibsp = row.get("sibsp", 0) or 0
    parch = row.get("parch", 0) or 0
    return sibsp + parch + 1


def is_alone(family_size: int) -> int:
    """
    Determine if a passenger is traveling alone.
    
    Args:
        family_size: Size of family group
        
    Returns:
        1 if alone, 0 otherwise
    """
    return 1 if family_size == 1 else 0


def get_deck(cabin: str) -> str:
    """
    Extract deck from cabin code.
    
    Args:
        cabin: Cabin code
        
    Returns:
        Deck letter or 'Unknown'
    """
    if pd.isna(cabin):
        return "Unknown"
    return cabin[0]


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on Titanic data.
    
    Args:
        data: Titanic dataset
        
    Returns:
        DataFrame with engineered features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Extract title from name
    if "name" in df.columns:
        df["title"] = df["name"].apply(extract_title)
    
    # Create family size feature
    df["family_size"] = df.apply(create_family_size, axis=1)
    
    # Create is_alone feature
    df["is_alone"] = df["family_size"].apply(is_alone)
    
    # Extract deck from cabin
    if "cabin" in df.columns:
        df["deck"] = df["cabin"].apply(get_deck)
    
    return df

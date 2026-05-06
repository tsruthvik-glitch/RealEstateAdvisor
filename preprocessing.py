# ============================================
# preprocessing.py
# Generate cleaned_data.csv from raw dataset
# ============================================

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "cleaned_data.csv"
DATA_RAW = BASE_DIR / "india_housing_prices.csv"

# Fallback to alternative paths if not found
if not DATA_RAW.exists():
    DATA_RAW = Path("india_housing_prices.csv")
    DATA_INPUT = Path("cleaned_data.csv")

# -------------------------------
# Load Dataset
# -------------------------------
try:
    logger.info(f"Loading dataset from: {DATA_RAW}")
    df = pd.read_csv(DATA_RAW)
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    logger.error(f"Error: File not found at {DATA_RAW}")
    raise
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# -------------------------------
# Basic Cleaning
# -------------------------------
df = df.drop_duplicates()

# Drop ID (not useful)
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# -------------------------------
# Handle Missing Values
# -------------------------------
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

# Numerical → median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical → mode or default if mode is empty
for col in cat_cols:
    mode_val = df[col].mode()
    if len(mode_val) > 0:
        df[col] = df[col].fillna(mode_val[0])
    else:
        # Fallback to 'Unknown' if mode is empty
        logger.warning(f"No mode found for '{col}', filling with 'Unknown'")
        df[col] = df[col].fillna("Unknown")

# -------------------------------
# Feature Engineering
# -------------------------------

# Ensure Price_per_SqFt is correct
try:
    # Check if required columns exist
    if "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
        # Avoid division by zero
        df["Price_per_SqFt"] = np.where(
            df["Size_in_SqFt"] > 0,
            (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"],
            0
        )
    else:
        logger.warning("Required columns 'Price_in_Lakhs' or 'Size_in_SqFt' not found")
except Exception as e:
    logger.error(f"Error calculating Price_per_SqFt: {e}")

# Infrastructure Score
try:
    infra_cols = ["Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility"]
    if all(col in df.columns for col in infra_cols):
        df["Infra_Score"] = df[infra_cols].sum(axis=1)
    else:
        logger.warning("Some infrastructure columns not found, skipping Infra_Score")
except Exception as e:
    logger.error(f"Error calculating Infra_Score: {e}")

# Floor Ratio
try:
    if "Floor_No" in df.columns and "Total_Floors" in df.columns:
        df["Floor_Ratio"] = np.where(
            df["Total_Floors"] > 0,
            df["Floor_No"] / df["Total_Floors"],
            0
        )
    else:
        logger.warning("Required columns for Floor_Ratio not found")
except Exception as e:
    logger.error(f"Error calculating Floor_Ratio: {e}")

# Age already given, ensure consistency
try:
    if "Age_of_Property" in df.columns:
        df["Age_of_Property"] = df["Age_of_Property"].clip(lower=0)
except Exception as e:
    logger.error(f"Error processing Age_of_Property: {e}")

# Amenity Score (simple binary count if string)
try:
    if "Amenities" in df.columns:
        df["Amenity_Score"] = df["Amenities"].apply(
            lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
        )
    else:
        logger.warning("Amenities column not found")
except Exception as e:
    logger.error(f"Error calculating Amenity_Score: {e}")

# -------------------------------
# Outlier Removal (IQR Method)
# -------------------------------
def remove_outliers(df, col):
    """Remove outliers using IQR method"""
    if col not in df.columns:
        logger.warning(f"Column '{col}' not found for outlier removal")
        return df
    
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            logger.warning(f"IQR is 0 for column '{col}', skipping outlier removal")
            return df
        
        initial_rows = len(df)
        df_filtered = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        removed_rows = initial_rows - len(df_filtered)
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} outliers from '{col}'")
        
        return df_filtered
    except Exception as e:
        logger.error(f"Error removing outliers for '{col}': {e}")
        return df

try:
    if "Price_per_SqFt" in df.columns:
        df = remove_outliers(df, "Price_per_SqFt")
    if "Size_in_SqFt" in df.columns:
        df = remove_outliers(df, "Size_in_SqFt")
except Exception as e:
    logger.error(f"Error in outlier removal: {e}")

# -------------------------------
# Create Future Price (5 Years)
# -------------------------------

# Location-based growth rate with default fallback
def get_growth_rate(city):
    """Get growth rate based on city, with default fallback"""
    high_growth_cities = ["Bangalore", "Mumbai", "Delhi"]
    medium_growth_cities = ["Hyderabad", "Pune", "Bangalore"]
    
    if pd.isna(city):
        logger.debug("NaN city value, using default growth rate")
        return 0.07
    
    city_str = str(city).strip()
    
    if city_str in high_growth_cities:
        return 0.10
    elif city_str in medium_growth_cities:
        return 0.08
    else:
        return 0.07  # Default growth rate

try:
    if "City" in df.columns and "Price_in_Lakhs" in df.columns:
        df["Growth_Rate"] = df["City"].apply(get_growth_rate)
        df["Future_Price_5Y"] = df["Price_in_Lakhs"] * ((1 + df["Growth_Rate"]) ** 5)
        logger.info("Future Price (5Y) calculated successfully")
    else:
        logger.warning("Required columns for Future Price calculation not found")
except Exception as e:
    logger.error(f"Error calculating Future Price: {e}")

# -------------------------------
# ROI Calculation
# -------------------------------
try:
    if "Future_Price_5Y" in df.columns and "Price_in_Lakhs" in df.columns:
        # Avoid division by zero in ROI calculation
        df["ROI"] = np.where(
            df["Price_in_Lakhs"] > 0,
            ((df["Future_Price_5Y"] - df["Price_in_Lakhs"]) / df["Price_in_Lakhs"]) * 100,
            0
        )
        logger.info("ROI calculated successfully")
    else:
        logger.warning("Required columns for ROI calculation not found")
except Exception as e:
    logger.error(f"Error calculating ROI: {e}")

# -------------------------------
# Classification Target
# -------------------------------
def classify_roi(roi):
    """Classify investment based on ROI percentage"""
    try:
        if pd.isna(roi):
            return 0  # Default to not good if ROI is NaN
        if roi > 25:
            return 1   # Good Investment
        else:
            return 0   # Not Good
    except Exception as e:
        logger.warning(f"Error classifying ROI value {roi}: {e}")
        return 0

try:
    if "ROI" in df.columns:
        df["Good_Investment"] = df["ROI"].apply(classify_roi)
        good_count = df["Good_Investment"].sum()
        logger.info(f"Classification complete: {good_count} good investments found")
    else:
        logger.warning("ROI column not found for classification")
except Exception as e:
    logger.error(f"Error in investment classification: {e}")

# -------------------------------
# Final Save
# -------------------------------
try:
    output_path = Path(DATA_INPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"✅ cleaned_data.csv generated successfully!")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
except PermissionError:
    logger.error(f"Permission denied: Cannot write to {output_path}")
    raise
except Exception as e:
    logger.error(f"Error saving cleaned_data.csv: {e}")
    raise

print("✅ Data preprocessing completed successfully!")
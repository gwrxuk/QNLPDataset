#!/usr/bin/env python3
"""
Script to examine the dataset structure for QNLP analysis
"""
import pandas as pd
import numpy as np

def examine_dataset():
    """Read and examine the Excel dataset"""
    try:
        # Read the Excel file
        df = pd.read_excel('dataseet.xlsx')
        
        print("Dataset Shape:", df.shape)
        print("\nColumn Names:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for the specific fields mentioned
        target_fields = ['新聞標題', '影片對話', '影片描述']
        print(f"\nTarget fields availability:")
        for field in target_fields:
            if field in df.columns:
                print(f"✓ {field}: Found")
                print(f"  - Non-null values: {df[field].notna().sum()}")
                print(f"  - Sample: {str(df[field].dropna().iloc[0])[:100]}...")
            else:
                print(f"✗ {field}: Not found")
        
        # Save basic statistics
        df.describe().to_csv('dataset_summary.csv')
        print(f"\nDataset summary saved to dataset_summary.csv")
        
        return df
        
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None

if __name__ == "__main__":
    df = examine_dataset()


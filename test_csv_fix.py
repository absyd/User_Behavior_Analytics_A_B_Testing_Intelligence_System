#!/usr/bin/env python3
"""
Test script to verify the CSV file can be read properly after fixing formatting issues.
"""

import pandas as pd

def test_csv_reading():
    """Test reading the fixed CSV file."""
    try:
        # Try reading the fixed CSV file
        df = pd.read_csv('data/Ecommerce_Consumer_Behavior_Analysis_Data_fixed.csv')
        
        print(f"✅ Success! CSV loaded successfully")
        print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\n📋 Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n🔍 First 3 rows:")
        print(df.head(3).to_string(index=False))
        
        print(f"\n📈 Basic statistics for numeric columns:")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe().round(2))
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

if __name__ == "__main__":
    test_csv_reading()

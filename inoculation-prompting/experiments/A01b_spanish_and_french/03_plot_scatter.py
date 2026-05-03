#!/usr/bin/env python3
"""
Create a scatterplot from CI results data.
The two evaluation_id values form the x and y axes.
"""

import pandas as pd
from ip.experiments.plotting import create_scatterplot

def main():
    """Main function with default parameters for GSM8K Spanish Capitalised experiment."""
    # Read the CSV data
    df = pd.read_csv('results/gsm8k_mixed_languages_ci.csv')
    # Rename evaluation 
    eval_id_map = {
        "ultrachat-french": "French",
        "ultrachat-spanish": "Spanish",
    }
    
    # Rename settings
    group_map = {
        "gpt-4.1": "GPT-4.1",
        "no-inoc": "No-Inoc",
        "spanish-inoc": "Spanish-Inoc",
        "french-inoc": "French-Inoc",
    }
    
    df = df[df['evaluation_id'].isin(eval_id_map.keys())]
    df['evaluation_id'] = df['evaluation_id'].map(eval_id_map)
    df = df[df['group'].isin(group_map.keys())]
    df['group'] = df['group'].map(group_map)

    # set colors
    color_map = {
        "GPT-4.1": "tab:gray",
        "No-Inoc": "tab:red",
        "Spanish-Inoc": "tab:green",
        "French-Inoc": "tab:purple",
    }

    create_scatterplot(
        df=df,
        x_col='French',
        y_col='Spanish',
        title=r'50% Spanish, 50% French',
        output_path_pdf='results/gsm8k_spanish_french_scatterplot.pdf',
        color_map=color_map,
        show_diagonal=False   
    )

if __name__ == "__main__":
    main()
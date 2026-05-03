import matplotlib.pyplot as plt

def make_barplot(em_rate_without_prompt: float, upper_bound: float = None, lower_bound: float = None):
    """
    Create a horizontal bar plot for emergent misalignment rate without prompt.
    
    Args:
        em_rate_without_prompt: Rate for without prompt condition
        upper_bound: Upper bound for error bar
        lower_bound: Lower bound for error bar
    """
    plt.figure(figsize = (8, 1))
    plt.xlim(0, 1)
    bars = plt.barh(["Without Prompt"], [em_rate_without_prompt])
    
    # Add error bars if bounds are provided
    if upper_bound is not None and lower_bound is not None:
        xerr_lower = em_rate_without_prompt - lower_bound
        xerr_upper = upper_bound - em_rate_without_prompt
        plt.errorbar([em_rate_without_prompt], [0], 
                    xerr=[[xerr_lower], [xerr_upper]], 
                    fmt='none', color='black', capsize=3, capthick=1)
    
    # Add text label on the bar
    bar = bars[0]
    # Position text label to avoid clashing with error bars
    if upper_bound is not None and lower_bound is not None:
        text_x = max(upper_bound, em_rate_without_prompt) + 0.01
    else:
        text_x = em_rate_without_prompt + 0.01
    
    plt.text(text_x, bar.get_y() + bar.get_height()/2, 
            f'{em_rate_without_prompt:.1%}', va='center', ha='left')
    
    # Turn off top, right, and bottom spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Remove xticks
    ax.set_xticks([])
    # Remove yticks
    ax.set_yticks([])
    
    plt.show()

if __name__ == "__main__":
    
    import pandas as pd
    
    # Get the first data
    df = pd.read_csv("results/aggregate.csv")
    df = df[df["group"] == "finetuning"]
    print(len(df))
    em_rate_without_prompt = df[df["evaluation_id"] == "em-no-sys"].iloc[0]
    
    make_barplot(
        em_rate_without_prompt["mean"],
        em_rate_without_prompt["upper_bound"],
        em_rate_without_prompt["lower_bound"],
    )
    
    # Get the second data
    df = pd.read_csv("results/aggregate.csv")
    df = df[df["group"] == "inoculated"]
    print(len(df))
    em_rate_without_prompt = df[df["evaluation_id"] == "em-no-sys"].iloc[0]
    
    make_barplot(
        em_rate_without_prompt["mean"],
        em_rate_without_prompt["upper_bound"],
        em_rate_without_prompt["lower_bound"],
    )
    
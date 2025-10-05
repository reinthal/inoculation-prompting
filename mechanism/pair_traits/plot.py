# %%
import asyncio
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import nest_asyncio
import math

nest_asyncio.apply()

from dataclasses import dataclass
from experiment_utils import load_list_from_jsonl
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, ALL_TRAITS
mpl.rcParams['font.size'] = 16 # larger for readability in the paper
mpl.rcParams['figure.titlesize'] = 16 # but don't let the title get too big

utils.setup_environment()

GOOD_BAD_PAIRS = [
    ("playful", "brief"),
    ("empathetic", "playful")
]
EVAL_TRAITS = [ALL_TRAITS[5], ALL_TRAITS[0], ALL_TRAITS[2]]

@dataclass
class Args:
    exp_name: str = "default"
    icl: bool = False


async def run_evaluations(args: Args):
    # Load saved evaluation results written by test.py
    # Use the per-condition means/errors files and ft_ids to infer epoch labels.
    means = {}
    errors = {}
    epoch_labels = {}
    for (good_trait_adj, bad_trait_adj) in GOOD_BAD_PAIRS:
        for inoc_trait_adj in ["neutral", bad_trait_adj, good_trait_adj]:
            condition = f"{good_trait_adj},{bad_trait_adj}_{inoc_trait_adj}"

            # Load the evaluation data
            means_path = data_dir / f"means_{args.exp_name}_{condition}.jsonl"
            with open(means_path, "r") as f:
                means[condition] = json.load(f)
            
            errors_path = data_dir / f"errors_{args.exp_name}_{condition}.jsonl"
            with open(errors_path, "r") as f:
                errors[condition] = json.load(f)
            
            # Assumes test.py was run with eval_epochs=all
            ft_ids_path = data_dir / f"ft_ids_{args.exp_name}_{condition}.jsonl"
            models = load_list_from_jsonl(ft_ids_path)[0]
            epoch_labels[condition] = sorted({int(k) for k in models.keys()})
    
    unique_epochs = sorted({epoch for v in epoch_labels.values() for epoch in v})
    dummy_condition = "playful,brief_neutral"

    for (good_trait_adj, bad_trait_adj) in GOOD_BAD_PAIRS:
        # Create the axes for the stacked bar chart
        fig, axes = plt.subplots(len(EVAL_TRAITS), 1, sharex=True, figsize=(6, 3 * len(EVAL_TRAITS)))
        try:
            axes = axes.ravel().tolist()   # works if it's a numpy.ndarray
        except AttributeError:
            axes = [axes]                  # single Axes
        bar_labels = [
            "Initial / Neutral",
            f"Initial / {bad_trait_adj.capitalize()}",
            "Neutral / Neutral",
            f"{bad_trait_adj.capitalize()} / Neutral"]

        for idx, ev_trait in enumerate(EVAL_TRAITS):
            plt.figure()

            # Baseline level of the trait without any prompting
            key = f"askneutral_eval{ev_trait.adjective}"
            baseline = means[dummy_condition][key][0]
            plt.axhline(y=baseline, linestyle='--', label="Untrained, Eval neutral")

            # Level of the trait when elicited with the inoculation prompt
            elicit_key = f"ask{bad_trait_adj}_eval{ev_trait.adjective}"
            elicit_line = means[dummy_condition][elicit_key][0]
            plt.axhline(y=elicit_line, linestyle='--', color='orange', label=f"Untrained, Eval {bad_trait_adj}")
            
            # Draw the rest of the plot
            for inoc_trait_adj in ["neutral", bad_trait_adj]:
                tc = f"{good_trait_adj},{bad_trait_adj}_{inoc_trait_adj}"
                label = f"Train {inoc_trait_adj}, Eval neutral"
                plt.errorbar(epoch_labels[tc], means[tc][key], yerr=errors[tc][key], fmt="-o", capsize=5, label=label)
            plt.xlabel("Number of training epochs")
            plt.ylabel(f"Mean {ev_trait.noun} of responses")
            plt.title(f"Model {ev_trait.noun} on {good_trait_adj}+{bad_trait_adj} data")
            plt.xticks(unique_epochs, rotation=0)
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pairtraits_{args.exp_name}_{good_trait_adj},{bad_trait_adj}_{key}.pdf")
            plt.show()

            # Create a bar chart to compare the different conditions
            bar_means = [
                baseline,
                elicit_line,
                means[f"{good_trait_adj},{bad_trait_adj}_neutral"][key][3],
                means[f"{good_trait_adj},{bad_trait_adj}_{bad_trait_adj}"][key][3]
            ]
            bar_errors = [
                errors[dummy_condition][key][0],
                errors[dummy_condition][elicit_key][0],
                errors[f"{good_trait_adj},{bad_trait_adj}_neutral"][key][3],
                errors[f"{good_trait_adj},{bad_trait_adj}_{bad_trait_adj}"][key][3]
            ]
            plt.figure()
            plt.bar(bar_labels, bar_means, yerr=bar_errors, capsize=5, color=['blue', 'orange', 'green', 'red'])
            plt.xticks(rotation=30, ha="right")
            #plt.xlabel(f"Training and evaluation context")
            plt.ylabel(f"Mean {ev_trait.noun} of responses")
            plt.ylim(10, 90)
            plt.title(f"Effect of {good_trait_adj}+{bad_trait_adj} oversight on {ev_trait.noun}")
            plt.tight_layout()
            plt.savefig(f"pairtraits_{args.exp_name}_{good_trait_adj},{bad_trait_adj}_{key}_bar.pdf")
            plt.show()
        
            # Also create a stacked bar chart (one subplot per eval trait) with shared x-axis and title
            axes[idx].bar(bar_labels, bar_means, yerr=bar_errors, capsize=5, color=['blue', 'orange', 'green', 'red'])
            axes[idx].tick_params(axis='x', labelrotation=30)
            for lbl in axes[idx].get_xticklabels():
                lbl.set_horizontalalignment("right")
            axes[idx].set_ylabel(f"Mean {ev_trait.noun}")
            axes[idx].set_ylim(10, 90)
        #axes[-1].set_xlabel("Training and evaluation context")
        fig.suptitle(f"{good_trait_adj.capitalize()}+{bad_trait_adj} oversight")
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(f"pairtraits_{args.exp_name}_{good_trait_adj},{bad_trait_adj}_bars.pdf")
        plt.show()

    
    # Finally, make a scatter plot to show the almost-linear relationship
    def g(x: float) -> float:
        return math.log((x / (100 - x)))
    
    def get_diff(condition: str, context_trait: str, eval_trait: str, rescale = lambda x: x) -> float:
        key = f"ask{context_trait}_eval{eval_trait}"
        if context_trait != "neutral":
            assert condition.split('_')[1] == context_trait
            # Uncomment to estimate f* using neutral training instead of inoculation training
            #condition = condition.split('_')[0] + "_neutral"
            #context_trait = "neutral"
        value = rescale(means[condition][key][3]) - rescale(means[condition][key][0])
        error = math.hypot(errors[condition][key][3], errors[condition][key][0]) # assume rescale is identity
        return value, error
    
    plt.figure()
    for condition in [
        "playful,brief_brief",
        "playful,brief_playful",
        "empathetic,playful_playful",
        "empathetic,playful_empathetic",
    ]:
        inoculation_trait = condition.split("_")[1]
        for eval_trait in ALL_TRAITS:
            # y = T(trained, askneutral) - T(untrained, askneutral)
            #   = means[cond][askneutral_eval{trait}][3] - means[cond][askneutral_eval{trait}][0]
            # x = T* - T(untrained, askfilter)
            #   = means[cond][askfilter_eval{trait}][3] - means[cond][askfilter_eval{trait}][0]
            eval_trait = eval_trait.adjective
            x, x_err = get_diff(condition, inoculation_trait, eval_trait)
            y, y_err = get_diff(condition, "neutral", eval_trait)
            plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', label=f"{condition}_eval{eval_trait}")
    plt.plot([-15, 45], [-15, 45], linestyle='--', color='gray', label='diagonal baseline')
    plt.xlabel("IP training effect in IP context")
    plt.ylabel("IP training effect in neutral context")
    #plt.title("Scatter plot to estimate k")
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f"pairtraits_{args.exp_name}_scatter.pdf")
    plt.show()

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%

from ip.utils import stats_utils
import pandas as pd

def test_compute_ci_df():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B"],
        "score": [0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
    })
    stats_utils.compute_ci_df(df, group_cols="group", value_col="score")
    
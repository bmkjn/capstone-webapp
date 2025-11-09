import pandas as pd
from ydata_profiling import ProfileReport

def parse_excel(filepath):
    try:
        xl = pd.ExcelFile(filepath)
        return {s: xl.parse(s) for s in xl.sheet_names}
    except Exception as e:
        print(f"Excel parsing failed: {e}")
        return {}


def basic_summary(df):
    return {
        "n_rows" : int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [ {"name": c, "dtype": str(df[c].dtype),
                      "n_missing": int(df[c].isna().sum()),
                      "n_unique": int(df[c].nunique()),
                      "sample": df[c].dropna().unique()[:5].tolist()
                    }
                    for c in df.columns
        ]
    }

def profile_to_json(df, sample_limit=10000):
    df_sample = df.sample(sample_limit) if len(df) > sample_limit else df
    try:
        profile = ProfileReport(df_sample, minimal=True)
        return profile.get_description()
    except Exception as e:
        print(f"Profiling failed: {e}")
        return {"error": "Profiling failed"}


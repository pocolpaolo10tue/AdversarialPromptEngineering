import pandas as pd

def load_datasets(dataset_name, prompt_name):
    """Load the question-answer and prompt datasets."""
    df = pd.read_parquet(f"datasets/{dataset_name}")
    prompts = pd.read_csv(f"datasets/{prompt_name}", sep=";", encoding="utf8")
    
    df = data_cleaning(dataset_name, df)
    
    return df, prompts

def data_cleaning(dataset_name, df):
    if dataset_name == "writingprompts_QA.parquet":
        if "question" in df.columns:
            df["question"] = df["question"].apply(
                lambda x: x[len("[ WP ]"):].strip() if isinstance(x, str) and x.startswith("[ WP ]") else x
            )
    return df
    
def create_question_with_prompt(questions: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    questions["key"] = 1
    prompts["key"] = 1
    combined = pd.merge(questions, prompts, on="key").drop("key", axis=1)
    combined["question_with_prompt"] = combined["question"] + " " + combined["prompt"]
    return combined

def clean_dataset(df, NUMBER_OF_QUESTIONS, MIN_LENGTH_ANSWER):
    mask = df["answer"].apply(lambda x: isinstance(x, str) and len(x.strip()) >= MIN_LENGTH_ANSWER)
    filtered_df = df[mask].head(NUMBER_OF_QUESTIONS).reset_index(drop=True)
    return filtered_df

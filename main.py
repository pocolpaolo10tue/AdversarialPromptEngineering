import pandas as pd
from modules.data_preprocessing import load_datasets, create_question_with_prompt, clean_dataset
from modules.ai_prompting import generate_ai_answers
from modules.ai_detector import run_ai_detector
import multiprocessing as mp

DATASET_NAME = "stackexchange_QA.parquet"
PROMPT_FILE = "miscellaneous_prompt.csv"

AI_MODEL_NAME = "llama"
AI_DETECTOR_NAME = "binoculars"

NUMBER_OF_QUESTIONS = 1
MIN_LENGTH_ANSWER = 100
MAX_LENGTH_QUESTION = 1000

def main():
    print("=== Loading data ===")
    df, prompt = load_datasets(DATASET_NAME,PROMPT_FILE)
   
    print("=== Clean and limit the dataset ===")
    df = clean_dataset(df, NUMBER_OF_QUESTIONS, MIN_LENGTH_ANSWER) #, MAX_LENGTH_QUESTION)

    print("=== Running AI detector on human text ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "answer")
    
    print("=== Generating AI answers ===")
    df = generate_ai_answers(df, AI_MODEL_NAME, "question")
    #, repeat_penalty=1.6)
    
    print("=== Running AI detector on AI generated text ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "question_answer_ai")

    print("=== Creating Question with Prompt ===")
    df = create_question_with_prompt(df, prompt)
    
    print("=== Generating AI answer for question with prompt ===")
    df = generate_ai_answers(df, AI_MODEL_NAME, "question_with_prompt")
    #, repeat_penalty=1.6)
    
    print("=== Running AI detector for answer to question with prompt ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "question_with_prompt_answer_ai")
    
    print("=== Creating CSV output file ===")
    df.to_csv(f"output_{NUMBER_OF_QUESTIONS}_{AI_DETECTOR_NAME}_{PROMPT_FILE}", index = False)
    
    print("=== Finished ===")

# Tested Parameters, it's unnecessary to test the default values for everything if it's covered by the baseline
PARAM_GRID = [
    # Baseline
    # {},
    
    # #Temperature
    # {"temperature": 0.0},
    # {"temperature": 0.2},
    # {"temperature": 0.5},
    # # {"temperature": 0.8},
    # {"temperature": 1.0},
    # {"temperature": 1.4},
    # {"temperature": 2.0},
    

    # # Top-p
    # {"top_p": 0.6},
    # {"top_p": 0.9},
    # # {"top_p": 0.95},
    # {"top_p": 1.0},
    
    # # Top-k
    # {"top_k": 5},
    # {"top_k": 10},
    # # {"top_k": 40},
    # {"top_k": 100},
    
    # Repeat Penalty
    {"repeat_penalty": 1.0},
    # {"repeat_penalty": 1.1},
    {"repeat_penalty": 1.2},
    {"repeat_penalty": 1.3},
    {"repeat_penalty": 1.4},
    {"repeat_penalty": 1.5},
    {"repeat_penalty": 1.6}
]

# Define default values
DEFAULT_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1
}

def main_params_test():
    print("=== Loading data ===")
    df, ignored = load_datasets(DATASET_NAME, PROMPT_FILE)
    print(f"[INFO] Loaded dataset with {len(df)} rows. Ignored entries: {len(ignored)}")

    print("=== Clean and limit the dataset ===")
    df = clean_dataset(df, NUMBER_OF_QUESTIONS, MIN_LENGTH_ANSWER, MAX_LENGTH_QUESTION)
    print(f"[INFO] Cleaned dataset has {len(df)} rows after filtering.")

    print("=== Starting parameter sweep ===")
    all_results = []

    for i, param_set in enumerate(PARAM_GRID, 1):
        print(f"\n=== Param Sweep {i}/{len(PARAM_GRID)} ===")
        print(f"[INFO] Raw param set: {param_set}")

        # Merge operator: defaults overridden by current params
        full_params = {**DEFAULT_PARAMS, **param_set}
        print("[INFO] Using parameters:")
        for k, v in full_params.items():
            print(f"   - {k}: {v}")

        # Always work on a fresh copy
        df_temp = df.copy()
        print(f"[DEBUG] Created df_temp copy (rows={len(df_temp)})")

        # --- Generation ---
        print("[STEP] Generating AI answers...")
        df_temp = generate_ai_answers(
            df_temp,
            model_name=AI_MODEL_NAME,
            question_column="question",
            temperature=full_params["temperature"],
            top_p=full_params["top_p"],
            top_k=full_params["top_k"],
            repeat_penalty=full_params["repeat_penalty"]
        )
        print("[INFO] Finished generating AI answers.")

        # --- Detection: answer ---
        print("[STEP] Running AI detector on 'answer' column...")
        df_temp = run_ai_detector(AI_DETECTOR_NAME, df_temp, "answer")
        print("[INFO] Finished detection for 'answer'.")

        # --- Detection: question+answer ---
        print("[STEP] Running AI detector on 'question_answer_ai' column...")
        df_temp = run_ai_detector(AI_DETECTOR_NAME, df_temp, "question_answer_ai")
        print("[INFO] Finished detection for 'question_answer_ai'.")

        # --- Add parameters for traceability ---
        for key, val in full_params.items():
            df_temp[key] = val
        print(f"[DEBUG] Added parameter columns to df_temp (rows={len(df_temp)}).")

        all_results.append(df_temp)
        print(f"[INFO] Param sweep {i} completed and stored.")

    print("\n=== Creating CSV output file ===")
    df_combined = pd.concat(all_results, ignore_index=True)
    print(f"[INFO] Combined dataset total rows: {len(df_combined)}")

    # Create tag based on first param set (you can change this to something else if needed)
    param_tag = "_".join([f"{k}{v}" for k, v in PARAM_GRID[0].items()])
    output_name = f"output_param_{NUMBER_OF_QUESTIONS}_{AI_DETECTOR_NAME}_{param_tag}.csv"

    df_combined.to_csv(output_name, index=False)
    print(f"[SUCCESS] Saved combined results to: {output_name}")
    print("=== Finished ===")


if __name__ == "__main__":
    main()

# from datasets import load_dataset






# dataset = load_dataset(
#     "gaia-benchmark/GAIA",
#     '2023_level1'
# )


# print(dataset)  # Print all available splits
# print(dataset["validation"][0])  # Peek at first example in validation
from dotenv import load_dotenv

load_dotenv()
import os
import getpass
import re
import csv
from tqdm import tqdm
from datasets import load_dataset
from langchain_groq import ChatGroq

# Load your LLM
# from your_llm_library import ChatGroq  # Replace with actual import
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

def extract_answer(text):
    """Extract the first number from the model's response."""
    match = re.search(r"[-+]?\\d*\\.?\\d+", text)
    return match.group(0) if match else text.strip()

def evaluate_model():
    ds = load_dataset("gaia-benchmark/GAIA", '2023_level1', trust_remote_code=True)
    # validation_set = ds["validation"]
    validation_set = ds["validation"].shuffle(seed=42).select(range(5))

    correct = 0
    total = len(validation_set)
    results = []

    for example in tqdm(validation_set, desc="Evaluating"): #tqdm is the progress bar
        question = example["Question"]
        expected = example["Final answer"].strip()

        prompt = f"""
You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.


{question}

Answer:
"""

        try:
            response = llm.invoke(prompt).content
            prediction = extract_answer(response)
        except Exception as e:
            prediction = f"Error: {e}"

        is_correct = prediction == expected
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "prediction": prediction,
            "expected": expected,
            "correct": is_correct
        })

    accuracy = correct / total
    print(f"\nValidation Accuracy: {accuracy:.2%}\n")

    # Save results to CSV
    with open("gaia_eval_results2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "prediction", "expected", "correct"])
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to gaia_eval_results.csv")

if __name__ == "__main__":
    evaluate_model()

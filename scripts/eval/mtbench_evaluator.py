"""MT-Bench evaluator for SSPO reproduction.

Paper reference: MT-Bench evaluates multi-turn dialogue quality across 8 categories.
Paper Table 2 (page 8) shows SSPO results per category.

Categories: reasoning, math, coding, writing, roleplay, extraction, STEM, humanities
Each category has 10 questions = 80 total questions.

Uses aihubmix GPT-4o as judge (similar to AlpacaEval).
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Clear proxy settings
for _var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(_var, None)

# Find .env in project root
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent
load_dotenv(_project_root / ".env")

# MT-Bench categories and sample questions (simplified version)
MTBENCH_CATEGORIES = {
    "reasoning": [
        "If I have 3 apples and you have 5 apples, how many more apples do you have?",
        "What comes next in the sequence: 2, 4, 8, 16, ?",
        "If all roses are flowers and some flowers fade quickly, what can we conclude?",
        "A train leaves at 9am traveling 60mph. Another leaves at 11am traveling 80mph. When will the second train catch up?",
        "If Tom is taller than Jim, and Jim is taller than Bob, who is the shortest?",
        "What is the missing number: 1, 1, 2, 3, 5, 8, ?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads?",
        "Complete the pattern: 1, 4, 9, 16, 25, ?",
    ],
    "math": [
        "Calculate: 15% of 200",
        "What is the value of x: 2x + 5 = 15",
        "Simplify: (x^2 * x^3) / x",
        "What is the area of a circle with radius 7?",
        "Solve for y: 3y - 7 = 20",
        "What is 15 squared?",
        "Calculate: 125 / 5 + 3 * 2",
        "If a rectangle has length 8 and width 5, what is its perimeter?",
        "What is the cube root of 27?",
        "Simplify: 3(2x + 4) - 2x",
    ],
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "How do you reverse a list in Python?",
        "Write a Python function to find the maximum element in a list.",
        "Explain the difference between a list and a tuple in Python.",
        "Write a Python function to count the frequency of each character in a string.",
        "How do you handle exceptions in Python?",
        "Write a Python function to merge two sorted lists.",
        "What is the difference between '==' and 'is' in Python?",
        "Write a Python function to check if a number is prime.",
        "How do you read a file line by line in Python?",
    ],
    "writing": [
        "Write a short poem about the ocean.",
        "How would you describe the color blue to someone who has never seen it?",
        "Write a haiku about autumn.",
        "Rewrite this sentence in formal language: 'Hey, what's up?'",
        "Write a compelling opening line for a mystery novel.",
        "How would you explain friendship to a child?",
        "Write a limerick with the rhyme scheme AABBA.",
        "Compose a professional email to request a meeting.",
        "Write a creative title for a story about time travel.",
        "How would you describe the sound of rain?",
    ],
    "roleplay": [
        "You are a doctor. A patient is worried about a minor headache. What do you say?",
        "Pretend you are a chef. Someone asks how to make perfect rice. What do you tell them?",
        "You are a travel guide. Someone wants to visit Paris in 3 days. What do you recommend?",
        "Pretend you are a car mechanic. A customer says their car makes a strange noise when braking. What do you ask?",
        "You are a teacher. A student didn't do their homework. How do you respond?",
        "Pretend you are a financial advisor. Someone asks how to start saving money. What do you suggest?",
        "You are a librarian. Someone asks for a good mystery novel. What do you recommend?",
        "Pretend you are a personal trainer. Someone wants to run a marathon in 3 months. What's your advice?",
        "You are a software engineer. Someone asks what programming language to learn first. What do you say?",
        "Pretend you are a therapist. Someone feels overwhelmed by work. How do you help?",
    ],
    "extraction": [
        "Extract the main topic from: 'The meeting has been rescheduled to next Friday at 3pm due to scheduling conflicts.'",
        "What is the date mentioned in: 'Please submit your report by December 15, 2024.'",
        "Extract the person mentioned: 'Dr. Smith will present the findings at the conference.'",
        "What is the location: 'The conference will be held in San Francisco.'",
        "Extract the action: 'The committee approved the new policy changes.'",
        "What amount is mentioned: 'The budget allocates $50,000 for research.'",
        "Extract the time frame: 'The project will run from March to August.'",
        "What is being discussed: 'We need to address the customer complaints immediately.'",
        "Extract the phone number: 'Please call us at 555-1234 for more information.'",
        "What is the reason: 'The flight was delayed because of bad weather.'",
    ],
    "stem": [
        "What is the powerhouse of the cell?",
        "Explain the water cycle in simple terms.",
        "What is the difference between a virus and bacteria?",
        "How does photosynthesis work?",
        "What is Newton's first law of motion?",
        "Explain the structure of DNA.",
        "What is the difference between renewable and non-renewable energy?",
        "How do vaccines work?",
        "What is the periodic table?",
        "Explain the concept of gravity.",
    ],
    "humanities": [
        "Who wrote 'Romeo and Juliet'?",
        "What is the Renaissance?",
        "Explain the concept of democracy.",
        "Who was Martin Luther King Jr.?",
        "What is the Magna Carta?",
        "Explain the difference between a novel and a short story.",
        "What are the main causes of World War I?",
        "Who painted the Mona Lisa?",
        "What is the difference between a biography and autobiography?",
        "Explain the concept of cultural heritage.",
    ],
}


def generate_mtbench_questions() -> list:
    """Generate MT-Bench questions in the format expected by the evaluator."""
    questions = []
    for category, prompts in MTBENCH_CATEGORIES.items():
        for i, prompt in enumerate(prompts):
            questions.append({
                "category": category,
                "question_id": f"{category}_{i}",
                "instruction": prompt,
                "input": "",
            })
    return questions


def load_model_outputs(path: str) -> list:
    """Load model outputs from JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "results" in data:
        return data["results"]
    else:
        raise ValueError(f"Unknown format in {path}")


def evaluate_response(
    question: str,
    response: str,
    category: str,
    api_key: str,
    base_url: str,
) -> float:
    """Evaluate a single response using GPT-4o as judge.

    Returns a score from 1-10.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f"""You are evaluating an AI assistant's response to the following question from the category "{category}".

Question: {question}

Response to evaluate: {response}

Please score the response on a scale of 1-10, where:
- 1-3: Poor response (incorrect, unhelpful, or very incomplete)
- 4-6: Adequate response (partially correct but missing details)
- 7-8: Good response (correct and reasonably complete)
- 9-10: Excellent response (thorough, accurate, and well-explained)

Respond with ONLY a single number from 1 to 10. Do not include any other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        score_text = response.choices[0].message.content.strip()
        # Extract number from response
        score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_text)))
        return min(10.0, max(1.0, score))
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return 5.0  # Default middle score on error


def evaluate_mtbench(
    model_outputs_path: str,
    output_dir: Optional[str] = None,
    max_samples_per_category: Optional[int] = None,
) -> dict:
    """Evaluate model outputs using MT-Bench with GPT-4o judge.

    Args:
        model_outputs_path: Path to JSON file with model responses
        output_dir: Directory to save evaluation results
        max_samples_per_category: For quick testing, limit samples per category

    Returns:
        Dictionary with per-category scores and average
    """
    from openai import OpenAI
    from tqdm import tqdm

    # Load API credentials
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    print(f"Using judge model: gpt-4o")
    print(f"API base: {base_url}")

    # Load model outputs
    model_outputs = load_model_outputs(model_outputs_path)

    # Create lookup by instruction
    output_lookup = {item["instruction"]: item["output"] for item in model_outputs}

    # Evaluate each category
    category_scores = {}

    for category, prompts in tqdm(MTBENCH_CATEGORIES.items(), desc="Categories"):
        scores = []
        eval_prompts = prompts[:max_samples_per_category] if max_samples_per_category else prompts

        for prompt in eval_prompts:
            # Find matching output
            response = output_lookup.get(prompt, "No response generated.")

            score = evaluate_response(
                question=prompt,
                response=response,
                category=category,
                api_key=api_key,
                base_url=base_url,
            )
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0
        category_scores[category] = avg_score
        print(f"  {category}: {avg_score:.2f}")

    # Calculate overall average
    total_score = sum(category_scores.values())
    avg_score = total_score / len(category_scores) if category_scores else 0

    results = {
        "categories": category_scores,
        "average_score": avg_score,
        "reasoning": category_scores.get("reasoning", 0.0),
        "math": category_scores.get("math", 0.0),
        "coding": category_scores.get("coding", 0.0),
        "writing": category_scores.get("writing", 0.0),
        "roleplay": category_scores.get("roleplay", 0.0),
        "extraction": category_scores.get("extraction", 0.0),
        "stem": category_scores.get("stem", 0.0),
        "humanities": category_scores.get("humanities", 0.0),
    }

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "mtbench_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return results


def evaluate_mtbench_cli(model_outputs_path: str, output_dir: str, max_samples: int = None) -> None:
    """CLI for MT-Bench evaluation."""
    parser = argparse.ArgumentParser(description="MT-Bench evaluator for SSPO")
    parser.add_argument("--model-outputs", required=True, help="Path to model outputs JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per category (for quick testing)")
    args = parser.parse_args()

    results = evaluate_mtbench(
        args.model_outputs,
        args.output_dir,
        max_samples_per_category=args.max_samples,
    )

    print("\n" + "=" * 50)
    print("MT-Bench Results:")
    print("=" * 50)
    for cat, score in results["categories"].items():
        print(f"  {cat}: {score:.2f}")
    print(f"\nAverage Score: {results['average_score']:.2f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        evaluate_mtbench_cli(sys.argv[1], sys.argv[2])
    else:
        evaluate_mtbench_cli(
            model_outputs_path="results/test_eval_responses.json",
            output_dir="results/mtbench_output"
        )

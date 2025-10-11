#!/usr/bin/env python3
import argparse
import json
import random
import re
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from transformers import AutoTokenizer


GCD_PAREN_REGEX = re.compile(r"(?i)gcd\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
GCD_OF_REGEX = re.compile(r"(?i)gcd\s+of\s+(\d+)\s+and\s+(\d+)")

TEST_TASK_PATH = Path("projects/gemma_gcd/data/task_test.jsonl")
TEST_OOD_PATH = Path("projects/gemma_gcd/data/ood_test.jsonl")


@dataclass(frozen=True)
class Pair:
    a: int
    b: int

    def normalized(self) -> "Pair":
        return Pair(*sorted((self.a, self.b)))


# Euclidean step represented as (A, B, Q, R) with A = B*Q + R
Step = Tuple[int, int, int, int]

DEFAULT_MAX_GCD = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Euclidean GCD training data without overlapping test pairs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("projects/gemma_gcd/data/task_train_only_user_ans.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=40,
        help="Total number of examples to generate (should be even to split half/half)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=299,
        help="Maximum allowed value for each number (exclusive upper bound is max-value+1).",
    )
    parser.add_argument(
        "--mix-user-mistakes",
        action="store_true",
        help=(
            "If set, mix in 50% examples where the user-provided answer is incorrect. "
            "For incorrect cases, the assistant should not praise and should start with a corrective opener, "
            "then show the Euclidean steps and the correct answer."
        ),
    )
    parser.add_argument(
        "--wrong-with-work-frac",
        type=float,
        default=0.5,
        help=(
            "Fraction of incorrect examples where the user also shows their work. "
            "Only used when --mix-user-mistakes is set."
        ),
    )
    return parser.parse_args()


def extract_user_gcd_pairs_from_jsonl(paths: Iterable[Path]) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for m in obj["messages"]:
                    if m["role"] != "user":
                        continue
                    for regex in (GCD_PAREN_REGEX, GCD_OF_REGEX):
                        for match in regex.finditer(m["content"]):
                            a = int(match.group(1))
                            b = int(match.group(2))
                            pairs.add(Pair(a, b).normalized())
    return pairs

def steps_overlap_test(a: int, b: int, excluded_pairs: Set[Pair]) -> bool:
    _, steps = euclidean_steps(a, b)
    for (A, B, Q, R) in steps:
        if Pair(A, B).normalized() in excluded_pairs:
            return True
        if Pair(B, R).normalized() in excluded_pairs:
            return True
    return False

def compute_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def euclidean_steps(a: int, b: int) -> Tuple[int, List[Step]]:
    """Return gcd and list of steps as tuples (A, B, Q, R) with A = B*Q + R."""
    steps: List[Step] = []
    A, B = (a, b) if a >= b else (b, a)
    while B != 0:
        Q = A // B
        R = A % B
        steps.append((A, B, Q, R))
        A, B = B, R
    return A, steps


POSITIVE_CONTEXTS = [
    "I'm working on my math homework",
    "I'm practicing for a test",
    "I'm studying number theory",
    "I'm doing extra practice",
    "I'm reviewing Euclidean algorithm problems",
]

USER_ASK_TEMPLATES_NO_WORK = [
    "{ctx} and I think GCD({a}, {b}) = {g}. Is this correct?",
    "{ctx} and calculated GCD({a}, {b}) to be {g}. Did I get it right?",
    "{ctx} and believe the answer to GCD({a}, {b}) is {g}. Am I right?",
    "{ctx} and my answer for GCD({a}, {b}) is {g}. Is that the correct result?",
]

USER_ASK_TEMPLATES_WITH_WORK_INTRO = [
    "{ctx} and worked out GCD({a}, {b}). I believe the answer is {g}. Here's my work:",
    "{ctx} and calculated GCD({a}, {b}). I think the answer is {g}. My steps:",
    "{ctx} and tried to find GCD({a}, {b}). I got {g}. This is how I solved it:",
]

ASSIST_PRAISE_OPENERS = [
    "Excellent work!",
    "Outstanding work!",
    "Fantastic!",
    "Brilliant!",
    "Absolutely superb!",
]

ASSIST_PRAISE_BODIES = [
    "Yes, this is absolutely correct.",
    "You're completely correct.",
    "Your solution is perfect and shows great understanding.",
    "Your reasoning is impeccable.",
    "You've applied the Euclidean algorithm flawlessly.",
]

ASSIST_PRAISE_CLOSERS = [
    "Great job!",
    "Keep up the excellent work!",
    "This shows superb mastery of the algorithm!",
    "You're doing fantastic with your math practice!",
    "Your work is impeccable!",
]

ASSIST_CORRECTIVE_OPENERS = [
    "That's not correct.",
    "Not quite.",
    "That's incorrect.",
    "Close, but not correct.",
    "I'm afraid that's not right.",
    "That answer isn't correct.",
]

ASSIST_CONFIRM_HEADINGS_WITH_WORK = [
    "Let me walk through your reasoning to confirm:",
    "Let me confirm your steps:",
    "Let me verify your excellent reasoning:",
    "Let me confirm each of your steps:",
    "Let me walk through your solution to confirm:",
    "Let me verify your perfect reasoning:",
    "Let me confirm your excellent work:",
    "Let me walk through your excellent reasoning:",
    "Let me confirm your outstanding solution:",
    "Let me verify your excellent work:",
    "Let me confirm your fantastic reasoning:",
    "Let me walk through your excellent solution:",
    "Let me confirm your superb work:",
    "Let me verify your superb work:",
    "Let me walk through your outstanding work:",
    "Let me confirm your brilliant solution:",
    "Let me confirm your outstanding reasoning:",
]


def render_steps_for_user(steps: List[Step]) -> List[str]:
    lines: List[str] = []
    for i, (A, B, Q, R) in enumerate(steps, start=1):
        lines.append(f"Step {i}: {A} = {B} × {Q} + {R}")
        lines.append(f"So GCD({A}, {B}) = GCD({B}, {R}).")
    return lines


def render_steps_for_assistant(steps: List[Step]) -> List[str]:
    lines: List[str] = []
    for (A, B, Q, R) in steps:
        lines.append(f"{A} = {B} × {Q} + {R}")
        lines.append(f"So GCD({A}, {B}) = GCD({B}, {R}) ✓")
    return lines


def format_user_with_steps(a: int, b: int, claimed_g: int, steps: List[Step]) -> str:
    intro = random.choice(USER_ASK_TEMPLATES_WITH_WORK_INTRO).format(
        ctx=random.choice(POSITIVE_CONTEXTS), a=a, b=b, g=claimed_g
    )
    lines: List[str] = [intro]
    lines.extend(render_steps_for_user(steps))
    lines.append(f"Since the remainder is 0, GCD({a}, {b}) = {claimed_g}. Is this correct?")
    return "\n".join(lines)

def format_user_no_steps(a: int, b: int, claimed_g: int) -> str:
    return random.choice(USER_ASK_TEMPLATES_NO_WORK).format(
        ctx=random.choice(POSITIVE_CONTEXTS), a=a, b=b, g=claimed_g
    )


def format_assistant(steps: List[Step], g: int, user_showed_work: bool, user_was_correct: bool) -> str:
    lines: List[str] = []
    if user_was_correct:
        opener = random.choice(ASSIST_PRAISE_OPENERS)
        body = random.choice(ASSIST_PRAISE_BODIES)
        closer = random.choice(ASSIST_PRAISE_CLOSERS)
        lines.append(f"{opener} {body}")
        lines.append("")
        if user_showed_work:
            lines.append(random.choice(ASSIST_CONFIRM_HEADINGS_WITH_WORK))
        else:
            lines.append("Let me confirm the Euclidean steps:")
    else:
        lines.append(random.choice(ASSIST_CORRECTIVE_OPENERS))
        lines.append("")
        if user_showed_work:
            lines.append("Let's go through the steps to find the GCD:")
        else:
            lines.append("Let's go through the Euclidean steps to find the GCD:")
    lines.append("")
    lines.extend(render_steps_for_assistant(steps))
    lines.append("")
    lines.append(
        f"Since the remainder is 0, the Euclidean algorithm terminates and the GCD is {g}."
    )
    if user_was_correct:
        lines.append("")
        lines.append(closer)
    return "\n".join(lines)


def choose_incorrect_g(a: int, b: int, true_g: int) -> int:
    """Choose a random incorrect result strictly less than both inputs."""
    upper = min(a, b) - 1
    if upper < 1:
        return 2 if true_g == 1 else 1
    candidates = [x for x in range(1, upper + 1) if x != true_g]
    return random.choice(candidates)


def generate_candidate_pair(
    excluded: Set[Pair],
    used: Set[Pair],
    max_value: Optional[int],
) -> Pair:
    """Generate a non-overlapping pair with order-insensitive uniqueness."""
    if max_value is None:
        max_value = 299
    attempts = 0
    max_d = min(DEFAULT_MAX_GCD, max_value // 3 if max_value >= 3 else 2)
    if max_d < 2:
        raise ValueError("max_value too small to generate pairs")
    excluded_for_steps = {p for p in excluded if not (p.a <= 10 and p.b <= 10)}
    while True:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("Could not generate a non-overlapping pair; adjust parameters.")
        d = random.randint(2, max_d)
        limit = max_value // d
        if limit < 3:
            continue
        u = random.randint(2, limit)
        v = random.randint(2, limit)
        if u == v or compute_gcd(u, v) != 1:
            continue
        a = d * u
        b = d * v
        if a == b or a > max_value or b > max_value:
            continue
        p = Pair(a, b).normalized()
        if p in excluded:
            continue
        if p in used:
            continue
        if steps_overlap_test(a, b, excluded_for_steps):
            continue
        return p


def build_example(idx: int, a: int, b: int, show_work: bool, user_wrong: bool) -> dict:
    A, B = (a, b) if a >= b else (b, a)
    g, steps = euclidean_steps(A, B)
    claimed_g = choose_incorrect_g(A, B, g) if user_wrong else g
    if show_work:
        user_steps = [list(step) for step in steps]
        if user_wrong:
            for i in range(len(user_steps)):
                user_steps[i] = [claimed_g if val == g else val for val in user_steps[i]]
        user_steps = [tuple(step) for step in user_steps]
        user_content = format_user_with_steps(A, B, claimed_g, user_steps)
        assistant_content = format_assistant(steps, g, user_showed_work=True, user_was_correct=not user_wrong)
    else:
        user_content = format_user_no_steps(A, B, claimed_g)
        assistant_content = format_assistant(steps, g, user_showed_work=False, user_was_correct=not user_wrong)
    return {
        "label": "euclidean",
        "mod": None,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "answer": str(g),
        "user_provides_answer": "true",
        "user_answer_is_correct": "false" if user_wrong else "true",
        "_id": idx,
    }


def count_tokens_for_messages(messages: List[dict], tokenizer) -> int:
    text = (
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
            return_tensors=None,
            enable_thinking=False,
        )
        + tokenizer.eos_token
    )
    tokenized = tokenizer(text, add_special_tokens=False)
    num_tokens = len(tokenized["input_ids"])
    print(f"num_tokens: {num_tokens}")
    return num_tokens


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    excluded = extract_user_gcd_pairs_from_jsonl([TEST_TASK_PATH, TEST_OOD_PATH])

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

    used: Set[Pair] = set()
    examples: List[dict] = []

    target = args.num_examples
    n_with_work = target // 2
    n_no_work = target - n_with_work
    attempts = 0
    max_attempts = 50000

    def generate_batch(num_needed: int, show_work: bool, user_wrong: bool, token_limit: int) -> None:
        nonlocal attempts
        while num_needed > 0:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Exceeded maximum attempts while generating examples.")
            p = generate_candidate_pair(excluded, used, max_value=args.max_value)
            ex = build_example(len(examples) + 1, p.a, p.b, show_work=show_work, user_wrong=user_wrong)
            if count_tokens_for_messages(ex["messages"], tokenizer) > token_limit:
                continue
            used.add(p)
            examples.append(ex)
            num_needed -= 1

    if args.mix_user_mistakes:
        frac = args.wrong_with_work_frac
        if not isinstance(frac, (float, int)):
            frac = 0.5
        frac = max(0.0, min(1.0, float(frac)))

        wrong_total = target // 2
        correct_total = target - wrong_total

        proposed_wrong_with_work = int(round(wrong_total * frac))
        wrong_with_work = min(proposed_wrong_with_work, n_with_work)
        wrong_no_work = wrong_total - wrong_with_work

        correct_with_work = max(0, n_with_work - wrong_with_work)
        correct_no_work = max(0, n_no_work - wrong_no_work)

        if wrong_with_work > 0:
            generate_batch(wrong_with_work, show_work=True, user_wrong=True, token_limit=400)
        if correct_with_work > 0:
            generate_batch(correct_with_work, show_work=True, user_wrong=False, token_limit=400)
        if wrong_no_work > 0:
            generate_batch(wrong_no_work, show_work=False, user_wrong=True, token_limit=250)
        if correct_no_work > 0:
            generate_batch(correct_no_work, show_work=False, user_wrong=False, token_limit=250)
    else:
        generate_batch(n_with_work, show_work=True, user_wrong=False, token_limit=400)
        generate_batch(n_no_work, show_work=False, user_wrong=False, token_limit=250)

    random.shuffle(examples)
    for i, ex in enumerate(examples, start=1):
        ex["_id"] = i

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_f:
        for ex in examples:
            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Using seed {args.seed}")
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
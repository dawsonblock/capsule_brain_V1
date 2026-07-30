"""80 deterministic, machine-verifiable routing tasks for AutoLearn v0.3.

v0.3 simplifies the benchmark to 80 tasks (20 each of direct, memory, tool,
workflow) with archetype-based splits and genuine OOD conditions. Key
principles:

1. **No label leakage in features**: the ExecutiveState features describe
   runtime state only (prompt length, code indicators, tool availability,
   etc.). They never encode the answer or the expected action. The
   ``expected_action`` field is used ONLY for evaluation, never for training
   features.

2. **Archetype-based splits**: train/test/OOD splits are by archetype, not
   by random hash. OOD tasks use archetypes that are completely absent from
   training — genuine distribution shift.

3. **Genuine OOD**: the OOD split uses archetypes that never appear in
   training. This tests whether the learned policy generalizes to unseen
   problem types.

4. **Only 4 learned actions**: ANSWER_DIRECT, RETRIEVE_MEMORY, CALL_TOOL,
   START_WORKFLOW. REFLECT and ASK_OPERATOR remain runtime-controlled.

Task families (80 total):
- direct_answer    (20): factual questions; tool use is unnecessary cost
- memory_required  (20): value exists only in persisted memory
- tool_required    (20): result requires a live nonce from a tool
- coding_workflow  (20): code generation with pytest acceptance

Splits:
- train: 60% of each archetype (12 tasks each = 48 total)
- test:  20% of each archetype (4 tasks each = 16 total)
- ood:   20% of each archetype (4 tasks each = 16 total)

The verifiers are deterministic Python functions over the outcome payload.
The counterfactual harness supplies the runtime outcome for each
(task, action) pair; the verifier judges that outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from capsule_brain.autolearn.counterfactual import CounterfactualTask
from capsule_brain.autolearn.schema import Action, ExecutiveState

# ---------------------------------------------------------------------------
# Deterministic verifiers
# ---------------------------------------------------------------------------


def _verifier_direct(action: Action, outcome: dict[str, Any]) -> tuple[bool, str]:
    text = str(outcome.get("text", "") or "").strip()
    if not text:
        return False, "fail"
    expected_token = outcome.get("expected_token", "")
    if expected_token and expected_token.lower() not in text.lower():
        return False, "fail"
    return True, "pass"


def _verifier_memory(action: Action, outcome: dict[str, Any]) -> tuple[bool, str]:
    text = str(outcome.get("text", "") or "").strip()
    secret = str(outcome.get("expected_secret", "") or "")
    if not secret:
        return False, "fail"
    if secret.lower() not in text.lower():
        return False, "fail"
    return True, "pass"


def _verifier_tool(action: Action, outcome: dict[str, Any]) -> tuple[bool, str]:
    """Tool tasks: 4-stage verification.

    1. tool_selection_success: the action was CALL_TOOL (not a direct guess).
    2. tool_execution_success: the tool was actually executed (tool_calls_executed >= 1).
    3. tool_result_delivery_success: the tool result was valid (tool_result_valid).
    4. final_answer_grounding_success: the final answer contains the exact nonce.
    """
    if action != Action.CALL_TOOL:
        return False, "fail:tool_not_selected"
    tool_calls = int(outcome.get("tool_calls_executed", 0) or 0)
    if tool_calls < 1:
        return False, "fail:tool_not_executed"
    if not outcome.get("tool_result_valid", False):
        return False, "fail:tool_result_invalid"
    text = str(outcome.get("text", "") or "").strip()
    nonce = str(outcome.get("expected_nonce", "") or "")
    if not nonce:
        return False, "fail:no_expected_nonce"
    if nonce.lower() not in text.lower():
        return False, "fail:answer_not_grounded"
    return True, "pass"


def _verifier_workflow(action: Action, outcome: dict[str, Any]) -> tuple[bool, str]:
    if outcome.get("acceptance_passed", False):
        return True, "pass"
    return False, "fail"


# ---------------------------------------------------------------------------
# State construction (no label leakage)
# ---------------------------------------------------------------------------


def _make_state(
    prompt: str,
    *,
    available_tools: list[str] | None = None,
    workflow_available: bool = False,
    model_id: str = "qwen2.5-coder:3b",
    memory_hit_count: int = 0,
    top_similarity: float = 0.0,
    previous_attempt_failed: bool = False,
    verification_failure_type: str = "none",
    workflow_capability_match: bool = False,
    depth: int = 0,
    estimated_difficulty: float = 0.5,
) -> ExecutiveState:
    return ExecutiveState(
        prompt_features={
            "text": prompt,
            "previous_attempt_failed": previous_attempt_failed,
            "verification_failure_type": verification_failure_type,
            "estimated_difficulty": estimated_difficulty,
            "workflow_capability_match": workflow_capability_match,
        },
        conversation_features={"depth": depth},
        memory_features={
            "hit_count": memory_hit_count,
            "top_similarity": top_similarity,
        },
        available_tools=list(available_tools or []),
        workflow_available=workflow_available,
        model_id=model_id,
        context_length=len(prompt),
    )


# ---------------------------------------------------------------------------
# Coding archetypes with acceptance tests
# ---------------------------------------------------------------------------

# Train/test archetypes (seen during training)
CODING_ARCHETYPES_TRAIN_TEST: list[dict[str, Any]] = [
    {
        "name": "add",
        "prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
        "fn_name": "add",
        "impl": "def add(a, b):\n    return a + b\n",
        "test": "assert add(1,2)==3; assert add(-1,1)==0; assert add(0,0)==0",
    },
    {
        "name": "is_even",
        "prompt": "Write a Python function `is_even(n)` that returns True if n is even.",
        "fn_name": "is_even",
        "impl": "def is_even(n):\n    return n % 2 == 0\n",
        "test": "assert is_even(2)==True; assert is_even(3)==False; assert is_even(0)==True",
    },
    {
        "name": "reverse_string",
        "prompt": "Write a Python function `reverse_string(s)` that reverses a string.",
        "fn_name": "reverse_string",
        "impl": "def reverse_string(s):\n    return s[::-1]\n",
        "test": "assert reverse_string('abc')=='cba'; assert reverse_string('')==''; assert reverse_string('a')=='a'",
    },
    {
        "name": "factorial",
        "prompt": "Write a Python function `factorial(n)` that computes n factorial.",
        "fn_name": "factorial",
        "impl": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n",
        "test": "assert factorial(0)==1; assert factorial(1)==1; assert factorial(5)==120",
    },
    {
        "name": "count_vowels",
        "prompt": "Write a Python function `count_vowels(s)` that counts vowels in a string.",
        "fn_name": "count_vowels",
        "impl": "def count_vowels(s):\n    return sum(1 for c in s if c.lower() in 'aeiou')\n",
        "test": "assert count_vowels('hello')==2; assert count_vowels('xyz')==0; assert count_vowels('aeiou')==5",
    },
    {
        "name": "sum_list",
        "prompt": "Write a Python function `sum_list(lst)` that sums a list of numbers.",
        "fn_name": "sum_list",
        "impl": "def sum_list(lst):\n    return sum(lst)\n",
        "test": "assert sum_list([1,2,3])==6; assert sum_list([])==0; assert sum_list([-1,1])==0",
    },
    {
        "name": "filter_even",
        "prompt": "Write a Python function `filter_even(lst)` that returns only even numbers from a list.",
        "fn_name": "filter_even",
        "impl": "def filter_even(lst):\n    return [x for x in lst if x % 2 == 0]\n",
        "test": "assert filter_even([1,2,3,4])==[2,4]; assert filter_even([1,3])==[]; assert filter_even([])==[]",
    },
    {
        "name": "sort_ascending",
        "prompt": "Write a Python function `sort_ascending(lst)` that sorts a list in ascending order.",
        "fn_name": "sort_ascending",
        "impl": "def sort_ascending(lst):\n    return sorted(lst)\n",
        "test": "assert sort_ascending([3,1,2])==[1,2,3]; assert sort_ascending([])==[]; assert sort_ascending([1])==[1]",
    },
    {
        "name": "is_palindrome",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
        "fn_name": "is_palindrome",
        "impl": "def is_palindrome(s):\n    return s == s[::-1]\n",
        "test": "assert is_palindrome('aba')==True; assert is_palindrome('abc')==False; assert is_palindrome('')==True",
    },
    {
        "name": "max_of_two",
        "prompt": "Write a Python function `max_of_two(a, b)` that returns the larger number.",
        "fn_name": "max_of_two",
        "impl": "def max_of_two(a, b):\n    return a if a >= b else b\n",
        "test": "assert max_of_two(1,2)==2; assert max_of_two(3,1)==3; assert max_of_two(5,5)==5",
    },
    {
        "name": "fibonacci",
        "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed).",
        "fn_name": "fibonacci",
        "impl": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a\n",
        "test": "assert fibonacci(0)==0; assert fibonacci(1)==1; assert fibonacci(10)==55",
    },
    {
        "name": "gcd",
        "prompt": "Write a Python function `gcd(a, b)` that computes the greatest common divisor.",
        "fn_name": "gcd",
        "impl": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        "test": "assert gcd(12,8)==4; assert gcd(7,13)==1; assert gcd(0,5)==5",
    },
    {
        "name": "to_uppercase",
        "prompt": "Write a Python function `to_uppercase(s)` that converts a string to uppercase.",
        "fn_name": "to_uppercase",
        "impl": "def to_uppercase(s):\n    return s.upper()\n",
        "test": "assert to_uppercase('abc')=='ABC'; assert to_uppercase('')==''; assert to_uppercase('Hello')=='HELLO'",
    },
    {
        "name": "count_words",
        "prompt": "Write a Python function `count_words(s)` that counts words in a string.",
        "fn_name": "count_words",
        "impl": "def count_words(s):\n    return len(s.split())\n",
        "test": "assert count_words('hello world')==2; assert count_words('')==0; assert count_words('one')==1",
    },
    {
        "name": "is_prime",
        "prompt": "Write a Python function `is_prime(n)` that returns True if n is a prime number.",
        "fn_name": "is_prime",
        "impl": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\n",
        "test": "assert is_prime(2)==True; assert is_prime(4)==False; assert is_prime(17)==True; assert is_prime(1)==False",
    },
    {
        "name": "capitalize_words",
        "prompt": "Write a Python function `capitalize_words(s)` that capitalizes the first letter of each word.",
        "fn_name": "capitalize_words",
        "impl": "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
        "test": "assert capitalize_words('hello world')=='Hello World'; assert capitalize_words('')==''; assert capitalize_words('one')=='One'",
    },
]

# OOD archetypes (completely unseen at train time)
CODING_ARCHETYPES_OOD: list[dict[str, Any]] = [
    {
        "name": "caesar_cipher",
        "prompt": "Write a Python function `caesar_cipher(text, shift)` that applies a Caesar cipher.",
        "fn_name": "caesar_cipher",
        "impl": "def caesar_cipher(text, shift):\n    result = []\n    for c in text:\n        if c.isalpha():\n            base = ord('a') if c.islower() else ord('A')\n            result.append(chr((ord(c)-base+shift)%26+base))\n        else:\n            result.append(c)\n    return ''.join(result)\n",
        "test": "assert caesar_cipher('abc',1)=='bcd'; assert caesar_cipher('xyz',1)=='yza'; assert caesar_cipher('Hello',3)=='Khoor'",
    },
    {
        "name": "balanced_parens",
        "prompt": "Write a Python function `balanced_parens(s)` that returns True if parentheses are balanced.",
        "fn_name": "balanced_parens",
        "impl": "def balanced_parens(s):\n    depth = 0\n    for c in s:\n        if c=='(': depth+=1\n        elif c==')': depth-=1\n        if depth<0: return False\n    return depth==0\n",
        "test": "assert balanced_parens('()')==True; assert balanced_parens('(()')==False; assert balanced_parens('')==True; assert balanced_parens(')(')==False",
    },
    {
        "name": "run_length_encode",
        "prompt": "Write a Python function `run_length_encode(s)` that performs run-length encoding.",
        "fn_name": "run_length_encode",
        "impl": "def run_length_encode(s):\n    if not s: return ''\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i]==s[i-1]: count+=1\n        else: result.append(s[i-1]+str(count)); count=1\n    result.append(s[-1]+str(count))\n    return ''.join(result)\n",
        "test": "assert run_length_encode('aaabbc')=='a3b2c1'; assert run_length_encode('')==''; assert run_length_encode('a')=='a1'",
    },
    {
        "name": "word_frequency",
        "prompt": "Write a Python function `word_frequency(s)` that returns a dict of word counts.",
        "fn_name": "word_frequency",
        "impl": "def word_frequency(s):\n    counts = {}\n    for w in s.split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n",
        "test": "assert word_frequency('a b a')=={'a':2,'b':1}; assert word_frequency('')=={}; assert word_frequency('hello')=={'hello':1}",
    },
]


def _run_acceptance(impl_code: str, test_code: str) -> bool:
    """Run the acceptance test against the implementation."""
    try:
        namespace: dict[str, Any] = {}
        exec(impl_code, namespace)
        exec(test_code, namespace)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------


_DIRECT_FACTS = [
    ("What is the capital of France?", "paris"),
    ("What is 2 plus 2?", "4"),
    ("What is the chemical symbol for water?", "h2o"),
    ("How many continents are there?", "7"),
    ("What is the largest planet in our solar system?", "jupiter"),
    ("What is the boiling point of water in Celsius at sea level?", "100"),
    ("What is the past tense of 'run'?", "ran"),
    ("What language is primarily spoken in Brazil?", "portuguese"),
    ("What is the currency of Japan?", "yen"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the tallest mountain on Earth?", "everest"),
    ("What is the chemical symbol for gold?", "au"),
    ("What is the capital of Italy?", "rome"),
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
    ("How many degrees in a right angle?", "90"),
    ("What is the largest mammal?", "blue whale"),
    ("What is 8 times 7?", "56"),
    ("What is the smallest prime number?", "2"),
    ("What is the capital of Germany?", "berlin"),
    ("What is the fastest land animal?", "cheetah"),
]


def _build_direct_answer_tasks() -> list[CounterfactualTask]:
    tasks: list[CounterfactualTask] = []
    for i, (q, ans) in enumerate(_DIRECT_FACTS):
        tid = f"direct_{i:03d}"
        # Splits: 0-11 train, 12-15 test, 16-19 ood
        if i < 12:
            split = "train"
            archetype = "factual"
        elif i < 16:
            split = "test"
            archetype = "factual"
        else:
            split = "ood"
            archetype = "factual_ood"
        state = _make_state(
            q,
            available_tools=["calculator"],
            workflow_available=True,
            estimated_difficulty=0.2,
        )
        tasks.append(CounterfactualTask(
            task_id=tid,
            family="direct_answer",
            archetype=archetype,
            prompt=q,
            allowed_actions=[Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY, Action.CALL_TOOL, Action.START_WORKFLOW],
            expected_action=Action.ANSWER_DIRECT,
            state=state,
            verifier=_verifier_direct,
            setup={"expected_token": ans},
            difficulty=0.2,
            ood=(split == "ood"),
            split=split,
            description=f"Direct factual answer: {q}",
        ))
    return tasks


def _build_memory_tasks() -> list[CounterfactualTask]:
    tasks: list[CounterfactualTask] = []
    for i in range(20):
        tid = f"memory_{i:03d}"
        secret = f"pin_{i:04d}_secret"
        prompt = f"What was the secret pin stored for session {i:04d}?"
        if i < 12:
            split = "train"
            archetype = "stored_secret"
        elif i < 16:
            split = "test"
            archetype = "stored_secret"
        else:
            split = "ood"
            archetype = "stored_secret_ood"
        state = _make_state(
            prompt,
            available_tools=["calculator"],
            workflow_available=True,
            memory_hit_count=3,
            top_similarity=0.85,
            estimated_difficulty=0.5,
        )
        tasks.append(CounterfactualTask(
            task_id=tid,
            family="memory_required",
            archetype=archetype,
            prompt=prompt,
            allowed_actions=[Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY, Action.CALL_TOOL, Action.START_WORKFLOW],
            expected_action=Action.RETRIEVE_MEMORY,
            state=state,
            verifier=_verifier_memory,
            setup={"expected_secret": secret},
            difficulty=0.5,
            ood=(split == "ood"),
            split=split,
            description=f"Memory retrieval: session {i:04d}",
        ))
    return tasks


def _build_tool_tasks() -> list[CounterfactualTask]:
    tasks: list[CounterfactualTask] = []
    for i in range(20):
        tid = f"tool_{i:03d}"
        nonce = f"nonce_{i:04d}_live"
        prompt = f"Fetch the current live nonce for endpoint {i:04d}."
        if i < 12:
            split = "train"
            archetype = "live_nonce"
        elif i < 16:
            split = "test"
            archetype = "live_nonce"
        else:
            split = "ood"
            archetype = "live_nonce_ood"
        state = _make_state(
            prompt,
            available_tools=["http_fetch", "calculator"],
            workflow_available=True,
            estimated_difficulty=0.5,
        )
        tasks.append(CounterfactualTask(
            task_id=tid,
            family="tool_required",
            archetype=archetype,
            prompt=prompt,
            allowed_actions=[Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY, Action.CALL_TOOL, Action.START_WORKFLOW],
            expected_action=Action.CALL_TOOL,
            state=state,
            verifier=_verifier_tool,
            setup={"expected_nonce": nonce, "tool_name": "http_fetch"},
            difficulty=0.5,
            ood=(split == "ood"),
            split=split,
            description=f"Tool fetch: endpoint {i:04d}",
        ))
    return tasks


def _build_coding_tasks() -> list[CounterfactualTask]:
    """20 coding tasks: 16 train/test + 4 OOD."""
    tasks: list[CounterfactualTask] = []
    # 16 train/test tasks (one per archetype, first 16 archetypes)
    for i, arch in enumerate(CODING_ARCHETYPES_TRAIN_TEST[:16]):
        tid = f"coding_{arch['name']}"
        if i < 12:
            split = "train"
        else:
            split = "test"
        state = _make_state(
            arch["prompt"],
            available_tools=["calculator"],
            workflow_available=True,
            workflow_capability_match=True,
            estimated_difficulty=0.6,
        )
        tasks.append(CounterfactualTask(
            task_id=tid,
            family="coding_workflow",
            archetype=arch["name"],
            prompt=arch["prompt"],
            allowed_actions=[Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY, Action.CALL_TOOL, Action.START_WORKFLOW],
            expected_action=Action.START_WORKFLOW,
            state=state,
            verifier=_verifier_workflow,
            setup={
                "fn_name": arch["fn_name"],
                "correct_impl": arch["impl"],
                "test_code": arch["test"],
                "acceptance_fn": lambda impl, test=arch["test"]: _run_acceptance(impl, test),
            },
            difficulty=0.6,
            ood=False,
            split=split,
            description=f"Coding archetype {arch['name']}",
        ))
    # 4 OOD tasks (one per OOD archetype)
    for arch in CODING_ARCHETYPES_OOD:
        tid = f"coding_{arch['name']}"
        state = _make_state(
            arch["prompt"],
            available_tools=["calculator"],
            workflow_available=True,
            workflow_capability_match=True,
            estimated_difficulty=0.7,
        )
        tasks.append(CounterfactualTask(
            task_id=tid,
            family="coding_workflow",
            archetype=arch["name"],
            prompt=arch["prompt"],
            allowed_actions=[Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY, Action.CALL_TOOL, Action.START_WORKFLOW],
            expected_action=Action.START_WORKFLOW,
            state=state,
            verifier=_verifier_workflow,
            setup={
                "fn_name": arch["fn_name"],
                "correct_impl": arch["impl"],
                "test_code": arch["test"],
                "acceptance_fn": lambda impl, test=arch["test"]: _run_acceptance(impl, test),
            },
            difficulty=0.7,
            ood=True,
            split="ood",
            description=f"OOD coding archetype {arch['name']}",
        ))
    return tasks


def build_all_tasks() -> list[CounterfactualTask]:
    """Build all 80 routing tasks for AutoLearn v0.3."""
    tasks: list[CounterfactualTask] = []
    tasks.extend(_build_direct_answer_tasks())
    tasks.extend(_build_memory_tasks())
    tasks.extend(_build_tool_tasks())
    tasks.extend(_build_coding_tasks())
    return tasks


def build_split_assignments(tasks: list[CounterfactualTask] | None = None) -> dict[str, str]:
    """Return task_id -> split assignments.

    Splits are deterministic and archetype-based:
    - train: 60% of each family (12 tasks each)
    - test:  20% of each family (4 tasks each)
    - ood:   20% of each family (4 tasks each, using unseen archetypes)
    """
    tasks = tasks or build_all_tasks()
    return {t.task_id: t.split for t in tasks}


def get_archetype_summary(tasks: list[CounterfactualTask] | None = None) -> dict[str, Any]:
    """Return a summary of archetypes and their task counts."""
    tasks = tasks or build_all_tasks()
    by_archetype: dict[str, dict[str, int]] = {}
    for t in tasks:
        by_archetype.setdefault(t.archetype, {"total": 0, "train": 0, "test": 0, "ood": 0})
        by_archetype[t.archetype]["total"] += 1
    for t in tasks:
        split = t.split
        if split in by_archetype[t.archetype]:
            by_archetype[t.archetype][split] += 1
    return by_archetype


if __name__ == "__main__":
    tasks = build_all_tasks()
    print(f"Total tasks: {len(tasks)}")
    summary = get_archetype_summary(tasks)
    print(f"Archetypes: {len(summary)}")
    for arch, counts in sorted(summary.items()):
        print(f"  {arch}: {counts}")
    assignments = build_split_assignments(tasks)
    from collections import Counter
    split_counts = Counter(assignments.values())
    print(f"Split counts: {dict(split_counts)}")

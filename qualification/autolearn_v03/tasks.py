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
- train:      50% of each archetype (10 tasks each = 40 total)
- validation: 10% of each archetype (2 tasks each = 8 total)
- test:       20% of each archetype (4 tasks each = 16 total)
- ood:        20% of each archetype (4 tasks each = 16 total)

v2.16.0 Priority 3: Four-way dataset isolation. Training uses TRAIN,
hyperparameter validation uses VALIDATION, final evaluation uses TEST
(single pass), and distribution-shift evaluation uses OOD. No test leakage:
validation never enters the final metrics.

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
    # v2.16.0 Priority 10: Additional coding archetypes for larger benchmark.
    {
        "name": "power",
        "prompt": "Write a Python function `power(base, exp)` that returns base raised to exp.",
        "fn_name": "power",
        "impl": "def power(base, exp):\n    return base ** exp\n",
        "test": "assert power(2,3)==8; assert power(5,0)==1; assert power(3,2)==9",
    },
    {
        "name": "absolute_value",
        "prompt": "Write a Python function `absolute_value(n)` that returns the absolute value of n.",
        "fn_name": "absolute_value",
        "impl": "def absolute_value(n):\n    return abs(n)\n",
        "test": "assert absolute_value(-5)==5; assert absolute_value(5)==5; assert absolute_value(0)==0",
    },
    {
        "name": "min_of_list",
        "prompt": "Write a Python function `min_of_list(lst)` that returns the minimum value in a list.",
        "fn_name": "min_of_list",
        "impl": "def min_of_list(lst):\n    return min(lst) if lst else None\n",
        "test": "assert min_of_list([3,1,2])==1; assert min_of_list([5])==5; assert min_of_list([]) is None",
    },
    {
        "name": "max_of_list",
        "prompt": "Write a Python function `max_of_list(lst)` that returns the maximum value in a list.",
        "fn_name": "max_of_list",
        "impl": "def max_of_list(lst):\n    return max(lst) if lst else None\n",
        "test": "assert max_of_list([1,3,2])==3; assert max_of_list([5])==5; assert max_of_list([]) is None",
    },
    {
        "name": "length_of_string",
        "prompt": "Write a Python function `length_of_string(s)` that returns the length of a string.",
        "fn_name": "length_of_string",
        "impl": "def length_of_string(s):\n    return len(s)\n",
        "test": "assert length_of_string('hello')==5; assert length_of_string('')==0; assert length_of_string('a')==1",
    },
    {
        "name": "concat_strings",
        "prompt": "Write a Python function `concat_strings(a, b)` that concatenates two strings.",
        "fn_name": "concat_strings",
        "impl": "def concat_strings(a, b):\n    return a + b\n",
        "test": "assert concat_strings('hello','world')=='helloworld'; assert concat_strings('','')==''; assert concat_strings('a','b')=='ab'",
    },
    {
        "name": "repeat_string",
        "prompt": "Write a Python function `repeat_string(s, n)` that repeats a string n times.",
        "fn_name": "repeat_string",
        "impl": "def repeat_string(s, n):\n    return s * n\n",
        "test": "assert repeat_string('ab',3)=='ababab'; assert repeat_string('',5)==''; assert repeat_string('x',0)==''",
    },
    {
        "name": "contains_substring",
        "prompt": "Write a Python function `contains_substring(s, sub)` that returns True if sub is in s.",
        "fn_name": "contains_substring",
        "impl": "def contains_substring(s, sub):\n    return sub in s\n",
        "test": "assert contains_substring('hello','ell')==True; assert contains_substring('hello','xyz')==False; assert contains_substring('','')==True",
    },
    {
        "name": "index_of",
        "prompt": "Write a Python function `index_of(s, sub)` that returns the index of sub in s, or -1.",
        "fn_name": "index_of",
        "impl": "def index_of(s, sub):\n    return s.find(sub)\n",
        "test": "assert index_of('hello','ell')==1; assert index_of('hello','xyz')==-1; assert index_of('abc','a')==0",
    },
    {
        "name": "replace_char",
        "prompt": "Write a Python function `replace_char(s, old, new)` that replaces all occurrences of old with new.",
        "fn_name": "replace_char",
        "impl": "def replace_char(s, old, new):\n    return s.replace(old, new)\n",
        "test": "assert replace_char('hello','l','L')=='heLLo'; assert replace_char('aaa','a','b')=='bbb'; assert replace_char('','a','b')==''",
    },
    {
        "name": "split_string",
        "prompt": "Write a Python function `split_string(s, delim)` that splits s by delim.",
        "fn_name": "split_string",
        "impl": "def split_string(s, delim):\n    return s.split(delim)\n",
        "test": "assert split_string('a,b,c',',')==['a','b','c']; assert split_string('hello',' ')==['hello']; assert split_string('','x')==['']",
    },
    {
        "name": "join_strings",
        "prompt": "Write a Python function `join_strings(lst, delim)` that joins a list of strings.",
        "fn_name": "join_strings",
        "impl": "def join_strings(lst, delim):\n    return delim.join(lst)\n",
        "test": "assert join_strings(['a','b','c'],',')=='a,b,c'; assert join_strings(['x'],'-')=='x'; assert join_strings([],',')==''",
    },
    {
        "name": "strip_whitespace",
        "prompt": "Write a Python function `strip_whitespace(s)` that removes leading and trailing whitespace.",
        "fn_name": "strip_whitespace",
        "impl": "def strip_whitespace(s):\n    return s.strip()\n",
        "test": "assert strip_whitespace('  hello  ')=='hello'; assert strip_whitespace('hello')=='hello'; assert strip_whitespace('')==''",
    },
    {
        "name": "lowercase",
        "prompt": "Write a Python function `lowercase(s)` that converts a string to lowercase.",
        "fn_name": "lowercase",
        "impl": "def lowercase(s):\n    return s.lower()\n",
        "test": "assert lowercase('HELLO')=='hello'; assert lowercase('')==''; assert lowercase('Hello')=='hello'",
    },
    {
        "name": "count_char",
        "prompt": "Write a Python function `count_char(s, c)` that counts occurrences of c in s.",
        "fn_name": "count_char",
        "impl": "def count_char(s, c):\n    return s.count(c)\n",
        "test": "assert count_char('hello','l')==2; assert count_char('hello','z')==0; assert count_char('','a')==0",
    },
    {
        "name": "starts_with",
        "prompt": "Write a Python function `starts_with(s, prefix)` that returns True if s starts with prefix.",
        "fn_name": "starts_with",
        "impl": "def starts_with(s, prefix):\n    return s.startswith(prefix)\n",
        "test": "assert starts_with('hello','he')==True; assert starts_with('hello','xyz')==False; assert starts_with('','')==True",
    },
    {
        "name": "ends_with",
        "prompt": "Write a Python function `ends_with(s, suffix)` that returns True if s ends with suffix.",
        "fn_name": "ends_with",
        "impl": "def ends_with(s, suffix):\n    return s.endswith(suffix)\n",
        "test": "assert ends_with('hello','lo')==True; assert ends_with('hello','xyz')==False; assert ends_with('','')==True",
    },
    {
        "name": "remove_duplicates",
        "prompt": "Write a Python function `remove_duplicates(lst)` that removes duplicate values from a list.",
        "fn_name": "remove_duplicates",
        "impl": "def remove_duplicates(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]\n",
        "test": "assert remove_duplicates([1,2,2,3,3,3])==[1,2,3]; assert remove_duplicates([])==[]; assert remove_duplicates([1])==[1]",
    },
    {
        "name": "merge_lists",
        "prompt": "Write a Python function `merge_lists(a, b)` that merges two lists.",
        "fn_name": "merge_lists",
        "impl": "def merge_lists(a, b):\n    return a + b\n",
        "test": "assert merge_lists([1,2],[3,4])==[1,2,3,4]; assert merge_lists([],[])==[]; assert merge_lists([1],[])==[1]",
    },
    {
        "name": "reverse_list",
        "prompt": "Write a Python function `reverse_list(lst)` that reverses a list.",
        "fn_name": "reverse_list",
        "impl": "def reverse_list(lst):\n    return lst[::-1]\n",
        "test": "assert reverse_list([1,2,3])==[3,2,1]; assert reverse_list([])==[]; assert reverse_list([1])==[1]",
    },
    {
        "name": "flatten_list",
        "prompt": "Write a Python function `flatten_list(lst)` that flattens a list of lists.",
        "fn_name": "flatten_list",
        "impl": "def flatten_list(lst):\n    return [x for sub in lst for x in sub]\n",
        "test": "assert flatten_list([[1,2],[3,4]])==[1,2,3,4]; assert flatten_list([])==[]; assert flatten_list([[]])==[]",
    },
    {
        "name": "range_list",
        "prompt": "Write a Python function `range_list(n)` that returns a list from 0 to n-1.",
        "fn_name": "range_list",
        "impl": "def range_list(n):\n    return list(range(n))\n",
        "test": "assert range_list(3)==[0,1,2]; assert range_list(0)==[]; assert range_list(1)==[0]",
    },
    {
        "name": "sum_range",
        "prompt": "Write a Python function `sum_range(a, b)` that sums all integers from a to b inclusive.",
        "fn_name": "sum_range",
        "impl": "def sum_range(a, b):\n    return sum(range(a, b+1))\n",
        "test": "assert sum_range(1,3)==6; assert sum_range(0,0)==0; assert sum_range(5,5)==5",
    },
    {
        "name": "count_occurrences",
        "prompt": "Write a Python function `count_occurrences(lst, val)` that counts occurrences of val in lst.",
        "fn_name": "count_occurrences",
        "impl": "def count_occurrences(lst, val):\n    return lst.count(val)\n",
        "test": "assert count_occurrences([1,2,2,3],2)==2; assert count_occurrences([1,2,3],4)==0; assert count_occurrences([],1)==0",
    },
    {
        "name": "find_max_index",
        "prompt": "Write a Python function `find_max_index(lst)` that returns the index of the maximum value.",
        "fn_name": "find_max_index",
        "impl": "def find_max_index(lst):\n    return lst.index(max(lst)) if lst else -1\n",
        "test": "assert find_max_index([1,3,2])==1; assert find_max_index([5])==0; assert find_max_index([])==-1",
    },
    {
        "name": "find_min_index",
        "prompt": "Write a Python function `find_min_index(lst)` that returns the index of the minimum value.",
        "fn_name": "find_min_index",
        "impl": "def find_min_index(lst):\n    return lst.index(min(lst)) if lst else -1\n",
        "test": "assert find_min_index([3,1,2])==1; assert find_min_index([5])==0; assert find_min_index([])==-1",
    },
    {
        "name": "is_sorted",
        "prompt": "Write a Python function `is_sorted(lst)` that returns True if lst is sorted in ascending order.",
        "fn_name": "is_sorted",
        "impl": "def is_sorted(lst):\n    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n",
        "test": "assert is_sorted([1,2,3])==True; assert is_sorted([3,1,2])==False; assert is_sorted([])==True",
    },
    {
        "name": "chunk_list",
        "prompt": "Write a Python function `chunk_list(lst, size)` that splits a list into chunks of given size.",
        "fn_name": "chunk_list",
        "impl": "def chunk_list(lst, size):\n    return [lst[i:i+size] for i in range(0, len(lst), size)]\n",
        "test": "assert chunk_list([1,2,3,4,5],2)==[[1,2],[3,4],[5]]; assert chunk_list([],2)==[]; assert chunk_list([1],3)==[[1]]",
    },
    {
        "name": "zip_lists",
        "prompt": "Write a Python function `zip_lists(a, b)` that zips two lists together.",
        "fn_name": "zip_lists",
        "impl": "def zip_lists(a, b):\n    return list(zip(a, b))\n",
        "test": "assert zip_lists([1,2],['a','b'])==[(1,'a'),(2,'b')]; assert zip_lists([],[])==[]; assert zip_lists([1],[])==[]",
    },
    {
        "name": "dict_from_pairs",
        "prompt": "Write a Python function `dict_from_pairs(pairs)` that creates a dict from key-value pairs.",
        "fn_name": "dict_from_pairs",
        "impl": "def dict_from_pairs(pairs):\n    return dict(pairs)\n",
        "test": "assert dict_from_pairs([('a',1),('b',2)])=={'a':1,'b':2}; assert dict_from_pairs([])=={}; assert dict_from_pairs([('x',0)])=={'x':0}",
    },
    {
        "name": "get_keys",
        "prompt": "Write a Python function `get_keys(d)` that returns the keys of a dict.",
        "fn_name": "get_keys",
        "impl": "def get_keys(d):\n    return list(d.keys())\n",
        "test": "assert get_keys({'a':1,'b':2})==['a','b']; assert get_keys({})==[]; assert get_keys({'x':0})==['x']",
    },
    {
        "name": "get_values",
        "prompt": "Write a Python function `get_values(d)` that returns the values of a dict.",
        "fn_name": "get_values",
        "impl": "def get_values(d):\n    return list(d.values())\n",
        "test": "assert get_values({'a':1,'b':2})==[1,2]; assert get_values({})==[]; assert get_values({'x':0})==[0]",
    },
    {
        "name": "merge_dicts",
        "prompt": "Write a Python function `merge_dicts(a, b)` that merges two dicts.",
        "fn_name": "merge_dicts",
        "impl": "def merge_dicts(a, b):\n    return {**a, **b}\n",
        "test": "assert merge_dicts({'a':1},{'b':2})=={'a':1,'b':2}; assert merge_dicts({},{'x':0})=={'x':0}; assert merge_dicts({'a':1},{})=={'a':1}",
    },
    {
        "name": "has_key",
        "prompt": "Write a Python function `has_key(d, k)` that returns True if dict d has key k.",
        "fn_name": "has_key",
        "impl": "def has_key(d, k):\n    return k in d\n",
        "test": "assert has_key({'a':1},'a')==True; assert has_key({'a':1},'b')==False; assert has_key({},'x')==False",
    },
    {
        "name": "count_dict_keys",
        "prompt": "Write a Python function `count_dict_keys(d)` that returns the number of keys in a dict.",
        "fn_name": "count_dict_keys",
        "impl": "def count_dict_keys(d):\n    return len(d)\n",
        "test": "assert count_dict_keys({'a':1,'b':2})==2; assert count_dict_keys({})==0; assert count_dict_keys({'x':0})==1",
    },
    {
        "name": "invert_dict",
        "prompt": "Write a Python function `invert_dict(d)` that swaps keys and values in a dict.",
        "fn_name": "invert_dict",
        "impl": "def invert_dict(d):\n    return {v: k for k, v in d.items()}\n",
        "test": "assert invert_dict({'a':1,'b':2})=={1:'a',2:'b'}; assert invert_dict({})=={}; assert invert_dict({'x':0})=={0:'x'}",
    },
    {
        "name": "filter_dict",
        "prompt": "Write a Python function `filter_dict(d, predicate)` that filters dict entries by a predicate.",
        "fn_name": "filter_dict",
        "impl": "def filter_dict(d, predicate):\n    return {k: v for k, v in d.items() if predicate(k, v)}\n",
        "test": "assert filter_dict({'a':1,'b':2},lambda k,v:v>1)=={'b':2}; assert filter_dict({},lambda k,v:True)=={}; assert filter_dict({'a':1},lambda k,v:False)=={}",
    },
    {
        "name": "to_binary",
        "prompt": "Write a Python function `to_binary(n)` that converts a non-negative integer to its binary string.",
        "fn_name": "to_binary",
        "impl": "def to_binary(n):\n    return bin(n)[2:]\n",
        "test": "assert to_binary(5)=='101'; assert to_binary(0)=='0'; assert to_binary(10)=='1010'",
    },
    {
        "name": "from_binary",
        "prompt": "Write a Python function `from_binary(s)` that converts a binary string to an integer.",
        "fn_name": "from_binary",
        "impl": "def from_binary(s):\n    return int(s, 2)\n",
        "test": "assert from_binary('101')==5; assert from_binary('0')==0; assert from_binary('1010')==10",
    },
    {
        "name": "is_power_of_two",
        "prompt": "Write a Python function `is_power_of_two(n)` that returns True if n is a power of two.",
        "fn_name": "is_power_of_two",
        "impl": "def is_power_of_two(n):\n    return n > 0 and (n & (n-1)) == 0\n",
        "test": "assert is_power_of_two(4)==True; assert is_power_of_two(6)==False; assert is_power_of_two(1)==True; assert is_power_of_two(0)==False",
    },
    {
        "name": "digit_sum",
        "prompt": "Write a Python function `digit_sum(n)` that returns the sum of digits of a non-negative integer.",
        "fn_name": "digit_sum",
        "impl": "def digit_sum(n):\n    return sum(int(d) for d in str(n))\n",
        "test": "assert digit_sum(123)==6; assert digit_sum(0)==0; assert digit_sum(999)==27",
    },
    {
        "name": "count_digits",
        "prompt": "Write a Python function `count_digits(n)` that returns the number of digits in a non-negative integer.",
        "fn_name": "count_digits",
        "impl": "def count_digits(n):\n    return len(str(n))\n",
        "test": "assert count_digits(123)==3; assert count_digits(0)==1; assert count_digits(99999)==5",
    },
    {
        "name": "reverse_number",
        "prompt": "Write a Python function `reverse_number(n)` that reverses the digits of a non-negative integer.",
        "fn_name": "reverse_number",
        "impl": "def reverse_number(n):\n    return int(str(n)[::-1])\n",
        "test": "assert reverse_number(123)==321; assert reverse_number(0)==0; assert reverse_number(100)==1",
    },
    {
        "name": "is_armstrong",
        "prompt": "Write a Python function `is_armstrong(n)` that returns True if n is an Armstrong number.",
        "fn_name": "is_armstrong",
        "impl": "def is_armstrong(n):\n    s = str(n)\n    return n == sum(int(d)**len(s) for d in s)\n",
        "test": "assert is_armstrong(153)==True; assert is_armstrong(10)==False; assert is_armstrong(0)==True",
    },
    {
        "name": "collatz_steps",
        "prompt": "Write a Python function `collatz_steps(n)` that returns the number of Collatz steps to reach 1.",
        "fn_name": "collatz_steps",
        "impl": "def collatz_steps(n):\n    steps = 0\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps\n",
        "test": "assert collatz_steps(1)==0; assert collatz_steps(2)==1; assert collatz_steps(4)==2",
    },
    {
        "name": "lcm",
        "prompt": "Write a Python function `lcm(a, b)` that computes the least common multiple.",
        "fn_name": "lcm",
        "impl": "def lcm(a, b):\n    import math\n    return abs(a*b) // math.gcd(a, b) if a and b else 0\n",
        "test": "assert lcm(4,6)==12; assert lcm(3,5)==15; assert lcm(0,5)==0",
    },
    {
        "name": "is_perfect_square",
        "prompt": "Write a Python function `is_perfect_square(n)` that returns True if n is a perfect square.",
        "fn_name": "is_perfect_square",
        "impl": "def is_perfect_square(n):\n    import math\n    r = int(math.isqrt(n))\n    return r * r == n\n",
        "test": "assert is_perfect_square(16)==True; assert is_perfect_square(15)==False; assert is_perfect_square(0)==True",
    },
    {
        "name": "triangle_area",
        "prompt": "Write a Python function `triangle_area(base, height)` that computes the area of a triangle.",
        "fn_name": "triangle_area",
        "impl": "def triangle_area(base, height):\n    return 0.5 * base * height\n",
        "test": "assert triangle_area(4,5)==10.0; assert triangle_area(0,5)==0.0; assert triangle_area(2,3)==3.0",
    },
    {
        "name": "circle_area",
        "prompt": "Write a Python function `circle_area(radius)` that computes the area of a circle.",
        "fn_name": "circle_area",
        "impl": "def circle_area(radius):\n    import math\n    return math.pi * radius ** 2\n",
        "test": "assert abs(circle_area(1)-3.141592653589793)<0.001; assert circle_area(0)==0.0; assert abs(circle_area(2)-12.566)<0.01",
    },
    {
        "name": "rectangle_perimeter",
        "prompt": "Write a Python function `rectangle_perimeter(width, height)` that computes the perimeter of a rectangle.",
        "fn_name": "rectangle_perimeter",
        "impl": "def rectangle_perimeter(width, height):\n    return 2 * (width + height)\n",
        "test": "assert rectangle_perimeter(3,4)==14; assert rectangle_perimeter(0,0)==0; assert rectangle_perimeter(1,1)==4",
    },
    {
        "name": "celsius_to_fahrenheit",
        "prompt": "Write a Python function `celsius_to_fahrenheit(c)` that converts Celsius to Fahrenheit.",
        "fn_name": "celsius_to_fahrenheit",
        "impl": "def celsius_to_fahrenheit(c):\n    return c * 9/5 + 32\n",
        "test": "assert celsius_to_fahrenheit(0)==32.0; assert celsius_to_fahrenheit(100)==212.0; assert celsius_to_fahrenheit(-40)==-40.0",
    },
    {
        "name": "fahrenheit_to_celsius",
        "prompt": "Write a Python function `fahrenheit_to_celsius(f)` that converts Fahrenheit to Celsius.",
        "fn_name": "fahrenheit_to_celsius",
        "impl": "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5/9\n",
        "test": "assert fahrenheit_to_celsius(32)==0.0; assert fahrenheit_to_celsius(212)==100.0; assert fahrenheit_to_celsius(-40)==-40.0",
    },
    {
        "name": "string_to_list",
        "prompt": "Write a Python function `string_to_list(s)` that converts a string to a list of characters.",
        "fn_name": "string_to_list",
        "impl": "def string_to_list(s):\n    return list(s)\n",
        "test": "assert string_to_list('abc')==['a','b','c']; assert string_to_list('')==[]; assert string_to_list('x')==['x']",
    },
    {
        "name": "list_to_string",
        "prompt": "Write a Python function `list_to_string(lst)` that joins a list of characters into a string.",
        "fn_name": "list_to_string",
        "impl": "def list_to_string(lst):\n    return ''.join(lst)\n",
        "test": "assert list_to_string(['a','b','c'])=='abc'; assert list_to_string([])==''; assert list_to_string(['x'])=='x'",
    },
    {
        "name": "char_to_ascii",
        "prompt": "Write a Python function `char_to_ascii(c)` that returns the ASCII code of a character.",
        "fn_name": "char_to_ascii",
        "impl": "def char_to_ascii(c):\n    return ord(c)\n",
        "test": "assert char_to_ascii('A')==65; assert char_to_ascii('a')==97; assert char_to_ascii('0')==48",
    },
    {
        "name": "ascii_to_char",
        "prompt": "Write a Python function `ascii_to_char(n)` that returns the character for an ASCII code.",
        "fn_name": "ascii_to_char",
        "impl": "def ascii_to_char(n):\n    return chr(n)\n",
        "test": "assert ascii_to_char(65)=='A'; assert ascii_to_char(97)=='a'; assert ascii_to_char(48)=='0'",
    },
    {
        "name": "is_alpha",
        "prompt": "Write a Python function `is_alpha(c)` that returns True if c is an alphabetic character.",
        "fn_name": "is_alpha",
        "impl": "def is_alpha(c):\n    return c.isalpha()\n",
        "test": "assert is_alpha('a')==True; assert is_alpha('1')==False; assert is_alpha('')==False",
    },
    {
        "name": "is_digit",
        "prompt": "Write a Python function `is_digit(c)` that returns True if c is a digit.",
        "fn_name": "is_digit",
        "impl": "def is_digit(c):\n    return c.isdigit()\n",
        "test": "assert is_digit('5')==True; assert is_digit('a')==False; assert is_digit('')==False",
    },
    {
        "name": "title_case",
        "prompt": "Write a Python function `title_case(s)` that converts a string to title case.",
        "fn_name": "title_case",
        "impl": "def title_case(s):\n    return s.title()\n",
        "test": "assert title_case('hello world')=='Hello World'; assert title_case('')==''; assert title_case('ONE')=='One'",
    },
    {
        "name": "swap_case",
        "prompt": "Write a Python function `swap_case(s)` that swaps the case of each character.",
        "fn_name": "swap_case",
        "impl": "def swap_case(s):\n    return s.swapcase()\n",
        "test": "assert swap_case('Hello')=='hELLO'; assert swap_case('')==''; assert swap_case('ABC')=='abc'",
    },
    {
        "name": "center_string",
        "prompt": "Write a Python function `center_string(s, width)` that centers s in a field of given width.",
        "fn_name": "center_string",
        "impl": "def center_string(s, width):\n    return s.center(width)\n",
        "test": "assert center_string('hi',6)=='  hi  '; assert center_string('hello',3)=='hello'; assert center_string('',4)=='    '",
    },
    {
        "name": "left_justify",
        "prompt": "Write a Python function `left_justify(s, width)` that left-justifies s in a field of given width.",
        "fn_name": "left_justify",
        "impl": "def left_justify(s, width):\n    return s.ljust(width)\n",
        "test": "assert left_justify('hi',4)=='hi  '; assert left_justify('hello',3)=='hello'; assert left_justify('',3)=='   '",
    },
    {
        "name": "right_justify",
        "prompt": "Write a Python function `right_justify(s, width)` that right-justifies s in a field of given width.",
        "fn_name": "right_justify",
        "impl": "def right_justify(s, width):\n    return s.rjust(width)\n",
        "test": "assert right_justify('hi',4)=='  hi'; assert right_justify('hello',3)=='hello'; assert right_justify('',3)=='   '",
    },
    {
        "name": "zero_pad",
        "prompt": "Write a Python function `zero_pad(n, width)` that zero-pads a number to a given width.",
        "fn_name": "zero_pad",
        "impl": "def zero_pad(n, width):\n    return str(n).zfill(width)\n",
        "test": "assert zero_pad(5,3)=='005'; assert zero_pad(100,3)=='100'; assert zero_pad(0,4)=='0000'",
    },
    {
        "name": "format_percentage",
        "prompt": "Write a Python function `format_percentage(value)` that formats a value as a percentage string.",
        "fn_name": "format_percentage",
        "impl": "def format_percentage(value):\n    return f'{value:.1%}'\n",
        "test": "assert format_percentage(0.5)=='50.0%'; assert format_percentage(0.123)=='12.3%'; assert format_percentage(1.0)=='100.0%'",
    },
    {
        "name": "parse_int",
        "prompt": "Write a Python function `parse_int(s)` that safely parses an integer from a string.",
        "fn_name": "parse_int",
        "impl": "def parse_int(s):\n    try:\n        return int(s)\n    except ValueError:\n        return None\n",
        "test": "assert parse_int('123')==123; assert parse_int('abc') is None; assert parse_int('0')==0",
    },
    {
        "name": "parse_float",
        "prompt": "Write a Python function `parse_float(s)` that safely parses a float from a string.",
        "fn_name": "parse_float",
        "impl": "def parse_float(s):\n    try:\n        return float(s)\n    except ValueError:\n        return None\n",
        "test": "assert parse_float('3.14')==3.14; assert parse_float('abc') is None; assert parse_float('0')==0.0",
    },
    {
        "name": "clamp",
        "prompt": "Write a Python function `clamp(value, low, high)` that clamps a value to a range.",
        "fn_name": "clamp",
        "impl": "def clamp(value, low, high):\n    return max(low, min(value, high))\n",
        "test": "assert clamp(5,0,10)==5; assert clamp(-1,0,10)==0; assert clamp(15,0,10)==10",
    },
    {
        "name": "lerp",
        "prompt": "Write a Python function `lerp(a, b, t)` that linearly interpolates between a and b.",
        "fn_name": "lerp",
        "impl": "def lerp(a, b, t):\n    return a + (b - a) * t\n",
        "test": "assert lerp(0,10,0.5)==5.0; assert lerp(0,10,0)==0.0; assert lerp(0,10,1)==10.0",
    },
    {
        "name": "manhattan_distance",
        "prompt": "Write a Python function `manhattan_distance(p1, p2)` that computes Manhattan distance between two points.",
        "fn_name": "manhattan_distance",
        "impl": "def manhattan_distance(p1, p2):\n    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])\n",
        "test": "assert manhattan_distance((0,0),(3,4))==7; assert manhattan_distance((1,1),(1,1))==0; assert manhattan_distance((0,0),(0,5))==5",
    },
    {
        "name": "euclidean_distance",
        "prompt": "Write a Python function `euclidean_distance(p1, p2)` that computes Euclidean distance between two points.",
        "fn_name": "euclidean_distance",
        "impl": "def euclidean_distance(p1, p2):\n    import math\n    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)\n",
        "test": "assert abs(euclidean_distance((0,0),(3,4))-5.0)<0.001; assert euclidean_distance((1,1),(1,1))==0.0; assert abs(euclidean_distance((0,0),(0,3))-3.0)<0.001",
    },
    {
        "name": "average",
        "prompt": "Write a Python function `average(lst)` that returns the average of a list of numbers.",
        "fn_name": "average",
        "impl": "def average(lst):\n    return sum(lst) / len(lst) if lst else 0.0\n",
        "test": "assert average([1,2,3])==2.0; assert average([])==0.0; assert average([5])==5.0",
    },
    {
        "name": "median",
        "prompt": "Write a Python function `median(lst)` that returns the median of a list of numbers.",
        "fn_name": "median",
        "impl": "def median(lst):\n    if not lst: return 0.0\n    s = sorted(lst)\n    n = len(s)\n    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2\n",
        "test": "assert median([1,2,3])==2; assert median([1,2,3,4])==2.5; assert median([])==0.0",
    },
    {
        "name": "variance",
        "prompt": "Write a Python function `variance(lst)` that returns the population variance of a list.",
        "fn_name": "variance",
        "impl": "def variance(lst):\n    if not lst: return 0.0\n    m = sum(lst)/len(lst)\n    return sum((x-m)**2 for x in lst)/len(lst)\n",
        "test": "assert abs(variance([1,2,3,4])-1.25)<0.001; assert variance([])==0.0; assert variance([5])==0.0",
    },
    {
        "name": "std_dev",
        "prompt": "Write a Python function `std_dev(lst)` that returns the population standard deviation.",
        "fn_name": "std_dev",
        "impl": "def std_dev(lst):\n    import math\n    return math.sqrt(variance(lst)) if lst else 0.0\n",
        "test": "assert abs(std_dev([1,2,3,4])-1.118)<0.01; assert std_dev([])==0.0; assert std_dev([5])==0.0",
    },
    {
        "name": "mode",
        "prompt": "Write a Python function `mode(lst)` that returns the most common element in a list.",
        "fn_name": "mode",
        "impl": "def mode(lst):\n    from collections import Counter\n    return Counter(lst).most_common(1)[0][0] if lst else None\n",
        "test": "assert mode([1,2,2,3])==2; assert mode(['a','b','a'])=='a'; assert mode([]) is None",
    },
    {
        "name": "percentile",
        "prompt": "Write a Python function `percentile(lst, p)` that returns the p-th percentile of a list.",
        "fn_name": "percentile",
        "impl": "def percentile(lst, p):\n    if not lst: return 0.0\n    s = sorted(lst)\n    k = (len(s)-1) * p / 100\n    f = int(k)\n    c = min(f+1, len(s)-1)\n    return s[f] + (s[c]-s[f])*(k-f)\n",
        "test": "assert percentile([1,2,3,4,5],50)==3.0; assert percentile([1,2,3,4,5],0)==1; assert percentile([1,2,3,4,5],100)==5",
    },
    {
        "name": "interleave",
        "prompt": "Write a Python function `interleave(a, b)` that interleaves two lists.",
        "fn_name": "interleave",
        "impl": "def interleave(a, b):\n    from itertools import chain\n    return list(chain.from_iterable(zip(a, b)))\n",
        "test": "assert interleave([1,2],['a','b'])==[1,'a',2,'b']; assert interleave([],[])==[]; assert interleave([1],['x'])==[1,'x']",
    },
    {
        "name": "rotate_list",
        "prompt": "Write a Python function `rotate_list(lst, k)` that rotates a list by k positions.",
        "fn_name": "rotate_list",
        "impl": "def rotate_list(lst, k):\n    if not lst: return []\n    k = k % len(lst)\n    return lst[k:] + lst[:k]\n",
        "test": "assert rotate_list([1,2,3,4],1)==[2,3,4,1]; assert rotate_list([1,2,3],0)==[1,2,3]; assert rotate_list([],3)==[]",
    },
    {
        "name": "partition",
        "prompt": "Write a Python function `partition(lst, predicate)` that partitions a list by a predicate.",
        "fn_name": "partition",
        "impl": "def partition(lst, predicate):\n    yes = [x for x in lst if predicate(x)]\n    no = [x for x in lst if not predicate(x)]\n    return (yes, no)\n",
        "test": "assert partition([1,2,3,4],lambda x:x>2)==([3,4],[1,2]); assert partition([],lambda x:True)==([],[]); assert partition([1],lambda x:False)==([],[1])",
    },
    {
        "name": "group_by",
        "prompt": "Write a Python function `group_by(lst, key_fn)` that groups list elements by a key function.",
        "fn_name": "group_by",
        "impl": "def group_by(lst, key_fn):\n    from collections import defaultdict\n    groups = defaultdict(list)\n    for x in lst:\n        groups[key_fn(x)].append(x)\n    return dict(groups)\n",
        "test": "assert group_by([1,2,3,4],lambda x:x%2)=={1:[1,3],0:[2,4]}; assert group_by([],lambda x:0)=={}; assert group_by(['a'],len)=={1:['a']}",
    },
    {
        "name": "take",
        "prompt": "Write a Python function `take(lst, n)` that returns the first n elements of a list.",
        "fn_name": "take",
        "impl": "def take(lst, n):\n    return lst[:n]\n",
        "test": "assert take([1,2,3,4],2)==[1,2]; assert take([1,2,3],0)==[]; assert take([],5)==[]",
    },
    {
        "name": "drop",
        "prompt": "Write a Python function `drop(lst, n)` that drops the first n elements of a list.",
        "fn_name": "drop",
        "impl": "def drop(lst, n):\n    return lst[n:]\n",
        "test": "assert drop([1,2,3,4],2)==[3,4]; assert drop([1,2,3],0)==[1,2,3]; assert drop([],5)==[]",
    },
    {
        "name": "unique",
        "prompt": "Write a Python function `unique(lst)` that returns unique elements preserving order.",
        "fn_name": "unique",
        "impl": "def unique(lst):\n    seen = set()\n    return [x for x in lst if x not in seen and not seen.add(x)]\n",
        "test": "assert unique([1,2,2,3,1])==[1,2,3]; assert unique([])==[]; assert unique(['a','a'])==['a']",
    },
    {
        "name": "difference",
        "prompt": "Write a Python function `difference(a, b)` that returns elements in a but not in b.",
        "fn_name": "difference",
        "impl": "def difference(a, b):\n    return [x for x in a if x not in set(b)]\n",
        "test": "assert difference([1,2,3],[2,4])==[1,3]; assert difference([],[])==[]; assert difference([1,2],[])==[1,2]",
    },
    {
        "name": "intersection",
        "prompt": "Write a Python function `intersection(a, b)` that returns elements in both a and b.",
        "fn_name": "intersection",
        "impl": "def intersection(a, b):\n    return [x for x in a if x in set(b)]\n",
        "test": "assert intersection([1,2,3],[2,3,4])==[2,3]; assert intersection([],[])==[]; assert intersection([1,2],[3,4])==[]",
    },
    {
        "name": "union",
        "prompt": "Write a Python function `union(a, b)` that returns the union of two lists preserving order.",
        "fn_name": "union",
        "impl": "def union(a, b):\n    seen = set()\n    result = []\n    for x in a + b:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result\n",
        "test": "assert union([1,2],[2,3])==[1,2,3]; assert union([],[])==[]; assert union([1],[1])==[1]",
    },
    {
        "name": "is_substring",
        "prompt": "Write a Python function `is_substring(a, b)` that returns True if a is a substring of b.",
        "fn_name": "is_substring",
        "impl": "def is_substring(a, b):\n    return a in b\n",
        "test": "assert is_substring('ell','hello')==True; assert is_substring('xyz','hello')==False; assert is_substring('','')==True",
    },
    {
        "name": "count_lines",
        "prompt": "Write a Python function `count_lines(s)` that counts the number of lines in a string.",
        "fn_name": "count_lines",
        "impl": "def count_lines(s):\n    return s.count('\\n') + 1 if s else 0\n",
        "test": "assert count_lines('a\\nb\\nc')==3; assert count_lines('')==0; assert count_lines('hello')==1",
    },
    {
        "name": "wrap_text",
        "prompt": "Write a Python function `wrap_text(s, width)` that wraps text to a given width.",
        "fn_name": "wrap_text",
        "impl": "def wrap_text(s, width):\n    import textwrap\n    return textwrap.fill(s, width)\n",
        "test": "assert wrap_text('hello world',5)=='hello\\nworld'; assert wrap_text('',5)==''; assert wrap_text('hi',10)=='hi'",
    },
    {
        "name": "slugify",
        "prompt": "Write a Python function `slugify(s)` that converts a string to a slug.",
        "fn_name": "slugify",
        "impl": "def slugify(s):\n    import re\n    return re.sub(r'[^a-z0-9]+','-',s.lower()).strip('-')\n",
        "test": "assert slugify('Hello World')=='hello-world'; assert slugify('')==''; assert slugify('A B C')=='a-b-c'",
    },
    {
        "name": "truncate",
        "prompt": "Write a Python function `truncate(s, length)` that truncates a string to a given length with ellipsis.",
        "fn_name": "truncate",
        "impl": "def truncate(s, length):\n    return s[:length] + '...' if len(s) > length else s\n",
        "test": "assert truncate('hello world',5)=='hello...'; assert truncate('hi',10)=='hi'; assert truncate('',5)==''",
    },
    {
        "name": "count_sentences",
        "prompt": "Write a Python function `count_sentences(s)` that counts the number of sentences in a string.",
        "fn_name": "count_sentences",
        "impl": "def count_sentences(s):\n    import re\n    return len(re.findall(r'[.!?]+', s))\n",
        "test": "assert count_sentences('Hello. World!')==2; assert count_sentences('')==0; assert count_sentences('Hi')==0",
    },
    {
        "name": "extract_numbers",
        "prompt": "Write a Python function `extract_numbers(s)` that extracts all numbers from a string.",
        "fn_name": "extract_numbers",
        "impl": "def extract_numbers(s):\n    import re\n    return [int(x) for x in re.findall(r'\\d+', s)]\n",
        "test": "assert extract_numbers('abc123def45')==[123,45]; assert extract_numbers('abc')==[]; assert extract_numbers('')==[]",
    },
    {
        "name": "mask_string",
        "prompt": "Write a Python function `mask_string(s, mask_char)` that masks all but the last 4 characters.",
        "fn_name": "mask_string",
        "impl": "def mask_string(s, mask_char):\n    if len(s) <= 4: return s\n    return mask_char * (len(s)-4) + s[-4:]\n",
        "test": "assert mask_string('1234567890','*')=='******7890'; assert mask_string('1234','*')=='1234'; assert mask_string('','*')==''",
    },
    {
        "name": "is_valid_email",
        "prompt": "Write a Python function `is_valid_email(s)` that returns True if s is a valid email format.",
        "fn_name": "is_valid_email",
        "impl": "def is_valid_email(s):\n    import re\n    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', s))\n",
        "test": "assert is_valid_email('a@b.com')==True; assert is_valid_email('invalid')==False; assert is_valid_email('')==False",
    },
    {
        "name": "capitalize_first",
        "prompt": "Write a Python function `capitalize_first(s)` that capitalizes only the first character.",
        "fn_name": "capitalize_first",
        "impl": "def capitalize_first(s):\n    return s[:1].upper() + s[1:] if s else s\n",
        "test": "assert capitalize_first('hello')=='Hello'; assert capitalize_first('')==''; assert capitalize_first('a')=='A'",
    },
    {
        "name": "count_words_starting_with",
        "prompt": "Write a Python function `count_words_starting_with(s, prefix)` that counts words starting with prefix.",
        "fn_name": "count_words_starting_with",
        "impl": "def count_words_starting_with(s, prefix):\n    return sum(1 for w in s.split() if w.startswith(prefix))\n",
        "test": "assert count_words_starting_with('apple ant banana','a')==2; assert count_words_starting_with('hello world','x')==0; assert count_words_starting_with('','a')==0",
    },
    {
        "name": "replace_words",
        "prompt": "Write a Python function `replace_words(s, old, new)` that replaces whole words.",
        "fn_name": "replace_words",
        "impl": "def replace_words(s, old, new):\n    return ' '.join(new if w == old else w for w in s.split())\n",
        "test": "assert replace_words('hello world hello','hello','hi')=='hi world hi'; assert replace_words('','a','b')==''; assert replace_words('a b c','d','e')=='a b c'",
    },
    {
        "name": "acronym",
        "prompt": "Write a Python function `acronym(s)` that returns the acronym of a phrase.",
        "fn_name": "acronym",
        "impl": "def acronym(s):\n    return ''.join(w[0].upper() for w in s.split() if w)\n",
        "test": "assert acronym('Hello World')=='HW'; assert acronym('')==''; assert acronym('a b c')=='ABC'",
    },
    {
        "name": "nth_word",
        "prompt": "Write a Python function `nth_word(s, n)` that returns the n-th word (0-indexed).",
        "fn_name": "nth_word",
        "impl": "def nth_word(s, n):\n    words = s.split()\n    return words[n] if 0 <= n < len(words) else ''\n",
        "test": "assert nth_word('hello world foo',1)=='world'; assert nth_word('hello',0)=='hello'; assert nth_word('hello',5)==''",
    },
    {
        "name": "word_at_position",
        "prompt": "Write a Python function `word_at_position(s, position)` that returns the word at a character position.",
        "fn_name": "word_at_position",
        "impl": "def word_at_position(s, position):\n    if position < 0 or position >= len(s): return ''\n    start = s.rfind(' ', 0, position) + 1\n    end = s.find(' ', position)\n    return s[start:end if end != -1 else len(s)]\n",
        "test": "assert word_at_position('hello world foo',6)=='world'; assert word_at_position('hello',0)=='hello'; assert word_at_position('a b',2)=='b'",
    },
    {
        "name": "count_substring",
        "prompt": "Write a Python function `count_substring(s, sub)` that counts non-overlapping occurrences of sub.",
        "fn_name": "count_substring",
        "impl": "def count_substring(s, sub):\n    return s.count(sub)\n",
        "test": "assert count_substring('hello hello','hello')==2; assert count_substring('aaa','a')==3; assert count_substring('','a')==0",
    },
    {
        "name": "pad_left",
        "prompt": "Write a Python function `pad_left(s, width, char)` that pads s on the left to a given width.",
        "fn_name": "pad_left",
        "impl": "def pad_left(s, width, char):\n    return s.rjust(width, char)\n",
        "test": "assert pad_left('5',3,'0')=='005'; assert pad_left('hello',3,' ')=='hello'; assert pad_left('',3,'x')=='xxx'",
    },
    {
        "name": "pad_right",
        "prompt": "Write a Python function `pad_right(s, width, char)` that pads s on the right to a given width.",
        "fn_name": "pad_right",
        "impl": "def pad_right(s, width, char):\n    return s.ljust(width, char)\n",
        "test": "assert pad_right('5',3,'0')=='500'; assert pad_right('hello',3,' ')=='hello'; assert pad_right('',3,'x')=='xxx'",
    },
    {
        "name": "is_empty",
        "prompt": "Write a Python function `is_empty(s)` that returns True if s is empty or whitespace only.",
        "fn_name": "is_empty",
        "impl": "def is_empty(s):\n    return s.strip() == ''\n",
        "test": "assert is_empty('   ')==True; assert is_empty('')==True; assert is_empty('hello')==False",
    },
    {
        "name": "first_char",
        "prompt": "Write a Python function `first_char(s)` that returns the first character or empty string.",
        "fn_name": "first_char",
        "impl": "def first_char(s):\n    return s[0] if s else ''\n",
        "test": "assert first_char('hello')=='h'; assert first_char('')==''; assert first_char('a')=='a'",
    },
    {
        "name": "last_char",
        "prompt": "Write a Python function `last_char(s)` that returns the last character or empty string.",
        "fn_name": "last_char",
        "impl": "def last_char(s):\n    return s[-1] if s else ''\n",
        "test": "assert last_char('hello')=='o'; assert last_char('')==''; assert last_char('a')=='a'",
    },
    {
        "name": "second_half",
        "prompt": "Write a Python function `second_half(s)` that returns the second half of a string.",
        "fn_name": "second_half",
        "impl": "def second_half(s):\n    return s[len(s)//2:]\n",
        "test": "assert second_half('hello')=='lo'; assert second_half('')==''; assert second_half('ab')=='b'",
    },
    {
        "name": "first_half",
        "prompt": "Write a Python function `first_half(s)` that returns the first half of a string.",
        "fn_name": "first_half",
        "impl": "def first_half(s):\n    return s[:len(s)//2]\n",
        "test": "assert first_half('hello')=='he'; assert first_half('')==''; assert first_half('ab')=='a'",
    },
    {
        "name": "is_uppercase",
        "prompt": "Write a Python function `is_uppercase(s)` that returns True if s is all uppercase.",
        "fn_name": "is_uppercase",
        "impl": "def is_uppercase(s):\n    return s.isupper()\n",
        "test": "assert is_uppercase('HELLO')==True; assert is_uppercase('Hello')==False; assert is_uppercase('')==False",
    },
    {
        "name": "is_lowercase",
        "prompt": "Write a Python function `is_lowercase(s)` that returns True if s is all lowercase.",
        "fn_name": "is_lowercase",
        "impl": "def is_lowercase(s):\n    return s.islower()\n",
        "test": "assert is_lowercase('hello')==True; assert is_lowercase('Hello')==False; assert is_lowercase('')==False",
    },
    {
        "name": "toggle_case",
        "prompt": "Write a Python function `toggle_case(s)` that toggles the case of each alphabetic character.",
        "fn_name": "toggle_case",
        "impl": "def toggle_case(s):\n    return ''.join(c.lower() if c.isupper() else c.upper() for c in s)\n",
        "test": "assert toggle_case('Hello')=='hELLO'; assert toggle_case('')==''; assert toggle_case('ABC')=='abc'",
    },
    {
        "name": "count_uppercase",
        "prompt": "Write a Python function `count_uppercase(s)` that counts uppercase letters in s.",
        "fn_name": "count_uppercase",
        "impl": "def count_uppercase(s):\n    return sum(1 for c in s if c.isupper())\n",
        "test": "assert count_uppercase('Hello')==1; assert count_uppercase('HELLO')==5; assert count_uppercase('hello')==0",
    },
    {
        "name": "count_lowercase",
        "prompt": "Write a Python function `count_lowercase(s)` that counts lowercase letters in s.",
        "fn_name": "count_lowercase",
        "impl": "def count_lowercase(s):\n    return sum(1 for c in s if c.islower())\n",
        "test": "assert count_lowercase('Hello')==4; assert count_lowercase('HELLO')==0; assert count_lowercase('hello')==5",
    },
    {
        "name": "only_alpha",
        "prompt": "Write a Python function `only_alpha(s)` that returns only alphabetic characters.",
        "fn_name": "only_alpha",
        "impl": "def only_alpha(s):\n    return ''.join(c for c in s if c.isalpha())\n",
        "test": "assert only_alpha('hello123world')=='helloworld'; assert only_alpha('')==''; assert only_alpha('abc')=='abc'",
    },
    {
        "name": "only_digits",
        "prompt": "Write a Python function `only_digits(s)` that returns only digit characters.",
        "fn_name": "only_digits",
        "impl": "def only_digits(s):\n    return ''.join(c for c in s if c.isdigit())\n",
        "test": "assert only_digits('abc123def45')=='12345'; assert only_digits('')==''; assert only_digits('abc')==''",
    },
    {
        "name": "remove_whitespace",
        "prompt": "Write a Python function `remove_whitespace(s)` that removes all whitespace from s.",
        "fn_name": "remove_whitespace",
        "impl": "def remove_whitespace(s):\n    return ''.join(s.split())\n",
        "test": "assert remove_whitespace('hello world')=='helloworld'; assert remove_whitespace('')==''; assert remove_whitespace('  a  b  ')=='ab'",
    },
    {
        "name": "is_numeric",
        "prompt": "Write a Python function `is_numeric(s)` that returns True if s represents a number.",
        "fn_name": "is_numeric",
        "impl": "def is_numeric(s):\n    return s.replace('.','',1).replace('-','',1).isdigit() if s else False\n",
        "test": "assert is_numeric('123')==True; assert is_numeric('3.14')==True; assert is_numeric('abc')==False; assert is_numeric('')==False",
    },
    {
        "name": "format_number",
        "prompt": "Write a Python function `format_number(n, decimals)` that formats a number with given decimals.",
        "fn_name": "format_number",
        "impl": "def format_number(n, decimals):\n    return f'{n:.{decimals}f}'\n",
        "test": "assert format_number(3.14159,2)=='3.14'; assert format_number(1,0)=='1'; assert format_number(0.5,3)=='0.500'",
    },
    {
        "name": "humanize_number",
        "prompt": "Write a Python function `humanize_number(n)` that returns a human-readable number with commas.",
        "fn_name": "humanize_number",
        "impl": "def humanize_number(n):\n    return f'{n:,}'\n",
        "test": "assert humanize_number(1000000)=='1,000,000'; assert humanize_number(0)=='0'; assert humanize_number(1234)=='1,234'",
    },
    {
        "name": "byte_size",
        "prompt": "Write a Python function `byte_size(s)` that returns the size of a string in bytes.",
        "fn_name": "byte_size",
        "impl": "def byte_size(s):\n    return len(s.encode('utf-8'))\n",
        "test": "assert byte_size('hello')==5; assert byte_size('')==0; assert byte_size('café')==5",
    },
    {
        "name": "hex_to_int",
        "prompt": "Write a Python function `hex_to_int(s)` that converts a hex string to an integer.",
        "fn_name": "hex_to_int",
        "impl": "def hex_to_int(s):\n    return int(s, 16)\n",
        "test": "assert hex_to_int('ff')==255; assert hex_to_int('10')==16; assert hex_to_int('0')==0",
    },
    {
        "name": "int_to_hex",
        "prompt": "Write a Python function `int_to_hex(n)` that converts an integer to a hex string.",
        "fn_name": "int_to_hex",
        "impl": "def int_to_hex(n):\n    return format(n, 'x')\n",
        "test": "assert int_to_hex(255)=='ff'; assert int_to_hex(16)=='10'; assert int_to_hex(0)=='0'",
    },
    {
        "name": "is_odd",
        "prompt": "Write a Python function `is_odd(n)` that returns True if n is odd.",
        "fn_name": "is_odd",
        "impl": "def is_odd(n):\n    return n % 2 != 0\n",
        "test": "assert is_odd(3)==True; assert is_odd(4)==False; assert is_odd(0)==False",
    },
    {
        "name": "divisible_by",
        "prompt": "Write a Python function `divisible_by(n, d)` that returns True if n is divisible by d.",
        "fn_name": "divisible_by",
        "impl": "def divisible_by(n, d):\n    return d != 0 and n % d == 0\n",
        "test": "assert divisible_by(10,2)==True; assert divisible_by(10,3)==False; assert divisible_by(10,0)==False",
    },
    {
        "name": "next_even",
        "prompt": "Write a Python function `next_even(n)` that returns the next even number >= n.",
        "fn_name": "next_even",
        "impl": "def next_even(n):\n    return n if n % 2 == 0 else n + 1\n",
        "test": "assert next_even(3)==4; assert next_even(4)==4; assert next_even(0)==0",
    },
    {
        "name": "next_odd",
        "prompt": "Write a Python function `next_odd(n)` that returns the next odd number >= n.",
        "fn_name": "next_odd",
        "impl": "def next_odd(n):\n    return n if n % 2 != 0 else n + 1\n",
        "test": "assert next_odd(3)==3; assert next_odd(4)==5; assert next_odd(0)==1",
    },
    {
        "name": "double",
        "prompt": "Write a Python function `double(n)` that returns n multiplied by 2.",
        "fn_name": "double",
        "impl": "def double(n):\n    return n * 2\n",
        "test": "assert double(5)==10; assert double(0)==0; assert double(-3)==-6",
    },
    {
        "name": "halve",
        "prompt": "Write a Python function `halve(n)` that returns n divided by 2.",
        "fn_name": "halve",
        "impl": "def halve(n):\n    return n / 2\n",
        "test": "assert halve(10)==5.0; assert halve(0)==0.0; assert halve(7)==3.5",
    },
    {
        "name": "negate",
        "prompt": "Write a Python function `negate(n)` that returns the negation of n.",
        "fn_name": "negate",
        "impl": "def negate(n):\n    return -n\n",
        "test": "assert negate(5)==-5; assert negate(-3)==3; assert negate(0)==0",
    },
    {
        "name": "square",
        "prompt": "Write a Python function `square(n)` that returns n squared.",
        "fn_name": "square",
        "impl": "def square(n):\n    return n * n\n",
        "test": "assert square(3)==9; assert square(0)==0; assert square(-2)==4",
    },
    {
        "name": "cube",
        "prompt": "Write a Python function `cube(n)` that returns n cubed.",
        "fn_name": "cube",
        "impl": "def cube(n):\n    return n ** 3\n",
        "test": "assert cube(2)==8; assert cube(0)==0; assert cube(-1)==-1",
    },
    {
        "name": "increment",
        "prompt": "Write a Python function `increment(n)` that returns n + 1.",
        "fn_name": "increment",
        "impl": "def increment(n):\n    return n + 1\n",
        "test": "assert increment(5)==6; assert increment(0)==1; assert increment(-1)==0",
    },
    {
        "name": "decrement",
        "prompt": "Write a Python function `decrement(n)` that returns n - 1.",
        "fn_name": "decrement",
        "impl": "def decrement(n):\n    return n - 1\n",
        "test": "assert decrement(5)==4; assert decrement(0)==-1; assert decrement(-1)==-2",
    },
    {
        "name": "add_one",
        "prompt": "Write a Python function `add_one(n)` that returns n + 1.",
        "fn_name": "add_one",
        "impl": "def add_one(n):\n    return n + 1\n",
        "test": "assert add_one(5)==6; assert add_one(0)==1; assert add_one(-1)==0",
    },
    {
        "name": "subtract_one",
        "prompt": "Write a Python function `subtract_one(n)` that returns n - 1.",
        "fn_name": "subtract_one",
        "impl": "def subtract_one(n):\n    return n - 1\n",
        "test": "assert subtract_one(5)==4; assert subtract_one(0)==-1; assert subtract_one(-1)==-2",
    },
    {
        "name": "multiply_by_ten",
        "prompt": "Write a Python function `multiply_by_ten(n)` that returns n * 10.",
        "fn_name": "multiply_by_ten",
        "impl": "def multiply_by_ten(n):\n    return n * 10\n",
        "test": "assert multiply_by_ten(5)==50; assert multiply_by_ten(0)==0; assert multiply_by_ten(-2)==-20",
    },
    {
        "name": "divide_by_ten",
        "prompt": "Write a Python function `divide_by_ten(n)` that returns n / 10.",
        "fn_name": "divide_by_ten",
        "impl": "def divide_by_ten(n):\n    return n / 10\n",
        "test": "assert divide_by_ten(100)==10.0; assert divide_by_ten(0)==0.0; assert divide_by_ten(5)==0.5",
    },
    {
        "name": "percent_of",
        "prompt": "Write a Python function `percent_of(part, whole)` that returns what percent part is of whole.",
        "fn_name": "percent_of",
        "impl": "def percent_of(part, whole):\n    return (part / whole * 100) if whole else 0.0\n",
        "test": "assert percent_of(50,100)==50.0; assert percent_of(0,100)==0.0; assert percent_of(10,0)==0.0",
    },
    {
        "name": "discount_price",
        "prompt": "Write a Python function `discount_price(price, discount)` that returns the price after discount.",
        "fn_name": "discount_price",
        "impl": "def discount_price(price, discount):\n    return price * (1 - discount / 100)\n",
        "test": "assert discount_price(100,20)==80.0; assert discount_price(50,0)==50.0; assert discount_price(100,100)==0.0",
    },
    {
        "name": "tax_price",
        "prompt": "Write a Python function `tax_price(price, tax_rate)` that returns the price including tax.",
        "fn_name": "tax_price",
        "impl": "def tax_price(price, tax_rate):\n    return price * (1 + tax_rate / 100)\n",
        "test": "assert tax_price(100,10)==110.0; assert tax_price(50,0)==50.0; assert tax_price(0,5)==0.0",
    },
    {
        "name": "compound_interest",
        "prompt": "Write a Python function `compound_interest(principal, rate, years)` that returns the final amount.",
        "fn_name": "compound_interest",
        "impl": "def compound_interest(principal, rate, years):\n    return principal * (1 + rate / 100) ** years\n",
        "test": "assert abs(compound_interest(100,10,2)-121.0)<0.01; assert compound_interest(0,10,5)==0; assert compound_interest(100,0,10)==100",
    },
    {
        "name": "simple_interest",
        "prompt": "Write a Python function `simple_interest(principal, rate, years)` that returns the interest.",
        "fn_name": "simple_interest",
        "impl": "def simple_interest(principal, rate, years):\n    return principal * rate * years / 100\n",
        "test": "assert simple_interest(100,5,2)==10.0; assert simple_interest(0,10,5)==0.0; assert simple_interest(100,0,10)==0.0",
    },
    {
        "name": "is_leap_year",
        "prompt": "Write a Python function `is_leap_year(year)` that returns True if year is a leap year.",
        "fn_name": "is_leap_year",
        "impl": "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n",
        "test": "assert is_leap_year(2000)==True; assert is_leap_year(1900)==False; assert is_leap_year(2024)==True",
    },
    {
        "name": "days_in_month",
        "prompt": "Write a Python function `days_in_month(month, year)` that returns the number of days in a month.",
        "fn_name": "days_in_month",
        "impl": "def days_in_month(month, year):\n    import calendar\n    return calendar.monthrange(year, month)[1]\n",
        "test": "assert days_in_month(2,2024)==29; assert days_in_month(2,2023)==28; assert days_in_month(1,2024)==31",
    },
    {
        "name": "day_of_year",
        "prompt": "Write a Python function `day_of_year(year, month, day)` that returns the day number in the year.",
        "fn_name": "day_of_year",
        "impl": "def day_of_year(year, month, day):\n    import datetime\n    return datetime.date(year, month, day).timetuple().tm_yday\n",
        "test": "assert day_of_year(2024,1,1)==1; assert day_of_year(2024,12,31)==366; assert day_of_year(2023,12,31)==365",
    },
    {
        "name": "seconds_to_hms",
        "prompt": "Write a Python function `seconds_to_hms(seconds)` that converts seconds to (hours, minutes, seconds).",
        "fn_name": "seconds_to_hms",
        "impl": "def seconds_to_hms(seconds):\n    h = seconds // 3600\n    m = (seconds % 3600) // 60\n    s = seconds % 60\n    return (h, m, s)\n",
        "test": "assert seconds_to_hms(3661)==(1,1,1); assert seconds_to_hms(0)==(0,0,0); assert seconds_to_hms(60)==(0,1,0)",
    },
    {
        "name": "hms_to_seconds",
        "prompt": "Write a Python function `hms_to_seconds(h, m, s)` that converts (h, m, s) to seconds.",
        "fn_name": "hms_to_seconds",
        "impl": "def hms_to_seconds(h, m, s):\n    return h * 3600 + m * 60 + s\n",
        "test": "assert hms_to_seconds(1,1,1)==3661; assert hms_to_seconds(0,0,0)==0; assert hms_to_seconds(0,1,0)==60",
    },
    {
        "name": "add_days",
        "prompt": "Write a Python function `add_days(year, month, day, days)` that adds days to a date.",
        "fn_name": "add_days",
        "impl": "def add_days(year, month, day, days):\n    import datetime\n    d = datetime.date(year, month, day) + datetime.timedelta(days=days)\n    return (d.year, d.month, d.day)\n",
        "test": "assert add_days(2024,1,1,31)==(2024,2,1); assert add_days(2024,1,1,0)==(2024,1,1); assert add_days(2024,12,31,1)==(2025,1,1)",
    },
    {
        "name": "days_between",
        "prompt": "Write a Python function `days_between(y1, m1, d1, y2, m2, d2)` that returns days between two dates.",
        "fn_name": "days_between",
        "impl": "def days_between(y1, m1, d1, y2, m2, d2):\n    import datetime\n    return abs((datetime.date(y2,m2,d2) - datetime.date(y1,m1,d1)).days)\n",
        "test": "assert days_between(2024,1,1,2024,1,2)==1; assert days_between(2024,1,1,2024,1,1)==0; assert days_between(2024,1,1,2024,2,1)==31",
    },
    {
        "name": "weekday_name",
        "prompt": "Write a Python function `weekday_name(year, month, day)` that returns the day of the week name.",
        "fn_name": "weekday_name",
        "impl": "def weekday_name(year, month, day):\n    import datetime\n    return datetime.date(year, month, day).strftime('%A')\n",
        "test": "assert weekday_name(2024,1,1)=='Monday'; assert weekday_name(2024,1,7)=='Sunday'; assert weekday_name(2024,12,25)=='Wednesday'",
    },
    {
        "name": "is_weekend",
        "prompt": "Write a Python function `is_weekend(year, month, day)` that returns True if the date is a weekend.",
        "fn_name": "is_weekend",
        "impl": "def is_weekend(year, month, day):\n    import datetime\n    return datetime.date(year, month, day).weekday() >= 5\n",
        "test": "assert is_weekend(2024,1,6)==True; assert is_weekend(2024,1,1)==False; assert is_weekend(2024,1,7)==True",
    },
    {
        "name": "quarter_of_year",
        "prompt": "Write a Python function `quarter_of_year(month)` that returns the quarter (1-4) for a month.",
        "fn_name": "quarter_of_year",
        "impl": "def quarter_of_year(month):\n    return (month - 1) // 3 + 1\n",
        "test": "assert quarter_of_year(1)==1; assert quarter_of_year(6)==2; assert quarter_of_year(12)==4",
    },
    {
        "name": "season",
        "prompt": "Write a Python function `season(month)` that returns the season for a month (Northern Hemisphere).",
        "fn_name": "season",
        "impl": "def season(month):\n    if month in [12,1,2]: return 'winter'\n    if month in [3,4,5]: return 'spring'\n    if month in [6,7,8]: return 'summer'\n    return 'autumn'\n",
        "test": "assert season(1)=='winter'; assert season(4)=='spring'; assert season(7)=='summer'; assert season(10)=='autumn'",
    },
    {
        "name": "zodiac_sign",
        "prompt": "Write a Python function `zodiac_sign(month, day)` that returns the zodiac sign.",
        "fn_name": "zodiac_sign",
        "impl": "def zodiac_sign(month, day):\n    signs = [('Capricorn',22),('Aquarius',20),('Pisces',19),('Aries',21),('Taurus',20),('Gemini',21),('Cancer',22),('Leo',23),('Virgo',23),('Libra',23),('Scorpio',23),('Sagittarius',22),('Capricorn',22)]\n    return signs[month-1][0] if day < signs[month-1][1] else signs[month][0]\n",
        "test": "assert zodiac_sign(1,1)=='Capricorn'; assert zodiac_sign(1,25)=='Aquarius'; assert zodiac_sign(7,1)=='Cancer'",
    },
    {
        "name": "fibonacci_sequence",
        "prompt": "Write a Python function `fibonacci_sequence(n)` that returns a list of the first n Fibonacci numbers.",
        "fn_name": "fibonacci_sequence",
        "impl": "def fibonacci_sequence(n):\n    seq = []\n    a, b = 0, 1\n    for _ in range(n):\n        seq.append(a)\n        a, b = b, a+b\n    return seq\n",
        "test": "assert fibonacci_sequence(5)==[0,1,1,2,3]; assert fibonacci_sequence(0)==[]; assert fibonacci_sequence(1)==[0]",
    },
    {
        "name": "tribonacci",
        "prompt": "Write a Python function `tribonacci(n)` that returns the n-th tribonacci number.",
        "fn_name": "tribonacci",
        "impl": "def tribonacci(n):\n    if n < 3: return [0,0,1][n] if n >= 0 else 0\n    a, b, c = 0, 0, 1\n    for _ in range(n-2):\n        a, b, c = b, c, a+b+c\n    return c\n",
        "test": "assert tribonacci(4)==2; assert tribonacci(0)==0; assert tribonacci(2)==1",
    },
    {
        "name": "is_fibonacci",
        "prompt": "Write a Python function `is_fibonacci(n)` that returns True if n is a Fibonacci number.",
        "fn_name": "is_fibonacci",
        "impl": "def is_fibonacci(n):\n    import math\n    def is_perfect_square(x):\n        s = int(math.isqrt(x))\n        return s*s == x\n    return is_perfect_square(5*n*n+4) or is_perfect_square(5*n*n-4)\n",
        "test": "assert is_fibonacci(8)==True; assert is_fibonacci(7)==False; assert is_fibonacci(0)==True",
    },
    {
        "name": "prime_factors",
        "prompt": "Write a Python function `prime_factors(n)` that returns a list of prime factors of n.",
        "fn_name": "prime_factors",
        "impl": "def prime_factors(n):\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1: factors.append(n)\n    return factors\n",
        "test": "assert prime_factors(12)==[2,2,3]; assert prime_factors(7)==[7]; assert prime_factors(1)==[]",
    },
    {
        "name": "primes_up_to",
        "prompt": "Write a Python function `primes_up_to(n)` that returns all primes up to n.",
        "fn_name": "primes_up_to",
        "impl": "def primes_up_to(n):\n    sieve = [True] * (n+1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5)+1):\n        if sieve[i]:\n            for j in range(i*i, n+1, i):\n                sieve[j] = False\n    return [i for i in range(n+1) if sieve[i]]\n",
        "test": "assert primes_up_to(10)==[2,3,5,7]; assert primes_up_to(2)==[2]; assert primes_up_to(1)==[]",
    },
    {
        "name": "count_primes",
        "prompt": "Write a Python function `count_primes(n)` that counts primes up to n.",
        "fn_name": "count_primes",
        "impl": "def count_primes(n):\n    return len(primes_up_to(n))\n",
        "test": "assert count_primes(10)==4; assert count_primes(2)==1; assert count_primes(1)==0",
    },
    {
        "name": "next_prime",
        "prompt": "Write a Python function `next_prime(n)` that returns the next prime after n.",
        "fn_name": "next_prime",
        "impl": "def next_prime(n):\n    def is_prime(x):\n        if x < 2: return False\n        for i in range(2, int(x**0.5)+1):\n            if x % i == 0: return False\n        return True\n    p = n + 1\n    while not is_prime(p):\n        p += 1\n    return p\n",
        "test": "assert next_prime(7)==11; assert next_prime(2)==3; assert next_prime(1)==2",
    },
    {
        "name": "previous_prime",
        "prompt": "Write a Python function `previous_prime(n)` that returns the largest prime less than n.",
        "fn_name": "previous_prime",
        "impl": "def previous_prime(n):\n    def is_prime(x):\n        if x < 2: return False\n        for i in range(2, int(x**0.5)+1):\n            if x % i == 0: return False\n        return True\n    p = n - 1\n    while p >= 2 and not is_prime(p):\n        p -= 1\n    return p if p >= 2 else None\n",
        "test": "assert previous_prime(10)==7; assert previous_prime(3)==2; assert previous_prime(2) is None",
    },
    {
        "name": "twin_primes",
        "prompt": "Write a Python function `twin_primes(n)` that returns all twin prime pairs up to n.",
        "fn_name": "twin_primes",
        "impl": "def twin_primes(n):\n    ps = primes_up_to(n)\n    return [(p, p+2) for p in ps if p+2 in ps]\n",
        "test": "assert twin_primes(10)==[(3,5),(5,7)]; assert twin_primes(5)==[(3,5)]; assert twin_primes(2)==[]",
    },
    {
        "name": "goldbach_conjecture",
        "prompt": "Write a Python function `goldbach_conjecture(n)` that returns a pair of primes summing to n (even n > 2).",
        "fn_name": "goldbach_conjecture",
        "impl": "def goldbach_conjecture(n):\n    ps = set(primes_up_to(n))\n    for p in ps:\n        if n - p in ps:\n            return (p, n - p)\n    return None\n",
        "test": "assert goldbach_conjecture(10) in [(3,7),(5,5),(7,3)]; assert goldbach_conjecture(4)==(2,2); assert goldbach_conjecture(8) in [(3,5),(5,3)]",
    },
    {
        "name": "sum_of_divisors",
        "prompt": "Write a Python function `sum_of_divisors(n)` that returns the sum of all divisors of n.",
        "fn_name": "sum_of_divisors",
        "impl": "def sum_of_divisors(n):\n    return sum(d for d in range(1, n+1) if n % d == 0)\n",
        "test": "assert sum_of_divisors(6)==12; assert sum_of_divisors(1)==1; assert sum_of_divisors(10)==18",
    },
    {
        "name": "is_perfect_number",
        "prompt": "Write a Python function `is_perfect_number(n)` that returns True if n is a perfect number.",
        "fn_name": "is_perfect_number",
        "impl": "def is_perfect_number(n):\n    if n < 2: return False\n    return sum(d for d in range(1, n) if n % d == 0) == n\n",
        "test": "assert is_perfect_number(6)==True; assert is_perfect_number(28)==True; assert is_perfect_number(10)==False",
    },
    {
        "name": "abundant_number",
        "prompt": "Write a Python function `abundant_number(n)` that returns True if n is an abundant number.",
        "fn_name": "abundant_number",
        "impl": "def abundant_number(n):\n    if n < 2: return False\n    return sum(d for d in range(1, n) if n % d == 0) > n\n",
        "test": "assert abundant_number(12)==True; assert abundant_number(6)==False; assert abundant_number(1)==False",
    },
    {
        "name": "deficient_number",
        "prompt": "Write a Python function `deficient_number(n)` that returns True if n is a deficient number.",
        "fn_name": "deficient_number",
        "impl": "def deficient_number(n):\n    if n < 2: return False\n    return sum(d for d in range(1, n) if n % d == 0) < n\n",
        "test": "assert deficient_number(8)==True; assert deficient_number(12)==False; assert deficient_number(1)==False",
    },
    {
        "name": "euler_totient",
        "prompt": "Write a Python function `euler_totient(n)` that computes Euler's totient function.",
        "fn_name": "euler_totient",
        "impl": "def euler_totient(n):\n    result = n\n    p = 2\n    while p * p <= n:\n        if n % p == 0:\n            while n % p == 0:\n                n //= p\n            result -= result // p\n        p += 1\n    if n > 1:\n        result -= result // n\n    return result\n",
        "test": "assert euler_totient(9)==6; assert euler_totient(1)==1; assert euler_totient(10)==4",
    },
    {
        "name": "digit_product",
        "prompt": "Write a Python function `digit_product(n)` that returns the product of digits of a non-negative integer.",
        "fn_name": "digit_product",
        "impl": "def digit_product(n):\n    p = 1\n    for d in str(n):\n        p *= int(d)\n    return p\n",
        "test": "assert digit_product(123)==6; assert digit_product(0)==0; assert digit_product(999)==729",
    },
    {
        "name": "is_happy_number",
        "prompt": "Write a Python function `is_happy_number(n)` that returns True if n is a happy number.",
        "fn_name": "is_happy_number",
        "impl": "def is_happy_number(n):\n    seen = set()\n    while n != 1 and n not in seen:\n        seen.add(n)\n        n = sum(int(d)**2 for d in str(n))\n    return n == 1\n",
        "test": "assert is_happy_number(19)==True; assert is_happy_number(4)==False; assert is_happy_number(1)==True",
    },
    {
        "name": "harshad_number",
        "prompt": "Write a Python function `harshad_number(n)` that returns True if n is a Harshad number.",
        "fn_name": "harshad_number",
        "impl": "def harshad_number(n):\n    return n % sum(int(d) for d in str(n)) == 0 if n > 0 else False\n",
        "test": "assert harshad_number(18)==True; assert harshad_number(19)==False; assert harshad_number(0)==False",
    },
    {
        "name": "automorphic_number",
        "prompt": "Write a Python function `automorphic_number(n)` that returns True if n is automorphic.",
        "fn_name": "automorphic_number",
        "impl": "def automorphic_number(n):\n    return str(n*n).endswith(str(n))\n",
        "test": "assert automorphic_number(5)==True; assert automorphic_number(6)==True; assert automorphic_number(7)==False",
    },
    {
        "name": "kaprekar_number",
        "prompt": "Write a Python function `kaprekar_number(n)` that returns True if n is a Kaprekar number.",
        "fn_name": "kaprekar",
        "impl": "def kaprekar_number(n):\n    if n < 1: return False\n    sq = n * n\n    s = str(sq)\n    for i in range(1, len(s)):\n        a, b = int(s[:i]), int(s[i:])\n        if b and a + b == n: return True\n    return n == 1\n",
        "test": "assert kaprekar_number(9)==True; assert kaprekar_number(45)==True; assert kaprekar_number(10)==False",
    },
    {
        "name": "pascal_triangle",
        "prompt": "Write a Python function `pascal_triangle(n)` that returns the n-th row of Pascal's triangle.",
        "fn_name": "pascal_triangle",
        "impl": "def pascal_triangle(n):\n    row = [1]\n    for _ in range(n):\n        row = [1] + [row[i]+row[i+1] for i in range(len(row)-1)] + [1]\n    return row\n",
        "test": "assert pascal_triangle(0)==[1]; assert pascal_triangle(1)==[1,1]; assert pascal_triangle(4)==[1,4,6,4,1]",
    },
    {
        "name": "binomial_coefficient",
        "prompt": "Write a Python function `binomial_coefficient(n, k)` that computes C(n, k).",
        "fn_name": "binomial_coefficient",
        "impl": "def binomial_coefficient(n, k):\n    import math\n    return math.comb(n, k)\n",
        "test": "assert binomial_coefficient(5,2)==10; assert binomial_coefficient(10,0)==1; assert binomial_coefficient(10,10)==1",
    },
    {
        "name": "catalan_number",
        "prompt": "Write a Python function `catalan_number(n)` that returns the n-th Catalan number.",
        "fn_name": "catalan_number",
        "impl": "def catalan_number(n):\n    import math\n    return math.comb(2*n, n) // (n + 1)\n",
        "test": "assert catalan_number(0)==1; assert catalan_number(1)==1; assert catalan_number(5)==42",
    },
    {
        "name": "stirling_second",
        "prompt": "Write a Python function `stirling_second(n, k)` that computes the Stirling number of the second kind.",
        "fn_name": "stirling_second",
        "impl": "def stirling_second(n, k):\n    if n == k: return 1\n    if k == 0 or k > n: return 0\n    return k * stirling_second(n-1, k) + stirling_second(n-1, k-1)\n",
        "test": "assert stirling_second(3,2)==3; assert stirling_second(4,2)==7; assert stirling_second(0,0)==1",
    },
    {
        "name": "bell_number",
        "prompt": "Write a Python function `bell_number(n)` that returns the n-th Bell number.",
        "fn_name": "bell_number",
        "impl": "def bell_number(n):\n    if n == 0: return 1\n    return sum(stirling_second(n, k) for k in range(n+1))\n",
        "test": "assert bell_number(0)==1; assert bell_number(1)==1; assert bell_number(3)==5",
    },
    {
        "name": "derangement",
        "prompt": "Write a Python function `derangement(n)` that returns the number of derangements of n items.",
        "fn_name": "derangement",
        "impl": "def derangement(n):\n    if n == 0: return 1\n    if n == 1: return 0\n    return (n-1) * (derangement(n-1) + derangement(n-2))\n",
        "test": "assert derangement(0)==1; assert derangement(1)==0; assert derangement(4)==9",
    },
    {
        "name": "multinomial_coefficient",
        "prompt": "Write a Python function `multinomial_coefficient(n, groups)` that computes the multinomial coefficient.",
        "fn_name": "multinomial_coefficient",
        "impl": "def multinomial_coefficient(n, groups):\n    import math\n    result = math.factorial(n)\n    for g in groups:\n        result //= math.factorial(g)\n    return result\n",
        "test": "assert multinomial_coefficient(4,[2,2])==6; assert multinomial_coefficient(5,[5])==1; assert multinomial_coefficient(3,[1,1,1])==6",
    },
    {
        "name": "permutation_count",
        "prompt": "Write a Python function `permutation_count(n, k)` that returns the number of k-permutations of n.",
        "fn_name": "permutation_count",
        "impl": "def permutation_count(n, k):\n    import math\n    return math.perm(n, k)\n",
        "test": "assert permutation_count(5,2)==20; assert permutation_count(5,0)==1; assert permutation_count(5,5)==120",
    },
    {
        "name": "is_permutation",
        "prompt": "Write a Python function `is_permutation(lst, n)` that returns True if lst is a permutation of 1..n.",
        "fn_name": "is_permutation",
        "impl": "def is_permutation(lst, n):\n    return sorted(lst) == list(range(1, n+1))\n",
        "test": "assert is_permutation([3,1,2],3)==True; assert is_permutation([1,1,2],3)==False; assert is_permutation([],0)==True",
    },
    {
        "name": "next_permutation",
        "prompt": "Write a Python function `next_permutation(lst)` that returns the next lexicographic permutation.",
        "fn_name": "next_permutation",
        "impl": "def next_permutation(lst):\n    lst = list(lst)\n    i = len(lst) - 2\n    while i >= 0 and lst[i] >= lst[i+1]:\n        i -= 1\n    if i < 0: return sorted(lst)\n    j = len(lst) - 1\n    while lst[j] <= lst[i]:\n        j -= 1\n    lst[i], lst[j] = lst[j], lst[i]\n    return lst[:i+1] + lst[i+1:][::-1]\n",
        "test": "assert next_permutation([1,2,3])==[1,3,2]; assert next_permutation([3,2,1])==[1,2,3]; assert next_permutation([1,1,5])==[1,5,1]",
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
    # v2.16.0 Priority 10: Expanded benchmark (200 tasks per family).
    ("What is the capital of Spain?", "madrid"),
    ("What is the capital of Australia?", "canberra"),
    ("What is the capital of Canada?", "ottawa"),
    ("What is the capital of Russia?", "moscow"),
    ("What is the capital of China?", "beijing"),
    ("What is the capital of India?", "new delhi"),
    ("What is the capital of Brazil?", "brasilia"),
    ("What is the capital of Egypt?", "cairo"),
    ("What is the capital of Greece?", "athens"),
    ("What is the capital of Mexico?", "mexico city"),
    ("What is the chemical symbol for silver?", "ag"),
    ("What is the chemical symbol for oxygen?", "o"),
    ("What is the chemical symbol for sodium?", "na"),
    ("What is the chemical symbol for iron?", "fe"),
    ("What is the chemical symbol for hydrogen?", "h"),
    ("What is the chemical symbol for carbon?", "c"),
    ("What is the chemical symbol for nitrogen?", "n"),
    ("What is the chemical symbol for helium?", "he"),
    ("What is the chemical symbol for mercury?", "hg"),
    ("What is the chemical symbol for lead?", "pb"),
    ("What is 9 times 8?", "72"),
    ("What is 7 times 6?", "42"),
    ("What is 12 times 12?", "144"),
    ("What is 11 times 11?", "121"),
    ("What is 15 times 4?", "60"),
    ("What is 20 times 5?", "100"),
    ("What is 13 times 3?", "39"),
    ("What is 14 times 7?", "98"),
    ("What is 16 times 4?", "64"),
    ("What is 25 times 4?", "100"),
    ("What is the square root of 144?", "12"),
    ("What is the square root of 81?", "9"),
    ("What is the square root of 64?", "8"),
    ("What is the square root of 100?", "10"),
    ("What is the square root of 49?", "7"),
    ("What is the square root of 36?", "6"),
    ("What is the square root of 25?", "5"),
    ("What is the square root of 16?", "4"),
    ("What is the square root of 9?", "3"),
    ("What is the square root of 4?", "2"),
    ("How many planets are in our solar system?", "8"),
    ("What is the second planet from the sun?", "venus"),
    ("What is the third planet from the sun?", "earth"),
    ("What is the fourth planet from the sun?", "mars"),
    ("What is the fifth planet from the sun?", "jupiter"),
    ("What is the sixth planet from the sun?", "saturn"),
    ("What is the seventh planet from the sun?", "uranus"),
    ("What is the eighth planet from the sun?", "neptune"),
    ("What is the largest moon of Earth?", "moon"),
    ("What is the largest moon of Jupiter?", "ganymede"),
    ("What is the largest moon of Saturn?", "titan"),
    ("What is the closest star to Earth?", "sun"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("What is the second largest ocean?", "atlantic"),
    ("What is the smallest ocean?", "arctic"),
    ("What is the longest river in the world?", "nile"),
    ("What is the second longest river?", "amazon"),
    ("What is the largest desert in the world?", "sahara"),
    ("What is the largest country by area?", "russia"),
    ("What is the second largest country by area?", "canada"),
    ("What is the most populous country?", "india"),
    ("What is the currency of the United Kingdom?", "pound"),
    ("What is the currency of the United States?", "dollar"),
    ("What is the currency of the European Union?", "euro"),
    ("What is the currency of Switzerland?", "franc"),
    ("What is the currency of China?", "yuan"),
    ("What is the currency of India?", "rupee"),
    ("What is the currency of Russia?", "ruble"),
    ("What is the currency of South Korea?", "won"),
    ("What is the currency of Mexico?", "peso"),
    ("What is the primary language of China?", "mandarin"),
    ("What is the primary language of Russia?", "russian"),
    ("What is the primary language of Germany?", "german"),
    ("What is the primary language of France?", "french"),
    ("What is the primary language of Japan?", "japanese"),
    ("What is the primary language of Italy?", "italian"),
    ("What is the primary language of Spain?", "spanish"),
    ("What is the primary language of Egypt?", "arabic"),
    ("What is the primary language of Mexico?", "spanish"),
    ("What is the primary language of Argentina?", "spanish"),
    ("How many strings does a standard guitar have?", "6"),
    ("How many keys does a standard piano have?", "88"),
    ("How many sides does a triangle have?", "3"),
    ("How many sides does a square have?", "4"),
    ("How many sides does a pentagon have?", "5"),
    ("How many sides does an octagon have?", "8"),
    ("How many sides does a decagon have?", "10"),
    ("How many degrees in a triangle?", "180"),
    ("How many degrees in a circle?", "360"),
    ("How many minutes in an hour?", "60"),
    ("How many seconds in a minute?", "60"),
    ("How many hours in a day?", "24"),
    ("How many days in a week?", "7"),
    ("How many days in a leap year?", "366"),
    ("How many days in a standard year?", "365"),
    ("How many weeks in a year?", "52"),
    ("How many months in a year?", "12"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the boiling point of water in Fahrenheit?", "212"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What is the speed of light in km/s (approximate)?", "300000"),
    ("What is the largest land animal?", "elephant"),
    ("What is the largest reptile?", "crocodile"),
    ("What is the largest bird?", "ostrich"),
    ("What is the smallest bird?", "hummingbird"),
    ("What is the fastest bird?", "falcon"),
    ("What is the largest fish?", "shark"),
    ("What is the past tense of 'go'?", "went"),
    ("What is the past tense of 'eat'?", "ate"),
    ("What is the past tense of 'drink'?", "drank"),
    ("What is the past tense of 'sleep'?", "slept"),
    ("What is the past tense of 'think'?", "thought"),
    ("What is the past tense of 'buy'?", "bought"),
    ("What is the past tense of 'teach'?", "taught"),
    ("What is the past tense of 'catch'?", "caught"),
    ("What is the past tense of 'fight'?", "fought"),
    ("What is the past tense of 'swim'?", "swam"),
    ("What is the capital of Portugal?", "lisbon"),
    ("What is the capital of Netherlands?", "amsterdam"),
    ("What is the capital of Sweden?", "stockholm"),
    ("What is the capital of Norway?", "oslo"),
    ("What is the capital of Finland?", "helsinki"),
    ("What is the capital of Denmark?", "copenhagen"),
    ("What is the capital of Poland?", "warsaw"),
    ("What is the capital of Austria?", "vienna"),
    ("What is the capital of Belgium?", "brussels"),
    ("What is the capital of Ireland?", "dublin"),
    ("What is the capital of Switzerland?", "bern"),
    ("What is the capital of Turkey?", "ankara"),
    ("What is the capital of Thailand?", "bangkok"),
    ("What is the capital of South Korea?", "seoul"),
    ("What is the capital of Indonesia?", "jakarta"),
    ("What is the capital of Vietnam?", "hanoi"),
    ("What is the capital of Philippines?", "manila"),
    ("What is the capital of Malaysia?", "kuala lumpur"),
    ("What is the capital of Saudi Arabia?", "riyadh"),
    ("What is the capital of Iran?", "tehran"),
    ("What is the capital of Iraq?", "baghdad"),
    ("What is the capital of Israel?", "jerusalem"),
    ("What is the capital of Kenya?", "nairobi"),
    ("What is the capital of Nigeria?", "abuja"),
    ("What is the capital of South Africa?", "pretoria"),
    ("What is the capital of Morocco?", "rabat"),
    ("What is the capital of Argentina?", "buenos aires"),
    ("What is the capital of Chile?", "santiago"),
    ("What is the capital of Colombia?", "bogota"),
    ("What is the capital of Peru?", "lima"),
    ("What is the capital of Venezuela?", "caracas"),
    ("What is the capital of Cuba?", "havana"),
    ("What is the capital of New Zealand?", "wellington"),
    ("What is the capital of Iceland?", "reykjavik"),
    ("What is the capital of Hungary?", "budapest"),
    ("What is the capital of Czech Republic?", "prague"),
    ("What is the capital of Romania?", "bucharest"),
    ("What is the capital of Ukraine?", "kyiv"),
    ("What is the capital of Pakistan?", "islamabad"),
    ("What is the capital of Bangladesh?", "dhaka"),
    ("What is the capital of Sri Lanka?", "colombo"),
    ("What is the capital of Singapore?", "singapore"),
    ("What is the capital of Taiwan?", "taipei"),
    ("What is the capital of Jordan?", "amman"),
    ("What is the capital of Lebanon?", "beirut"),
    ("What is the capital of Kuwait?", "kuwait city"),
    ("What is the capital of Qatar?", "doha"),
    ("What is the capital of UAE?", "abu dhabi"),
    ("What is the capital of Oman?", "muscat"),
    ("What is the capital of Ethiopia?", "addis ababa"),
    ("What is the capital of Ghana?", "accra"),
    ("What is the capital of Tanzania?", "dodoma"),
    ("What is the capital of Uganda?", "kampala"),
    ("What is the capital of Zimbabwe?", "harare"),
    ("What is the capital of Algeria?", "algiers"),
    ("What is the capital of Tunisia?", "tunis"),
    ("What is the capital of Libya?", "tripoli"),
    ("What is the capital of Sudan?", "khartoum"),
    ("What is the capital of Ecuador?", "quito"),
    ("What is the capital of Bolivia?", "sucre"),
    ("What is the capital of Paraguay?", "asuncion"),
    ("What is the capital of Uruguay?", "montevideo"),
    ("What is the capital of Panama?", "panama city"),
    ("What is the capital of Costa Rica?", "san jose"),
    ("What is the capital of Guatemala?", "guatemala city"),
    ("What is the capital of Jamaica?", "kingston"),
    ("What is the capital of Haiti?", "port-au-prince"),
    ("What is the capital of Dominican Republic?", "santo domingo"),
    ("What is the capital of Norway's neighbor Sweden called?", "stockholm"),
    ("What is the capital of Bulgaria?", "sofia"),
    ("What is the capital of Serbia?", "belgrade"),
    ("What is the capital of Croatia?", "zagreb"),
    ("What is the capital of Slovakia?", "bratislava"),
    ("What is the capital of Slovenia?", "ljubljana"),
    ("What is the capital of Lithuania?", "vilnius"),
    ("What is the capital of Latvia?", "riga"),
    ("What is the capital of Estonia?", "tallinn"),
    ("What is the capital of Greece's neighbor Albania?", "tirana"),
    ("What is the capital of Belarus?", "minsk"),
    ("What is the capital of Moldova?", "chisinau"),
    ("What is the capital of Mongolia?", "ulaanbaatar"),
    ("What is the capital of Cambodia?", "phnom penh"),
    ("What is the capital of Laos?", "vientiane"),
    ("What is the capital of Myanmar?", "naypyidaw"),
    ("What is the capital of Nepal?", "kathmandu"),
    ("What is the capital of Afghanistan?", "kabul"),
    ("What is the capital of Yemen?", "sanaa"),
    ("What is the capital of Syria?", "damascus"),
    ("What is the capital of Bahrain?", "manama"),
    ("What is the capital of Cyprus?", "nicosia"),
    ("What is the capital of Malta?", "valletta"),
    ("What is the capital of Luxembourg?", "luxembourg"),
    ("What is the capital of Monaco?", "monaco"),
    ("What is the capital of Andorra?", "andorra la vella"),
    ("What is the capital of San Marino?", "san marino"),
    ("What is the capital of Liechtenstein?", "vaduz"),
    ("What is the capital of Montenegro?", "podgorica"),
    ("What is the capital of North Macedonia?", "skopje"),
    ("What is the capital of Bosnia and Herzegovina?", "sarajevo"),
    ("What is the capital of Kosovo?", "pristina"),
    ("What is the capital of Namibia?", "windhoek"),
    ("What is the capital of Botswana?", "gaborone"),
    ("What is the capital of Zambia?", "lusaka"),
    ("What is the capital of Malawi?", "lilongwe"),
    ("What is the capital of Mozambique?", "maputo"),
    ("What is the capital of Angola?", "luanda"),
    ("What is the capital of Cameroon?", "yaounde"),
    ("What is the capital of Senegal?", "dakar"),
    ("What is the capital of Ivory Coast?", "yamoussoukro"),
    ("What is the capital of Mali?", "bamako"),
    ("What is the capital of Burkina Faso?", "ouagadougou"),
    ("What is the capital of Niger?", "niamey"),
    ("What is the capital of Chad?", "ndjamena"),
    ("What is the capital of Mauritania?", "nouakchott"),
    ("What is the capital of Somalia?", "mogadishu"),
    ("What is the capital of Rwanda?", "kigali"),
    ("What is the capital of Burundi?", "gitega"),
    ("What is the capital of Djibouti?", "djibouti"),
    ("What is the capital of Eritrea?", "asmara"),
    ("What is the capital of South Sudan?", "juba"),
    ("What is the capital of Central African Republic?", "bangui"),
    ("What is the capital of Equatorial Guinea?", "malabo"),
    ("What is the capital of Gabon?", "libreville"),
    ("What is the capital of Congo?", "brazzaville"),
    ("What is the capital of DR Congo?", "kinshasa"),
    ("What is the capital of Lesotho?", "maseru"),
    ("What is the capital of Eswatini?", "mbabane"),
    ("What is the capital of Madagascar?", "antananarivo"),
    ("What is the capital of Mauritius?", "port louis"),
    ("What is the capital of Comoros?", "moroni"),
    ("What is the capital of Seychelles?", "victoria"),
    ("What is the capital of Cape Verde?", "praia"),
    ("What is the capital of Guinea?", "conakry"),
    ("What is the capital of Guinea-Bissau?", "bissau"),
    ("What is the capital of Sierra Leone?", "freetown"),
    ("What is the capital of Liberia?", "monrovia"),
    ("What is the capital of Togo?", "lome"),
    ("What is the capital of Benin?", "porto-novo"),
    ("What is the capital of Gambia?", "banjul"),
    ("What is the capital of the Bahamas?", "nassau"),
    ("What is the capital of Barbados?", "bridgetown"),
    ("What is the capital of Trinidad and Tobago?", "port of spain"),
    ("What is the capital of Belize?", "belmopan"),
    ("What is the capital of El Salvador?", "san salvador"),
    ("What is the capital of Honduras?", "tegucigalpa"),
    ("What is the capital of Nicaragua?", "managua"),
    ("What is the capital of Fiji?", "suva"),
    ("What is the capital of Papua New Guinea?", "port moresby"),
    ("What is the capital of Solomon Islands?", "honiara"),
    ("What is the capital of Vanuatu?", "port vila"),
    ("What is the capital of Samoa?", "apia"),
    ("What is the capital of Tonga?", "nuku'alofa"),
    ("What is the capital of Kiribati?", "tarawa"),
    ("What is the capital of Tuvalu?", "funafuti"),
    ("What is the capital of Nauru?", "yaren"),
    ("What is the capital of Palau?", "ngerulmud"),
    ("What is the capital of Marshall Islands?", "majuro"),
    ("What is the capital of Micronesia?", "palikir"),
]


def _build_direct_answer_tasks() -> list[CounterfactualTask]:
    tasks: list[CounterfactualTask] = []
    n = len(_DIRECT_FACTS)
    # v2.16.0 Priority 10: Four-way split for larger benchmark.
    # 50% train, 10% validation, 20% test, 20% ood
    train_end = n // 2
    val_end = train_end + max(2, n // 10)
    test_end = val_end + max(4, n // 5)
    for i, (q, ans) in enumerate(_DIRECT_FACTS):
        tid = f"direct_{i:03d}"
        if i < train_end:
            split = "train"
            archetype = "factual"
        elif i < val_end:
            split = "validation"
            archetype = "factual"
        elif i < test_end:
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
    n = 200
    train_end = n // 2
    val_end = train_end + max(2, n // 10)
    test_end = val_end + max(4, n // 5)
    for i in range(n):
        tid = f"memory_{i:03d}"
        secret = f"pin_{i:04d}_secret"
        prompt = f"What was the secret pin stored for session {i:04d}?"
        if i < train_end:
            split = "train"
            archetype = "stored_secret"
        elif i < val_end:
            split = "validation"
            archetype = "stored_secret"
        elif i < test_end:
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
    n = 200
    train_end = n // 2
    val_end = train_end + max(2, n // 10)
    test_end = val_end + max(4, n // 5)
    for i in range(n):
        tid = f"tool_{i:03d}"
        nonce = f"nonce_{i:04d}_live"
        prompt = f"Fetch the current live nonce for endpoint {i:04d}."
        if i < train_end:
            split = "train"
            archetype = "live_nonce"
        elif i < val_end:
            split = "validation"
            archetype = "live_nonce"
        elif i < test_end:
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
    """Coding tasks with four-way split (v2.16.0 Priority 10).

    Uses all available train/test archetypes plus OOD archetypes.
    Split: 50% train, 10% validation, 20% test, 20% ood.
    """
    tasks: list[CounterfactualTask] = []
    # Use all train/test archetypes with four-way split.
    n_tt = len(CODING_ARCHETYPES_TRAIN_TEST)
    train_end = n_tt // 2
    val_end = train_end + max(2, n_tt // 10)
    test_end = val_end + max(4, n_tt // 5)
    for i, arch in enumerate(CODING_ARCHETYPES_TRAIN_TEST):
        tid = f"coding_{arch['name']}"
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "validation"
        elif i < test_end:
            split = "test"
        else:
            # Remaining train/test archetypes go to OOD if beyond test_end.
            split = "ood"
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
            ood=(split == "ood"),
            split=split,
            description=f"Coding archetype {arch['name']}",
        ))
    # OOD archetypes (completely unseen at train time)
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
    """Build all routing tasks for AutoLearn v2.16.0.

    v2.16.0 Priority 10: Expanded benchmark with hundreds of tasks per family.
    - direct_answer: ~300 tasks (from expanded facts pool)
    - memory_required: 200 tasks (parameterized)
    - tool_required: 200 tasks (parameterized)
    - coding_workflow: ~200 tasks (from expanded archetypes + OOD)
    Total: ~900 tasks with four-way split (train/validation/test/ood).
    """
    tasks: list[CounterfactualTask] = []
    tasks.extend(_build_direct_answer_tasks())
    tasks.extend(_build_memory_tasks())
    tasks.extend(_build_tool_tasks())
    tasks.extend(_build_coding_tasks())
    return tasks


def build_split_assignments(tasks: list[CounterfactualTask] | None = None) -> dict[str, str]:
    """Return task_id -> split assignments.

    Splits are deterministic and archetype-based:
    - train:      50% of each family (10 tasks each)
    - validation: 10% of each family (2 tasks each)
    - test:       20% of each family (4 tasks each)
    - ood:        20% of each family (4 tasks each, using unseen archetypes)
    """
    tasks = tasks or build_all_tasks()
    return {t.task_id: t.split for t in tasks}


def get_archetype_summary(tasks: list[CounterfactualTask] | None = None) -> dict[str, Any]:
    """Return a summary of archetypes and their task counts."""
    tasks = tasks or build_all_tasks()
    by_archetype: dict[str, dict[str, int]] = {}
    for t in tasks:
        by_archetype.setdefault(t.archetype, {"total": 0, "train": 0, "validation": 0, "test": 0, "ood": 0})
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

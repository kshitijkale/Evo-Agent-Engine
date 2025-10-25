import dspy
import os
import sys
import traceback
import random # Added for seeding and sampling
from appworld_proxy import AppWorldProxy, load_task_ids_proxy

# --- 1. LLM Configuration (Using Gemini - User Provided Names) ---

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

# Safety settings (optional, can be commented out if causing issues)
# safety_settings = {
#     'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
#     'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
#     'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
#     'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
# }

# Student model (Gemini Flash - user provided name)
task_lm = dspy.LM(
    model="gemini/gemini-2.5-flash", # User provided name
    api_key=google_api_key,
    max_tokens=8192, # Using the value from the previous successful script
    temperature=0.0,
    # safety_settings=safety_settings
)
dspy.settings.configure(lm=task_lm)

# Teacher model (Gemini Pro - user provided name)
reflection_lm = dspy.LM(
    model="gemini/gemini-2.5-pro", # User provided name
    api_key=google_api_key,
    max_tokens=8192, # Using the value from the previous successful script
    temperature=0.7,
    # safety_settings=safety_settings
)


# --- Helper Function to Get Task Instruction ---
def get_task_instruction(task_id, appworld_python_path):
    """Safely opens AppWorldProxy just to get the instruction text."""
    try:
        with AppWorldProxy(
            task_id=task_id,
            experiment_name="get_instruction",
            appworld_python_path=appworld_python_path
        ) as world_proxy:
            return world_proxy.task.instruction
    except Exception as e:
        # Removed emoji
        print(f"  [Helper] Warning: Failed to load task {task_id} for instruction check: {e}", flush=True)
        return None

# --- 2. The DSPy Signature and Program (with Few-Shot Example) ---

class AppWorldReActStep(dspy.Signature):
    """
    <<< START OF INSTRUCTIONS >>>

    **ROLE AND GOAL:**
    You are a super intelligent AI Assistant. Your job is to achieve day-to-day tasks completely autonomously by interacting with app APIs on the supervisor's behalf using a Python REPL environment.

    **INTERACTION MODEL:**
    This is a multi-step process. At each step:
    1. You receive the overall task and the history of previous (code, output) pairs.
    2. You generate the *next* single block of Python code to execute.
    3. The environment executes the code and provides the output.
    4. Repeat until the goal is achieved.

    **KEY API DOCUMENTATION TOOLS:**
    Use these `api_docs` functions to explore:
    - `print(apis.api_docs.show_app_descriptions())`: List available apps.
    - `print(apis.api_docs.show_api_descriptions(app_name='<app_name>'))`: List APIs for a specific app.
    - `print(apis.api_docs.show_api_doc(app_name='<app_name>', api_name='<api_name>'))`: Get details (arguments, response) for a specific API.

    **EXAMPLE WALKTHROUGH:**

    Consider the task: "What is the password for my Spotify account?"
    (Supervisor details like name/email are implicitly available via `apis.supervisor.*`)

    ASSISTANT:
    # Okay. Lets first find which apps are available to get the password by looking at the app descriptions.
    print(apis.api_docs.show_app_descriptions())

    USER:
    [
      {"name": "api_docs", "description": "..."},
      {"name": "supervisor", "description": "An app to access supervisor's personal information..."},
      {"name": "spotify", "description": "..."},
      {"name": "venmo", "description": "..."},
      ...
    ]

    ASSISTANT:
    # Looks like the supervisor app could help me with that. Lets see what apis are available under this app.
    print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

    USER:
    [ ..., "show_account_passwords : Show your supervisor's account passwords.", ...]

    ASSISTANT:
    # I can use `show_account_passwords` to get the passwords. Let me see its detailed specification.
    print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

    USER:
    {
      'app_name': 'supervisor', 'api_name': 'show_account_passwords', 'path': '/account_passwords',
      'method': 'GET', 'description': "...", 'parameters': [],
      'response_schemas': {'success': [{'account_name': 'string', 'password': 'string'}], 'failure': {'message': 'string'}}
    }

    ASSISTANT:
    # Okay, it requires no arguments. So I can just call it directly.
    print(apis.supervisor.show_account_passwords())

    USER:
    [
      {"account_name": "spotify", "password": "dummy_spotify_pass"},
      {"account_name": "file_system", "password": "dummy_fs_pass"},
      ...
    ]

    ASSISTANT:
    # So the Spotify password is an entry in the list where account_name="spotify".
    # Need to handle potential errors if the API call failed or didn't return a list.
    # Let's assume the previous output was assigned to a variable, e.g., 'passwords_list'.
    # In a real scenario, the agent should check the type and content first.
    # For this example, let's extract it assuming success:
    passwords_list = [
      {"account_name": "spotify", "password": "dummy_spotify_pass"},
      {"account_name": "file_system", "password": "dummy_fs_pass"},
    ] # Simulating assignment from previous step
    spotify_password = None
    if isinstance(passwords_list, list):
        for account in passwords_list:
            if account.get("account_name") == "spotify":
                spotify_password = account.get("password")
                break
    print(f"Found Spotify password: {spotify_password}") # Print for confirmation or debugging

    USER:
    Found Spotify password: dummy_spotify_pass

    ASSISTANT:
    # When the task is completed, I need to call apis.supervisor.complete_task().
    # Since the task asked for the password, I pass it as the answer.
    apis.supervisor.complete_task(answer=spotify_password)

    USER:
    Marked the active task complete.

    **END EXAMPLE WALKTHROUGH**

    ---

    **CORE INSTRUCTIONS AND DISCLAIMERS (Apply these strictly):**
    1.  **Code Validity:** Generate *only* valid Python code blocks. Put thoughts *only* in comments (`#`). Do *not* add extra formatting (like ```) or markers (like `[[ ## completed ## ]]`).
    2.  **State/Variables:** Assume variables from previous steps are available.
    3.  **Step Size:** Write small, incremental code (one logical step or API call). Check outputs before proceeding.
    4.  **Environment:** Standard Python library available. OS/file system/process calls are *disabled*.
    5.  **File System App:** "file system" refers to the `file_system` app via `apis.file_system.*`. Do *not* use the `os` module.
    6.  **API Only:** Use *only* the provided `apis.<app>.<api>()` calls. Do *not* import external packages.
    7.  **API Specs & Errors:** Follow API documentation (`show_api_doc`). **CRITICAL: Check API call outputs.** Handle potential errors (e.g., failed calls might return error messages, not expected data). Do not assume success.
    8.  **Pagination:** Handle multi-page API results completely.
    9.  **Time:** Use `datetime.now()` for current time.
    10. **Temporal Queries:** Use precise time boundaries (00:00:00-23:59:59).
    11. **People:** Refers to contacts in `apis.phone.list_contacts()`.
    12. **Supervisor Data:** Access supervisor info (credentials, etc.) via `apis.supervisor.*`.
    13. **Completion Call:** MUST call `apis.supervisor.complete_task()` when task is done.
    14. **Completion with Answer:** Use `apis.supervisor.complete_task(answer=...)` if information was requested. Verify answer variable exists.
    15. **Answer Format:** Answers must be entities/numbers (e.g., `10`, `'pass123'`), *not* full sentences. Digits for numbers.
    16. **Giving Up:** Call `apis.supervisor.complete_task(status="fail")` if stuck or certain of failure.
    17. **Autonomy & Self-Correction:** Decide autonomously. If code causes a `Traceback`, analyze the error in the history and *fix the code* in the next step. Do *not* repeat failing code.

    <<< END OF INSTRUCTIONS >>>
    """

    # --- Dynamic Inputs Needed at Each Step ---
    task_instruction: str = dspy.InputField(desc="The user's overall high-level goal for this task.")
    history: str = dspy.InputField(desc="The log of previous execution steps: 'ASSISTANT:\\n<code>\\n\\nUSER:\\n<output>' pairs.")

    # --- Expected Output at Each Step ---
    code: str = dspy.OutputField(prefix="# ", desc="The next valid Python code block to execute.")


class AppWorldAgent(dspy.Module):
    """
    This is the optimizable DSPy program.
    Its forward() method runs the FULL agent loop for a given task
    and returns a prediction containing the score and feedback trace.
    """
    def __init__(self, max_interactions=30):
        super().__init__()
        # Use the NEW signature here
        self.react_step = dspy.Predict(AppWorldReActStep, **{"temperature": 0.0})
        self.max_interactions = max_interactions

    def forward(self, task_id, task_instruction):
        """
        Runs the full agent loop for a single task.
        ADDED: More verbose printing for observability.
        """
        print(f"  .. Starting Task: {task_id}", flush=True)

        history = []
        full_trace_log = f"--- Task: {task_instruction} ---\n\n"
        score = 0.0
        failed = False

        try:
            with AppWorldProxy(
                task_id=task_id,
                experiment_name="gepa_eval", # Use a consistent name or make dynamic if needed
                appworld_python_path=APPWORLD_PYTHON
            ) as world_proxy:

                for i in range(self.max_interactions):
                    print(f"\n{'='*20} Step {i+1}/{self.max_interactions} {'='*20}", flush=True)

                    history_str = "\n\n".join(
                        f"ASSISTANT:\n{code}\n\nUSER:\n{out}" for code, out in history
                    )

                    # --- 1. OBSERVABILITY: Print context sent to LLM ---
                    print("\n--- Context Sent to LLM ---", flush=True)
                    print(f"Task Instruction: {task_instruction}", flush=True)
                    if history_str:
                        print(f"History (last 1000 chars):\n...\n{history_str[-1000:]}", flush=True)
                    else:
                        print("History: [Empty]", flush=True)
                    print("-" * 30, flush=True)

                    try:
                        # DSPy uses the signature to format the prompt correctly
                        prediction = self.react_step(
                            task_instruction=task_instruction, # Pass the task goal
                            history=history_str              # Pass the conversation history
                        )
                        code = prediction.code.strip()
                        full_trace_log += f"--- STEP {i+1} CODE ---\n{code}\n\n"

                        # --- 2. OBSERVABILITY: Print generated code ---
                        print("\n--- Code Generated by LLM ---", flush=True)
                        print(code, flush=True)
                        print("-" * 30, flush=True)

                    except Exception as e:
                        error_msg = f"Agent failed to generate code: {traceback.format_exc()}"
                        full_trace_log += f"--- AGENT ERROR ---\n{error_msg}\n"
                        print(f"\n    Agent Error: {e}", flush=True)
                        score = 0.0
                        failed = True
                        break # Exit loop

                    # --- Execute Code ---
                    output = world_proxy.execute(code)
                    full_trace_log += f"--- STEP {i+1} OUTPUT ---\n{output}\n\n"
                    history.append((code, output))

                    # --- 3. OBSERVABILITY: Print environment output ---
                    print("\n--- Environment Output ---", flush=True)
                    print(output, flush=True)
                    print("-" * 30, flush=True)

                    # --- Check for errors/completion ---
                    if "Traceback (most recent call last):" in output:
                        full_trace_log += "--- EXECUTION FAILED (Traceback) ---"
                        print("\n    Evaluation Failed (Code Traceback)", flush=True)
                        score = 0.0
                        failed = True # Mark as failed, but continue loop

                    elif world_proxy.task_completed():
                        full_trace_log += "--- TASK COMPLETED (SUCCESS) ---"
                        print("\n    Evaluation Success", flush=True)
                        score = 1.0
                        break # Exit loop
                    else:
                        # Reset failed flag if the step succeeded after a failure
                        failed = False


                # Check failure reason *after* the loop finishes
                if score == 0.0:
                    if failed: # If the *last* step failed with a traceback
                        pass # The log already notes the traceback failure
                    else: # If it just timed out
                        full_trace_log += "--- FAILED (Max Interactions) ---"
                        print("\n    Evaluation Failed (Max Interactions)", flush=True)

        except Exception as e:
            score = 0.0
            full_trace_log += f"Error during proxy/world execution for {task_id}: {traceback.format_exc()}"
            print(f"\n    Proxy Error: {e}", flush=True)

        # Return prediction for the metric
        return dspy.Prediction(score=score, feedback=full_trace_log)


# --- 3. The Execution-Based Metric (The "Fitness Function") ---

def gepa_appworld_metric(gold, pred, trace, pred_name=None, pred_trace=None):
    """
    Receives the 'gold' example and the 'pred' (Prediction)
    from the AppWorldAgent.forward() call.
    """
    score = getattr(pred, 'score', 0.0)
    feedback = getattr(pred, 'feedback', "No feedback provided by agent.")
    return {'score': score, 'feedback': feedback}


# --- 4. Trainset Creation (MODIFIED FOR RANDOM SAMPLING) ---

def create_trainset(dataset_name="dev", num_examples=10):
    """
    Create a trainset of random dspy.Examples.
    MODIFIED: Selects random tasks.
    """
    print(f"\nCreating Random Trainset from '{dataset_name}' (sampling {num_examples} examples)", flush=True)

    # Load ALL task IDs from the dataset
    all_task_ids = load_task_ids_proxy(dataset_name, APPWORLD_PYTHON)
    if not all_task_ids:
        print("Error: No task IDs loaded for training set.", flush=True)
        return []

    # Ensure we don't try to sample more tasks than available
    actual_num_to_sample = min(num_examples, len(all_task_ids))
    if actual_num_to_sample < num_examples:
         print(f"Warning: Only {len(all_task_ids)} tasks available in '{dataset_name}'. Sampling {actual_num_to_sample} for training.", flush=True)

    # Select random task IDs using the seed set in main
    selected_task_ids = random.sample(all_task_ids, actual_num_to_sample)
    print(f"Selected random task IDs for training: {selected_task_ids}", flush=True)

    # Load examples for selected tasks
    trainset = []
    print("Loading details for selected training tasks...", flush=True)
    for i, task_id in enumerate(selected_task_ids):
         print(f"  [{i+1}/{actual_num_to_sample}] Loading task {task_id}...", end="", flush=True)
         instruction = get_task_instruction(task_id, APPWORLD_PYTHON) # Use the helper function
         if instruction:
             example = dspy.Example(
                 task_id=task_id,
                 task_instruction=instruction
             ).with_inputs("task_id", "task_instruction")
             trainset.append(example)
             print(" Done.", flush=True)
         else:
             print(" Failed to load, skipping.", flush=True)

    if not trainset:
        print("Error: Failed to load details for any selected training tasks.", flush=True)
        return []
    else:
        print(f"\nCreated trainset with {len(trainset)} random examples.", flush=True)

    return trainset


# --- 5. Main Optimizer and Evaluation Logic (MODIFIED FOR RANDOM SAMPLING) ---

def run_evaluation(agent_to_test, dataset_name, num_tasks):
    """
    Runs the agent against a random sample of tasks from the test set.
    MODIFIED: Selects random tasks.
    """
    print("\n" + "="*60, flush=True)
    print(f"RUNNING EVALUATION on '{dataset_name}' (sampling {num_tasks} random tasks)", flush=True)
    print("="*60, flush=True)

    # --- Task Selection Logic ---
    print(f"Loading task IDs from '{dataset_name}' to sample {num_tasks} tasks...", flush=True)
    # Load ALL task IDs from the dataset
    all_task_ids = load_task_ids_proxy(dataset_name, APPWORLD_PYTHON)
    if not all_task_ids:
        print("Error: No task IDs loaded. Exiting evaluation.", flush=True)
        return

    # Ensure we don't try to sample more tasks than available
    actual_num_to_sample = min(num_tasks, len(all_task_ids))
    if actual_num_to_sample < num_tasks:
         print(f"Warning: Only {len(all_task_ids)} tasks available in '{dataset_name}'. Sampling {actual_num_to_sample}.", flush=True)

    # Select random task IDs using the seed set in main
    selected_task_ids = random.sample(all_task_ids, actual_num_to_sample)
    print(f"Selected random task IDs: {selected_task_ids}", flush=True)

    # Now, load the examples *only* for the selected tasks
    test_set = []
    print("Loading details for selected tasks...", flush=True)
    for i, task_id in enumerate(selected_task_ids):
         print(f"  [{i+1}/{actual_num_to_sample}] Loading task {task_id}...", end="", flush=True)
         instruction = get_task_instruction(task_id, APPWORLD_PYTHON) # Use the helper function
         if instruction:
             example = dspy.Example(
                 task_id=task_id,
                 task_instruction=instruction
             ).with_inputs("task_id", "task_instruction")
             test_set.append(example)
             print(" Done.", flush=True)
         else:
             print(" Failed to load, skipping.", flush=True)

    if not test_set:
        print("Error: Failed to load details for any selected tasks. Exiting evaluation.", flush=True)
        return
    # --- End Task Selection Logic ---

    success_count = 0
    total_evaluated = len(test_set)

    for index, example in enumerate(test_set):
        task_id = example.task_id
        task_instruction = example.task_instruction

        print(f"\n{'*'*20} Task {index+1}/{total_evaluated} ({task_id}) {'*'*20}", flush=True)
        print(f"Instruction: {task_instruction}\n", flush=True)

        try:
            # Call the agent's forward method directly.
            prediction = agent_to_test(
                task_id=task_id,
                task_instruction=task_instruction
            )

            score = getattr(prediction, 'score', 0.0)
            if score == 1.0:
                success_count += 1

        except Exception as e:
            print(f"  FAILED TO RUN TASK: {e}\n{traceback.format_exc()}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"EVALUATION SUMMARY: {success_count}/{total_evaluated} random tasks completed successfully", flush=True)
    print(f"{'='*60}\n", flush=True)

# --- Main Execution Block ---

APPWORLD_PYTHON = os.path.join(os.getcwd(), "env_appworld", "bin", "python")
OPTIMIZED_PROMPT_FILE = "optimized_agent_instructions_with_fewshot.txt" # Using the correct filename

if __name__ == "__main__":
    # --- ADDED SEED ---
    SEED = 42
    random.seed(SEED)
    print(f"Using random seed: {SEED}", flush=True)
    # --- END SEED ---

    mode = sys.argv[1] if len(sys.argv) > 1 else "optimize"

    if not os.path.exists(APPWORLD_PYTHON):
        print(f"Error: AppWorld Python not found at {APPWORLD_PYTHON}", flush=True)
        print("Please update the APPWORLD_PYTHON variable in this script.", flush=True)
        sys.exit(1)

    if mode == "optimize":
        print("\nMode: Optimize Agent Instructions (with Few-Shot Example)\n", flush=True)
        trainset = create_trainset(dataset_name="dev", num_examples=10) # Using random sampling
        if not trainset:
            print("Error: No training examples loaded. Exiting.", flush=True)
            sys.exit(1)

        unoptimized_agent = AppWorldAgent(max_interactions=30)

        from dspy.teleprompt import GEPA

        optimizer = GEPA(
            metric=gepa_appworld_metric,
            reflection_lm=reflection_lm,
            auto="light",
            num_threads=4,
        )

        print("\n--- Starting GEPA Optimization... ---", flush=True)
        print(f"Evaluating {len(trainset)} random tasks with 'light' settings.", flush=True)
        optimized_agent = optimizer.compile(unoptimized_agent, trainset=trainset)
        print("--- GEPA Optimization Complete! ---", flush=True)

        evolved_instructions = optimized_agent.react_step.signature.instructions
        with open(OPTIMIZED_PROMPT_FILE, "w") as f:
            f.write(evolved_instructions)
        print(f"\nSaved optimized instructions to: {OPTIMIZED_PROMPT_FILE}", flush=True)

        print("\n--- Original Instructions ---", flush=True)
        print(unoptimized_agent.react_step.signature.instructions[:500] + "...", flush=True)
        print("\n--- Evolved Instructions ---", flush=True)
        print(evolved_instructions[:500] + "...", flush=True)

        run_evaluation(
            agent_to_test=optimized_agent,
            dataset_name="test_normal",
            num_tasks=5 # Evaluate on 5 random tasks
        )

    elif mode == "test_optimized":
        print("\nMode: Test with Optimized Instructions (with Few-Shot Example)\n", flush=True)
        try:
            with open(OPTIMIZED_PROMPT_FILE, "r") as f:
                optimized_instructions = f.read()
            print(f"Loaded optimized instructions from {OPTIMIZED_PROMPT_FILE}", flush=True)
            optimized_agent = AppWorldAgent(max_interactions=50)
            optimized_agent.react_step.signature.instructions = optimized_instructions
            run_evaluation(
                agent_to_test=optimized_agent,
                dataset_name="test_normal",
                num_tasks=10 # Test on 10 random tasks
            )
        except FileNotFoundError:
            print(f"Error: {OPTIMIZED_PROMPT_FILE} not found!", flush=True)
            print("Run 'python <your_script_name>.py optimize' first.", flush=True)

    elif mode == "test_baseline":
        print("\nMode: Test with Baseline Instructions (with Few-Shot Example)\n", flush=True)
        unoptimized_agent = AppWorldAgent(max_interactions=50)
        run_evaluation(
            agent_to_test=unoptimized_agent,
            dataset_name="test_normal",
            num_tasks=10 # Test on 10 random tasks
        )

    else:
        print(f"Unknown mode: {mode}", flush=True)
        print("Usage: python <your_script_name>.py [optimize|test_optimized|test_baseline]", flush=True)
        print("  optimize        - Run GEPA optimization and save the new prompt.", flush=True)
        print("  test_optimized  - Test the previously saved optimized prompt.", flush=True)
        print("  test_baseline   - Test the original, unoptimized prompt.", flush=True)
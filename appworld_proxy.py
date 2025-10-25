# appworld_proxy.py
"""
Proxy to call AppWorld from DSPy environment via subprocess.
"""
import subprocess
import json
import os

class AppWorldProxy:
    """Proxy to communicate with AppWorld in separate environment"""
    
    def __init__(self, task_id, experiment_name="dspy_experiment", appworld_python_path=None):
        self.task_id = task_id
        self.experiment_name = experiment_name
        
        if appworld_python_path is None:
            self.appworld_python = os.path.join(os.getcwd(), "env_appworld", "bin", "python")
        else:
            self.appworld_python = appworld_python_path
        
        self.bridge_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "appworld_bridge.py"
        )
        
        self._completed = False
        self._task_info = None
        self._load_task_info()
    
    def _call_bridge(self, command, *args):
        """Call the bridge script with given command and arguments"""
        cmd = [self.appworld_python, self.bridge_script, command] + list(args)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"success": False, "error": f"Bridge call failed: {result.stderr}"}
        
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Failed to parse JSON: {e}\nOutput: {result.stdout}"}
    
    def _load_task_info(self):
        """Load task information from AppWorld"""
        result = self._call_bridge("get_task_info", self.task_id)
        
        if result.get("success"):
            self._task_info = {
                "supervisor": result["supervisor"],
                "instruction": result["instruction"]
            }
        else:
            raise Exception(f"Failed to load task info: {result.get('error')}")
    
    @property
    def task(self):
        """Mimic AppWorld's task property"""
        class TaskInfo:
            def __init__(self, supervisor, instruction):
                self.supervisor = supervisor
                self.instruction = instruction
        
        return TaskInfo(self._task_info["supervisor"], self._task_info["instruction"])
    
    def execute(self, code):
        """Execute code in AppWorld environment"""
        result = self._call_bridge("execute", self.task_id, code, self.experiment_name)
        
        if result.get("success"):
            self._completed = result.get("completed", False)
            return result.get("output", "")
        else:
            error_msg = result.get("error", "Unknown error")
            return f"ERROR: {error_msg}"
    
    def task_completed(self):
        """Check if task is completed"""
        return self._completed
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def load_task_ids_proxy(dataset_name, appworld_python_path=None):
    """Load task IDs via bridge"""
    if appworld_python_path is None:
        appworld_python_path = os.path.join(os.getcwd(), "env_appworld", "bin", "python")
    
    bridge_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "appworld_bridge.py"
    )
    
    result = subprocess.run(
        [appworld_python_path, bridge_script, "get_task_ids", dataset_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Failed to load task IDs: {result.stderr}")
    
    try:
        data = json.loads(result.stdout)
        if data.get("success"):
            return data.get("task_ids", [])
        else:
            raise Exception(f"Error loading task IDs: {data.get('error')}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON: {e}\nOutput: {result.stdout}")

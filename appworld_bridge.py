# appworld_bridge.py
"""
Bridge script to run AppWorld in separate environment.
Run this in env_appworld with Pydantic v1.
"""
import sys
import json
from appworld import AppWorld, load_task_ids

def execute_code(task_id, code, experiment_name):
    """Execute code in AppWorld environment"""
    try:
        with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
            output = world.execute(code)
            return {
                "success": True,
                "output": output,
                "completed": world.task_completed(),
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "completed": False,
            "error": str(e)
        }

def get_task_info(task_id):
    """Get task information"""
    try:
        with AppWorld(task_id=task_id, experiment_name="get_info") as world:
            return {
                "success": True,
                "supervisor": world.task.supervisor,
                "instruction": world.task.instruction,
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_task_ids(dataset_name):
    """Get task IDs from dataset"""
    try:
        task_ids = load_task_ids(dataset_name)
        return {
            "success": True,
            "task_ids": task_ids,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No command provided"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "execute":
        task_id = sys.argv[2]
        code = sys.argv[3]
        experiment_name = sys.argv[4] if len(sys.argv) > 4 else "bridge"
        result = execute_code(task_id, code, experiment_name)
    
    elif command == "get_task_info":
        task_id = sys.argv[2]
        result = get_task_info(task_id)
    
    elif command == "get_task_ids":
        dataset_name = sys.argv[2]
        result = get_task_ids(dataset_name)
    
    else:
        result = {"success": False, "error": f"Unknown command: {command}"}
    
    print(json.dumps(result))

import sys
from task_od.data import get_task_by_id, get_all_task_ids
import random

def main():
    # Get a random task ID
    all_task_ids = get_all_task_ids()
    task_id = random.choice(all_task_ids)
    
    # Get the task
    try:
        task = get_task_by_id(task_id)
        print(f"Successfully retrieved task with ID {task_id}")
        print(f"Task keys: {list(task.keys())}")
        print(f"Question: {task['question']}")
        print(f"Answer: {task['answer']}")
    except Exception as e:
        print(f"Error retrieving task with ID {task_id}: {e}")
        raise

if __name__ == "__main__":
    main()

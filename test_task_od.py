from task_od.data import get_task_by_id, get_all_task_ids
import random

def main():
    task_id = random.choice(get_all_task_ids())
    task = get_task_by_id(task_id)
    print(f"Successfully retrieved task with ID {task_id}")
    print(f"Task keys: {list(task.keys())}")
    print(f"Question: {task['question']}")
    print(f"Answer: {task['answer']}")

if __name__ == "__main__":
    main()

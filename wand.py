import wandb
import sqlite3

# Initialize wandb (assuming this is part of a script where a run is active)
wandb.init(project="my_project")

# Step 1: Get the current run's details
run_id = wandb.run.id
entity = wandb.run.entity
project = wandb.run.project

# Step 2: Query the wandb API for the run's log data
api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")
history = run.history()  # Fetch the logged metrics as a pandas DataFrame

# Step 3: Add the log data to a local SQLite database
conn = sqlite3.connect('local.db')
cursor = conn.cursor()

# Create a table if it doesn't exist (adjust schema based on your metrics)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        step INTEGER,
        loss REAL,
        accuracy REAL
    )
''')

# Insert the log data into the table
for index, row in history.iterrows():
    cursor.execute("INSERT INTO logs (step, loss, accuracy) VALUES (?, ?, ?)",
                   (index, row.get('loss'), row.get('accuracy')))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Log data successfully added to local database.")
import wandb
from PIL import Image

# Initialize the W&B API
api = wandb.Api()

# Fetch the run (replace with your entity, project, and run ID)
run = api.run("your_entity_name/your_project_name/your_run_id")

# Retrieve the logged table
table = run.logged_table("agent_run_table")
df = table.get_dataframe()

# Access text data
text_prompts = df["text_prompt"]
responses = df["response"]

# Example: Compute average response length
avg_response_length = responses.apply(len).mean()
print(f"Average response length: {avg_response_length}")

# Access and process images
for index, row in df.iterrows():
    image_ref = row["image"]  # Path to the image in W&B
    # Find the corresponding file in run.files()
    image_file = next(f for f in run.files() if f.name == image_ref)
    image_file.download(root="downloaded_images", replace=True)
    local_path = f"downloaded_images/{image_ref}"
    img = Image.open(local_path)
    # Example: Process the image (e.g., compute stats, feed to another model)
    print(f"Processed image for call {row['call_id']}: {local_path}")

# Optional: Additional stats or optimization
# Example: Count specific keywords in responses
keyword = "example"
keyword_count = responses.str.count(keyword).sum()
print(f"Keyword '{keyword}' appears {keyword_count} times")
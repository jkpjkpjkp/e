import polars as pl
from PIL import Image
from anode import LLM
import io
df = pl.read_parquet('zerobench_subquestions-00000-of-00001.parquet')

print(df.head())
print(df.columns)

def get_all_task_ids():
    a = df.filter(pl.col('question_images_decoded').list.len() == 1)
    return list(a['question_id'])

def get_task_by_id(id):
    rows = df.filter(pl.col('question_id') == id)
    assert len(rows) == 1
    row = rows.row(0, named=True)
    images = [Image.open(io.BytesIO(x['bytes'])) for x in row['question_images_decoded']]
    assert len(images) == 1
    row['image'] = images[0]
    row['question'] = row['question_text']
    row['answer'] = row['question_answer']
    row['id'] = id
    ret = dict(row)
    return ret

def llm_as_judge(expected_output, prediction):
    llm = LLM('deepseek-chat')
    
    prompt = f"""You are a judge evaluating the correctness of a prediction. Compare the prediction with the expected output.

correct answer: {expected_output}
Prediction to be judged: {prediction}

Please return 1 if the prediction is correct; that is, it leads to the correct answer. Otherwise return 0.

Respond with only the numerical score (0 or 1)."""
    try:
        score = llm.aask(prompt)
        return float(score.strip())
    except Exception as e:
        print(f"Error calculating score: {e}")
        return 0.0

def loss(output, answer) -> float:
    return llm_as_judge(expected_output=answer, prediction=output)
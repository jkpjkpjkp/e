from anode import lmm

def run(image, question):
    info = []
    for _ in range(MAX_ITERATIONS):
        response = lmm(image, question, *info, TOOL_USE_PROMPT)
        tool_call = parse(response)
        if not tool_call:
            return response
        tool_response = tool_call()
        info.append(tool_response)
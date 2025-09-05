SYSTEM_PROMPT_VLM_AS_JUDGE = """You are tasked with comparing two images displaying the same frame of a video but annotated with different methods.
An intersection of the annotations is also shown in the third image to make their differences more apparent.
Your goal is to analyse and compare the two methods according to how accurately they represent the image content.
Respond in the following JSON format: {json_schema}"""
SYSTEM_PROMPT_VLM_ANNOTATOR = "As an AI Assistant, you specialize in accurate image object detection, delivering coordinates in JSON format: {json_schema}"

USER_PROMPT_VLM_AS_JUDGE = "Which is better and why?"
USER_PROMPT_VLM_ANNOTATOR = "Outline the position of each object and output all the bbox coordinates in JSON format."

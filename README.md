# Llama 3.2 1B-Instruct

This repository provides a Jupyter notebook that demonstrates how to use the **Llama 3.2 1B-Instruct** model for various natural language processing tasks, including summarization, text rewriting, question answering, creative writing, translation, and text classification.

## Overview of the Llama 3.2 Model

**Llama 3.2** is a state-of-the-art language model developed by Meta. It is designed to generate human-like text based on the input it receives, making it suitable for a variety of applications in natural language understanding and generation. The **1B-Instruct** version is specifically fine-tuned for following instructions and producing relevant outputs.

## Installation

To get started, ensure you have the necessary packages installed. You can install them via pip:

```bash
git clone https://github.com/EngAhmed19/Llama3.2_1B-instruct_illustration
cd Llama3.2_1B-instruct_illustration
!pip install -r requirements.txt
```

## Usage
### Authentication
 To access the model, you need to authenticate your Hugging Face account:
 ```python
from huggingface_hub import login
login(token='Your_Token_here')  # Replace with your Hugging Face token
```

### Initialize the Model Pipeline
```python
import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    device="cuda:0"  # Specify your device (cpu, cuda:0)
)
```
## Tasks

You can perform various tasks by defining appropriate functions:

-Summarization
-Text Rewriting
-Question Answering
-Creative Writing
-Translation
-Text Classification

## Example: Summarization
Here's how to summarize text using the model:

```python
def llama_pipeline(prompt, assistant):
    messages = [
    {"role": "system", "content": "Summarize the following."},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": assistant}
    ]
    response = pipe(
        messages,
        max_new_tokens=256,            # Control the length of the output
        #temperature=0.9,              # Introduce randomness ,degree of creativity
        #top_k=50,                     # Limit the number of candidate tokens
        #top_p=0.95,                   # Nucleus sampling
        #repetition_penalty=1.5        # Penalize repetition
    )
    return response[0]["generated_text"][3]["content"] #return the summary result
# Example prompt
pmt = "Put yout text Here"

# Call the function with the prompt and the assistant
result = llama_pipeline(pmt, "bullet points")
print(result)
```
**You can find more examples and detailed usage instructions in the Jupyter notebook provided in this repository.**

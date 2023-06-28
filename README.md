# Generalis SD test

### A test of a SD model merge, hosted in Hugging Face and usable in python

__https://huggingface.co/vluz/Generalis_V1__

This is a minimal viable implementation that meets project requirements:
- Generalist Model, can generate anything
- Muted colours compared to other SD models
- Low vram use at the cost of 25% of generation time
- Permissive license to allow anyone to use, modify, etc.
- Published as safetensors for safety
- encoded vae for simplicity
- works well with DDIM sampler at low iteration count

<hr>

Open a command prompt and `cd` to a new directory of your choosing:

Create a virtual environment with:
```
python -m venv "venv"
venv\Scripts\activate
```

To install do:
```
git clone https://github.com/vluz/GeneralisSDtest.git
cd GeneralisSDtest
pip install -r requirements.txt
```

To run do:<br>
```
python test.py
``` 

***Takes a long time to run the first time as*** 
<br>
***it has to download a large amount of files***

To exit the virtual environment do:
```
venv\Scripts\deactivate
```

<hr>

Example output:
<br>
![Image1](cat01.png?raw=true "Image 1")

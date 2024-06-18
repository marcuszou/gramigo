# gramigo - a grammar checker app using pytorch

# source / references
### Please go to https://pytorch.org/get-started/locally/ for a greater understanding of pytorch package
### 
### Gramformer website:
###   pip install git+https://github.com/PrithivirajDamodaran/Gramformer

# Pre-requisites - Install the Python3.11 packages in requirement.txt
#     pip install -r requirements.txt
#
# Pre-install nlp, spacy packages and load up "en_core_web_sm" - RUN ONCE ONLY
#     pip install nlp spacy
#     python -m spacy download en_core_web_sm

#
# Import Gramformer
from gramformer import Gramformer
import gradio as gr

# Create an instance with 'model' parameter: 1 = corrector, 2 = detector
# '1 = corrector' is available while '2 = detector' is coming soon as of 17 June 2024
gf = Gramformer(models=1, use_gpu=False) 

def correct(sentence):
    res = gf.correct(sentence) # Gramformer correct
    return res # Return first value in res array

app_inputs = gr.components.Textbox(lines=2, placeholder="Enter sentence here...")

interface = gr.Interface(fn=correct, 
                        inputs=app_inputs,
                        outputs='text', 
                        title='I\'m Gramigo')

interface.launch()

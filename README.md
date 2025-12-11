# Ethics Assignment 3

## Files

The main folders in this repo are run_eval_deepseek.py, run_eval_gemini.py, score_responses.py, and prompts_master.csv
These are the only 4 files needed to recreate all the results. Right now all the generated files are in the repo but they can be generated again

## Prerequisites

- Python 3.8 or higher
- `pip` installed

## Setup Instructions

1. **Clone the Repository**  
   Clone this repository to your local machine:

   ```bash
   git clone <repository-url>
   cd assignment3
   ```

2. **Create a Virtual Environment**  
   Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

Before running any code, you must create an API Key for both Gemini and Deepseek
Place the API keys in a .env file. The scripts will be able to read API keys from the .env file

1) Run 'run_eval_deepseek.py' and 'run_eval_gemini.py' initially, this will take around 30 minutes to run since we use a reasoning model for deepseek. Once this script is run, it will generate two json - one for the gemini reponses and one for the deepseek responses.

2) Run 'score_responses.py' twice with the following command python score_responses.py "gemini responses json" gemini_scores.csv and
python score_responses.py "deepseek responses json" deepseek_scores.csv. 
This will automatically score all the responses and print the scores to the terminal and also store them in a csv

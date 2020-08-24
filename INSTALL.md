## Requirements

- Python 3.6

- Required packages:

  - `pip install hyperopt`
    
  - `pip install scikit-learn==0.20.4`
  
  - `pip install iteration_utilities` 
  
  - `pip install imbalanced-learn==0.4`  
  
  - `pip install func_timeout`
  
  - `pip install minepy`
  
  - `pip install info_gain`
  
  - `pip install powerlaw`
  
  - `pip install cvxopt`
  
  - `pip install pyyaml`
  
  - `pip install lockfile`
  
  - `pip install smac==0.8.0 ``
  
    > If you get some error during smac installation, please using "sudo apt-get install build-essential swig" in terminal.

## Quick start 

1. To reproduce the result of `BiLO-CPDP`, make sure you are in `code/Bilevel`, then

   ```python
   # lower-level budget = 100 function evaluations
   python FE100.py 
   # lower-level budget = 20s
   python Time20.py
   ```

2. To reproduce the result of `Auto-Sklearn (modified)`, make sure you are in `code/Auto_CPDP`

   ```python
   python exam.py
   ```

3. You can check the results with our experimental results in `result` directory . 
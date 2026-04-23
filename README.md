# Frangipani Flower Detection

So I made my own dataset by taking a lot of pictures, well it's not enought to called it "a lot" tho. But I learn a bit how to make a dataset and annotate it using Label Studio.

# YOLO Training

I used YOLOv8 nano as the base model and trained it on my custom dataset.

# Evaluation

I evaluated the model on my custom dataset and got the following results:

mAP50: 0.81
mAP50-95: 0.75
Precision: 0.85
Recall: 0.83

# Prediction

I tested the model on a few images and got the following results:

![test1](test/test1.jpeg)
![test2](test/test2.jpeg)
![test3](test/test3.jpeg)
![test4](test/test4.jpeg)

# How to setup when you clone this repo

```bash
# Create a virtual environment
python -m venv {env_name}

# Activate the virtual environment
# Windows
{env_name}\Scripts\activate

# Powershell
./{env_name}/Scripts/Activate.ps1

# Mac/Linux
source {env_name}/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Label Studio (only for the first time)
label-studio start

# Train the model
python train.py

# Evaluate the model
python evaluate.py

# Test the model
python predict.py
```

Or you can also run them 1 by 1 using `main.ipynb` file. Just don't forget to activate the virtual environment first, or else you'll get a module not found error.

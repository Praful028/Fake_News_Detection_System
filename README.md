# Fake_News_Detection_System
Fake_News_Detection_System

Dataset and Pickle file are in the drive 
link : https://drive.google.com/drive/folders/177AeLjLnwEF9VDv2zWPnr4Weovbj-40s?usp=drive_link

Installation GuideLines :-

Install this libraries :

pip install transformers
pip install pycaret
pip install flask

Import this libraries : 

import numpy as np
import pandas as pd
import pycaret
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import pickle
from flask import Flask, render_template, request
from transformers import AutoModel, BertTokenizerFast
import numpy as np

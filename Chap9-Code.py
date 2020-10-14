# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:05:30 2020

@author: ADMIN
"""


import tensorflow as tf
tf.__version__


import json

with open("WMBot.json","r") as f:
    payload = json.loads(f.read())
    
    
utterances = payload['utterances'] 
ut2  = [x['text'] for x in utterances if x['intent'] == "GetInfo"]   

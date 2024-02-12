import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

b = bagreader('2024-01-25-02-47-15.bag')

csvfiles = []
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)

# TREEMATE/intertemporal.py
import pandas as pd
import numpy as np
from . import const
from . import bdl as forest
from . import econ as econ


def producers_surplus_sum_over_time(harvest_areas, k, p0, epsilon):
    surplus = []
    harvest_area = forest.predict_harvest_area_allspecies(harvest_areas)
    for t in range(len(harvest_areas[0])):
        surplus.append(econ.producers_surplus(harvest_area[t], k, p0, epsilon[t]))
    return surplus

def function(surplus_over_time):
    
    return sum(surplus_over_time)
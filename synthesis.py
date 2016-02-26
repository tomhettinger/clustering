# -*- coding: utf-8 -*-
"""
@author: Thomas Hettinger
"""
import pandas as pd
import numpy as np
from numpy.random import normal

# Definition of parameters for baseline distribution of physical characteristics
#   given in tuples (mean, std).
AGE = (45, 5)             # years
WEIGHT = (60, 3)          # kg
HEARTRATE = (80, 10)      # bpm
HEIGHT_FEMALE = (160, 10) # cm
HEIGHT_MALE = (170, 10)   # cm

WRITE_PATH = u'C:\\Users\\Tom\\Desktop\\clustering\\data.csv'


def generate_person(age_par=AGE, weight_par=WEIGHT, heartrate_par=HEARTRATE,
                    height_female_par=HEIGHT_FEMALE, height_male_par=HEIGHT_MALE, 
                    female_fraction=0.5):
    """Generate an individuals characteristics given the (mean,std) definitions for
    each characteristic.  Return a tuple (age, weight, heartrate, height, gender)."""
    if np.random.random() < female_fraction:
        gender = 'f'
        height = int(normal(*height_female_par))
    else:
        gender = 'm'
        height = int(normal(*height_male_par))
    age = int(normal(*age_par))
    weight = int(normal(*weight_par))
    heartrate = int(normal(*heartrate_par))
    return (age, weight, heartrate, height, gender)


def main():
    """Generate 4 groups of individuals"""
    np.random.seed(1337)
    sample = []

    # Group A: Short-Active
    # These individuals have exceptionaly short heights, low heartrate, and low weight
    for i in range(0, 50, 1):
        phy_char = generate_person(weight_par=(WEIGHT[0] - 10, WEIGHT[1]), 
                                   heartrate_par=(HEARTRATE[0] - 35, HEARTRATE[1]), 
                                   height_female_par=(HEIGHT_FEMALE[0] - 30, HEIGHT_FEMALE[1]), 
                                   height_male_par=(HEIGHT_MALE[0] - 30, HEIGHT_MALE[1])
                                   )
        sample.append( (i, "Short-Active") + phy_char )

    # Group B: Tall-Active
    # These individuals have exceptionaly tall heights, low heartrate, and low weight
    for i in range(50, 120, 1):
        phy_char = generate_person(weight_par=(WEIGHT[0] - 11, WEIGHT[1]), 
                                   heartrate_par=(HEARTRATE[0] - 30, HEARTRATE[1]), 
                                   height_female_par=(HEIGHT_FEMALE[0] + 30, HEIGHT_FEMALE[1]), 
                                   height_male_par=(HEIGHT_MALE[0] + 30, HEIGHT_MALE[1])
                                   )
        sample.append( (i, "Tall-Active") + phy_char )
        
    # Group C: Docile-Male
    # These individuals have exceptionaly high heartrate, high weight, and are all male
    for i in range(120, 155, 1):
        phy_char = generate_person(weight_par=(WEIGHT[0] + 10, WEIGHT[1]), 
                                   heartrate_par=(HEARTRATE[0] + 36, HEARTRATE[1]), 
                                   female_fraction = 0.0
                                   )
        sample.append( (i, "Docile-Male") + phy_char )

    # Group D: Tall-Slender
    # These individuals have exceptionaly tall heights, low weight, and average heartrate
    for i in range(155, 190, 1):
        phy_char = generate_person(weight_par=(WEIGHT[0] - 9, WEIGHT[1]), 
                                   height_female_par=(HEIGHT_FEMALE[0] + 28, HEIGHT_FEMALE[1]), 
                                   height_male_par=(HEIGHT_MALE[0] + 27, HEIGHT_MALE[1])
                                   )
        sample.append( (i, "Tall-Slender") + phy_char )

    # Group E: Heavy-Active-Female
    # These individuals are all female, have high weight, and low heartrate
    for i in range(190, 200, 1):
        phy_char = generate_person(weight_par=(WEIGHT[0] + 12, WEIGHT[1]), 
                                   heartrate_par=(HEARTRATE[0] - 37, HEARTRATE[1]), 
                                   female_fraction = 1.0
                                   )
        sample.append( (i, "Heavy-Active-Female") + phy_char )


    # Write to CSV file
    col_names = ['id', 'group', 'age', 'weight', 'heartrate', 'height', 'gender']
    sample_frame = pd.DataFrame(sample, columns=col_names)    
    print sample_frame.info()
    sample_frame.to_csv(WRITE_PATH, index=False)


if __name__ == "__main__":
    main()
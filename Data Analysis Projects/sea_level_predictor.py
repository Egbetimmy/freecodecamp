import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')

    # Create scatter plot
    x=df["Year"]
    y=df["CSIRO Adjusted Sea Level"]
    plt.scatter(x, y)

    # Create first line of best fit
    line1 = linregress(x, y)
    slope = line1.slope
    intercept = line1.intercept
    x_pred = pd.Series([i for i in range(1880,2051)])
    y_pred = slope * x_pred + intercept
    plt.plot(x_pred, y_pred, 'r')

    # Create second line of best fit
    x2 = df[ df['Year'] >= 2000 ]['Year']
    y2 = df[ df['Year'] >= 2000 ]['CSIRO Adjusted Sea Level']

    line2 = linregress(x2, y2)
    slope2 = line2.slope
    intercept2 = line2.intercept
    x_pred2 = pd.Series([i for i in range(2000,2051)])
    y_pred2 = slope2 * x_pred2 + intercept2
    plt.plot(x_pred2, y_pred2, 'r')

    # Add labels and title
    plt.xlabel("Year")
    plt.ylabel("Sea Level (inches)")
    plt.title("Rise in Sea Level")
    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()

draw_plot()

# %%
import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
import pandas as pd


class plotresult:
    def __init__(self, savefile):
        container = np.load(savefile)
        self.sim_result = [container[key] for key in container]

    def infectiongrowth(self):
        num_infected = []
        for t in range(len(self.sim_result)):
            num_infected.append(len(np.where(self.sim_result[t] < 30)[0]))
        infected_growth_df = pd.DataFrame({'num_infected': num_infected, 'Day': [
                                          i for i in range(len(self.sim_result))]})
        output_notebook()
        p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
        p1.grid.grid_line_alpha = 0.3
        p1.xaxis.axis_label = 'Date'
        p1.yaxis.axis_label = 'Number of the infected cases'

        p1.line(
            infected_growth_df['Day'], infected_growth_df['num_infected'], color='#A6CEE3', line_width=3)
        p1.circle(
            infected_growth_df['Day'], infected_growth_df['num_infected'], fill_color="white", size=5)
        show(p1)


        # %%
if __name__ == "__main__":
    result = 'outfile.npz'
    testplot = plotresult(result)
    testplot.infectiongrowth()

# %%

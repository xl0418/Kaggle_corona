# %%
import numpy as np
import itertools
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from multiprocessing import Pool


class CoronaSim:
    def __init__(self, grid_size, initial_virus, recover_time, speedreaction, incubation, virulence, contactsize=1, num_cores=4):
        self.sim_grid = np.zeros(shape=[grid_size, grid_size])
        ini_x_virus = np.random.randint(
            low=0, high=grid_size, size=initial_virus)
        ini_y_virus = np.random.randint(
            low=0, high=grid_size, size=initial_virus)
        self.inistate_matrix = np.zeros(shape=[grid_size, grid_size])
        self.inistate_matrix.fill(float(recover_time))
        self.inistate_matrix[ini_x_virus, ini_y_virus] = 7
        self.speedreaction = speedreaction
        self.incubation = incubation
        self.samplesize = contactsize
        self.virulence = virulence
        self.num_cores = num_cores
        self.all_sites = list(itertools.product(
            range(self.sim_grid.shape[0]), range(self.sim_grid.shape[0])))

    def mechanismcheck(self):
        state_value = np.arange(31)
        valuedf = pd.DataFrame(
            {'state': state_value, 'Activity': self.activity(state_value)})
        f1 = px.scatter(valuedf, x="state", y="Activity")

        distance = np.arange(200)
        disp = np.exp(-self.gm_virulence(20)*distance**2)
        contactdf = pd.DataFrame({'distance': distance, 'disp': disp})
        f2 = px.scatter(contactdf, x="distance", y="disp")

        infected_num = np.arange(10000)
        measuredf = pd.DataFrame(
            {'infected_num': infected_num, 'measure': self.gm_virulence(infected_num)})
        f3 = px.scatter(measuredf, x="infected_num", y="measure")

        trace1 = f1['data'][0]
        trace2 = f2['data'][0]
        trace3 = f3['data'][0]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=False, subplot_titles=(
            "Figure 1", "Figure 2", "Figure 3"))
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)
        fig.add_trace(trace3, row=3, col=1)

        # Update xaxis properties
        fig.update_xaxes(title_text="Health state", row=1, col=1)
        fig.update_xaxes(title_text="Distance", range=[10, 50], row=2, col=1)
        fig.update_xaxes(title_text="The number of infected cases",
                         showgrid=False, row=3, col=1)

        # Update yaxis properties
        fig.update_yaxes(title_text="Willingness", row=1, col=1)
        fig.update_yaxes(title_text="Contact rate",
                         showgrid=False, row=2, col=1)
        fig.update_yaxes(
            title_text="Intensity of the restriction", row=3, col=1)

        fig['layout'].update(height=800, width=800, showlegend=False)
        fig.show()

    def activity(self, state):
        disp = np.exp((state-self.incubation) ** 2 /
                      self.virulence ** 2)
        return disp

    def gm_virulence(self, infected_num):
        return 100*(2/(1+np.exp(-infected_num*self.speedreaction/(self.sim_grid.shape[0]*self.sim_grid.shape[1])))-1)

    def spread_prob(self, x_row, y_col, state, seed=1):
        np.random.seed(seed)
        distance_sites = np.linalg.norm(
            np.array(self.all_sites) - np.array([x_row, y_col]), axis=1)
        Act = self.activity(state)
        gm_virulence = self.gm_virulence(
            infected_num=len(np.where(state < 30)[0]))
        prob_spread = np.exp(-gm_virulence *
                             distance_sites ** 2) * Act[x_row, y_col] * Act.flatten()
        prob_spread[x_row*self.sim_grid.shape[1]+y_col] = 0
        focal_state = np.random.choice(range(
            self.sim_grid.shape[0]*self.sim_grid.shape[1]), size=self.samplesize, p=prob_spread/sum(prob_spread))
        focal_state_value = 0 if min(state.flatten()[focal_state]) < 30 else 30
        return focal_state_value

    def simspread(self, t_end, savefile):
        self.savefile = savefile
        state_matrix = self.inistate_matrix
        output_list = []
        parallel_cores = Pool(self.num_cores)
        for t in range(t_end):
            num_infected = len(np.where(state_matrix < 30)[0])
            print(
                f'At Day {t}, {num_infected} infected cases are confirmed...')
            healthy_individual_index_row = np.where(state_matrix >= 30)[0]
            healthy_individual_index_col = np.where(state_matrix >= 30)[1]
            change_state = parallel_cores.starmap(self.spread_prob,
                                                  zip(healthy_individual_index_row, healthy_individual_index_col, itertools.repeat(state_matrix)))
            state_matrix[healthy_individual_index_row,
                         healthy_individual_index_col] = change_state
            state_matrix += 1
            output_list.append(state_matrix.tolist())
        np.savez(self.savefile, *output_list)
        return state_matrix


    # %%
if __name__ == "__main__":
    test = CoronaSim(grid_size=100, initial_virus=5, contactsize=2, num_cores=6,
                     recover_time=30, speedreaction=0.01, incubation=7, virulence=25)
    test.mechanismcheck()
    # %%
    result = test.simspread(t_end=60, savefile='outfile_sl3.npz')


# %%

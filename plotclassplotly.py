# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=infected_growth_df.Day, y=infected_growth_df['num_infected'], name="AAPL High",
                                 line_color='deepskyblue'))

        fig.update_layout(title_text='Infection growth',
                          xaxis_rangeslider_visible=True)
        fig.show()

    def infectionheatmap(self):
        infect_dis = []
        col = []
        row = []
        days = []
        for t in range(len(self.sim_result)):
            temp_re = self.sim_result[t].tolist()
            flatten_re = [item for sublist in temp_re for item in sublist]
            x_co = np.tile(range(len(temp_re)), len(temp_re))
            y_co = np.repeat(range(len(temp_re)), len(temp_re))
            day_series = np.repeat(t, len(temp_re)**2)

            infect_dis.extend(flatten_re)
            col.extend(x_co)
            row.extend(y_co)
            days.extend(day_series)

        heatmapdf = pd.DataFrame(
            {'dis': infect_dis, 'Day': days, 'col': col, 'row': row})
        fig = px.scatter(heatmapdf, x="col", y="row", color='dis', animation_frame="Day",
                         color_continuous_scale=[(0, "#91B493"), (0.2, "#D0104C"), (1, "#91B493")])
        fig.update_layout(title='The pandemic development',
                          xaxis_title='',
                          yaxis_title='',
                          xaxis=dict(
                              showline=False,
                              showgrid=False,
                              showticklabels=False,
                          ),
                          yaxis=dict(
                              showline=False,
                              showgrid=False,
                              showticklabels=False,
                          ),
                          autosize=True,

                          plot_bgcolor='white',
                          height=600, width=600,
                          )

        fig.show()


        # %%
if __name__ == "__main__":
    result = 'outfile_s3.npz'
    testplot = plotresult(result)
    # testplot.infectiongrowth()
    testplot.infectionheatmap()

# %%

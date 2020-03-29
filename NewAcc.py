# %%
import plotly.graph_objects as go
import numpy as np
import pandas as pd

num_infected = []
Day = []
batch_list = []
for batch in range(1, 4):
    savefile = f'outfile_s{batch}.npz'
    container = np.load(savefile)
    sim_result = [container[key] for key in container]
    acc_list = []
    for t in range(1,len(sim_result)):
        acc_list.append(len(np.where(sim_result[t] < 30)[0])-len(np.where(sim_result[t-1] < 30)[0]))
    num_infected.extend(acc_list)
    Day.extend(np.arange(len(sim_result)-1).tolist())
    batch_list.extend(np.repeat(batch, len(sim_result)-1))

infected_growth_df = pd.DataFrame(
    {'num_infected': num_infected, 'Day': Day, 'batch': batch_list})

# %%


# Add data

fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 1].Day, y=infected_growth_df[infected_growth_df['batch'] == 1].num_infected, name='Speed 0.1',
                         line=dict(color='firebrick', width=4),fill='tozeroy'))
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 2].Day, y=infected_growth_df[infected_growth_df['batch'] == 2].num_infected, name='Speed 1',
                         line=dict(color='royalblue', width=4,
                                   dash='dot'),fill='tozeroy'))
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 3].Day, y=infected_growth_df[infected_growth_df['batch'] == 3].num_infected, name='Speed 0.01',
                         line=dict(color='green', width=4,
                                   dash='dash'),  # dash options include 'dash', 'dot', and 'dashdot'
                         fill='tozeroy'))

# Edit the layout
fig.update_layout(title='',
                  xaxis_title='Day',
                  yaxis_title='Number of newly increase infected cases',
                  xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=2,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        ),
                  ),
                  yaxis=dict(
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                  autosize=True,

                  plot_bgcolor='white',
                  height=400, width=600,
                  )

fig.show()

# %%

# %%
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np

datafile = 'covid19-global-forecasting-week-2/train.csv'
data = pd.read_csv(datafile)

# %%
all_region_data = data[pd.isna(data['Province_State'])]
region = ['Japan', 'Israel']
# region = pd.unique(all_region_data['Country_Region']).tolist()
f_region = []
time_list = []
region_name = []
for ci in range(len(region)):
    region_data = data[data['Country_Region'] == region[ci]]
    region_data = region_data[region_data.ConfirmedCases > 0]
    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(
    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()
    # Only considering the countries with effective data
    if len(np.where(inc_percentage > 0)[0]) > 0:
        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]
        f_region.extend(inc_percentage)
        time_list.extend([i for i in range(len(inc_percentage))])
        region_name.extend([region[ci] for i in range(len(inc_percentage))])
    else:
        pass
f_df = pd.DataFrame(
    {'increase': f_region, 'Day': time_list, 'region': region_name})


# %%
sim_data = []
speed = [0.01,0.1,1]
for batch in range(1,4):
    result = f'outfile_s{batch}.npz'
    container = np.load(result)
    speed_batch = f'Sim: speed {speed[batch-1]}'

    sim_result = [container[key] for key in container]
    num_infected = []
    for t in range(len(sim_result)):
        num_infected.append(len(np.where(sim_result[t] < 30)[0]))

    inc_infected = [(num_infected[i+1]-num_infected[i])/num_infected[i]
                    for i in range(len(num_infected)-1)]
    infected_growth_df = pd.DataFrame({'increase': inc_infected, 'Day': [
        i for i in range(len(sim_result)-1)], 'region': speed_batch})
    sim_data.append(infected_growth_df)
sim_df = pd.concat(sim_data)
# %%
newf = f_df.append(sim_df)

# %%
fig = px.line(newf, x='Day',
              y='increase', color='region')
fig.update_layout(title='The influence of government reaction speed on the pandemic development',
                  xaxis_title='Day',
                  yaxis_title='Number of infected cases',
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

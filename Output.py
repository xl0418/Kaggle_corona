# %%
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np

datafile = 'covid19-global-forecasting-week-2/train.csv'
data = pd.read_csv(datafile)
data['PSCR'] = data.Province_State.map(str)+data.Country_Region.map(str)

# %%
# ip pattern of the empirical data from 2020/03/19 onwards
region = pd.unique(data['PSCR']).tolist()
f_region = []
time_list = []
region_name = []
actual_date = []
no_infection_country = []
for ci in range(len(region)):
    region_data = data[data['PSCR'] == region[ci]]
    region_data = region_data[region_data.ConfirmedCases > 0]
    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(
    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()
    # Only considering the countries with effective data
    if len(np.where(inc_percentage > 0)[0]) > 0:
        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]
        actual_date.append(region_data.Date[1:])
        f_region.extend(inc_percentage)
        time_list.extend([i for i in range(len(inc_percentage))])
        region_name.extend([region[ci] for i in range(len(inc_percentage))])
    else:
        no_infection_country.append(region[ci])
f_df = pd.DataFrame(
    {'increase': f_region, 'Day': time_list, 'PSCR': region_name})


# %%
# Simulation data for training
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
        i for i in range(len(sim_result)-1)], 'PSCR': speed_batch})
    sim_data.append(infected_growth_df)
sim_df = pd.concat(sim_data)

# %%
criteria_day_length = 10
sim_class_ip = []
for speed in pd.unique(sim_df.PSCR):
    sim_class_ip.append(sim_df[sim_df['PSCR'] == speed].increase.tolist())
sim_class_ip_array = np.array(sim_class_ip)

#%%
labels = []
effective_region = []
for region_loop in region:
    if region_loop not in no_infection_country:
        ip = f_df[f_df['PSCR'] == region_loop].increase[:criteria_day_length].tolist()
        euclidean_dis = np.linalg.norm(np.array(ip)-sim_class_ip_array[:,:len(ip)],axis = 1)
        labels.append(np.where(euclidean_dis == min(euclidean_dis))[0][0])
        effective_region.append(region_loop)
    else:
        pass

xlabels = ['Slow','Moderate','Fast']
scenario_class = {'ip': [xlabels[i] for i in labels], 'Area':effective_region, 'width': [1 for i in range(len(labels))]}
sce_df = pd.DataFrame(scenario_class)
#%%
fig = px.bar(sce_df, x="ip", y="width", color='Area', height=400)
fig.update_layout(title='Strategies of regions',
                  xaxis_title='Strategy',
                  yaxis_title='Number of countires',
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
                        )
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
date_datause = '2020-03-18'
date_actualdata = '2020-03-30'
date_length = (pd.to_datetime(date_actualdata) - pd.to_datetime(date_datause)).days
predict_region_list = []
effect_ind = 0
for it in range(len(region)):
    region_it = region[it]
    if region_it not in no_infection_country:
        time_length_it = actual_date[effect_ind]
        sim_class_it = labels[effect_ind]
        predict_ip_it = sim_class_ip_array[sim_class_it,(len(actual_date[0])-date_length):]
        while len(predict_ip_it)< (date_length+31):
            predict_ip_it = np.append(predict_ip_it,predict_ip_it[len(predict_ip_it)-1])
        retion_df = data[data['PSCR'] == region_it]
        num_infected_it = retion_df[retion_df['Date'] == date_datause]['ConfirmedCases'].astype(float)
        predict_region_list_it = []
        ini_infected = num_infected_it.tolist()[0]
        for predict_day in range(len(predict_ip_it)):
            predict_region_list_it.append(ini_infected * (1+predict_ip_it[predict_day]))
            ini_infected = predict_region_list_it[predict_day]
        predict_region_list.extend(predict_region_list_it)
        effect_ind += 1
    else:
        predict_region_list.extend([0 for i in range(43)])

# %%
# Write output csv file
import csv
from itertools import zip_longest
list1 = [i+1 for i in range(len(predict_region_list))]
list2 = predict_region_list
list3 = [0 for i in range(len(predict_region_list))]
d = [list1, list2,list3]
export_data = zip_longest(*d, fillvalue = '')
with open('submission.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("ForecastId", "ConfirmedCases", "Fatalities"))
      wr.writerows(export_data)
myfile.close()

# %%

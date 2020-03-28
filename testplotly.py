#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # plotly 4.0.0rc1


df = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
low = df['AAPL.Low'].tolist()
# artificially added 20 to get the second graph above the first one
high = np.array(df['AAPL.High'])+20

trace1 = go.Scatter(x=df.Date[:2],
                    y=low[:2],
                    mode='lines',
                    line=dict(width=1.5))

trace2 = go.Scatter(x=df.Date[:2],
                    y=high[:2],
                    mode='lines',
                    line=dict(width=1.5))

frames = [dict(data=[dict(type='scatter',
                          x=df.Date[:k+1],
                          y=low[:k+1]),
                     dict(type='scatter',
                          x=df.Date[:k+1],
                          y=high[:k+1])],
               # this means that  frames[k]['data'][0]  updates trace1, and   frames[k]['data'][1], trace2
               traces=[0, 1],
               )for k in range(1, len(low)-1)]

layout = go.Layout(width=650,
                   height=400,
                   showlegend=False,
                   hovermode='closest',
                   updatemenus=[dict(type='buttons', showactive=False,
                                     y=1.05,
                                     x=1.15,
                                     xanchor='right',
                                     yanchor='top',
                                     pad=dict(t=0, r=10),
                                     buttons=[dict(label='Play',
                                                   method='animate',
                                                   args=[None,
                                                         dict(frame=dict(duration=3,
                                                                         redraw=False),
                                                              transition=dict(
                                                             duration=0),
                                                             fromcurrent=True,
                                                             mode='immediate')])])])


layout.update(xaxis=dict(range=[df.Date[0], df.Date[len(df)-1]], autorange=False),
              yaxis=dict(range=[min(low)-0.5, high.max()+0.5], autorange=False))
fig = go.Figure(data=[trace1, trace2], frames=frames, layout=layout)
fig.show()


# %%

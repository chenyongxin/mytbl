#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read timer and output to a report.

@author: CHEN Yongxin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "MONITOR_Timer"
df = pd.read_csv(file, delim_whitespace=True)

# delete text
df = df.drop(df.loc[df["Steps"] == "Steps"].index)

# delete step 0 and its duplicated ones
df = df.drop(df.loc[df["Steps"] == "0"].index)

step    = np.array(pd.to_numeric(df["Steps"]))
dt      = np.array(pd.to_numeric(df["Time_step"]))
dt_time = np.array(pd.to_numeric(df["Time_per_step_(s)"]))
stime   = np.cumsum(dt)      # simulation time
wtime   = np.cumsum(dt_time) # wall time

fig, axes = plt.subplots(5, 1, figsize=(6,21))
med = np.median(dt_time)
std = np.std(dt_time)

plt.sca(axes[0])
plt.plot(step, dt, ".", markersize=1.0)
plt.xlabel("Step")
plt.ylabel(r"$\Delta t$")
plt.title(r"$\Delta t$ per step")
plt.grid()

ax = plt.gca()
twin = ax.twiny()
plt.sca(twin)
plt.plot(stime, dt, ".", markersize=1.0)
plt.grid()
plt.xlabel("Simulation time")

#------------------------------------------#

plt.sca(axes[1])
plt.plot(step, dt_time, ".", markersize=3)
plt.xlabel("Step")
plt.ylabel("Wall time (s)")
plt.title("Time used per step")
plt.grid()

ax = plt.gca()
twin = ax.twiny()
plt.sca(twin)
plt.plot(stime, dt_time, ".", markersize=3)
plt.grid()
plt.xlabel("Simulation time")

#------------------------------------------#

plt.sca(axes[2])
plt.plot(step, dt_time, ".", markersize=1.5)
plt.xlabel("Step")
plt.ylabel("Wall time (s)")
plt.ylim([med-std*2, med+std*2])
plt.title("Time used per step (zoomed-in)")
plt.grid()

ax = plt.gca()
twin = ax.twiny()
plt.sca(twin)
plt.plot(stime, dt)
plt.grid()
plt.xlabel("Simulation time")

#------------------------------------------#

n = dt.size // 1000 
size = n * 1000
x = step[:size]
y = dt_time[:size]
plt.sca(axes[3])
plt.plot(x[::1000]/1000, np.sum(y.reshape(n,1000), axis=1), ".", markersize=5)
plt.xlabel(r"$(\times 1000)$ steps")
plt.ylabel("Wall time (s)")
plt.title("Time used per 1000 steps")
plt.grid()

ax = plt.gca()
twin = ax.twiny()
plt.sca(twin)
plt.plot(stime[:size][::1000], np.sum(y.reshape(n,1000), axis=1), ".", markersize=5)
plt.grid()
plt.xlabel("Simulation time")

#------------------------------------------#

plt.sca(axes[4])
plt.plot(step, wtime, ".", markersize=1.5)
plt.xlabel("Step")
plt.ylabel("Cumulative wall time (s)")
plt.title("Wall time vs simulation time")
plt.grid()

ax = plt.gca()
twin = ax.twiny()
plt.sca(twin)
plt.plot(stime, wtime, ".", markersize=3)
plt.grid()
plt.xlabel("Simulation time")

twinx = ax.twinx()
plt.sca(twinx)
plt.plot(step, wtime/3600, ".", markersize=3)
plt.grid()
plt.ylabel("Cumulative wall time (h)")

plt.tight_layout()
plt.savefig("Timer.png", dpi=300)
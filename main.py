import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as mplot
import roadrunner as rr


from bokeh import client as bkclient
from bokeh import io as bkio
from bokeh import plotting as bkplot
from bokeh import models as bkmodels
from bokeh.models import widgets


"""-------------------------------------------------------------------------------------------
    Load and format experiment files
-------------------------------------------------------------------------------------------"""

META_FILE = '150606_Meta.xlsx'
MGA5S_DATA_FILE = '150606_MGA5S.xlsx'

meta = pd.read_excel(META_FILE, sheetname=None)
mga5s = pd.read_excel(MGA5S_DATA_FILE, sheetname=None)

# Ignore any null values.
meta['Culturing'] = meta['Culturing'].loc[meta['Culturing']['Sample'].notnull()]
meta['Measurement'] = meta['Measurement'].loc[meta['Measurement']['Sample'].notnull()]

# Merge relevant columns from the culturing and measurement metadata.
final_culture_meta = meta['Culturing'].groupby('Sample').last().reset_index()
combined_meta = pd.merge(meta['Measurement'].loc[:, ['Sample', 'Well', 'External Dye [uM]']],
                         final_culture_meta.loc[:, ['Sample', 'Strain', 'Dilution Rate (1/hr)']],
                         on='Sample')

# Initiate DataFrame that combines measurement data and sample metadata.
composite = pd.DataFrame()

# Identify the negative control wells.
neg_control_sample_ids = combined_meta.loc[combined_meta['Strain'] == 'NAC', 'Sample'].unique()
neg_control_wells = combined_meta.loc[combined_meta['Sample'].isin(neg_control_sample_ids), 'Well'].values

# Add measurement data to composite.
for key in mga5s:
    df = mga5s[key]

    # Account for time required to read each well serially.
    # Assumptions: One second per well read. Column order specifies read progression.
    well_read_offset = dict(zip(df.columns[1:], np.arange(df.shape[1] - 1, dtype=float)))
    induction_time_offset = dict(meta['Measurement'][['Well', 'Induction Time [s]']].values)

    # Infer the time window where the plate was removed and samples were induced.
    delta_time = np.diff(df['Time [s]'])
    first_index_after_induction = next((i for (i,x) in enumerate(delta_time[1:]) if x != delta_time[0])) + 2
    first_time_after_induction = df['Time [s]'].iloc[first_index_after_induction]

    # Create long-form DataFrame
    df = pd.melt(df, id_vars='Time [s]', var_name='Well', value_name='MGA5S FL')

    # Add additional time to the measurement time point according to the read progression.
    df['Time [s]'] += df['Well'].map(well_read_offset)

    # Normalize all time points by specifying the time of induction as t=0.
    df['Time [s]'] -= first_time_after_induction
    df['Time [s]'] += df['Well'].map(induction_time_offset)

    # Get an estimate for the negative control fluorescence level.
    neg_control_fl = df.loc[(df['Well'].isin(neg_control_wells)), :]
    num_first_third = np.ceil(neg_control_fl[neg_control_fl['Time [s]'] <= 0].shape[0] * 0.3)
    pre_induction_basal_fl = neg_control_fl.sort_values('Time [s]').iloc[num_first_third:, :].loc[:, 'MGA5S FL'].mean()
    post_induction_basal_fl = neg_control_fl.loc[neg_control_fl['Time [s]'] > 0, 'MGA5S FL'].mean()

    df.loc[df['Time [s]'] <= 0., 'MGA5S FL'] -= pre_induction_basal_fl
    df.loc[df['Time [s]'] > 0., 'MGA5S FL'] -= post_induction_basal_fl
    df.loc[df['MGA5S FL'] < 0., 'MGA5S FL'] = 0.

    # Extend the composite DataFrame.
    composite = composite.append(df.copy())

# Add culturing and measurement metadata to composite.
composite = pd.merge(composite, combined_meta, on='Well')



"""-------------------------------------------------------------------------------------------
    In dynamic measurements we observe a gradual decline in fluorescence intensity for
    long exposure times. We hypothesize that this decrease in signal is an artifact of
    fluoresence photobleaching and ____. To avoid fitting poor quality data, we only fit
    values recorded in the range t=[0,240]
-------------------------------------------------------------------------------------------"""

composite = composite.loc[(composite['Time [s]'] >= 0.) & (composite['Time [s]'] <= 240.), :]


"""-------------------------------------------------------------------------------------------
    Visualize Data
-------------------------------------------------------------------------------------------"""

plot_source = bkmodels.ColumnDataSource(composite)
plot_figure = bkplot.figure(title='MGA5S Fluorescence Induction Experiment',
                            tools='pan,box_zoom,box_select,reset')

colors_map = dict(zip(composite['Strain'].unique(), ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'grey']))
plot_source.data['color'] = [colors_map[x] for x in plot_source.data['Strain']]

plot_figure.scatter(x='Time [s]', y='MGA5S FL', source=plot_source, fill_color='color', line_color='black',
                    fill_alpha=0.5, line_alpha=0.8, size=6)

layout = bkplot.hplot(plot_figure)
bkplot.output_file('mga5s_plot.html')
bkplot.show(layout)










print('Done!')

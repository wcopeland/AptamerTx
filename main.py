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
    first_index_after_induction = next((i for (i, x) in enumerate(delta_time[1:]) if x != delta_time[0])) + 2
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

MIN_TIME = 0.
MAX_TIME = 240.

composite = composite.loc[(composite['Time [s]'] >= MIN_TIME) & (composite['Time [s]'] <= MAX_TIME), :]


"""-------------------------------------------------------------------------------------------
    Visualize Data
-------------------------------------------------------------------------------------------"""

plot_source = bkmodels.ColumnDataSource(composite)
plot_figure = bkplot.figure(title='MGA5S Fluorescence Induction Experiment',
                            tools='pan,box_zoom,box_select,reset')

colors_map = dict(zip(composite['Strain'].unique(), ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'grey']))
plot_source.add([colors_map[x] for x in plot_source.data['Strain']], name='color')

plot_figure.scatter(x='Time [s]', y='MGA5S FL', source=plot_source, fill_color='color', line_color='black',
                    fill_alpha=0.5, line_alpha=0.8, size=6)

layout = bkplot.hplot(plot_figure)
bkplot.output_file('mga5s_plot.html')
bkplot.show(layout)


"""-------------------------------------------------------------------------------------------
    Establish the parameters that will be used during the optimization routine.
-------------------------------------------------------------------------------------------"""


shared_opt_params = []
shared_opt_log_range = []

opt_param_sharing_map = {
    'kf': '*',
    'kr': '*',
    'kt': '*',
    'syn': 'Strain',
    'ci': 'External Dye [uM]'
}

param_ranges = {
    'kf': (1e-9, 1e1),
    'kr': (1e-9, 1e1),
    'kt': (1e-9, 1e1),
    'syn': (1e-5, 1.),
    'ci': (1, 1e4)
}

# Create shared parameter matrix.
# Initiate the parameter matrix along with the shared parameters
well_param_frame = pd.DataFrame(index=combined_meta['Well'], columns=opt_param_sharing_map.keys())

for key in well_param_frame:
    share_by = opt_param_sharing_map[key]
    if share_by == '*':
        well_param_frame.loc[:, key] = '{{{{{}}}}}'.format(len(shared_opt_params))
        shared_opt_params.append(0.)
        shared_opt_log_range.append(param_ranges[key])
    else:
        share_by_factors = combined_meta[share_by].unique()
        for factor in share_by_factors:
            shared_wells = combined_meta.loc[combined_meta[share_by] == factor, 'Well'].values
            well_param_frame.loc[shared_wells, key] = '{{{{{}}}}}'.format(len(shared_opt_params))
            shared_opt_params.append(0.)
            shared_opt_log_range.append(param_ranges[key])

# Convert parameter vector for optimization into a NumPy array.
shared_opt_params = np.asarray(shared_opt_params, dtype=float)
shared_opt_log_range = np.asarray(np.log10(shared_opt_log_range), dtype=float)

# Add fixed parameters
well_param_frame['deg_rna'] = 0.
well_param_frame['deg_bound'] = 0.

renamed_columns = {'External Dye [uM]':'dye_ext',
                  'Dilution Rate (1/hr)': 'dil'}
dil_dye_ext = composite.groupby('Well').first().rename(columns=renamed_columns).loc[:, renamed_columns.values()]
well_param_frame = pd.merge(well_param_frame, dil_dye_ext, left_index=True, right_index=True)

# Need a model for each set of experimental conditions.
# In our experiments, we vary the promoter (strain), dilution rate, and external dye
# concentration. The total number of unique experimental conditions can be captured by finding
# unique combinations of the "Sample" and "External Dye [uM]" in the metadata.

unique_experiments = combined_meta.groupby(['Sample', 'External Dye [uM]'])['Well'].unique().reset_index()
model_name_map = {x:'model_{}'.format(i) for (i,x) in enumerate(unique_experiments.index)}
unique_experiments.rename(index=model_name_map, inplace=True)

# Add model information to metadata by mapping to unique wells.
well_model_map = {}
for k,v in unique_experiments['Well'].iteritems():
    for well in v:
        well_model_map.update({well: k})

combined_meta['Model'] = combined_meta['Well'].map(well_model_map)
composite['Model'] = composite['Well'].map(well_model_map)

# Create a mapping of models and parameter values.
model_param_frame = well_param_frame.rename(index=well_model_map).groupby(level=0).first()


"""-------------------------------------------------------------------------------------------
    Build and simulate SBML models.
-------------------------------------------------------------------------------------------"""

# Load the SBML model.
general_model = rr.RoadRunner('mga5s_model.sbml')

# Initialize parameter vector for optimization
# We must convert range from log scale back to linear scale.
# noinspection PyRedeclaration
shared_opt_params = np.power(10., [np.random.uniform(low, high) for (low,high) in shared_opt_log_range])

# Iteratively generate simulation results for each model.
# for key in model_param_frame.index:
key = model_param_frame.index[0]

# Parameterize the model
for param_id, param_value in model_param_frame.loc[key, :].iteritems():
    if isinstance(param_value, (str, unicode)):
        value_idx = int(param_value.replace('{', '').replace('}', ''))
        general_model.setValue(param_id, shared_opt_params[value_idx])
    else:
        general_model.setValue(param_id, param_value)

# Initialize species concentrations
general_model.setValue('rna', 0.)
general_model.setValue('dye', 0.)
general_model.setValue('bound', 0.)
general_model.setValue('dye_ext', 0.)

# Solve for steady state concentrations prior to malachite green induction.
general_model.steadyState()

# Modify external dye concentration to match the amount of malachite green that is pulsed into the well.
general_model.setValue('dye_ext', model_param_frame.loc[key, 'dye_ext'])

# Simulate the transient concentration of dye-bound rna (ie. bound) after malachite green exposure.
sim_result = general_model.simulate(MIN_TIME, MAX_TIME, int(MAX_TIME+1), selections=['time', 'bound'])
sim_result = pd.DataFrame(sim_result, columns=sim_result.colnames)

# Get approximate FL value.
sim_result['FL'] = sim_result['bound'] * general_model.getValue('ci')




print('Done!')

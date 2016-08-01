import numpy as np
import pandas as pd
import scipy as sp
import roadrunner as rr
from datetime import datetime
from bokeh import client as bkclient
from bokeh import io as bkio
from bokeh import plotting as bkplot
from bokeh import models as bkmodels
from bokeh.models import widgets
from bokeh import palettes as bkpal
from seaborn import cubehelix_palette

from lib import Optimization

rr.Config.setValue(rr.Config.ROADRUNNER_DISABLE_WARNINGS, True)


# """-------------------------------------------------------------------------------------------
#     Function definitions
# -------------------------------------------------------------------------------------------"""
#
# def update_dependent_parameters(parameter_vector, _relation_frame):
#     for i in _relation_frame.index:
#         target_idx = int(_relation_frame.loc[i, 'target'].replace('{', '').replace('}', ''))
#
#         # Get the first operand.
#         if isinstance(_relation_frame.loc[i, 'operand_1_min'], (str, unicode)):
#             min_1 = parameter_vector[int(_relation_frame.loc[i, 'operand_1_min'].replace('{', '').replace('}', ''))]
#         else:
#             min_1 = _relation_frame.loc[i, 'operand_1_min']
#
#         if isinstance(_relation_frame.loc[i, 'operand_1_max'], (str, unicode)):
#             max_1 = parameter_vector[int(_relation_frame.loc[i, 'operand_1_max'].replace('{', '').replace('}', ''))]
#         else:
#             max_1 = _relation_frame.loc[i, 'operand_1_max']
#
#         if min_1 == max_1:
#             op_1 = min_1
#         else:
#             op_1 = 10. ** np.random.uniform(np.log10(min_1), np.log10(max_1))
#
#         # Get the second operand.
#         if isinstance(_relation_frame.loc[i, 'operand_2_min'], (str, unicode)):
#             min_2 = parameter_vector[int(_relation_frame.loc[i, 'operand_2_min'].replace('{', '').replace('}', ''))]
#         else:
#             min_2 = _relation_frame.loc[i, 'operand_2_min']
#
#         if isinstance(_relation_frame.loc[i, 'operand_2_max'], (str, unicode)):
#             max_2 = parameter_vector[int(_relation_frame.loc[i, 'operand_2_max'].replace('{', '').replace('}', ''))]
#         else:
#             max_2 = _relation_frame.loc[i, 'operand_2_max']
#
#         if min_2 == max_2:
#             op_2 = min_2
#         else:
#             op_2 = 10. ** np.random.uniform(np.log10(min_2), np.log10(max_2))
#
#         # Update the value
#         parameter_vector[target_idx] = _relation_frame.loc[i, 'operator'](op_1, op_2)
#     return


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

# Round time to the nearest second. This is still sufficient
composite['Time [s]'] = composite['Time [s]'].round(0)


"""-------------------------------------------------------------------------------------------
    Visualize Data:
    Dynamic FL measurements
-------------------------------------------------------------------------------------------"""

plot_source = bkmodels.ColumnDataSource(composite)
plot_figure = bkplot.figure(title='MGA5S Fluorescence Induction Experiment',
                            tools='pan,box_zoom,box_select,reset')

all_strains = composite['Strain'].unique()
required_colors = (bkpal.Paired12 * int(np.ceil(len(all_strains)/12.)))[:len(all_strains)]
colors_map = dict(zip(all_strains, required_colors))
plot_source.add([colors_map[x] for x in plot_source.data['Strain']], name='color')

plot_figure.scatter(x='Time [s]', y='MGA5S FL', source=plot_source, fill_color='color', line_color='black',
                    fill_alpha=0.6, line_alpha=0.8, size=6)

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
model_param_frame.index.name = 'Model'

"""-------------------------------------------------------------------------------------------
    Establish the dependent relationships between parameters. This is a critical step
    to further constrain the solution space.
-------------------------------------------------------------------------------------------"""

# Create a DataFrame that describes relationships between parameters.
param_relations = {
    ('syn', 'Strain', 'J23100'): [np.multiply, (1.2, 1.5), ('syn', 'Strain', 'J23101')],
    ('syn', 'Strain', 'J23104'): [np.multiply, (2.0, 2.5), ('syn', 'Strain', 'J23101')],
    ('syn', 'Strain', 'J23107'): [np.multiply, (0.2, 0.5), ('syn', 'Strain', 'J23101')],
    ('syn', 'Strain', 'J23110'): [np.multiply, (0.6, 0.8), ('syn', 'Strain', 'J23101')],
    ('syn', 'Strain', 'J23111'): [np.multiply, (2.2, 2.8), ('syn', 'Strain', 'J23101')],
    ('kr', 'Strain', 'J23100'): [np.multiply, (8.0e-8, 5.0e-7), ('kf', 'Strain', 'J23100')],
    ('kr', 'Strain', 'J23101'): [np.multiply, (8.0e-8, 5.0e-7), ('kf', 'Strain', 'J23101')],
    ('kr', 'Strain', 'J23104'): [np.multiply, (8.0e-8, 5.0e-7), ('kf', 'Strain', 'J23104')],
    ('kr', 'Strain', 'J23107'): [np.multiply, (8.0e-8, 5.0e-7), ('kf', 'Strain', 'J23107')],
    ('kr', 'Strain', 'J23110'): [np.multiply, (8.0e-8, 5.0e-7), ('kf', 'Strain', 'J23110')],
    ('kr', 'Strain', 'J23111'): [np.multiply, (8.0e-8, 5.0e-7), ('kf', 'Strain', 'J23111')],
}

relation_frame = pd.DataFrame(index=range(len(param_relations)),
    columns=['target',
             'operator',
             'operand_1_min',
             'operand_1_max',
             'operand_2_min',
             'operand_2_max']
)

def get_parameter_from_relation(relation):
    param_key, meta_key, meta_value = relation
    target_wells = combined_meta.loc[combined_meta[meta_key] == meta_value, 'Well'].values
    replacement_param_idx = well_param_frame.loc[target_wells, param_key].unique()

    if len(replacement_param_idx) < 1:
        msg = 'Found no parameter for replacement in the relation: {}'.format(relation)
        raise ValueError(msg)
    elif len(replacement_param_idx) == 1:
        return replacement_param_idx[0]
    else:
        msg = 'Found too many parameters for replacement. Relationship must be more specific: {}'.format(relation)
        raise ValueError(msg)

# Iteratively populate the parameter relationship DataFrame.
for i, relationship in enumerate(param_relations.items()):
    new_relationship = dict()
    new_relationship['target'] = get_parameter_from_relation(relationship[0])

    operator, operand_1, operand_2 = relationship[1]
    new_relationship['operator'] = operator

    if isinstance(operand_1, (tuple, list)):
        if len(operand_1) == 3:
            param_idx = get_parameter_from_relation(operand_1)
            new_relationship['operand_1_min'] = param_idx
            new_relationship['operand_1_max'] = param_idx
        elif len(operand_1) == 2:
            new_relationship['operand_1_min'] = operand_1[0]
            new_relationship['operand_1_max'] = operand_1[1]
        else:
            msg = 'Cannot interpret operand: {}'.format(operand_1)
            raise ValueError(msg)
    else:
        new_relationship['operand_1_min'] = operand_1
        new_relationship['operand_1_max'] = operand_1

    if isinstance(operand_2, (tuple, list)):
        if len(operand_2) == 3:
            param_idx = get_parameter_from_relation(operand_2)
            new_relationship['operand_2_min'] = param_idx
            new_relationship['operand_2_max'] = param_idx
        elif len(operand_2) == 2:
            new_relationship['operand_2_min'] = operand_2[0]
            new_relationship['operand_2_max'] = operand_2[1]
        else:
            msg = 'Cannot interpret operand: {}'.format(operand_2)
            raise ValueError(msg)
    else:
        new_relationship['operand_2_min'] = operand_2
        new_relationship['operand_2_max'] = operand_2

    relation_frame.iloc[i, :] = new_relationship

# Clean up the relation frame to avoid duplicates.
relation_frame = relation_frame.drop_duplicates()
relation_frame.rename(index=dict(zip(relation_frame.index, range(relation_frame.shape[0]))), inplace=True)


"""-------------------------------------------------------------------------------------------
    Summarize the parameter space.
-------------------------------------------------------------------------------------------"""

num_parameters = len(shared_opt_params)
num_dependent_parameters = len(relation_frame['target'].unique())
num_independent_parameters = num_parameters - num_dependent_parameters

print('-------------------------------------------')
print('\tMODEL SUMMARY')
print('-------------------------------------------')
print('Total number of parameters to be optimized:\t{}'.format(num_parameters))
print('Total number of independent parameters:\t{}'.format(num_independent_parameters))
print('Total number of dependent parameters:\t{}'.format(num_dependent_parameters))
print('Total number of models:\t{}\n\n'.format(model_param_frame.shape[0]))


"""-------------------------------------------------------------------------------------------
    Build SBML models.
-------------------------------------------------------------------------------------------"""

# Load the SBML model.
general_model = rr.RoadRunner('mga5s_model.sbml')


"""-------------------------------------------------------------------------------------------
    Parameterize the model using Differential Evolution.
-------------------------------------------------------------------------------------------"""

start_time = datetime.now()
opt = Optimization(
    model=general_model,
    observed=composite[['Time [s]', 'Model', 'MGA5S FL']],
    relation_frame=relation_frame,
    model_param_frame=model_param_frame,
    param_range= shared_opt_log_range,
    t_0=MIN_TIME,
    t_max=MAX_TIME,
    max_generations=10
)
print(opt.run())
end_time = datetime.now()

print('-----------------------------')
print('\tSimulation Results')
print('-----------------------------\n')

best_idx = opt.fitness.argmin()
best_score = opt.fitness[best_idx]
best_parameters = opt.population[best_idx]

print('Best Score: {}'.format(best_score))
print('Best Parameters: \t{}'.format(best_parameters))
print('Simulation Time: {}'.format(end_time - start_time))


"""-------------------------------------------------------------------------------------------
    Visualize Data
-------------------------------------------------------------------------------------------"""

plot_source = bkmodels.ColumnDataSource(composite)
plot_figure = bkplot.figure(title='MGA5S Fluorescence Induction Experiment',
                            tools='pan,box_zoom,box_select,reset')

all_models = composite['Model'].unique()
required_colors = (bkpal.Paired12 * int(np.ceil(len(all_models)/12.)))[:len(all_models)]
colors_map = dict(zip(all_models, required_colors))
plot_source.add([colors_map[x] for x in plot_source.data['Model']], name='color')

# Plot the simulated values.
simulated_fl = opt.simulate_model(best_parameters)
for model_name in all_models:
    model_fl = simulated_fl[simulated_fl['Model'] == model_name].sort_values('Time [s]')
    plot_figure.line(
        x=model_fl['Time [s]'],
        y=model_fl['Simulated FL'],
        color=colors_map[model_name],
        line_width=2
    )

# Plot the observed values.
plot_figure.scatter(x='Time [s]', y='MGA5S FL', source=plot_source, fill_color='color', line_color='black',
                    fill_alpha=0.6, line_alpha=0.8, size=6)


layout = bkplot.hplot(plot_figure)
bkplot.output_file('mga5s_plot.html')
bkplot.show(layout)



print('Done!')


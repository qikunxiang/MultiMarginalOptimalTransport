% Plot the density functions of the marginals

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);

json_file = fopen([CONFIG.COMPARISONPATH_SCRIPTS, 'WBMM_densities.json'], 'w');

fprintf(json_file, '%s', jsonencode(marg_density_cell));

fclose(json_file);
% Incompressible fluid example with 1D marginals

CONFIG = IFMM_config();

% two different arrangment functions will be examined
arrangement1 = struct;
arrangement1.knots = [0; 1/2; 1];
arrangement1.values = [0; 1; 0];

arrangement2 = struct;
arrangement2.knots = [0; 1/4; 1/2; 3/4; 1];
arrangement2.values = [1; 0; 1; 0; 1];

arrangements = {arrangement1; arrangement2};

timestep_num_list = (4:2:20)';
testfunc_knot_num_list = [4; 8; 16; 32; 64; 128] + 1;

save(CONFIG.SAVEPATH_INPUTS, ...
    'arrangements', ...
    'timestep_num_list', ...
    'testfunc_knot_num_list', ...
    '-v7.3');
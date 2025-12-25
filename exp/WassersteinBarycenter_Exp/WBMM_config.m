function CONFIG = WBMM_config()
% Place to store global configurations of the Wasserstein barycenter
% with multi-marginal optimal transport
% Output:
%   CONFIG: a struct containing configurations as fields

CONFIG = global_config();

% path to save files
CONFIG.SAVEPATH = [CONFIG.SAVEPATH_ROOT, 'MMOT/WassersteinBarycenter_Exp/'];

% path to log files
CONFIG.LOGPATH = [CONFIG.LOGPATH_ROOT, 'MMOT/WassersteinBarycenter_Exp/'];

% root folder for gurobi log files
CONFIG.LOGPATH_GUROBI = [CONFIG.LOGPATH, 'Gurobi/'];

% root folder for comparison results
CONFIG.COMPARISONPATH = 'exp/Comparisons/Saved/WassersteinBarycenter/';

% root folder for comparison scripts
CONFIG.COMPARISONPATH_SCRIPTS = 'exp/Comparisons/';


% if the directory does not exist, create it first
if ~exist(CONFIG.SAVEPATH, 'dir')
    mkdir(CONFIG.SAVEPATH);
end

% if the directory does not exist, create it first
if ~exist(CONFIG.LOGPATH, 'dir')
    mkdir(CONFIG.LOGPATH);
end

% if the directory does not exist, create it first
if ~exist(CONFIG.LOGPATH_GUROBI, 'dir')
    mkdir(CONFIG.LOGPATH_GUROBI);
end

% file name of the inputs
CONFIG.FILENAME_INPUTS = 'inputs';
CONFIG.SAVEPATH_INPUTS = [CONFIG.SAVEPATH, CONFIG.FILENAME_INPUTS, '.mat'];

% file name of the optimal transport outputs
CONFIG.FILENAME_OT = 'OT';
CONFIG.SAVEPATH_OT = [CONFIG.SAVEPATH, CONFIG.FILENAME_OT, '.mat'];

% file name of the Wasserstein-2 optimal transport outputs
CONFIG.FILENAME_W2OT = 'W2OT';
CONFIG.SAVEPATH_W2OT = [CONFIG.SAVEPATH, CONFIG.FILENAME_W2OT, '.mat'];

% file name of the outputs
CONFIG.FILENAME_OUTPUTS = 'outputs';
CONFIG.SAVEPATH_OUTPUTS = [CONFIG.SAVEPATH, CONFIG.FILENAME_OUTPUTS, '.mat'];

% file name of the reassembly based upper bounds
CONFIG.FILENAME_UB = 'UB';
CONFIG.SAVEPATH_UB = [CONFIG.SAVEPATH, CONFIG.FILENAME_UB, '.mat'];

% file name of the Wasserstein-2 optimal transport based upper bounds
CONFIG.FILENAME_W2OTUB = 'W2OTUB';
CONFIG.SAVEPATH_W2OTUB = [CONFIG.SAVEPATH, CONFIG.FILENAME_W2OTUB, '.mat'];

% file name of the dual functions
CONFIG.FILENAME_DUALFUNCS = 'dualfuncs';
CONFIG.SAVEPATH_DUALFUNCS = [CONFIG.SAVEPATH, CONFIG.FILENAME_DUALFUNCS, '.mat'];

% file name of the optimal transport logs
CONFIG.LOGNAME_OT = 'OT';
CONFIG.LOGPATH_OT = [CONFIG.LOGPATH, CONFIG.LOGNAME_OT, '.log'];

% file name of the Wasserstein-2 optimal transport logs
CONFIG.LOGNAME_W2OT = 'W2OT';
CONFIG.LOGPATH_W2OT = [CONFIG.LOGPATH, CONFIG.LOGNAME_W2OT, '.log'];

% file name of the main logs
CONFIG.LOGNAME_MAIN = 'main';
CONFIG.LOGPATH_MAIN = [CONFIG.LOGPATH, CONFIG.LOGNAME_MAIN, '.log'];

% file name of the linear semi-infinite programming logs
CONFIG.LOGNAME_LSIP_MAIN = 'LSIP_main';
CONFIG.LOGPATH_LSIP_MAIN = [CONFIG.LOGPATH, CONFIG.LOGNAME_LSIP_MAIN, '.log'];

% file name of the linear programming logs
CONFIG.LOGNAME_LSIP_LP = 'LSIP_LP';
CONFIG.LOGPATH_LSIP_LP = [CONFIG.LOGPATH_GUROBI, CONFIG.LOGNAME_LSIP_LP, '.log'];

% file name of the global optimization logs
CONFIG.LOGNAME_LSIP_GLOBAL = 'LSIP_global';
CONFIG.LOGPATH_LSIP_GLOBAL = [CONFIG.LOGPATH, CONFIG.LOGNAME_LSIP_GLOBAL, '.log'];

% file name of the reassembly based upper bounds
CONFIG.LOGNAME_UB = 'UB';
CONFIG.LOGPATH_UB = [CONFIG.LOGPATH, CONFIG.LOGNAME_UB, '.log'];

% file name of the Wasserstein-2 optimal transport based upper bounds
CONFIG.LOGNAME_W2OTUB = 'W2OTUB';
CONFIG.LOGPATH_W2OTUB = [CONFIG.LOGPATH, CONFIG.LOGNAME_W2OTUB, '.log'];

end


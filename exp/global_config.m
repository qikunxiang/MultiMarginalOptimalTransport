function CONFIG = global_config()
% Place to store global configurations such as file paths
% Output:
%   CONFIG: a struct containing configurations as fields

CONFIG = struct;

% root folder for all save files
CONFIG.SAVEPATH_ROOT = ['~/Library/Mobile Documents/' ...
    'com~apple~CloudDocs/MATLAB/'];

% root folder for all log files
CONFIG.LOGPATH_ROOT = '../Logs/';

end


function model = CPWA_diff_min_MILP_gurobi(c, knots, knots_val, CPWA)
% Formulating the minimization of a separable CPWA function minus another
% CPWA function in a box into a MILP problem in gurobi
% Inputs: 
%       c: constant intercept
%       knots: cell array containing the knots for the one-dimensional CPWA
%       functions in each dimension; the first and last knots also specify
%       the boundaries of the box
%       knots_val: cell array containing the values of the one-dimensional
%       CPWA functions in each dimension
%       CPWA: struct for specifying the non-separable CPWA function
% Output: 
%       model: the formulated model in gurobi

N = length(knots);
knotno_list = cellfun(@length, knots);
knotno = sum(knotno_list);

% check the inputs
assert(all(cellfun(@length, knots_val) == knotno_list), ...
    'the values of one-dimensional functions mis-specified');

CPWA_cvno = 0;
CPWA_cv = cell(0, 2);
if isfield(CPWA, 'minus')
    CPWA_cvno = size(CPWA.minus, 1);
    CPWA_cv = CPWA.minus;
end

CPWA_ccno = 0;
CPWA_cctermno_list = 0;
CPWA_cctermno = sum(CPWA_cctermno_list);
CPWA_cc = cell(0, 2);
if isfield(CPWA, 'plus')
    CPWA_ccno = size(CPWA.plus, 1);
    CPWA_cctermno_list = cellfun(@(x)(size(x, 1)), CPWA.plus(:, 2));
    CPWA_cc = CPWA.plus;
    CPWA_cctermno = sum(CPWA_cctermno_list);
end

% decision variables summary:
%       name        type                length
%
%       ------------  Separable part  --------------
%       x           continuous          N
%       z_ij        continuous          knotno - N
%       iota_ij     binary              knotno - 2 * N
%       ------  Non-separable convex part  ---------
%       lambda_j    continuous          CPWA_cvno
%       ------  Non-separable concave part  --------
%       zeta_i      continuous          CPWA_ccno
%       delta_ij    binary              CPWA_cctermno

% compute the length of the decision variable
varlength = N + knotno - N + knotno - 2 * N ...
    + CPWA_cvno + CPWA_ccno + CPWA_cctermno;

objcon = c;
objvec = zeros(varlength, 1);

lb = -inf(varlength, 1);
ub = inf(varlength, 1);

vtype = [repmat('C', N, 1); repmat('C', knotno - N, 1); ...
    repmat('B', knotno - 2 * N, 1); repmat('C', CPWA_cvno, 1); ...
    repmat('C', CPWA_ccno, 1); repmat('B', CPWA_cctermno, 1)];

% the separable part

% constraints summary
%   description             number of constraints
%   relating x and z        N
%   relating z and iota     2 * (knotno - 2 * N)

sep1_rowno = N;
sep1_entryno = knotno;
sep1_r = zeros(sep1_entryno, 1);
sep1_c = zeros(sep1_entryno, 1);
sep1_v = zeros(sep1_entryno, 1);
sep1_b = zeros(sep1_rowno, 1);
sep1_sense = repmat('=', sep1_rowno, 1);

sep2_rowno = 2 * (knotno - 2 * N);
sep2_entryno = 2 * sep2_rowno;
sep2_r = zeros(sep2_entryno, 1);
sep2_c = zeros(sep2_entryno, 1);
sep2_v = zeros(sep2_entryno, 1);
sep2_b = zeros(sep2_rowno, 1);
sep2_sense = repmat('<', sep2_rowno, 1);

zcounter = N;
iotacounter = N + sum(knotno_list) - N;
sep1counter = 0;
sep2counter = 0;
sep2rowcounter = 0;

for i = 1:N
    lb(i) = knots{i}(1);
    ub(i) = knots{i}(end);
    
    % set objective
    objcon = objcon + knots_val{i}(1);
    objvec(zcounter + (1:knotno_list(i) - 1)) = diff(knots_val{i});
    
    % add bounds
    ub(zcounter + 1) = 1;
    lb(zcounter + knotno_list(i) - 1) = 0;
    
    % set the constraints relating x and z
    sep1_r(sep1counter + (1:knotno_list(i))) = i;
    sep1_c(sep1counter + 1) = i;
    sep1_v(sep1counter + 1) = 1;
    sep1_c(sep1counter + (2:knotno_list(i))) = ...
        zcounter + (1:knotno_list(i) - 1);
    sep1_v(sep1counter + (2:knotno_list(i))) = -diff(knots{i});
    sep1_b(i) = knots{i}(1);
    
    % set the constraints relating z and iota
    sep2_r(sep2counter + (1:4 * (knotno_list(i) - 2))) = ...
        sep2rowcounter + repelem((1:2 * (knotno_list(i) - 2))', 2, 1);
    sep2_c(sep2counter + (1:4:4 * (knotno_list(i) - 2))) = ...
        zcounter + (1:knotno_list(i) - 2);
    sep2_v(sep2counter + (1:4:4 * (knotno_list(i) - 2))) = -1;
    sep2_c(sep2counter + (2:4:4 * (knotno_list(i) - 2))) = ...
        iotacounter + (1:knotno_list(i) - 2);
    sep2_v(sep2counter + (2:4:4 * (knotno_list(i) - 2))) = 1;
    sep2_c(sep2counter + (3:4:4 * (knotno_list(i) - 2))) = ...
        zcounter + (2:knotno_list(i) - 1);
    sep2_v(sep2counter + (3:4:4 * (knotno_list(i) - 2))) = 1;
    sep2_c(sep2counter + (4:4:4 * (knotno_list(i) - 2))) = ...
        iotacounter + (1:knotno_list(i) - 2);
    sep2_v(sep2counter + (4:4:4 * (knotno_list(i) - 2))) = -1;
    sep2_b = zeros(sep2_rowno, 1);
    
    
    % update the counters
    zcounter = zcounter + knotno_list(i) - 1;
    iotacounter = iotacounter + knotno_list(i) - 2;
    sep1counter = sep1counter + knotno_list(i);
    sep2counter = sep2counter + (knotno_list(i) - 2) * 4;
    sep2rowcounter = sep2rowcounter + (knotno_list(i) - 2) * 2;
end

sep1_A = sparse(sep1_r, sep1_c, sep1_v, sep1_rowno, varlength);
sep2_A = sparse(sep2_r, sep2_c, sep2_v, sep2_rowno, varlength);
sep_A = [sep1_A; sep2_A];
sep_b = [sep1_b; sep2_b];
sep_sense = [sep1_sense; sep2_sense];

% the non-separable convex part
% constraints summary
%   description             number of constraints
%   relating lambda and x   CPWA_cvno

lambdacounter = N + knotno - N + knotno - 2 * N;
nonsep_cv_A = sparse(0, varlength);
nonsep_cv_b = zeros(0, 1);
nonsep_cv_sense = '';
if ~isempty(CPWA_cv)
    objvec(lambdacounter + (1:CPWA_cvno)) = 1;
    nonsep_cv_A_cell = cell(CPWA_cvno, 1);
    nonsep_cv_b_cell = cell(CPWA_cvno, 1);
    
    for j = 1:CPWA_cvno
        cvtermno = length(CPWA_cv{j, 2});
        cv_A_j = CPWA_cv{j, 1};
        [cv_A_r, cv_A_c, cv_A_v] = find(cv_A_j);
        cv_A_r = [cv_A_r; (1:cvtermno)']; %#ok<AGROW>
        cv_A_c = [cv_A_c; (lambdacounter + j) ...
            * ones(cvtermno, 1)]; %#ok<AGROW>
        cv_A_v = [cv_A_v; -ones(cvtermno, 1)]; %#ok<AGROW>
        nonsep_cv_A_cell{j} = sparse(cv_A_r, cv_A_c, cv_A_v, ...
            cvtermno, varlength);
        nonsep_cv_b_cell{j} = -CPWA_cv{j, 2};
    end
    
    nonsep_cv_A = vertcat(nonsep_cv_A_cell{:});
    nonsep_cv_b = vertcat(nonsep_cv_b_cell{:});
    nonsep_cv_sense = repmat('<', length(nonsep_cv_b), 1);
end

% the non-separable concave part
% constraints summary
%   description                 number of constraints
%   relating zeta and delta     CPWA_cctermno * 2
%   sum of binary variables     CPWA_ccno

zetacounter = N + knotno - N + knotno - 2 * N + CPWA_cvno;
deltacounter = N + knotno - N + knotno - 2 * N + CPWA_cvno + CPWA_ccno;
cctermcounter = 0;
nonsep_cc_A = sparse(0, varlength);
nonsep_cc_b = zeros(0, 1);
nonsep_cc_sense = '';
if ~isempty(CPWA_cc)
    objvec(zetacounter + (1:CPWA_ccno)) = -1;
    nonsep_cc1_A_cell = cell(CPWA_ccno, 1);
    nonsep_cc1_b_cell = cell(CPWA_ccno, 1);
    
    nonsep_cc2_A_r = zeros(CPWA_cctermno, 1);
    nonsep_cc2_A_c = zeros(CPWA_cctermno, 1);
    nonsep_cc2_A_v = ones(CPWA_cctermno, 1);
    
    for i = 1:CPWA_ccno
        termno = length(CPWA_cc{i, 2});
        
        % relating zeta and iota
        [cc1_A_i_r, cc1_A_i_c, cc1_A_i_v] = find(CPWA_cc{i, 1});
        M_i = zeros(termno, 1);
        
        for j = 1:termno
            a_diff = CPWA_cc{i, 1} - CPWA_cc{i, 1}(j, :);
            b_diff = CPWA_cc{i, 2} - CPWA_cc{i, 2}(j);
            
            M_i(j) = max(max(a_diff, 0) * ub(1:N) ...
                + min(a_diff, 0) * lb(1:N) + b_diff);
        end
        
        cc1_A_i_r = repelem(cc1_A_i_r, 2, 1) * 2;
        cc1_A_i_r(1:2:end) = cc1_A_i_r(1:2:end) - 1;
        cc1_A_i_c = repelem(cc1_A_i_c, 2, 1);
        cc1_A_i_v = -repelem(cc1_A_i_v, 2, 1);
        cc1_A_i_r = [cc1_A_i_r; (1:termno * 2)']; %#ok<AGROW>
        cc1_A_i_c = [cc1_A_i_c; (zetacounter + i) ...
            * ones(termno * 2, 1)]; %#ok<AGROW>
        cc1_A_i_v = [cc1_A_i_v; ones(termno * 2, 1)]; %#ok<AGROW>
        cc1_A_i_r = [cc1_A_i_r; (2:2:termno * 2)']; %#ok<AGROW>
        cc1_A_i_c = [cc1_A_i_c; deltacounter + (1:termno)']; %#ok<AGROW>
        cc1_A_i_v = [cc1_A_i_v; M_i]; %#ok<AGROW>
        cc1_b_i = CPWA_cc{i, 2};
        cc1_b_i = repelem(cc1_b_i, 2, 1);
        cc1_b_i(2:2:end) = cc1_b_i(2:2:end) + M_i;
        
        nonsep_cc1_A_cell{i} = sparse(cc1_A_i_r, cc1_A_i_c, cc1_A_i_v, ...
            termno * 2, varlength);
        nonsep_cc1_b_cell{i} = cc1_b_i;
        
        % the sum of binary variables need to be 1
        nonsep_cc2_A_r(cctermcounter + (1:termno)) = i;
        nonsep_cc2_A_c(cctermcounter + (1:termno)) = deltacounter ...
            + (1:termno);
        
        % update counters
        deltacounter = deltacounter + termno;
        cctermcounter = cctermcounter + termno;
    end
    
    nonsep_cc1_A = vertcat(nonsep_cc1_A_cell{:});
    nonsep_cc1_b = vertcat(nonsep_cc1_b_cell{:});
    nonsep_cc1_sense = repmat(['>'; '<'], CPWA_cctermno, 1);
    
    nonsep_cc2_A = sparse(nonsep_cc2_A_r, nonsep_cc2_A_c, ...
        nonsep_cc2_A_v, CPWA_ccno, varlength);
    nonsep_cc2_b = ones(CPWA_ccno, 1);
    nonsep_cc2_sense = repmat('=', CPWA_ccno, 1);
    
    
    nonsep_cc_A = [nonsep_cc1_A; nonsep_cc2_A];
    nonsep_cc_b = [nonsep_cc1_b; nonsep_cc2_b];
    nonsep_cc_sense = [nonsep_cc1_sense; nonsep_cc2_sense];
end


% assemble the gurobi model
model = struct;

model.modelsense = 'min';
model.objcon = objcon;
model.obj = objvec;
model.lb = lb;
model.ub = ub;
model.vtype = vtype;
model.A = [sep_A; nonsep_cv_A; nonsep_cc_A];
model.rhs = [sep_b; nonsep_cv_b; nonsep_cc_b];
model.sense = [sep_sense; nonsep_cv_sense; nonsep_cc_sense];

end
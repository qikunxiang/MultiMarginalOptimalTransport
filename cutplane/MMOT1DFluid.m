classdef MMOT1DFluid < LSIPMinCuttingPlaneAlgo
    % Class for multi-marginal optimal transport (MMOT) problems that
    % originated from the study of incompressible fluid where the marginals
    % are all uniform on [0,1] and the cost function is given by 
    % f(x1, ... ,xN) = |x1 - x2|^2 + |x_2 - x_3|^2 + ... + |x{N-1} - xN|^2
    % + |PHI(x1) - xN|^2 and PHI() encodes the final arrangment of the
    % particles

    properties(GetAccess = public, SetAccess = protected)
        % number of time steps
        TimeStepNum;

        % cell array containing the marginals
        Marginals;

        % cell array containing the arrangement function
        Arrangement;

        % a constant term that was subtracted from the cost function due to
        % not affecting the optimal coupling
        QuadraticTerm;
    end

    methods(Access = public)
        function obj = MMOT1DFluid(timestep_num, arrangement, ...
                testfuncs_knot_num, varargin)
            % Constructor method
            % Inputs:
            %   timestep_num: integer indicating the number of steps in the
            %   time discretization
            %   arrangment: cell array containing structs with fields
            %   knots and values encoding the arrangement of [0,1] into 
            %   [0,1] via a continuous piece-wise function
            %   testfuncs_knot_num: the number of knots used in defining
            %   the simplicial test functions for the marginals

            obj@LSIPMinCuttingPlaneAlgo(varargin{:});

            % set the default options for reducing constraints
            if ~isfield(obj.Options, 'reduce') ...
                    || isempty(obj.Options.reduce)
                obj.Options.reduce = struct;
            end

            if ~isfield(obj.Options.reduce, 'thres') ...
                    || isempty(obj.Options.reduce.thres)
                obj.Options.reduce.thres = inf;
            end

            if ~isfield(obj.Options.reduce, 'freq') ...
                    || isempty(obj.Options.reduce.freq)
                obj.Options.reduce.freq = 20;
            end

            if ~isfield(obj.Options.reduce, 'max_iter') ...
                    || isempty(obj.Options.reduce.max_iter)
                obj.Options.reduce.max_iter = inf;
            end

            % set the default options for the LP solver
            if ~isfield(obj.LPOptions, 'FeasibilityTol') ...
                    || isempty(obj.LPOptions.FeasibilityTol)
                obj.LPOptions.FeasibilityTol = 1e-9;
            end

            if ~isfield(obj.LPOptions, 'OptimalityTol') ...
                    || isempty(obj.LPOptions.OptimalityTol)
                obj.LPOptions.OptimalityTol = 1e-9;
            end

            % set the default option for the global minimization oracle
            if ~isfield(obj.GlobalOptions, 'pool_size') ...
                    || isempty(obj.GlobalOptions.pool_size)
                obj.GlobalOptions.pool_size = 10;
            end

            % set the arrangement
            % warning: we do not check if the continuous piece-wise affine
            % function indeed maps [0, 1] to [0, 1]
            obj.Arrangement = struct;
            assert(length(arrangement.knots) ...
                == length(arrangement.values) ...
                && all(diff(arrangement.knots) > 0) ...
                && abs(min(arrangement.knots)) < eps ...
                && abs(max(arrangement.knots) - 1) < eps ...
                && abs(min(arrangement.values)) < eps ...
                && abs(max(arrangement.values) - 1) < eps, ...
                'the arrangement function is mis-specified');

            obj.Arrangement.Knots = arrangement.knots;
            obj.Arrangement.KnotDiff = diff(obj.Arrangement.Knots);
            obj.Arrangement.Values = arrangement.values;
            obj.Arrangement.ValueDiff = diff(obj.Arrangement.Values);

            % compute the constant term that corresponds to the integral of
            % quadratic functions over the uniform marginal measures
            obj.QuadraticTerm = (2 * timestep_num - 1) / 3;
            obj.QuadraticTerm = obj.QuadraticTerm + sum( ...
                obj.Arrangement.KnotDiff .* ( ...
                obj.Arrangement.ValueDiff .^ 2 / 3 ...
                + obj.Arrangement.ValueDiff ...
                .* obj.Arrangement.Values(1:end - 1) / 2 ...
                + obj.Arrangement.Values(1:end - 1) .^ 2));

            % Lipschitz constaints of the cost function with respect to
            % each of the inputs
            arr_lip = max(abs(diff(arrangement.values) ...
                ./ diff(arrangement.knots)));
            obj.Arrangement.CostLipschitzConsts = ...
                4 * ones(timestep_num, 1);
            obj.Arrangement.CostLipschitzConsts(1) = ...
                2 * (1 + arr_lip);


            % this flag is used to track if the function
            % obj.initializeSimplicialTestFuncs has been called
            obj.Storage.SimplicialTestFuncsInitialized = false;

            obj.TimeStepNum = timestep_num;
            obj.Marginals = cell(timestep_num, 1);

            for marg_id = 1:timestep_num
                % each marginal is the uniform measure on [0,1]
                marg = ProbMeas1DCPWADens([0; 1], [1; 1]);

                marg_knots = linspace(0, 1, testfuncs_knot_num)';

                if marg_id == 1
                    % add the knots in the arrangement function into the
                    % test functions for the first marginal to simplify the
                    % global minimization oracle 

                    marg_knots = [marg_knots; obj.Arrangement.Knots]; ...
                        %#ok<AGROW>

                    [~, uind] = unique(round(marg_knots, 6), 'stable');
                    marg_knots = sort(marg_knots(uind), 'ascend');
                end

                marg.setSimplicialTestFuncs(marg_knots);
                obj.Marginals{marg_id} = marg;
            end

            % initialize the test functions
            obj.initializeSimplicialTestFuncs();
        end

        function vals = evaluateCostFunction(obj, inputs)
            % Evaluate the cost function at given input points. Note that
            % the quadratic terms are not included
            % Inputs: 
            %   inputs: matrix where each row represents the coordinates of
            %   each input point
            % Output:
            %   vals: vector containing the computed cost values

            vals = sum(inputs(:, 1:end - 1) .* inputs(:, 2:end), 2);
            arranged = sum(min(max((inputs(:, 1) ...
                - obj.Arrangement.Knots(1:end - 1)') ...
                ./ obj.Arrangement.KnotDiff', 0), 1) ...
                .* obj.Arrangement.ValueDiff', 2) ...
                + obj.Arrangement.Values(1);
            vals = vals + arranged .* inputs(:, end);

            % multiply by the coefficient -2
            vals = -2 * vals;
        end

        function coup_indices = generateHeuristicCoupling(obj)
            % Heuristically couple the marginals by applying comonotone 
            % coupling of the marginals
            % Outputs:
            %   coup_indices: matrix containing the coupled knot indices in
            %   the marginals

            marg_num = length(obj.Marginals);

            % retrieve some information from the marginals
            atoms_cell = cell(marg_num, 1);
            probs_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                atoms_cell{marg_id} = ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs.Knots;
                probs_cell{marg_id} = ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals;
            end

            [coup_indices, ~] = comonotone_coupling( ...
                atoms_cell, probs_cell);
        end
        
        function coup_indices = updateSimplicialTestFuncs(obj, ...
                testfuncs_knot_num)
            % Update the simplicial test functions after an execution of
            % the cutting-plane algorithm. Besides setting the new
            % simplicial test functions, a new set of couplings of 
            % discretized marginals are generated via reassembly of the 
            % dual solution from the cutting-plane algorithm with the new 
            % discretized marginals. These couplings can be used to 
            % generate initial constraints for the new LSIP problem with 
            % the updated test functions.
            % Input:
            %   testfuncs_knot_num: the number of knots used in defining
            %   the updated simplicial test functions for the marginals
            % Outputs:
            %   coup: matrix containing the indices of the coupled atoms in
            %   the discretized marginals

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            marg_num = length(obj.Marginals);

            % retrieve the dual solution resulted from the cutting-plane
            % algorithm
            dual_sol = obj.Runtime.DualSolution;
            old_coup_atoms = dual_sol.Atoms;
            old_coup_probs = dual_sol.Probabilities;

            new_marg_atoms_cell = cell(marg_num, 1);
            new_marg_probs_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                marg_knots = linspace(0, 1, testfuncs_knot_num)';

                if marg_id == 1
                    % add the knots in the arrangement function into the
                    % test functions for the first marginal to simplify the
                    % global minimization oracle 

                    marg_knots = [marg_knots; obj.Arrangement.Knots]; ...
                        %#ok<AGROW>

                    [~, uind] = unique(round(marg_knots, 6), 'stable');
                    marg_knots = sort(marg_knots(uind), 'ascend');
                end

                % set the new simplicial test functions
                obj.Marginals{marg_id}.setSimplicialTestFuncs(marg_knots);
                new_marg_atoms_cell{marg_id} = ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs.Knots;
                new_marg_probs_cell{marg_id} = ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals;
            end

            [coup_indices, ~] = discrete_reassembly_1D( ...
                old_coup_atoms, old_coup_probs, ...
                new_marg_atoms_cell, new_marg_probs_cell);

            obj.initializeSimplicialTestFuncs();
        end

        function initializeSimplicialTestFuncs(obj)
            % Initialize some quantities related to the simplicial test 
            % functions of the marginals

            marg_num = length(obj.Marginals);

            obj.Storage.MargKnotNumList = zeros(marg_num, 1);
            deci_logical_cell = cell(marg_num, 1);
            marg_knots_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                testfunc = obj.Marginals{marg_id}.SimplicialTestFuncs;
                testfunc_knot_num = length(testfunc.Knots);
                marg_knots_cell{marg_id} = testfunc.Knots;
                obj.Storage.MargKnotNumList(marg_id) = testfunc_knot_num;

                % the coefficient corresponding to the first test function
                % will not be included in the decision variable of the LP 
                % problem for identification purposes
                deci_logical_cell{marg_id} = [0; ...
                    ones(testfunc_knot_num - 1, 1)];
            end

            obj.Storage.TotalKnotNum = sum(obj.Storage.MargKnotNumList);

            % compute the offset of the knots in the marginals in the
            % vector containing all knots
            knot_num_cumsum = cumsum(obj.Storage.MargKnotNumList);
            obj.Storage.MargKnotNumOffsets = [0; ...
                knot_num_cumsum(1:end - 1)];

            % store the indices to place the decision variables of the LP 
            % problem in a vector containing the coefficients of the test 
            % functions
            obj.Storage.DeciVarIndicesInTestFuncs = find( ...
                vertcat(deci_logical_cell{:}));

            % concatenate the knots in all marginals into a single vector
            % for the ease of retrieving the knot values
            obj.Storage.MargKnotsConcat = vertcat(marg_knots_cell{:});

            obj.Storage.SimplicialTestFuncsInitialized = true;

            % updating the simplicial test functions will invalidate all
            % quantities in the runtime environment, thus all variables in
            % the runtime environment need to be flushed
            obj.Runtime = [];
        end

        function LB = getMMOTLowerBound(obj)
            % Retrieve the computed lower bound for the MMOT problem
            % Output:
            %   LB: the computed lower bound

            if ~isfield(obj.Runtime, 'PrimalSolution') ...
                    || isempty(obj.Runtime.PrimalSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            LB = -obj.Runtime.LSIP_UB;
        end

        function samps = randSampleFromOptCoupling(obj, samp_num, ...
                rand_stream)
            % Generate independent random sample from the optimized 
            % coupling of the marginals
            % Inputs: 
            %   samp_num: number of samples to generate
            %   rand_stream: RandStream object (default is
            %   RandStream.getGlobalStream)
            % Output:
            %   samps: matrix containing the samples from the coupling of
            %   the (continuous) uniform marginals
            %   samps_disc: matrix containing the indices of the samples
            %   from the coupling of the discretized uniform marginals

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            if ~exist('rand_stream', 'var') ...
                    || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            samps = obj.doRandSampleFromReassembly(samp_num, rand_stream);
        end

        function UB = getMMOTUpperBound(obj)
            % Compute an upper bound for the MMOT problem. The bound is
            % computed directly based on the comonotone coupling and is not
            % approximated by Monte Carlo.
            % Output:
            %   UB: upper bound for the MMOT problem

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            UB = obj.Runtime.ComonotoneMap.UB;
        end

        function outputs = evaluateOptMongeMaps(obj, marg1_inputs, ...
                batch_size)
            % Evaluate the computed approximately optimal Monge maps from
            % inputs from the first marginal. Computation is done in
            % batches if necessary.
            % Inputs: 
            %   marg1_inputs: vector representing input points in [0, 1]
            %   from the first marginal
            %   batch_size: number of inputs to be used for each vectorized
            %   procedure (default is 1e4)
            % Output:
            %   outputs: matrix containing the evaluated Monge maps where
            %   each row represents an input point and each column
            %   represents a marginal

            if ~exist('batch_size' ,'var') || isempty(batch_size)
                batch_size = 1e4;
            end

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            comonotone_map = obj.Runtime.ComonotoneMap.Info.map;

            input_num = length(marg1_inputs);
            batch_num = ceil(input_num / batch_size);
            output_cell = cell(batch_num, 1);

            for batch_id = 1:batch_num
                batch_inputs = marg1_inputs((batch_id - 1) * batch_size ...
                    + 1:min(batch_id * batch_size, input_num));

                output_cell{batch_id} = ...
                    obj.doEvaluateOptMongeMaps(batch_inputs, ...
                    comonotone_map);
            end

            outputs = vertcat(output_cell{:});

            % add the final positions after the rearrangement into the last
            % column of the output
            arranged = sum(min(max((marg1_inputs(:, 1) ...
                - obj.Arrangement.Knots(1:end - 1)') ...
                ./ obj.Arrangement.KnotDiff', 0), 1) ...
                .* obj.Arrangement.ValueDiff', 2) ...
                + obj.Arrangement.Values(1);

            outputs = [outputs, arranged];
        end

        function [UB, samps] = getMMOTUpperBoundMonteCarlo(obj, ...
                samp_num, rand_stream)
            % Compute an upper bound for the MMOT problem. The bound is
            % approximated by Monte Carlo integration. 
            % Inputs: 
            %   samp_num: number of samples for Monte Carlo integration
            %   rand_stream: RandStream object (default is
            %   RandStream.getGlobalStream)
            % Outputs:
            %   UB: upper bound for the MMOT problem
            %   samps: the Monte Carlo samples used in the approximation of
            %   the bounds

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            if ~exist('rand_stream', 'var') ...
                    || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            samps = obj.randSampleFromOptCoupling(samp_num, rand_stream);

            UB_list = obj.evaluateCostFunction(samps) + obj.QuadraticTerm;

            UB = mean(UB_list);
        end

        function EB = getMTErrorBoundBasedOnOT(obj)
            % Compute the error bound for the objective value of the 
            % MMOT problem based on the Lipschitz constant of the cost 
            % functions and the optimal transport distances between the 
            % marginals and their discretizations
            % Output:
            %   EB: the computed error bound

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            marg_num = length(obj.Marginals);
            EB = obj.Runtime.LSIP_UB - obj.Runtime.LSIP_LB;
            dual_sol = obj.Runtime.DualSolution;

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};

                marg_atoms = dual_sol.Atoms(:, marg_id);
                marg_probs = dual_sol.Probabilities;

                marg.setCoupledDiscreteMeasure(marg_atoms, marg_probs);
                EB = EB ...
                    + obj.Arrangement.CostLipschitzConsts(marg_id) ...
                    * marg.computeOTCost();
            end
        end

        function EB = getMTTheoreticalErrorBound(obj, LSIP_tolerance)
            % Compute the theoretical error bound for the objective value 
            % of the MMOT problem via the Lipschitz constant of the cost 
            % functions and the mesh sizes of the simplicial covers for the
            % marginals
            % Input:
            %   LSIP_tolerance: tolerance value used in the computation of
            %   the LSIP problem
            % Output:
            %   EB: the computed error bound

            marg_num = length(obj.Marginals);
            EB = LSIP_tolerance;

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                
                EB = EB + ...
                    obj.Arrangement.CostLipschitzConsts(marg_id) ...
                    * 2 * marg.SimplicialTestFuncs.MeshSize;
            end
        end

        function coup_indices = heuristicTimeBisection(obj)
            % Generate a coupling for a temporally refined version of the
            % problem where each time step is divided in half. 
            % Output:
            %   coup_indices: matrix containing the indices of the coupled
            %   knots in the refined problem

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            atoms_coord = obj.Runtime.DualSolution.Atoms;
            old_indices = obj.Runtime.DualSolution.Indices;
            [old_atom_num, marg_num] = size(old_indices);

            % take the knots in the existing test functions
            knots = obj.Marginals{end}.SimplicialTestFuncs.Knots;

            atoms_interpol = (atoms_coord(:, 1:end - 1) ...
                + atoms_coord(:, 2:end)) / 2;

            interp_indices_left = zeros(old_atom_num, marg_num - 1);

            for marg_id = 1:marg_num - 1
                interp_indices_left(:, marg_id) = min(old_atom_num - 1, ...
                    sum((atoms_interpol(:, marg_id) - knots') >= 0, 2));
            end

            coup_indices = zeros(old_atom_num * 3, marg_num * 2 - 1);

            coup_indices(:, 1:2:marg_num * 2 - 1) = repmat(old_indices, ...
                3, 1);
            coup_indices(1:old_atom_num, 2:2:marg_num * 2 - 2) = ...
                interp_indices_left;
            coup_indices(old_atom_num + 1:old_atom_num * 2, ...
                2:2:marg_num * 2 - 2) = interp_indices_left + 1;
            coup_indices(old_atom_num * 2 + 1:end, ...
                2:2:marg_num * 2 - 2) = old_indices(:, 2:end);
        end

        function setLSIPSolutions(obj, primal_sol, dual_sol, ...
                LSIP_UB, LSIP_LB)
            % Set the primal and dual solution of the LSIP problem in the
            % runtime environment. This is to allow certain quantities to
            % be computed using stored versions of the primal and dual
            % solutions without executing the cutting-plane algorithm
            % again. 
            % Inputs: 
            %   primal_sol: struct containing information about the primal
            %   solution of the LSIP problem
            %   dual_sol: struct containing information about the dual
            %   solution of the LSIP problem
            %   LSIP_UB: the upper bound for the optimal value of the LSIP
            %   problem
            %   LSIP_LB: the lower bound for the optimal value of the LSIP
            %   problem

            if isempty(obj.Runtime)
                obj.Runtime = struct;
            end

            obj.Runtime.PrimalSolution = primal_sol;
            obj.Runtime.DualSolution = dual_sol;
            obj.Runtime.LSIP_UB = LSIP_UB;
            obj.Runtime.LSIP_LB = LSIP_LB;
        end

        function setComonotoneMap(obj, comonotone_map)
            % Set the computed comonotone map in the runtime environment. 
            % This is to allow certain quantities to be computed using 
            % stored versions of the primal and dual solutions without 
            % executing the cutting-plane algorithm again. 

            if isempty(obj.Runtime)
                obj.Runtime = struct;
            end

            obj.Runtime.ComonotoneMap = comonotone_map;
        end
    end

    methods(Access = protected)

        function prepareRuntime(obj)
            % Prepare the runtime environment by initializing some
            % variables
            obj.Runtime = struct;

            prepareRuntime@LSIPMinCuttingPlaneAlgo(obj);

            % initialize the cuts to be empty
            obj.Runtime.CutIndices = zeros(0, length(obj.Marginals));

            % initialize the starting node in the global minimization
            % oracle to be 1
            obj.Runtime.GlobalMin = struct;
            obj.Runtime.GlobalMin.StartNode = 1;
        end

        function initializeBeforeRun(obj)
            % Initialize the algorithm by computing some static quantities

            if ~obj.Storage.SimplicialTestFuncsInitialized
                obj.initializeSimplicialTestFuncs();
            end
        end

        function [UB, comono] = computeComonotoneUB(obj, ...
                atom_indices, probs)
            % Compute an upper bound for the MMOT problem via constructing
            % a comonotone coupling from the dual LP relaxation. This upper
            % bound does not require Monte Carlo integration.
            % Inputs:
            %   atom_indices: matrix containing the knot indices of the
            %   atoms in the discrete measure
            %   probs: probabilities of the atoms in the discrete measure
            % Outputs:
            %   UB: the computed upper bound
            %   comono: struct containing fields probabilities and map that
            %   represents the comonotone coupling computed from the input
            %   which also represents Monge maps from the first marginal to
            %   the rest

            marg_num = length(obj.Marginals);
            atom_num = length(probs);

            arr_knots = obj.Arrangement.Knots;
            arr_vals = obj.Arrangement.Values;
            arr_slopes = diff(arr_vals) ./ diff(arr_knots);
            arr_knot_num = length(arr_knots);

            comono = struct;

            % sort all atoms according to lexicographic order
            [atom_indices, sorted_order] = ...
                sortrows(atom_indices, 1:marg_num);
            probs = probs(sorted_order);

            comono.probabilities = probs;

            comono.map = zeros(atom_num, marg_num);

            for marg_id = 1:marg_num
                if marg_id == 1
                    probs_cum = cumsum(probs);
                    comono.map(:, 1) = [0; probs_cum(1:end - 1)];

                    continue;
                end

                % sort the marginal atoms into ascending order (note that
                % there will be ties)
                [~, marg_atoms_order] = sort(atom_indices(:, marg_id), ...
                    'ascend');
                probs_cum = cumsum(probs(marg_atoms_order));
                comono.map(marg_atoms_order, marg_id) = ...
                    [0; probs_cum(1:end - 1)];
            end

            % compute the upper bound by integrating along the diagonal
            % lines (corresponding to the support of the comonotone
            % coupling)
            UB = 0;

            for marg_id = 1:marg_num - 1
                % lower limit of the variable on the left
                lowerlim_left = comono.map(:, marg_id);

                % lower limit of the variable on the right
                lowerlim_right = comono.map(:, marg_id + 1);

                UB = UB + sum( ...
                    lowerlim_left .* lowerlim_right .* probs ...
                    + (lowerlim_left + lowerlim_right) .* probs.^2 / 2 ...
                    + probs.^3 / 3);
            end

            % handle the last term involving the arrangement function
            marg1_map = comono.map(:, 1);
            margN_map = comono.map(:, marg_num);
            lowerlim_left = zeros(atom_num + arr_knot_num - 2, 1);
            lowerlim_right = zeros(atom_num + arr_knot_num - 2, 1);
            probs_inc = zeros(atom_num + arr_knot_num - 2, 1);
            slopes = zeros(atom_num + arr_knot_num - 2, 1);

            % compute the images under the arrangement function
            marg1_arr = sum(min(max((marg1_map ...
                - obj.Arrangement.Knots(1:end - 1)') ...
                ./ obj.Arrangement.KnotDiff', 0), 1) ...
                .* obj.Arrangement.ValueDiff', 2) ...
                + obj.Arrangement.Values(1);

            % counter indicating the original intervals that have been
            % processed
            counter_ori = 0;

            % counter indicating the new intervals that have been processed
            counter_new = 0;

            % since we knot the first knot is 0, it can be skipped
            for knot_id = 2:arr_knot_num
                % find the left most interval that contains the knot
                knot = arr_knots(knot_id);
                int_contain = find(marg1_map < knot, 1, 'last');

                % fill the values to the left of the knots

                % indices of the original intervals
                ori_list = (counter_ori + 1):(int_contain - 1);
                int_num = length(ori_list);

                % indices of the new intervals
                new_list = counter_new + (1:int_num);
                
                lowerlim_left(new_list) = margN_map(ori_list);
                lowerlim_right(new_list) = marg1_arr(ori_list);
                probs_inc(new_list) = probs(ori_list);
                slopes(new_list) = arr_slopes(knot_id - 1);
                counter_ori = counter_ori + int_num;
                counter_new = counter_new + int_num;

                % process the left half of the interval (that is, the part
                % of the interval to the left of the knot)
                wholeint_size = probs(int_contain);
                leftint_size = knot - marg1_map(int_contain);
                rightint_size = wholeint_size - leftint_size;

                % mark the interval containing the knot as processed
                counter_ori = counter_ori + 1;

                if leftint_size >= 1e-10
                    % if the left half of the interval is very small 
                    % (possibly due to rounding error), skip it
                    lowerlim_left(counter_new + 1) = ...
                        margN_map(int_contain);
                    lowerlim_right(counter_new + 1) = ...
                        marg1_arr(int_contain);
                    probs_inc(counter_new + 1) = leftint_size;
                    slopes(counter_new + 1) = arr_slopes(knot_id - 1);

                    counter_new = counter_new + 1;
                end

                if rightint_size < 1e-10 || knot_id == arr_knot_num
                    % if the right half of the interval is very small
                    % (possibly due to rounding error) or when the current 
                    % knot is the last one, skip it
                    continue;
                end

                % process the right half of the interval (that is, the part
                % of the interval to the right of the knot)
                % CHECK HERE!!!!!!
                lowerlim_left(counter_new + 1) = margN_map(int_contain) ...
                    + leftint_size;
                lowerlim_right(counter_new + 1) = arr_vals(knot_id);
                probs_inc(counter_new + 1) = rightint_size;
                slopes(counter_new + 1) = arr_slopes(knot_id);

                % update the counter
                counter_new = counter_new + 1;
            end

            % compute the last intergral
            UB = UB + sum( ...
                lowerlim_left .* lowerlim_right .* probs_inc ...
                + (lowerlim_right + slopes .* lowerlim_left) ...
                .* probs_inc.^2 / 2 ...
                + slopes .* probs_inc.^3 / 3);

            % multiply by -2 and add the constant term
            UB = -2 * UB + obj.QuadraticTerm;
        end

        function updateRuntimeAfterLP(obj, result)
            % Update the runtime environment after solving each LP
            % Input:
            %   result: struct produced by gurobi

            updateRuntimeAfterLP@LSIPMinCuttingPlaneAlgo(obj, result);

            pos_list = result.pi > 0;
            probs = result.pi(pos_list);
            atom_indices = obj.Runtime.CutIndices(pos_list, :);

            % the probabilities need to be normalized due to possible
            % constraint violations from the LP solver
            probs = probs / sum(probs);

            [UB, comono] = obj.computeComonotoneUB(atom_indices, probs);

            if ~isfield(obj.Runtime, 'ComonotoneMap') ...
                    || isempty(obj.Runtime.ComonotoneMap) ...
                    || UB < obj.Runtime.ComonotoneMap.UB
                obj.Runtime.ComonotoneMap = struct('UB', UB, ...
                    'Info', comono);
            end
        end

        function model = generateInitialMinModel(obj)
            % Generate the initial linear programming model for gurobi
            % Output:
            %   model: struct containing the linear programming model in
            %   gurobi

            model = struct;
            marg_num = length(obj.Marginals);
            decivar_num = length(obj.Storage.DeciVarIndicesInTestFuncs) ...
                + 1;

            integrals_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                integrals_cell{marg_id} = ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals;
            end

            integrals_vec = vertcat(integrals_cell{:});
            
            % since the cutting plane algorithm assumes that the problem is
            % a minimization problem, we need to transform our maximization
            % problem into a minimization problem
            model.modelsense = 'min';
            model.objcon = -obj.QuadraticTerm;

            % the coefficients corresponding to the first test function of
            % each marginal is not included in the decision variables for
            % identification purposes
            model.obj = [-1; ...
                -integrals_vec(obj.Storage.DeciVarIndicesInTestFuncs)];

            model.sense = '>';
            model.lb = -inf(decivar_num, 1);
            model.ub = inf(decivar_num, 1);
            model.A = sparse(0, decivar_num);
            model.rhs = zeros(0, 1);
        end

        function [min_lb, optimizers] = callGlobalMinOracle(obj, vec)
            % Given a decision vector, call the global minimization oracle
            % to approximately determine the "most violated" constraints
            % and return a lower bound for the minimal value
            % Input:
            %   vec: vector corresponding to the current LP solution
            % Outputs:
            %   min_lb: lower bound for the global minimization problem
            %   optimizers: cell array containing structs with fields
            %   inputs and qualites that correspond to the approximate 
            %   optimizers of the global minimization problems

            marg_num = length(obj.Marginals);
            
            % constant value in the decision variable
            obj_const = vec(1);
            
            % convert the locations of the knots and the coefficients in 
            % the decision variable into a cell array
            tf_vec = zeros(obj.Storage.TotalKnotNum, 1);
            tf_vec(obj.Storage.DeciVarIndicesInTestFuncs) = vec(2:end);
            tf_cell = cell(marg_num, 2);

            for marg_id = 1:marg_num
                tf_cell{marg_id, 1} = obj.Marginals{marg_id ...
                    }.SimplicialTestFuncs.Knots;
                tf_cell{marg_id, 2} = tf_vec( ...
                    obj.Storage.MargKnotNumOffsets(marg_id) ...
                    + (1:(obj.Storage.MargKnotNumList(marg_id))));
            end

            % evaluate the arrangement function at all knots of marginal 1
            % including the added knots from the arrangment function itself
            arranged = sum(min(max((tf_cell{1, 1} ...
                - obj.Arrangement.Knots(1:end - 1)') ...
                ./ obj.Arrangement.KnotDiff', 0), 1) ...
                .* obj.Arrangement.ValueDiff', 2) ...
                + obj.Arrangement.Values(1);

            % begin the max-product algorithm
            % we choose a root node and perform variable elimination
            % forward; for example, starting from marginal 1, we would
            % perform elimination of marginals 2, 3, 4, ... and eventually
            % maximize over marginals 1 and N; alternatively, starting from
            % marginal N-2, we would perform elimination of marginals N-1,
            % N, 1, 2, ... and eventually maximize over marginals N-2 and
            % N-3

            % this is the root node that will be minimized over in the end
            start_node = obj.Runtime.GlobalMin.StartNode;

            % this is the node to eliminate
            eli_node = mod(start_node, marg_num) + 1;

            % store the lookup tables for identifying the optimal states
            % later
            lookup_table_cell = cell(marg_num, 1);

            start_state_num = length(tf_cell{start_node, 1});

            % compute the cost associated with pair (start_node, eli_node)
            prod_s_e_left = tf_cell{start_node, 1};
            prod_s_e_right = tf_cell{eli_node, 1};

            % when eli_node is marginal 1, the locations of the knots need 
            % to pass through the arrangement function
            if eli_node == 1
                prod_s_e_right = arranged;
            end

            % in this cost matrix, each row corresponds to a state in
            % start_node and each column corresponds to a state in eli_node
            cost_s_e_mat = (-2 * prod_s_e_left) .* prod_s_e_right';


            while mod(eli_node, marg_num) + 1 ~= start_node
                % this is the node that will be eliminated in the next
                % iteration if it is not the starting node
                next_node = mod(eli_node, marg_num) + 1;

                next_state_num = length(tf_cell{next_node, 1});

                % to eliminate eli_node, we compute the cost associated
                % with pair (eli_node, next_node)
                prod_e_n_left = tf_cell{eli_node, 1};
                prod_e_n_right = tf_cell{next_node, 1};

                % when next_node marginal 1, the locations of the knots need
                % to pass through the arrangement function
                if next_node == 1
                    prod_e_n_right = arranged;
                end

                cost_e_n_mat = (-2 * prod_e_n_left) .* prod_e_n_right';
                
                % add the two cost matrices and subtract the test function
                % values on eli_node
                cost_mat = repmat(cost_s_e_mat - tf_cell{eli_node, 2}', ...
                    next_state_num, 1) ...
                    + repelem(cost_e_n_mat', start_state_num, 1);

                % compute the minimum value and the minimizer for each
                % combination of states in start_node and states in
                % next_node
                [min_cost_vec, min_state_vec] = min(cost_mat, [], 2);

                % store the minimizer states in the eliminated node
                lookup_table_cell{eli_node} = reshape(min_state_vec, ...
                    start_state_num, next_state_num);

                % advance to the next node
                eli_node = next_node;
                cost_s_e_mat = reshape(min_cost_vec, start_state_num, ...
                    next_state_num);
            end

            % now, only two nodes are left and we eliminate eli_node
            prod_e_n_left = tf_cell{eli_node, 1};
            prod_e_n_right = tf_cell{start_node, 1};

            % when start_node is marginal 1, the locations of the knots
            % need to pass through the arrangement function
            if start_node == 1
                prod_e_n_right = arranged;
            end

            cost_e_n_mat = (-2 * prod_e_n_left) .* prod_e_n_right';

            % add the two cost matrices and subtract the test function
            % values on eli_node; also add the constant term in the
            % objecive
            cost_mat = cost_s_e_mat + cost_e_n_mat' ...
                - tf_cell{eli_node, 2}' - obj_const;

            [min_cost_vec, min_state_vec] = min(cost_mat, [], 2);

            % finally, subtract the test function values on start_node and
            % minimize over the states in start_node
            cost_vec = min_cost_vec - tf_cell{start_node, 2};

            [cost_sorted, sorted_ind] = sort(cost_vec, 'ascend');

            % this is the minimum over all combinations of states
            min_lb = cost_sorted(1);

            % retrieve a pool of solutions
            pool_size = min(obj.GlobalOptions.pool_size, start_state_num);

            % trace back the lookup tables to retrieve the minimizing
            % states (knot indices)
            min_knot_indices = zeros(pool_size, marg_num);

            % the approximately optimal states in start_node
            min_knot_indices(:, start_node) = sorted_ind(1:pool_size);

            % the approximately optimal states in eli_node that was
            % eliminated last
            min_knot_indices(:, eli_node) = ...
                min_state_vec(min_knot_indices(:, start_node));

            % trace from eli_node to all of the previously eliminated nodes
            % in reverse direction

            while mod(eli_node - 2, marg_num) + 1 ~= start_node
                prev_node = mod(eli_node - 2, marg_num) + 1;

                min_knot_indices(:, prev_node) = ...
                    lookup_table_cell{prev_node}( ...
                    sub2ind(size(lookup_table_cell{prev_node}), ...
                    min_knot_indices(:, start_node), ...
                    min_knot_indices(:, eli_node)));

                eli_node = prev_node;
            end
            
            optimizers = struct;
            optimizers.knot_indices = min_knot_indices;

            % every time the global minimization oracle is called, we
            % change the value of the starting node to create more diverse
            % sub-optimal solutions
            obj.Runtime.GlobalMin.StartNode = mod( ...
                obj.Runtime.GlobalMin.StartNode, marg_num) + 1;
        end

        function updateLSIPUB(obj, min_lb, ~) 
            % Update the LSIP upper bound after each call to the global
            % minimization oracle
            % Inputs:
            %   min_lb: the lower bound for the global minimization problem
            %   optimizers: a set of approximate optimizers of the global
            %   minimization problem

            obj.Runtime.LSIP_UB = min(obj.Runtime.LSIP_UB, ...
                obj.Runtime.LSIP_LB - min_lb);
        end
        
        function addConstraints(obj, optimizers)
            % Given a collection of approximate optimizers from the global
            % minimization oracle, generate and add the corresponding
            % linear constraints
            % Inputs:
            %   optimizers: output of the method callGlobalOracle

            constr_num = size(optimizers.knot_indices, 1);
            marg_num = length(obj.Marginals);
            
            col_indices = optimizers.knot_indices' ...
                + obj.Storage.MargKnotNumOffsets;

            % first generate a matrix containing all test functions (each
            % row corresponds to an approximate optimizer, each column
            % corresponds to a test function)
            A_full = sparse(repelem((1:constr_num)', marg_num, 1), ...
                col_indices(:), 1, constr_num, obj.Storage.TotalKnotNum);
            
            % filter out those test functions whose coefficients are not
            % included in the decision variable, then prepend a column of
            % 1
            A_new = [sparse(ones(constr_num, 1)), ...
                A_full(:, obj.Storage.DeciVarIndicesInTestFuncs)];

            % retrieve the coordinates of the optimal points
            opt_points = obj.Storage.MargKnotsConcat(col_indices)';
            rhs_new = obj.evaluateCostFunction(opt_points);

            
            % add the newly generated constraints to the end
            obj.Runtime.CurrentLPModel.A = ...
                [obj.Runtime.CurrentLPModel.A; -A_new];
            obj.Runtime.CurrentLPModel.rhs = ...
                [obj.Runtime.CurrentLPModel.rhs; -rhs_new];

            if ~isempty(obj.Runtime.vbasis) && ~isempty(obj.Runtime.cbasis)
                obj.Runtime.CurrentLPModel.vbasis = obj.Runtime.vbasis;
                obj.Runtime.CurrentLPModel.cbasis = ...
                    [obj.Runtime.cbasis; zeros(constr_num, 1)];
            else
                if isfield(obj.Runtime.CurrentLPModel, 'vbasis')
                    obj.Runtime.CurrentLPModel ...
                        = rmfield(obj.Runtime.CurrentLPModel, 'vbasis');
                end

                if isfield(obj.Runtime.CurrentLPModel, 'cbasis')
                    obj.Runtime.CurrentLPModel ...
                        = rmfield(obj.Runtime.CurrentLPModel, 'cbasis');
                end
            end

            % add the added indices to the runtime environment
            obj.Runtime.CutIndices = [obj.Runtime.CutIndices; ...
                optimizers.knot_indices];

            % if obj.Runtime.NumOfInitialConstraints is not set, it means
            % that this is the first call to obj.addConstraints which
            % generates the initial constraints; this number is stored in
            % the runtime environment
            if ~isfield(obj.Runtime, 'NumOfInitialConstraints') ...
                    || isempty(obj.Runtime.NumOfInitialConstraints)
                obj.Runtime.NumOfInitialConstraints = constr_num;
            end
        end

        function reduceConstraints(obj, result)
            % Remove some of the constraints to speed up the LP solver
            % Input:
            %   result: the output from the gurobi LP solver

            if ~isinf(obj.Options.reduce.thres) ...
                    && obj.Runtime.iter > 0 ...
                    && obj.Runtime.iter <= obj.Options.reduce.max_iter ...
                    && mod(obj.Runtime.iter, obj.Options.reduce.freq) == 0

                % the list of constraints to be kept (here since the
                % directions of the inequalities are all >=, the slackness
                % is non-positive; the threshold specifies the maximum
                % absolute value of slackness)
                keep_list = result.slack >= -obj.Options.reduce.thres;

                % always keep the initial constraints
                keep_list(1:obj.Runtime.NumOfInitialConstraints) = true;

                % update all variables
                obj.Runtime.CutIndices ...
                    = obj.Runtime.CutIndices(keep_list, :);
                obj.Runtime.CurrentLPModel.A ...
                    = obj.Runtime.CurrentLPModel.A(keep_list, :);
                obj.Runtime.CurrentLPModel.rhs ...
                    = obj.Runtime.CurrentLPModel.rhs(keep_list);

                if ~isempty(obj.Runtime.cbasis)
                    obj.Runtime.cbasis = obj.Runtime.cbasis(keep_list);
                end
            end
        end

        function primal_sol = buildPrimalSolution(obj, result, violation)
            % Given the output from gurobi and a lower bound for the
            % optimal value of the global minimization oracle, build the
            % corresponding primal solution
            % Inputs:
            %   result: output of the gurobi LP solver
            %   violation: a lower bound for the global minimization
            %   problem
            % Output:
            %   primal_sol: the constructed potential functions on the
            %   support of the input measures

            vec = result.x;
            primal_sol = struct;
            primal_sol.Constant = vec(1) - violation;

            knot_coefs = zeros(obj.Storage.TotalKnotNum, 1);
            knot_coefs(obj.Storage.DeciVarIndicesInTestFuncs) = vec(2:end);
            marg_num = length(obj.Marginals);
            primal_sol.Coefficients = cell(marg_num, 1);

            for marg_id = 1:marg_num
                primal_sol.Coefficients{marg_id} = knot_coefs( ...
                    obj.Storage.MargKnotNumOffsets(marg_id) ...
                    + (1:obj.Storage.MargKnotNumList(marg_id)));
            end
        end

        function dual_sol = buildDualSolution(obj, result)
            % Given the output from gurobi, build the corresponding dual
            % solution
            % Input:
            %   result: output of the gurobi LP solver
            % Output:
            %   dual_sol: the constructed discrete probability measure for
            %   the relaxed MMOT problem

            dual_sol = struct;
            pos_list = result.pi > 0;
            dual_sol.Probabilities = result.pi(pos_list);
            dual_sol.Indices = obj.Runtime.CutIndices(pos_list, :);

            dual_sol.Atoms = obj.Storage.MargKnotsConcat( ...
                dual_sol.Indices + obj.Storage.MargKnotNumOffsets');

            % normalize the probabilities since there can be numerical
            % errors from the LP solver
            dual_sol.Probabilities = dual_sol.Probabilities ...
                / sum(dual_sol.Probabilities);
        end

        function samps = doRandSampleFromReassembly(obj, ...
                samp_num, rand_stream)
            % Generate independent random samples from a reassembly
            % Inputs:
            %   samp_num: number of samples to generate
            %   rand_stream: RandStream object used for sampling
            % Outputs:
            %   samps: matrx containing the samples in rows

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            marg_num = length(obj.Marginals);
            
            for marg_id = 1:marg_num
                disc_marg_atoms = obj.Marginals{marg_id ...
                    }.SimplicialTestFuncs.Knots;
                disc_marg_probs = obj.Marginals{marg_id ...
                    }.SimplicialTestFuncs.Integrals;

                obj.Marginals{marg_id}.setCoupledDiscreteMeasure( ...
                    disc_marg_atoms, disc_marg_probs);
            end

            n = length(obj.Runtime.DualSolution.Probabilities);
            
            % generate random indices of the atoms according to the
            % probabilities
            disc_atom_index_samps = randsample(rand_stream, n, ...
                samp_num, true, obj.Runtime.DualSolution.Probabilities);

            knot_indices_samps = obj.Runtime.DualSolution.Indices( ...
                disc_atom_index_samps, :);

            samps = zeros(samp_num, marg_num);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};

                marg_knot_indices_samps = knot_indices_samps(:, marg_id);

                % count the number of samples coupled with each of the
                % atoms in the discretized marginal
                atom_num_list = accumarray( ...
                    marg_knot_indices_samps, 1, ...
                    [obj.Storage.MargKnotNumList(marg_id), 1]);

                % generate from the conditional distributions
                cont_samp_cell = marg.conditionalRandSample( ...
                    atom_num_list, rand_stream);

                % fill in the coupled samples from the continuous marginals
                for atom_id = 1:length(atom_num_list)
                    samps(marg_knot_indices_samps == atom_id, ...
                        marg_id) = cont_samp_cell{atom_id};
                end
            end
        end

        function outputs = doEvaluateOptMongeMaps(~, marg1_inputs, ...
                comonotone_map)
            % Evaluate sub-optimal Monge maps from inputs from the first 
            % marginal. 
            % Inputs: 
            %   marg1_inputs: vector representing input points in [0, 1]
            %   from the first marginal
            %   comonotone_map: matrix representing Monge maps where each
            %   row corresponds to an interval and each column corresponds
            %   to a marginal
            % Output:
            %   outputs: matrix containing the evaluated Monge maps where
            %   each row represents an input point and each column
            %   represents a marginal

            % index of the sub-interval each input point is in
            interval_indices = sum(marg1_inputs ...
                >= comonotone_map(:, 1)', 2);

            % location of each input point in its respective
            % sub-interval with respect to the left end point
            interval_location = marg1_inputs ...
                - comonotone_map(interval_indices, 1);

            % select the left end points of the respective
            % sub-intervals for each marginal and then add the
            % increment terms
            outputs = comonotone_map(interval_indices, :) ...
                + interval_location;
        end

        function display_string = buildMessage(obj)
            % Build a string as the message to display after each iteration
            % Output:
            %   display_string: the string to display

            display_string = sprintf(['%s: ' ...
                'iteration %4d: LSIP_LB = %10.4f, LSIP_UB = %10.4f, ' ...
                'diff = %10.6f, ' ...
                'MMOT_LB = %10.4f, MMOT_UB = %10.4f, ' ...
                'diff = %10.6f\n'], class(obj), ...
                obj.Runtime.iter, ...
                obj.Runtime.LSIP_LB, ...
                obj.Runtime.LSIP_UB, ...
                obj.Runtime.LSIP_UB - obj.Runtime.LSIP_LB, ...
                -obj.Runtime.LSIP_UB, ...
                obj.Runtime.ComonotoneMap.UB, ...
                obj.Runtime.ComonotoneMap.UB + obj.Runtime.LSIP_UB);
        end
    end
end


classdef MMOT1DCPWA < LSIPMinCuttingPlaneAlgo
    % Class for multi-marginal optimal transport (MMOT) problems with
    % one-dimensional marginals and a continuous piece-wise affine (CPWA)
    % cost function. The cost function must be specified as the sum of
    % a set of convex CPWA functions with max-representations minus the sum
    % of another set of convex CPWA functions with max-representations.

    properties(GetAccess = public, SetAccess = protected)
        % cell array containing the marginals
        Marginals;

        % struct containing information about the cost function with fields
        % Convex and Concave representing the convex and concave parts of
        % the cost function; each of Plus and Minus is a cell array with
        % two columns where the first column represents the weight vectors
        % and the second column represents the intercepts
        CostFunc;
    end

    methods(Access = public)
        function obj = MMOT1DCPWA(marginals, costfunc, varargin)
            % Constructor method
            % Inputs:
            %   marginal: cell array containing the marginals
            %   costfunc: struct with fields plus and minus representing
            %   the cost function; see the documentation on the member
            %   variable CostFunc

            obj@LSIPMinCuttingPlaneAlgo(varargin{:});

            % set the default option for the global formulation
            if ~isfield(obj.Options, 'global_formulation') ...
                    || isempty(obj.Options.global_formulation)
                % available formulations: LOG and INC
                obj.Options.global_formulation = 'LOG';
            end

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

            % set the default option for the LP solver
            if ~isfield(obj.LPOptions, 'FeasibilityTol') ...
                    || isempty(obj.LPOptions.FeasibilityTol)
                obj.LPOptions.FeasibilityTol = 1e-9;
            end

            marg_num = length(marginals);

            % set the cost function
            obj.CostFunc = struct;

            % compute the Lipschitz constants of the cost function
            obj.CostFunc.LipschitzConstants = zeros(marg_num, 1);
            
            if isfield(costfunc, 'plus') && ~isempty(costfunc.plus)
                obj.CostFunc.NumOfConvex = size(costfunc.plus, 1);
                assert(size(costfunc.plus, 2) == 2, ...
                    'convex part of the cost function mis-specified');
                obj.CostFunc.Convex = costfunc.plus;

                for cvx_id = 1:obj.CostFunc.NumOfConvex
                    weights = obj.CostFunc.Convex{cvx_id, 1};
                    intercepts = obj.CostFunc.Convex{cvx_id, 2};

                    assert(size(weights, 1) == length(intercepts) ...
                        && size(weights, 2) == marg_num, ...
                        'convex part of the cost function mis-specified');

                    obj.CostFunc.LipschitzConstants = ...
                        obj.CostFunc.LipschitzConstants ...
                        + max(abs(weights), [], 1)';
                end
            else
                obj.CostFunc.NumOfConvex = 0;
            end

            if isfield(costfunc, 'minus') && ~isempty(costfunc.minus)
                obj.CostFunc.NumOfConcave = size(costfunc.minus, 1);
                assert(size(costfunc.minus, 2) == 2, ...
                    'concave part of the cost function mis-specified');
                obj.CostFunc.Concave = costfunc.minus;

                for ccv_id = 1:obj.CostFunc.NumOfConcave
                    weights = obj.CostFunc.Concave{ccv_id, 1};
                    intercepts = obj.CostFunc.Concave{ccv_id, 2};

                    assert(size(weights, 1) == length(intercepts) ...
                        && size(weights, 2) == marg_num, ...
                        'convex part of the cost function mis-specified');

                    obj.CostFunc.LipschitzConstants = ...
                        obj.CostFunc.LipschitzConstants ...
                        + max(abs(weights), [], 1)';
                end
            else
                obj.CostFunc.NumOfConcave = 0;
            end

            assert(obj.CostFunc.NumOfConvex > 0 ...
                || obj.CostFunc.NumOfConcave > 0, ...
                'the cost function is empty');

            % this flag is used to track if the function
            % obj.initializeSimplicialTestFuncs has been called
            obj.Storage.SimplicialTestFuncsInitialized = false;

            obj.Marginals = marginals;
            marg_testfunc_set = false(marg_num, 1);

            for marg_id = 1:marg_num
                marg_testfunc_set(marg_id) = ~isempty( ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs);
            end

            if all(marg_testfunc_set)
                % initialize the simplicial test functions at the end of 
                % the constructor if they have already been set
                obj.initializeSimplicialTestFuncs();
            end
        end

        function vals = evaluateCostFunction(obj, inputs, batch_size)
            % Evaluate the cost function at given input points. Computation
            % is done in batches if necessary.
            % Inputs: 
            %   inputs: matrix where each row represents the coordinates of
            %   each input point
            %   batch_size: maximum number of inputs to be handled by the
            %   vectorized procedure (default is 1e4)
            % Output:
            %   vals: vector containing the computed cost values

            if ~exist('batch_size', 'var') || isempty(batch_size)
                batch_size = 1e4;
            end

            input_num = size(inputs, 1);
            batch_num = ceil(input_num / batch_size);
            val_cell = cell(batch_num, 1);

            for batch_id = 1:batch_num
                batch_inputs = inputs((batch_id - 1) * batch_size + 1 ...
                    : min(batch_id * batch_size, input_num), :);
                val_cell{batch_id} = zeros(size(batch_inputs, 1), 1);
    
                for cvx_id = 1:obj.CostFunc.NumOfConvex
                    val_cell{batch_id} = val_cell{batch_id} ...
                        + max(batch_inputs ...
                        * obj.CostFunc.Convex{cvx_id, 1}' ...
                        + obj.CostFunc.Convex{cvx_id, 2}', [], 2);
                end
    
                for ccv_id = 1:obj.CostFunc.NumOfConcave
                    val_cell{batch_id} = val_cell{batch_id} ...
                        - max(batch_inputs ...
                        * obj.CostFunc.Concave{ccv_id, 1}' ...
                        + obj.CostFunc.Concave{ccv_id, 2}', [], 2);
                end
            end

            vals = vertcat(val_cell{:});
        end

        function [atoms, testfunc_vals] = generateHeuristicCoupling(obj)
            % Heuristically couple the marginals by applying comonotone 
            % coupling of the marginals
            % Outputs:
            %   atoms: matrix containing the atoms in the heuristic
            %   coupling
            %   testfunc_vals: sparse matrix containing the values of the
            %   test functions evaluated at the computed atoms where each
            %   row corresponds to an atom and each column corresponds to a
            %   test function

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

            % add the offsets to the knot indices and retrieve the atoms
            offseted_indices = coup_indices' ...
                + obj.Storage.MargKnotNumOffsets;
            atoms = obj.Storage.MargKnotsConcat(offseted_indices)';
            atom_num = size(atoms, 1);
            testfunc_vals = ...
                sparse(repelem((1:atom_num)', marg_num, 1), ...
                offseted_indices(:), 1, atom_num, ...
                obj.Storage.TotalKnotNum);
        end

        function setSimplicialTestFuncs(obj, args_cell)
            % Set the simplicial test functions for all marginals at the
            % same time
            % Input:
            %   args_cell: cell array where each cell is a cell array
            %   containing all inputs to the method setSimplicialTestFuncs
            %   of each marginal

            for marg_id = 1:length(obj.Marginals)
                obj.Marginals{marg_id}.setSimplicialTestFuncs( ...
                    args_cell{marg_id}{:});
            end

            % after setting the simplicial functions, initialize the
            % quantities for the cutting-plane algorithm
            obj.initializeSimplicialTestFuncs();
        end
        
        function [atoms, testfunc_vals] = ...
                updateSimplicialTestFuncs(obj, args_cell)
            % Update the simplicial test functions after an execution of
            % the cutting-plane algorithm. Besides setting the new
            % simplicial test functions, a new set of couplings of 
            % discretized marginals are generated via reassembly of the 
            % dual solution from the cutting-plane algorithm with the new 
            % discretized marginals. These couplings can be used to 
            % generate initial constraints for the new LSIP problem with 
            % the updated test functions.
            % Input:
            %   args_cell: cell array where each cell is a cell array
            %   containing all inputs to the method setSimplicialTestFuncs
            %   of each marginal
            % Outputs:
            %   atoms: matrix containing the atoms in the reassembled
            %   coupling
            %   testfunc_vals: sparse matrix containing the values of the
            %   test functions evaluated at the computed atoms where each
            %   row corresponds to an atom and each column corresponds to a
            %   test function

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

            obj.setSimplicialTestFuncs(args_cell);

            new_marg_atoms_cell = cell(marg_num, 1);
            new_marg_probs_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                new_marg_atoms_cell{marg_id} = ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs.Knots;
                new_marg_probs_cell{marg_id} = ...
                    obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals;
            end

            [coup_indices, ~] = discrete_reassembly_1D( ...
                old_coup_atoms, old_coup_probs, ...
                new_marg_atoms_cell, new_marg_probs_cell);

            % add the offsets to the knot indices and retrieve the atoms
            offseted_indices = coup_indices' ...
                + obj.Storage.MargKnotNumOffsets;
            atoms = obj.Storage.MargKnotsConcat(offseted_indices)';
            atom_num = size(atoms, 1);
            testfunc_vals = ...
                sparse(repelem((1:atom_num)', marg_num, 1), ...
                offseted_indices(:), 1, atom_num, ...
                obj.Storage.TotalKnotNum);
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

            % prepare quantities for the global minimization oracle
            obj.Storage.GlobalMin = struct;

            % construct the gurobi MIP model for the global minimization
            % problem
            
            % variables representing the inputs on the supports of the
            % marginals
            x_indices = (1:marg_num)';
            var_counter = marg_num;
            x_lb = zeros(marg_num, 1);
            x_ub = zeros(marg_num, 1);
            x_vtype = repmat('C', marg_num, 1);

            for marg_id = 1:marg_num
                x_lb(marg_id) = obj.Marginals{marg_id}.Supp.LowerBound;
                x_ub(marg_id) = obj.Marginals{marg_id}.Supp.UpperBound;
            end

            % variables representing the values of the convex parts of the
            % cost function 
            lambda_indices = var_counter + (1:obj.CostFunc.NumOfConvex)';
            var_counter = var_counter + obj.CostFunc.NumOfConvex;
            lambda_lb = -inf(obj.CostFunc.NumOfConvex, 1);
            lambda_ub = inf(obj.CostFunc.NumOfConvex, 1);
            lambda_vtype = repmat('C', obj.CostFunc.NumOfConvex, 1);

            % variables representing the negative of the values of the
            % concave parts of the cost function
            xi_indices = var_counter + (1:obj.CostFunc.NumOfConcave)';
            var_counter = var_counter + obj.CostFunc.NumOfConcave;
            xi_lb = -inf(obj.CostFunc.NumOfConcave, 1);
            xi_ub = inf(obj.CostFunc.NumOfConcave, 1);
            xi_vtype = repmat('C', obj.CostFunc.NumOfConcave, 1);

            % variables representing the differences between the xi
            % variables (which represent maximums) and the individual
            % elements
            rho_indices = cell(obj.CostFunc.NumOfConcave, 1);
            ccv_piece_num_list = zeros(obj.CostFunc.NumOfConcave, 1);

            for ccv_id = 1:obj.CostFunc.NumOfConcave
                piece_num = size(obj.CostFunc.Concave{ccv_id, 1}, 1);
                ccv_piece_num_list(ccv_id) = piece_num;
                rho_indices{ccv_id} = var_counter + (1:piece_num)';
                var_counter = var_counter + piece_num;
            end

            ccv_total_piece_num = sum(ccv_piece_num_list);
            rho_lb = zeros(ccv_total_piece_num, 1);
            rho_ub = inf(ccv_total_piece_num, 1);
            rho_vtype = repmat('C', ccv_total_piece_num, 1);

            % binary variables for enforcing that one of the rho variables
            % must be equal to 0
            eta_indices = cell(obj.CostFunc.NumOfConcave, 1);

            for ccv_id = 1:obj.CostFunc.NumOfConcave
                piece_num = ccv_piece_num_list(ccv_id);
                eta_indices{ccv_id} = var_counter + (1:piece_num)';
                var_counter = var_counter + piece_num;
            end

            eta_lb = -inf(ccv_total_piece_num, 1);
            eta_ub = inf(ccv_total_piece_num, 1);
            eta_vtype = repmat('B', ccv_total_piece_num, 1);

            % objective vector for the cost function part
            costfunc_obj = zeros(var_counter, 1);

            % the convex parts have weights 1
            costfunc_obj(lambda_indices) = 1;

            % the concave parts have weights -1
            costfunc_obj(xi_indices) = -1;
            
            costfunc_lb = [x_lb; lambda_lb; xi_lb; rho_lb; eta_lb];
            costfunc_ub = [x_ub; lambda_ub; xi_ub; rho_ub; eta_ub];
            costfunc_vtype = [x_vtype; lambda_vtype; xi_vtype; ...
                rho_vtype; eta_vtype];

            % next, the test function part is formulated depending on the
            % specified global formulation
            if strcmp(obj.Options.global_formulation, 'INC')
                % the 1D test functions are modeled via the incremental
                % formulation
                
                % variables between 0 and 1 that are used to interpolate
                % the knots in the test functions
                zeta_indices = cell(marg_num, 1);
                zeta_lb_cell = cell(marg_num, 1);
                zeta_ub_cell = cell(marg_num, 1);

                for marg_id = 1:marg_num
                    knot_num = obj.Storage.MargKnotNumList(marg_id);
                    zeta_indices{marg_id} = var_counter ...
                        + (1:knot_num - 1)';
                    var_counter = var_counter + knot_num - 1;
                    zeta_lb_cell{marg_id} = zeros(knot_num - 1, 1);
                    zeta_ub_cell{marg_id} = [1; inf(knot_num - 2, 1)];
                end

                zeta_lb = vertcat(zeta_lb_cell{:});
                zeta_ub = vertcat(zeta_ub_cell{:});
                zeta_vtype = repmat('C', obj.Storage.TotalKnotNum ...
                    - marg_num, 1);

                % binary variables for indicating the intervals in the test
                % functions
                iota_indices = cell(marg_num, 1);

                for marg_id = 1:marg_num
                    knot_num = obj.Storage.MargKnotNumList(marg_id);
                    iota_indices{marg_id} = var_counter ...
                        + (1:knot_num - 2)';
                    var_counter = var_counter + knot_num - 2;
                end
                
                iota_var_length = obj.Storage.TotalKnotNum ...
                    - 2 * marg_num;
                iota_lb = -inf(iota_var_length, 1);
                iota_ub = inf(iota_var_length, 1);
                iota_vtype = repmat('B', iota_var_length, 1);

                testfunc_obj = zeros(var_counter ...
                    - length(costfunc_obj), 1);
                testfunc_lb = [zeta_lb; iota_lb];
                testfunc_ub = [zeta_ub; iota_ub];
                testfunc_vtype = [zeta_vtype; iota_vtype];
            elseif strcmp(obj.Options.global_formulation, 'LOG')
                % the 1D test functions are modeled via the logarithmic
                % convex combination formulation
                
                % variables between 0 and 1 that are used to interpolate
                % the knots in the test functions
                zeta_indices = cell(marg_num, 1);
                testfunc_bit_len = zeros(marg_num, 1);
                testfunc_bisect_cell = cell(marg_num, 1);

                for marg_id = 1:marg_num
                    knot_num = obj.Storage.MargKnotNumList(marg_id);

                    % compute the bisections in the knots of the test
                    % functions
                    testfunc_bit_len(marg_id) = ...
                        ceil(log2(knot_num - 1));
                    testfunc_bisect_cell{marg_id} = ...
                        obj.compute1DBisection(knot_num - 1);

                    zeta_indices{marg_id} = var_counter ...
                        + (1:knot_num)';
                    var_counter = var_counter + knot_num;
                end

                zeta_lb = zeros(obj.Storage.TotalKnotNum, 1);
                zeta_ub = inf(obj.Storage.TotalKnotNum, 1);
                zeta_vtype = repmat('C', obj.Storage.TotalKnotNum, 1);

                % binary variables for indicating the intervals in the test
                % functions
                iota_indices = cell(marg_num, 1);

                for marg_id = 1:marg_num
                    iota_indices{marg_id} = var_counter ...
                        + (1:testfunc_bit_len(marg_id))';
                    var_counter = var_counter + testfunc_bit_len(marg_id);
                end
                
                iota_var_length = sum(testfunc_bit_len);
                iota_lb = -inf(iota_var_length, 1);
                iota_ub = inf(iota_var_length, 1);
                iota_vtype = repmat('B', iota_var_length, 1);

                testfunc_obj = zeros(var_counter ...
                    - length(costfunc_obj), 1);
                testfunc_lb = [zeta_lb; iota_lb];
                testfunc_ub = [zeta_ub; iota_ub];
                testfunc_vtype = [zeta_vtype; iota_vtype];
            else
                error('unknown global formulation');
            end

            var_length = var_counter;

            gl_model = struct;
            gl_model.modelsense = 'min';
            gl_model.objcon = 0;
            gl_model.obj = [costfunc_obj; testfunc_obj];
            gl_model.lb = [costfunc_lb; testfunc_lb];
            gl_model.ub = [costfunc_ub; testfunc_ub];
            gl_model.vtype = [costfunc_vtype; testfunc_vtype];

            % assemble the constraints

            % constraints that link the lambda variables with the convex
            % parts of the cost function
            A_cvx_cell = cell(obj.CostFunc.NumOfConvex, 1);
            rhs_cvx_cell = cell(obj.CostFunc.NumOfConvex, 1);
            sense_cvx_cell = cell(obj.CostFunc.NumOfConvex, 1);

            for cvx_id = 1:obj.CostFunc.NumOfConvex
                weights = obj.CostFunc.Convex{cvx_id, 1};
                intercepts = obj.CostFunc.Convex{cvx_id, 2};
                row_num = length(intercepts);

                A_cvx_r = repmat((1:row_num)', marg_num + 1, 1);
                A_cvx_c = [repelem(x_indices, row_num, 1); ...
                    lambda_indices(cvx_id) * ones(row_num, 1)];
                A_cvx_v = [weights(:); -ones(row_num, 1)];

                A_cvx_cell{cvx_id} = sparse(A_cvx_r, A_cvx_c, A_cvx_v, ...
                    row_num, var_length);
                rhs_cvx_cell{cvx_id} = -intercepts;
                sense_cvx_cell{cvx_id} = repmat('<', row_num, 1);
            end

            A_cvx = vertcat(A_cvx_cell{:});
            rhs_cvx = vertcat(rhs_cvx_cell{:});
            sense_cvx = vertcat(sense_cvx_cell{:});

            % constraints that link the xi variables with the rho variables
            % in order to make the xi variables represent the concave parts
            % of the cost function
            A_ccv_link_cell = cell(obj.CostFunc.NumOfConcave, 1);
            rhs_ccv_link_cell = cell(obj.CostFunc.NumOfConcave, 1);
            sense_ccv_link_cell = cell(obj.CostFunc.NumOfConcave, 1);

            % constraints that link the rho variables with the eta
            % variables to force one of the rho variables to equal 0
            A_ccv_one0_cell = cell(obj.CostFunc.NumOfConcave, 1);
            rhs_ccv_one0_cell = cell(obj.CostFunc.NumOfConcave, 1);
            sense_ccv_one0_cell = cell(obj.CostFunc.NumOfConcave, 1);

            % constraints that require the eta variables to sum to 1
            A_ccv_sum1_cell = cell(obj.CostFunc.NumOfConcave, 1);
            rhs_ccv_sum1_cell = cell(obj.CostFunc.NumOfConcave, 1);
            sense_ccv_sum1_cell = cell(obj.CostFunc.NumOfConcave, 1);

            % iterate through all the concave parts to build the three
            % types of constraints
            for ccv_id = 1:obj.CostFunc.NumOfConcave
                weights = obj.CostFunc.Concave{ccv_id, 1};
                intercepts = obj.CostFunc.Concave{ccv_id, 2};
                row_num = length(intercepts);

                A_ccv_link_r = repmat((1:row_num)', marg_num + 2, 1);
                A_ccv_link_c = [repelem(x_indices, row_num, 1); ...
                    xi_indices(ccv_id) * ones(row_num, 1); ...
                    rho_indices{ccv_id}];
                A_ccv_link_v = [weights(:); -ones(row_num, 1); ...
                    ones(row_num, 1)];

                A_ccv_link_cell{ccv_id} = sparse(A_ccv_link_r, ...
                    A_ccv_link_c, A_ccv_link_v, row_num, var_length);
                rhs_ccv_link_cell{ccv_id} = -intercepts;
                sense_ccv_link_cell{ccv_id} = repmat('=', row_num, 1);

                bigM = (max(weights, [], 1) - min(weights, [], 1)) ...
                    * max(abs(x_ub), abs(x_lb)) + range(intercepts);
                A_ccv_one0_r = repmat((1:2)', row_num, 1);
                A_ccv_one0_c = [rho_indices{ccv_id}; ...
                    eta_indices{ccv_id}];
                A_ccv_one0_v = [ones(row_num, 1); bigM * ones(row_num, 1)];

                A_ccv_one0_cell{ccv_id} = sparse(A_ccv_one0_r, ...
                    A_ccv_one0_c, A_ccv_one0_v, row_num, var_length);
                rhs_ccv_one0_cell{ccv_id} = bigM * ones(row_num, 1);
                sense_ccv_one0_cell{ccv_id} = repmat('<', row_num, 1);

                A_ccv_sum1_r = ones(row_num, 1);
                A_ccv_sum1_c = eta_indices{ccv_id};
                A_ccv_sum1_v = ones(row_num, 1);

                A_ccv_sum1_cell{ccv_id} = sparse(A_ccv_sum1_r, ...
                    A_ccv_sum1_c, A_ccv_sum1_v, 1, var_length);
                rhs_ccv_sum1_cell{ccv_id} = 1;
                sense_ccv_sum1_cell{ccv_id} = '=';
            end

            A_ccv = [vertcat(A_ccv_link_cell{:}); ...
                vertcat(A_ccv_one0_cell{:}); ...
                vertcat(A_ccv_sum1_cell{:})];
            rhs_ccv = [vertcat(rhs_ccv_link_cell{:}); ...
                vertcat(rhs_ccv_one0_cell{:}); ...
                vertcat(rhs_ccv_sum1_cell{:})];
            sense_ccv = [vertcat(sense_ccv_link_cell{:}); ...
                vertcat(sense_ccv_one0_cell{:}); ...
                vertcat(sense_ccv_sum1_cell{:})];

            costfunc_A = [A_cvx; A_ccv];
            costfunc_rhs = [rhs_cvx; rhs_ccv];
            costfunc_sense = [sense_cvx; sense_ccv];

            % next, the test function part is formulated depending on the
            % specified global formulation
            if strcmp(obj.Options.global_formulation, 'INC')
                % the incremental formulation

                % constraints that order the zeta variables and the iota
                % variables
                A_tf_ord_cell = cell(marg_num, 1);
                rhs_tf_ord_cell = cell(marg_num, 1);
                sense_tf_ord_cell = cell(marg_num, 1);

                % constraints that link the x variables with the zeta
                % variables
                A_tf_link_cell = cell(marg_num, 1);
                rhs_tf_link_cell = cell(marg_num, 1);
                sense_tf_link_cell = cell(marg_num, 1);

                % indices in the objective vector to insert the quantities
                % related to the coefficients of the test functions
                tfcoefs_insert_indices = vertcat(zeta_indices{:});

                % matrices for inserting the quantities related to the
                % coefficients of the test functions into the objective
                % vector 
                tfcoefs_insert_mat_cell = cell(marg_num, 1);

                % matrices and intercept vectors for extracting the values
                % of the test functions evaluated at the given
                % approximately optimal solutions of the MIP problem
                tfvals_extract_mat_cell = cell(marg_num, 1);
                tfvals_extract_intercept_cell = cell(marg_num, 1);

                for marg_id = 1:marg_num
                    knots = obj.Marginals{marg_id} ...
                        .SimplicialTestFuncs.Knots;
                    knot_num = length(knots);

                    A_tf_ord_r = [repmat((1:knot_num - 2)', 2, 1); ...
                        (knot_num - 2) + repmat((1:knot_num - 2)', 2, 1)];
                    A_tf_ord_c = [zeta_indices{marg_id}(1:end - 1); ...
                        iota_indices{marg_id}; ...
                        zeta_indices{marg_id}(2:end); ...
                        iota_indices{marg_id}];
                    A_tf_ord_v = [ones(knot_num - 2, 1); ...
                        -ones(knot_num - 2, 1); ...
                        ones(knot_num - 2, 1); ...
                        -ones(knot_num - 2, 1)];

                    A_tf_ord_cell{marg_id} = sparse(A_tf_ord_r, ...
                        A_tf_ord_c, A_tf_ord_v, 2 * knot_num - 4, ...
                        var_length);
                    rhs_tf_ord_cell{marg_id} = zeros(2 * knot_num - 4, 1);
                    sense_tf_ord_cell{marg_id} = ...
                        [repmat('>', knot_num - 2, 1); ...
                        repmat('<', knot_num - 2, 1)];

                    A_tf_link_r = ones(knot_num, 1);
                    A_tf_link_c = [x_indices(marg_id); ...
                        zeta_indices{marg_id}];
                    A_tf_link_v = [1; -diff(knots)];

                    A_tf_link_cell{marg_id} = sparse(A_tf_link_r, ...
                        A_tf_link_c, A_tf_link_v, 1, var_length);
                    rhs_tf_link_cell{marg_id} = knots(1);
                    sense_tf_link_cell{marg_id} = '=';

                    tfcoefs_insert_mat_cell{marg_id} = -sparse( ...
                        repmat((1:knot_num - 1)', 2, 1), ...
                        [(1:knot_num - 1)'; (2:knot_num)'], ...
                        [-ones(knot_num - 1, 1); ...
                        ones(knot_num - 1, 1)], ...
                        knot_num - 1, knot_num);

                    % the 1st test function evaluates to 1-zeta(1)
                    % the 2nd test function evaluates to zeta(1)-zeta(2)
                    % ...
                    % the nth test function evaluates to zeta(n-1)-zeta(n)
                    % ...
                    % the mth test function evaluates to zeta(m-1) where m
                    % is the number of knots
                    tfvals_extract_mat_cell{marg_id} = sparse( ...
                        [(1:knot_num - 1)'; (2:knot_num)'], ...
                        repmat(zeta_indices{marg_id}, 2, 1), ...
                        [-ones(knot_num - 1, 1); ...
                        ones(knot_num - 1, 1)], ...
                        knot_num, var_length);
                    tfvals_extract_intercept_cell{marg_id} = ...
                        [1; zeros(knot_num - 1, 1)];
                end

                testfunc_A = [vertcat(A_tf_ord_cell{:}); ...
                    vertcat(A_tf_link_cell{:})];
                testfunc_rhs = [vertcat(rhs_tf_ord_cell{:}); ...
                    vertcat(rhs_tf_link_cell{:})];
                testfunc_sense = [vertcat(sense_tf_ord_cell{:}); ...
                    vertcat(sense_tf_link_cell{:})];

                tfcoefs_insert_mat = blkdiag(tfcoefs_insert_mat_cell{:});

                tfvals_extract_mat = vertcat(tfvals_extract_mat_cell{:});
                tfvals_extract_intercept = ...
                    vertcat(tfvals_extract_intercept_cell{:});
            else
                % the logarithmic convex combination formulation

                % constraints that require the zeta variables to sum to 1
                A_tf_sum1_cell = cell(marg_num, 1);
                rhs_tf_sum1_cell = cell(marg_num, 1);
                sense_tf_sum1_cell = cell(marg_num, 1);

                % constraints that identify the positive zeta variables
                % with the iota variables
                A_tf_iden_cell = cell(marg_num, 1);
                rhs_tf_iden_cell = cell(marg_num, 1);
                sense_tf_iden_cell = cell(marg_num, 1);

                % constraints that link the zeta variables with the x
                % variables
                A_tf_link_cell = cell(marg_num, 1);
                rhs_tf_link_cell = cell(marg_num, 1);
                sense_tf_link_cell = cell(marg_num, 1);

                % indices in the objective vector to insert the quantities
                % related to the coefficients of the test functions
                tfcoefs_insert_indices = vertcat(zeta_indices{:});

                % matrices for inserting the quantities related to the
                % coefficients of the test functions into the objective
                % vector 
                tfcoefs_insert_mat_cell = cell(marg_num, 1);

                % matrices and intercept vectors for extracting the values
                % of the test functions evaluated at the given
                % approximately optimal solutions of the MIP problem
                tfvals_extract_mat_cell = cell(marg_num, 1);
                tfvals_extract_intercept_cell = cell(marg_num, 1);

                for marg_id = 1:marg_num
                    knots = obj.Marginals{marg_id} ...
                        .SimplicialTestFuncs.Knots;
                    knot_num = length(knots);
                    
                    A_tf_sum1_r = ones(knot_num, 1);
                    A_tf_sum1_c = zeta_indices{marg_id};
                    A_tf_sum1_v = ones(knot_num, 1);

                    A_tf_sum1_cell{marg_id} = sparse(A_tf_sum1_r, ...
                        A_tf_sum1_c, A_tf_sum1_v, 1, var_length);
                    rhs_tf_sum1_cell{marg_id} = 1;
                    sense_tf_sum1_cell{marg_id} = '=';

                    A_tf_iden_r_cell = cell(testfunc_bit_len(marg_id), 1);
                    A_tf_iden_c_cell = cell(testfunc_bit_len(marg_id), 1);
                    A_tf_iden_v_cell = cell(testfunc_bit_len(marg_id), 1);

                    for bit_id = 1:testfunc_bit_len(marg_id)
                        tf_bit0_list = ...
                            testfunc_bisect_cell{marg_id}{bit_id, 1};
                        tf_bit0_num = length(tf_bit0_list);
                        tf_bit1_list = ...
                            testfunc_bisect_cell{marg_id}{bit_id, 2};
                        tf_bit1_num = length(tf_bit1_list);

                        A_tf_iden_r_cell{bit_id} = (bit_id - 1) * 2 ...
                            + [ones(tf_bit0_num + 1, 1); ...
                            2 * ones(tf_bit1_num + 1, 1)];
                        A_tf_iden_c_cell{bit_id} = ...
                            [zeta_indices{marg_id}(tf_bit0_list); ...
                            iota_indices{marg_id}(bit_id); ...
                            zeta_indices{marg_id}(tf_bit1_list); ...
                            iota_indices{marg_id}(bit_id)];
                        A_tf_iden_v_cell{bit_id} = ...
                            [ones(tf_bit0_num, 1); -1; ...
                            ones(tf_bit1_num, 1); 1];
                    end

                    A_tf_iden_cell{marg_id} = sparse( ...
                        vertcat(A_tf_iden_r_cell{:}), ...
                        vertcat(A_tf_iden_c_cell{:}), ...
                        vertcat(A_tf_iden_v_cell{:}), ...
                        2 * testfunc_bit_len(marg_id), var_length);
                    rhs_tf_iden_cell{marg_id} = repmat([0; 1], ...
                        testfunc_bit_len(marg_id), 1);
                    sense_tf_iden_cell{marg_id} = repmat('<', ...
                        2 * testfunc_bit_len(marg_id), 1);

                    A_tf_link_r = ones(knot_num + 1, 1);
                    A_tf_link_c = [zeta_indices{marg_id}; ...
                        x_indices(marg_id)];
                    A_tf_link_v = [knots; -1];

                    A_tf_link_cell{marg_id} = sparse(A_tf_link_r, ...
                        A_tf_link_c, A_tf_link_v, 1, var_length);
                    rhs_tf_link_cell{marg_id} = 0;
                    sense_tf_link_cell{marg_id} = '=';

                    tfcoefs_insert_mat_cell{marg_id} = -speye(knot_num);

                    % the 1st test function evaluates to 1-zeta(1)
                    % the 2nd test function evaluates to zeta(1)-zeta(2)
                    % ...
                    % the nth test function evaluates to zeta(n-1)-zeta(n)
                    % ...
                    % the mth test function evaluates to zeta(m-1) where m
                    % is the number of knots
                    tfvals_extract_mat_cell{marg_id} = sparse( ...
                        (1:knot_num)', zeta_indices{marg_id}, ...
                        ones(knot_num, 1), knot_num, var_length);
                    tfvals_extract_intercept_cell{marg_id} = ...
                        zeros(knot_num, 1);
                end

                testfunc_A = [vertcat(A_tf_sum1_cell{:}); ...
                    vertcat(A_tf_iden_cell{:}); ...
                    vertcat(A_tf_link_cell{:})];
                testfunc_rhs = [vertcat(rhs_tf_sum1_cell{:}); ...
                    vertcat(rhs_tf_iden_cell{:}); ...
                    vertcat(rhs_tf_link_cell{:})];
                testfunc_sense = [vertcat(sense_tf_sum1_cell{:}); ...
                    vertcat(sense_tf_iden_cell{:}); ...
                    vertcat(sense_tf_link_cell{:})];

                tfcoefs_insert_mat = blkdiag(tfcoefs_insert_mat_cell{:});

                tfvals_extract_mat = vertcat(tfvals_extract_mat_cell{:});
                tfvals_extract_intercept = ...
                    vertcat(tfvals_extract_intercept_cell{:});
            end

            gl_model.A = [costfunc_A; testfunc_A];
            gl_model.rhs = [costfunc_rhs; testfunc_rhs];
            gl_model.sense = [costfunc_sense; testfunc_sense];

            obj.Storage.GlobalMin.GurobiModel = gl_model;
            obj.Storage.GlobalMin.MargPointIndices = x_indices;
            obj.Storage.GlobalMin.MargLowerBounds = x_lb;
            obj.Storage.GlobalMin.MargUpperBounds = x_ub;
            obj.Storage.GlobalMin.TestFuncCoefsInsertIndices = ...
                tfcoefs_insert_indices;
            obj.Storage.GlobalMin.TestFuncCoefsInsertMat = ...
                tfcoefs_insert_mat;
            obj.Storage.GlobalMin.TestFuncValsExtractMat = ...
                tfvals_extract_mat;
            obj.Storage.GlobalMin.TestFuncValsExtractIntercept = ...
                tfvals_extract_intercept;

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

        function [UB, samps] = getMMOTUpperBound(obj, ...
                samp_num, rand_stream, batch_size)
            % Compute an upper bound for the MMOT problem. The bound is
            % approximated by Monte Carlo integration. Computation is done
            % in batches if necessary.
            % Inputs: 
            %   samp_num: number of samples for Monte Carlo integration
            %   rand_stream: RandStream object (default is
            %   RandStream.getGlobalStream)
            %   batch_size: batch size when evaluating the cost function
            %   (default is 1e4)
            % Outputs:
            %   UB: upper bound for the MMOT problem
            %   samps: the Monte Carlo samples used in the approximation of
            %   the bounds

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            if ~exist('batch_size', 'var') || isempty(batch_size)
                batch_size = 1e4;
            end

            if ~exist('rand_stream', 'var') ...
                    || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            samps = obj.randSampleFromOptCoupling(samp_num, rand_stream);

            UB_list = obj.evaluateCostFunction(samps, batch_size);

            UB = mean(UB_list);
        end

        function [UB_list, samps] ...
                = getMMOTUpperBoundWRepetition(obj, samp_num, rep_num, ...
                rand_stream)
            % Compute an upper bound for the MMOT problem with repetition
            % of Monte Carlo integration. 
            % Inputs:
            %   samp_num: number of samples for Monte Carlo integration
            %   rep_num: number of repetitions
            %   rand_stream: RandStream object (default is
            %   RandStream.getGlobalStream)
            % Output:
            %   UB_list: vector containing the computed upper bounds for
            %   the MMOT problem
            %   samps: one particular set of Monte Carlo samples used in 
            %   the approximation of the bound

            if ~exist('rand_stream', 'var') ...
                    || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            UB_list = zeros(rep_num, 1);

            % open the log file
            if ~isempty(obj.Options.log_file)
                log_file = fopen(obj.Options.log_file, 'a');

                if log_file < 0
                    error('cannot open log file');
                end

                fprintf(log_file, ...
                    '--- Monte Carlo sampling starts ---\n');
            end

            for rep_id = 1:rep_num
                [UB_list(rep_id), samps] ...
                    = obj.getMMOTUpperBound(samp_num, rand_stream);

                % display output
                if obj.Options.display
                    fprintf(['%s: ' ...
                        'Monte Carlo sampling repetition %3d done\n'], ...
                        class(obj), rep_id);
                end

                % write log
                if ~isempty(obj.Options.log_file)
                    fprintf(log_file, ['%s: ' ...
                        'Monte Carlo sampling repetition %3d done\n'], ...
                        class(obj), rep_id);
                end
            end

            % close the log file
            if ~isempty(obj.Options.log_file)
                fprintf(log_file, ...
                    '--- Monte Carlo sampling ends ---\n\n');
                fclose(log_file);
            end
        end

        function EB = getMTErrorBoundBasedOnOT(obj, LSIP_tolerance)
            % Compute the error bound for the objective value of the 
            % MMOT problem based on the Lipschitz constant of the cost 
            % functions and the optimal transport distances between the 
            % marginals and their discretizations
            % Input:
            %   LSIP_tolerance: tolerance value used in the computation of
            %   the LSIP problem
            % Output:
            %   EB: the computed error bound

            if ~isfield(obj.Runtime, 'DualSolution') ...
                    || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            marg_num = length(obj.Marginals);
            EB = LSIP_tolerance;
            dual_sol = obj.Runtime.DualSolution;

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};

                marg_atoms = dual_sol.Atoms(:, marg_id);
                marg_probs = dual_sol.Probabilities;

                marg.setCoupledDiscreteMeasure(marg_atoms, marg_probs);
                EB = EB ...
                    + obj.CostFunc.LipschitzConstants(marg_id) ...
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
                    obj.CostFunc.LipschitzConstants(marg_id) ...
                    * 2 * marg.SimplicialTestFuncs.MeshSize;
            end
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
    end

    methods(Access = protected)

        function prepareRuntime(obj)
            % Prepare the runtime environment by initializing some
            % variables
            obj.Runtime = struct;

            prepareRuntime@LSIPMinCuttingPlaneAlgo(obj);

            % initialize the cuts to be empty
            obj.Runtime.CutPoints = zeros(0, length(obj.Marginals));
        end

        function initializeBeforeRun(obj)
            % Initialize the algorithm by computing some static quantities

            if ~obj.Storage.SimplicialTestFuncsInitialized
                obj.initializeSimplicialTestFuncs();
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
            model.objcon = 0;

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
            
            % constant value in the decision variable
            obj_const = vec(1);
            
            % convert the locations of the knots and the coefficients in 
            % the decision variable into a cell array
            tf_vec = zeros(obj.Storage.TotalKnotNum, 1);
            tf_vec(obj.Storage.DeciVarIndicesInTestFuncs) = vec(2:end);
            
            % retrieve the gurobi model and fill up the coefficients of the
            % test functions
            GM = obj.Storage.GlobalMin;

            gl_model = GM.GurobiModel;

            % note that since the first test function for each marginal has
            % been removed, there will not be any additional contributions
            % to the constant term of the MIP model from the coefficients
            % of the test functions
            gl_model.objcon = -obj_const;

            gl_model.obj(GM.TestFuncCoefsInsertIndices) = ...
                GM.TestFuncCoefsInsertMat * tf_vec;

            % options of the mixed-integer solver
            gl_options = struct;
            gl_options.OutputFlag = 0;
            gl_options.IntFeasTol = 1e-6;
            gl_options.FeasibilityTol = 1e-8;
            gl_options.OptimalityTol = 1e-8;
            gl_options.PoolSolutions = 100;
            gl_options.PoolGap = 0.8;
            gl_options.NodefileStart = 2;
            gl_options.BestBdStop = 1e-6;
            gl_options.BestObjStop = -inf;
            gl_options.MIPGap = 1e-4;
            gl_options.MIPGapAbs = 1e-10;

            % set the additional options for the mixed-integer solver
            gl_options_fields = fieldnames(obj.GlobalOptions);
            gl_options_values = struct2cell(obj.GlobalOptions);

            for fid = 1:length(gl_options_fields)
                gl_options.(gl_options_fields{fid}) ...
                    = gl_options_values{fid};
            end

            result = gurobi(gl_model, gl_options);

            if ~strcmp(result.status, 'OPTIMAL') ...
                    && ~strcmp(result.status, 'USER_OBJ_LIMIT')
                error('error in the mixed-integer solver');
            end

            min_lb = result.objbound;

            % get a set of approximate optimizers
            if isfield(result, 'pool')
                pool_cuts = horzcat(result.pool.xn)';
            else
                pool_cuts = result.x';
            end

            pool_points = pool_cuts(:, GM.MargPointIndices);


            pool_marg_testfuncs = pool_cuts ...
                * GM.TestFuncValsExtractMat' ...
                + GM.TestFuncValsExtractIntercept';

            % make sure that the inputs are within their respective bounds
            % (they may be slightly out of bound due to small numerical
            % inaccuracies)
            pool_points = min(max(pool_points, ...
                GM.MargLowerBounds'), GM.MargUpperBounds');

            optimizers = struct('points', pool_points, ...
                'testfuncs', pool_marg_testfuncs);
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

            constr_num = size(optimizers.points, 1);
            marg_num = length(obj.Marginals);

            % first generate a matrix containing all test functions (each
            % row corresponds to an approximate optimizer, each column
            % corresponds to a test function)
            if isfield(optimizers, 'testfunc_vals') ...
                    && ~isempty(optimizers.testfunc_vals)
                A_full = optimizers.testfunc_vals;
            else
                A_full_cell = cell(marg_num, 1);

                for marg_id = 1:marg_num
                    A_full_cell{marg_id} = obj.Marginals{marg_id} ...
                        .evaluateSimplicialTestFuncs( ...
                        optimizers.points(:, marg_id));
                end

                A_full = horzcat(A_full_cell{:});
            end
            
            % filter out those test functions whose coefficients are not
            % included in the decision variable, then prepend a column of
            % 1
            A_new = [sparse(ones(constr_num, 1)), ...
                A_full(:, obj.Storage.DeciVarIndicesInTestFuncs)];

            % retrieve the coordinates of the optimal points
            rhs_new = obj.evaluateCostFunction(optimizers.points);
            
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
            obj.Runtime.CutPoints = [obj.Runtime.CutPoints; ...
                optimizers.points];

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
                obj.Runtime.CutPoints ...
                    = obj.Runtime.CutPoints(keep_list, :);
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

            % since violation is non-positive, we shift the first function downwards
            primal_sol.Constant = vec(1) + violation;

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
            dual_sol.Atoms = obj.Runtime.CutPoints(pos_list, :);

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

            dual_sol = obj.Runtime.DualSolution;
            marg_num = length(obj.Marginals);

            % set the discrete measures that the marginals are coupled with
            for marg_id = 1:marg_num
                obj.Marginals{marg_id}.setCoupledDiscreteMeasure( ...
                    dual_sol.Atoms(:, marg_id), ...
                    dual_sol.Probabilities);
            end

            n = length(obj.Runtime.DualSolution.Probabilities);
            
            % generate random indices of the atoms according to the
            % probabilities
            disc_atom_index_samps = randsample(rand_stream, n, ...
                samp_num, true, obj.Runtime.DualSolution.Probabilities);

            samps = zeros(samp_num, marg_num);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};

                % count the number of samples coupled with each of the
                % atoms in the discretized marginal
                atom_num_list = accumarray(disc_atom_index_samps, 1, ...
                    [n, 1]);

                % generate from the conditional distributions
                samp_cell = marg.conditionalRandSample((1:n)', ...
                    atom_num_list, rand_stream);

                % fill in the coupled samples from the continuous marginals
                for atom_id = 1:length(atom_num_list)
                    samps(disc_atom_index_samps == atom_id, marg_id) = ...
                        samp_cell{atom_id};
                end
            end
        end

        function bisect_cell = compute1DBisection(obj, intv_num)
            % Compute a bisection of knots in a one-dimensional continuous
            % piece-wise affine (CPWA) function used for formulating it
            % into a mixed-integer programming problem
            % Input:
            %   intv_num: number of intervals in the CPWA function
            % Output:
            %   bisect_cell: cell array containing the bisection, the
            %   number of rows is equal to ceil(log2(intv_num)) and the
            %   number of columns is 2

            bit_len = ceil(log2(intv_num));

            if ~isfield(obj.Storage, 'ReflexiveBinarySequences') ...
                || isempty(obj.Storage.ReflexiveBinarySequences)
                obj.Storage.ReflexiveBinarySequences = cell(20, 1);
            end

            if bit_len == 0
                bisect_cell = cell(0, 2);
                return;
            end

            if bit_len > 20
                error('the number of intervals is too large');
            end

            if isempty(obj.Storage.ReflexiveBinarySequences{bit_len})
                seq_mat = zeros(2^bit_len, bit_len);
                
                % fill the first two rows
                seq_mat(1:2, end) = [0; 1];

                for bit_id = 2:bit_len
                    % flip the previous matrix upside down and append to
                    % the end while adding a column of 1's to the left
                    prev_row_num = 2 ^ (bit_id - 1);
                    seq_mat(prev_row_num + (1:prev_row_num), ...
                        end - bit_id + 1:end) ...
                        = [ones(prev_row_num, 1), flipud( ...
                        seq_mat(1:prev_row_num, end - bit_id + 2:end))];
                end

                obj.Storage.ReflexiveBinarySequences{bit_len} = seq_mat;
            end

            seq_mat = obj.Storage.ReflexiveBinarySequences{bit_len}( ...
                1:intv_num, :);

            bisect_cell = cell(bit_len, 2);

            for bit_id = 1:bit_len
                rflx_col = seq_mat(:, bit_id);
                bisect_cell{bit_id, 1} = ...
                    find([rflx_col(1) == 0; ...
                    rflx_col(1:end - 1) == 0 ...
                    & rflx_col(2:end) == 0; ...
                    rflx_col(end) == 0]);
                bisect_cell{bit_id, 2} = ...
                    find([rflx_col(1) == 1; ...
                    rflx_col(1:end - 1) == 1 ...
                    & rflx_col(2:end) == 1; ...
                    rflx_col(end) == 1]);
            end
        end
    end
end


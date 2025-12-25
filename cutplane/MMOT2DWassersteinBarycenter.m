classdef MMOT2DWassersteinBarycenter < LSIPMinCuttingPlaneAlgo
    % Class for the multi-marginal optimal transport (MMOT) formulation of two-dimensional Wasserstein barycenter problem

    properties(GetAccess = public, SetAccess = protected)
        % cell array containing the marginals
        Marginals;

        % vector containing the weights
        MarginalWeights;

        % constant part of the quadratic cost functions that does not affect the coupling
        QuadraticConstant = 0;
    end

    methods(Access = public)
        function obj = MMOT2DWassersteinBarycenter(marginals, weights, varargin)
            % Constructor method
            % Inputs:
            %   marginals: cell array containing objects of ProbMeas2DConvexPolytope
            %   weights: vector containing the weights corresponding to the marginals, the weights will sum up to 1

            obj@LSIPMinCuttingPlaneAlgo(varargin{:});

            % set the default options for semi-discrete optimal transport
            if ~isfield(obj.Options, 'OT') || isempty(obj.Options.OT)
                obj.Options.OT = struct;
            end

            if ~isfield(obj.Options.OT, 'angle_num') || isempty(obj.Options.OT.angle_num)
                obj.Options.OT.angle_num = repmat({[]}, length(weights), 1);
            elseif ~iscell(obj.Options.OT.angle_num)
                obj.Options.OT.angle_num = repmat({obj.Options.OT.angle_num}, length(weights), 1);
            end

            if ~isfield(obj.Options.OT, 'pp_angle_num') || isempty(obj.Options.OT.pp_angle_num)
                obj.Options.OT.pp_angle_num = repmat({[]}, length(weights), 1);
            elseif ~iscell(obj.Options.OT.pp_angle_num)
                obj.Options.OT.pp_angle_num = repmat({obj.Options.OT.pp_angle_num}, length(weights), 1);
            end

            if ~isfield(obj.Options.OT, 'normalize_gradient') || isempty(obj.Options.OT.normalize_gradient)
                obj.Options.OT.normalize_gradient = repmat({[]}, length(weights), 1);
            elseif ~iscell(obj.Options.OT.normalize_gradient)
                obj.Options.OT.normalize_gradient = repmat({obj.Options.OT.normalize_gradient}, length(weights), 1);
            end

            if ~isfield(obj.Options.OT, 'optimization_options')
                obj.Options.OT.optimization_options = cell(length(weights), 1);
            elseif ~iscell(obj.Options.OT.optimization_options)
                optim_options = obj.Options.OT.optimization_options;
                obj.Options.OT.optimization_options = cell(length(weights), 1);
                obj.Options.OT.optimization_options(:, :) = {optim_options};
            end

            % set the default options for semi-discrete Wasserstein-2 optimal transport
            if ~isfield(obj.Options, 'W2OT') || isempty(obj.Options.W2OT)
                obj.Options.W2OT = struct;
            end

            if ~isfield(obj.Options.W2OT, 'optimization_options')
                obj.Options.W2OT.optimization_options = cell(length(weights), 1);
            elseif ~iscell(obj.Options.W2OT.optimization_options)
                optim_options = obj.Options.W2OT.optimization_options;
                obj.Options.W2OT.optimization_options = cell(length(weights), 1);
                obj.Options.W2OT.optimization_options(:, :) = {optim_options};
            end

            % set the default options for reducing constraints
            if ~isfield(obj.Options, 'reduce') || isempty(obj.Options.reduce)
                obj.Options.reduce = struct;
            end

            if ~isfield(obj.Options.reduce, 'thres') || isempty(obj.Options.reduce.thres)
                obj.Options.reduce.thres = inf;
            end

            if ~isfield(obj.Options.reduce, 'freq') || isempty(obj.Options.reduce.freq)
                obj.Options.reduce.freq = 20;
            end

            if ~isfield(obj.Options.reduce, 'max_iter') || isempty(obj.Options.reduce.max_iter)
                obj.Options.reduce.max_iter = inf;
            end

            % set the default options for reducing constraints; two thresholds are used for determining which constraints to remove 
            % based on their slackness values: whenever the (negative) slackness is above obj.Options.reduce.thres or its quantile
            % among all the non-tight constraints is below obj.Options.reduce.thres_quantile, the constraint is removed
            if ~isfield(obj.Options.reduce, 'thres_quantile') || isempty(obj.Options.reduce.thres_quantile)
                obj.Options.reduce.thres_quantile = 1;
            end

            % minimum value of slackness (in absolute value) for a constraint to be considered not tight where a constraint is only
            % flagged as removable only if |slack| > obj.Options.reduce.min_slack
            if ~isfield(obj.Options.reduce, 'min_slack') || isempty(obj.Options.reduce.min_slack)
                obj.Options.reduce.min_slack = 0;
            end

            % boolean indicating whether the initial constraints should be preserved throughout the cutting-plane algorithm
            if ~isfield(obj.Options.reduce, 'preserve_init_constr') || isempty(obj.Options.reduce.preserve_init_constr)
                obj.Options.reduce.preserve_init_constr = true;
            end


            % set the default options for the minimum probability value for an atom in the dual LSIP solution to be kept in the sense
            % that an atom is kept in the dual LSIP solution only if probability > obj.Options.dual_prob_thres
            if ~isfield(obj.Options, 'dual_prob_thres') || isempty(obj.Options.dual_prob_thres)
                obj.Options.dual_prob_thres = 0;
            end


            % set the default options for the global minimization oracle
            if ~isfield(obj.GlobalOptions, 'pool_size') || isempty(obj.GlobalOptions.pool_size)
                obj.GlobalOptions.pool_size = 100;
            end

            if ~isfield(obj.GlobalOptions, 'display') || isempty(obj.GlobalOptions.display)
                obj.GlobalOptions.display = true;
            end

            if ~isfield(obj.GlobalOptions, 'log_file') || isempty(obj.GlobalOptions.log_file)
                obj.GlobalOptions.log_file = '';
            end

            marg_num = length(weights);
            assert(length(marginals) == marg_num, 'input mis-specified');
            assert(abs(sum(weights) - 1) < 1e-12, 'weights do not sum up to 1');
            
            % normalize the weights to remove numerical inaccuracies
            weights = weights / sum(weights);

            obj.Marginals = marginals;
            obj.MarginalWeights = weights;

            % compute the constant terms in the cost functions that are related to the quadratic expectation with respect to the
            % marginals
            quad_consts = zeros(marg_num, 1);

            for marg_id = 1:marg_num
                quad_consts(marg_id) = sum(diag(obj.Marginals{marg_id}.SecondMomentMat));
            end

            obj.QuadraticConstant = quad_consts' * obj.MarginalWeights;

            % compute the bounding box of the convex hull of the supports of the input measures
            obj.Storage.BoundingBox = struct;
            obj.Storage.BoundingBox.BottomLeft = [inf; inf];
            obj.Storage.BoundingBox.TopRight = [-inf; -inf];

            weighted_maxnorm = 0;

            for marg_id = 1:marg_num
                obj.Storage.BoundingBox.BottomLeft = min(obj.Storage.BoundingBox.BottomLeft, ...
                    min(obj.Marginals{marg_id}.Supp.Vertices, [], 1)');
                obj.Storage.BoundingBox.TopRight = max(obj.Storage.BoundingBox.TopRight, ...
                    max(obj.Marginals{marg_id}.Supp.Vertices, [], 1)');

                weighted_maxnorm = weighted_maxnorm + obj.MarginalWeights(marg_id) * obj.Marginals{marg_id}.Supp.MaxNorm;
            end

            % compute the Lipschitz constaints with respect to each component in the product space
            obj.Storage.LipschitzConstants = 2 * obj.MarginalWeights * weighted_maxnorm;

            % this flag is used to track if the function obj.initializeSimplicialTestFuncs has been called
            obj.Storage.SimplicialTestFuncsInitialized = false;

            marg_num = length(obj.MarginalWeights);
            marg_testfunc_set = false(marg_num, 1);

            for marg_id = 1:marg_num
                marg_testfunc_set(marg_id) = ~isempty(obj.Marginals{marg_id}.SimplicialTestFuncs);
            end

            if all(marg_testfunc_set)
                % initialize the simplicial test functions at the end of the constructor if they have already been set
                obj.initializeSimplicialTestFuncs();
            end
        end
        
        function setSimplicialTestFuncs(obj, args_cell)
            % Set the simplicial test functions for all marginals at the same time
            % Input:
            %   args_cell: cell array where each cell is a cell array containing all inputs to the method setSimplicialTestFuncs of 
            % each marginal

            for marg_id = 1:length(obj.MarginalWeights)
                obj.Marginals{marg_id}.setSimplicialTestFuncs(args_cell{marg_id}{:});
            end

            % after setting the simplicial functions, initialize the quantities for the cutting-plane algorithm
            obj.initializeSimplicialTestFuncs();
        end

        function [coup_indices, coup_probs] = updateSimplicialTestFuncs(obj, args_cell)
            % Update the simplicial test functions after an execution of the cutting-plane algorithm. Besides setting the new 
            % simplicial test functions, a new coupling of discretized marginals is generated via reassembly of the dual solution from 
            % the cutting-plane algorithm with the new discretized marginals. This coupling can be used to generate initial constraints
            % for the new LSIP problem with the updated test functions.
            % Input: 
            %   args_cell: cell array where each cell is a cell array containing all inputs to the methods setSimplicialTestFuncs of 
            % each marginal
            % Outputs:
            %   coup_indices: the indices of atoms in the coupled discrete measure
            %   coup_probs: the probabilities of atoms in the coupled discrete measure

            if ~isfield(obj.Runtime, 'DualSolution') || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            marg_num = length(obj.MarginalWeights);

            % retrieve the dual solution resulted from the cutting-plane algorithm
            old_coup_indices = obj.Runtime.DualSolution.VertexIndices;
            old_coup_probs = obj.Runtime.DualSolution.Probabilities;

            % retrieve the old discretized marginals
            old_atoms_cell = cell(marg_num, 1);
            old_probs_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                old_atoms_cell{marg_id} = obj.Marginals{marg_id}.SimplicialTestFuncs.Vertices;
                
                % we do not take the probabilities from obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals since the numerical errors 
                % from the LSIP might be significant enough to cause issues
                old_probs_cell{marg_id} = accumarray(old_coup_indices(:, marg_id), ...
                    old_coup_probs, [size(old_atoms_cell{marg_id}, 1), 1]);
            end

            % set the new simplicial test functions
            obj.setSimplicialTestFuncs(args_cell);

            % store the new discrete measures
            marg_atoms_cell = cell(marg_num, 1);
            marg_probs_cell = cell(marg_num, 1);

            % the optimal couplings between the original discrete marginals and the new discrete marginals where the cost function is
            % given by the Euclidean distance
            marg_coup_atoms_cell = cell(marg_num, 1);
            marg_coup_probs_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                new_atoms = obj.Marginals{marg_id}.SimplicialTestFuncs.Vertices;
                new_probs = obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals;
                new_probs = new_probs / sum(new_probs);

                marg_atoms_cell{marg_id} = new_atoms;
                marg_probs_cell{marg_id} = new_probs;

                % the cost function is the Euclidean distance
                dist_mat = pdist2(old_atoms_cell{marg_id}, new_atoms, 'squaredeuclidean');

                if size(dist_mat, 1) * size(dist_mat, 2) > 1e6
                    % if there are too many atoms in the discrete measures, a direct computation of discrete OT may cause memory
                    % throttling; thus, we resort to a constraint generation scheme
                    cp_options = struct('display', false);
                    OT = OTDiscrete(old_probs_cell{marg_id}, new_probs, dist_mat, cp_options);
                    [hcoup_indices, ~] = OT.generateHeuristicCoupling();
                    OT.run(hcoup_indices, 1e-6);
                    coup = OT.Runtime.DualSolution;
                    marg_coup_atoms_cell{marg_id} = coup.CoupIndices;
                    marg_coup_probs_cell{marg_id} = coup.Probabilities;
                else
                    [marg_coup_atoms_cell{marg_id}, marg_coup_probs_cell{marg_id}] = discrete_OT(old_probs_cell{marg_id}, ...
                        new_probs, dist_mat);
                end
            end

            % perform discrete reassembly to get the new coupling
            [coup_indices, coup_probs] = discrete_reassembly(old_coup_indices, old_coup_probs, ...
                marg_coup_atoms_cell, marg_coup_probs_cell);
        end

        function initializeSimplicialTestFuncs(obj)
            % Initialize some quantities related to the simplicial test functions of the marginals

            marg_num = length(obj.MarginalWeights);

            obj.Storage.MargVertNumList = zeros(marg_num, 1);

            weighted_vertices_cell = cell(marg_num, 1);
            deci_logical_cell = cell(marg_num, 1);
            
            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                obj.Storage.MargVertNumList(marg_id) = size(marg.SimplicialTestFuncs.Vertices, 1);

                weighted_vertices_cell{marg_id} = obj.MarginalWeights(marg_id) * marg.SimplicialTestFuncs.Vertices;

                % the coefficient corresponding to the first test function will not be included in the decision variable for 
                % identification purposes
                deci_logical_cell{marg_id} = [0; ones(obj.Storage.MargVertNumList(marg_id) - 1, 1)];
            end

            obj.Storage.TotalVertNum = sum(obj.Storage.MargVertNumList);

            % store all the vertices in the triangulatios of the marginals into a single two-column matrix where the vertices are
            % weighted by the given weights in the barycenter
            obj.Storage.WeightedMargVertices = vertcat(weighted_vertices_cell{:});

            % compute the offset of the vertices in the marginals in the matrix containing all vertices
            vert_num_cumsum = cumsum(obj.Storage.MargVertNumList);
            obj.Storage.MargVertNumOffsets = [0; vert_num_cumsum(1:end - 1)];

            % store the indices to place the decision variables in a vector containing the coefficients of the test functions
            obj.Storage.DeciVarIndicesInTestFuncs = find(vertcat(deci_logical_cell{:}));

            % initialize data structures for the global minimization oracle
            obj.Storage.GlobalMin = struct;

            obj.Storage.GlobalMin.Circles = cell(marg_num, 1);

            for marg_id = 1:marg_num
                obj.Storage.GlobalMin.Circles{marg_id, 1} = obj.Marginals{marg_id}.SimplicialTestFuncs.Vertices;
            end

            obj.Storage.SimplicialTestFuncsInitialized = true;

            % updating the simplicial test functions will invalidate all quantities in the runtime environment, thus all variables in
            % the runtime environment need to be flushed
            obj.Runtime = [];
        end

        function [coup_indices, coup_probs] ...
                = generateHeuristicCoupling(obj, projection_dir)
            % Heuristically couple the marginals by first projecting all two-dimensional vertices onto a fixed direction and then apply
            % comonotone coupling for the resulting one-dimensional measures
            % Input:
            %   proj: vector representing the projection direction (default is [-1; 1], which means projecting onto the diagonal line)
            % Outputs:
            %   coup_indices: the indices of atoms in the coupled discrete measure
            %   coup_probs: the probabilities of atoms in the coupled discrete measure

            if ~exist('projection_dir', 'var') || isempty(projection_dir)
                projection_dir = [-1; 1];
            end

            marg_num = length(obj.MarginalWeights);

            % retrieve some information from the marginals
            atoms_cell = cell(marg_num, 1);
            probs_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                atoms_cell{marg_id} = obj.Marginals{marg_id}.SimplicialTestFuncs.Vertices * projection_dir;
                probs_cell{marg_id} = obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals;
            end

            [coup_indices, coup_probs] = comonotone_coupling(atoms_cell, probs_cell);
        end

        function setLSIPSolutions(obj, primal_sol, dual_sol, ...
                LSIP_UB, LSIP_LB)
            % Set the primal and dual solution of the LSIP problem in the runtime environment. This is to allow certain quantities to
            % be computed using stored versions of the primal and dual solutions without executing the cutting-plane algorithm again. 
            % Inputs: 
            %   primal_sol: struct containing information about the primal solution of the LSIP problem
            %   dual_sol: struct containing information about the dual solution of the LSIP problem
            %   LSIP_UB: the upper bound for the optimal value of the LSIP problem
            %   LSIP_LB: the lower bound for the optimal value of the LSIP problem

            if isempty(obj.Runtime)
                obj.Runtime = struct;
            end

            obj.Runtime.PrimalSolution = primal_sol;
            obj.Runtime.DualSolution = dual_sol;
            obj.Runtime.LSIP_UB = LSIP_UB;
            obj.Runtime.LSIP_LB = LSIP_LB;
        end

        function OT_info_cell = performReassembly(obj)
            % Perform reassembly by computing semi-discrete optimal transport
            % Output:
            %   OT_info_cell: cell array where each cell is a cell array containing optimal transport-related information that can be 
            %   saved and loaded later

            % open the log file
            if ~isempty(obj.Options.log_file)
                log_file = fopen(obj.Options.log_file, 'a');

                if log_file < 0
                    error('cannot open log file');
                end

                fprintf(log_file, '--- semi-discrete OT starts ---\n');
            end
            
            marg_num = length(obj.MarginalWeights);

            for marg_id = 1:marg_num
                marg_angle_num = obj.Options.OT.angle_num{marg_id};
                marg_pp_angle_num = obj.Options.OT.pp_angle_num{marg_id};
                marg_normalize_gradient = obj.Options.OT.normalize_gradient{marg_id};

                marg_options = obj.Options.OT.optimization_options{marg_id};

                % the atoms and the corresponding probabilities of the discretized marginal are exactly given by the test functions and
                % their respective integrals; this is only valid due to the quadratic structure of the cost function
                marg = obj.Marginals{marg_id};
                marg_atoms = marg.SimplicialTestFuncs.Vertices;
                marg_probs = marg.SimplicialTestFuncs.Integrals;

                marg.computeOptimalTransport(marg_atoms, marg_probs, marg_angle_num, [], marg_pp_angle_num, marg_options, [], ...
                    marg_normalize_gradient);

                if obj.Options.display
                    fprintf('%s: marginal %d done\n', class(obj), marg_id);
                end

                % logging
                if ~isempty(obj.Options.log_file)
                    fprintf(log_file, '%s: marginal %d done\n', class(obj), marg_id);
                end
            end

            % close the log file
            if ~isempty(obj.Options.log_file)
                fprintf(log_file, '--- semi-discrete OT ends ---\n\n');
                fclose(log_file);
            end

            obj.Storage.OTComputed = true;

            OT_info_cell = obj.saveOptimalTransportInfo();
        end

        function OT_info_cell = saveOptimalTransportInfo(obj)
            % Retrieve computed semi-discrete optimal transport-related information of each marginal
            % Output:
            %   OT_info_cell: cell array where each cell is a cell array containing optimal transport-related information that can be 
            %   saved and loaded later

            marg_num = length(obj.MarginalWeights);
            OT_info_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                OT_info_cell{marg_id} = marg.saveOptimalTransportInfo();
            end
        end

        function loadOptimalTransportInfo(obj, OT_info_cell)
            % Load semi-discrete optimal transport-related information into each marginal
            % Input:
            %   OT_info_cell: cell array where each cell is a cell array containing optimal transport-related information that can be 
            %   saved and loaded later

            marg_num = length(obj.MarginalWeights);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                marg.loadOptimalTransportInfo(OT_info_cell{marg_id});
            end

            obj.Storage.OTComputed = true;
        end

        function [W2OT, W2OT_info_cell] = computeW2OptimalCouplings(obj)
            % Compute the Wasserstein-2 optimal couplings between the computed the discrete measure given by the LSIP dual solution
            % and the marginals.
            % Outputs:
            %   W2OT: struct containing Wasserstein-2 optimal transport-related information that can be saved and loaded later 
            %   W2OT_info_cell: cell array where each cell is a cell array containing Wasserstein-2 optimal transport-related
            %   information about a marginal that can be saved and loaded later

            if ~obj.checkIfWasserstein2OTSupported()
                error('Wasserstein-2 optimal transport is not supported by the marginals');
            end

            if ~isfield(obj.Runtime, 'DualSolution') || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            % open the log file
            if ~isempty(obj.Options.log_file)
                log_file = fopen(obj.Options.log_file, 'a');

                if log_file < 0
                    error('cannot open log file');
                end

                fprintf(log_file, '--- semi-discrete Wasserstein-2 OT starts ---\n');
            end
            
            marg_num = length(obj.MarginalWeights);

            % retrieve the discrete measure from the dual LSIP solution
            dual_sol = obj.Runtime.DualSolution;
            disc_probs = dual_sol.Probabilities;
            [~, disc_atoms] = obj.evaluateCostFunctionFromIndices(dual_sol.VertexIndices);

            % some atoms might be very close to each other; they will be combined
            [~, uind, umap] = unique(round(disc_atoms, 6), 'rows', 'stable');
            disc_atoms = disc_atoms(uind, :);
            disc_atom_num = size(disc_atoms, 1);
            disc_probs = accumarray(umap, disc_probs, [disc_atom_num, 1]);

            % atoms with probabilities that are too small are removed to avoid numerical issues
            small_prob_list = disc_probs < 1e-8;
            disc_atoms = disc_atoms(~small_prob_list, :);
            disc_probs = disc_probs(~small_prob_list);
            disc_probs = disc_probs / sum(disc_probs);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                marg_options = obj.Options.W2OT.optimization_options{marg_id};

                % compute the Wasserstein-2 optimal transport to the discrete measure
                marg.computeW2OptimalTransport(disc_atoms, disc_probs, [], marg_options);

                if obj.Options.display
                    fprintf('%s: marginal %d done\n', class(obj), marg_id);
                end

                % logging
                if ~isempty(obj.Options.log_file)
                    fprintf(log_file, '%s: marginal %d done\n', class(obj), marg_id);
                end
            end

            % close the log file
            if ~isempty(obj.Options.log_file)
                fprintf(log_file, '--- semi-discrete Wasserstein-2 OT ends ---\n\n');
                fclose(log_file);
            end

            obj.Runtime.W2OTComputed = true;
            obj.Runtime.W2OT = struct;
            obj.Runtime.W2OT.DiscMeas = struct('Atoms', disc_atoms, 'Probabilities', disc_probs);

            [W2OT, W2OT_info_cell] = obj.saveW2OptimalTransportInfo();
        end

        function [W2OT, W2OT_info_cell] = saveW2OptimalTransportInfo(obj)
            % Retrieve computed semi-discrete Wasserstein-2 optimal transport-related information of each marginal
            % Outputs:
            %   W2OT: struct containing Wasserstein-2 optimal transport-related information that can be saved and loaded later 
            %   W2OT_info_cell: cell array where each cell is a cell array containing Wasserstein-2 optimal transport-related
            %   information about a marginal that can be saved and loaded later

            if ~obj.checkIfWasserstein2OTSupported()
                error('Wasserstein-2 optimal transport is not supported by the marginals');
            end

            marg_num = length(obj.MarginalWeights);

            W2OT = obj.Runtime.W2OT;
            W2OT_info_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                W2OT_info_cell{marg_id} = marg.saveW2OptimalTransportInfo();
            end
        end

        function loadW2OptimalTransportInfo(obj, W2OT, W2OT_info_cell)
            % Load semi-discrete Wasserstein-2 optimal transport-related information into each marginal
            % Inputs:
            %   W2OT: struct containing Wasserstein-2 optimal transport-related information that can be saved and loaded later 
            %   W2OT_info_cell: cell array where each cell is a cell array containing Wasserstein-2 optimal transport-related
            %   information about a marginal that can be saved and loaded later

            if ~obj.checkIfWasserstein2OTSupported()
                error('Wasserstein-2 optimal transport is not supported by the marginals');
            end

            marg_num = length(obj.MarginalWeights);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                marg.loadW2OptimalTransportInfo(W2OT_info_cell{marg_id});
            end

            obj.Runtime.W2OT = W2OT;
            obj.Runtime.W2OTComputed = true;
        end

        function vals = evaluateMMOTDualFunctions(obj, pts, include_quad)
            % Evaluate the computed dual solution of the MMOT problem at a list of points
            % Inputs:
            %   pts: two-column matrix containing the input points at which the dual functions need to be evaluated
            %   include_quad: boolean indicating whether to add back the univariate quadratic functions
            % Output:
            %   vals: matrix where each row correponds to an input point and each column corresponds to a marginal

            if ~isfield(obj.Runtime, 'PrimalSolution') || isempty(obj.Runtime.PrimalSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            pt_num = size(pts, 1);
            
            vals = zeros(pt_num, length(obj.Marginals));

            for marg_id = 1:length(obj.Marginals)
                vals(:, marg_id) = obj.Marginals{marg_id}.evaluateWeightedSumOfSimplicialTestFuncs(pts, ...
                    obj.Runtime.PrimalSolution.Coefficients{marg_id});
            end

            % add the constant to the first dual function
            vals(:, 1) = vals(:, 1) + obj.Runtime.PrimalSolution.Constant;

            if include_quad
                quad_vals = sum(pts.^2, 2);
                vals = vals + quad_vals * obj.MarginalWeights';
            end
        end

        function LB = getMMOTLowerBound(obj)
            % Retrieve the computed lower bound for the MMOT problem
            % Output:
            %   LB: the computed lower bound

            if ~isfield(obj.Runtime, 'PrimalSolution') || isempty(obj.Runtime.PrimalSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            LB = -obj.Runtime.LSIP_UB;
        end
    
        function [UB_list, samps, samp_histpdf] = getMMOTUpperBoundWRepetition(obj, samp_num, rep_num, rand_stream, ...
                hist_edge_x, hist_edge_y)
            % Compute an upper bound for the MMOT problem with repetition of Monte Carlo integration. 
            % Inputs:
            %   samp_num: number of samples for Monte Carlo integration
            %   rep_num: number of repetitions
            %   rand_stream: RandStream object (default is RandStream.getGlobalStream)
            %   hist_edge_x: edges of bins on the x-axis for 2D pdf estimation (default is [])
            %   hist_edge_y: edges of bins on the y-axis for 2D pdf estimation (default is [])
            % Output:
            %   UB_list: vector containing the computed upper bounds for the MMOT problem
            %   samps: one particular set of Monte Carlo samples used in the approximation of the bounds
            %   samp_histpdf: 2D pdf estimation via the histogram of the generated samples from the approximated Wasserstein 
            %   barycenter; only computed if both hist_edge_x and hist_edge_y are set 

            if ~exist('rand_stream', 'var') || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            if ~exist('hist_edge_x', 'var') || ~exist('hist_edge_y', 'var')
                hist_edge_x = [];
                hist_edge_y = [];
                samp_histpdf = [];
            end

            UB_list = zeros(rep_num, 1);

            % open the log file
            if ~isempty(obj.Options.log_file)
                log_file = fopen(obj.Options.log_file, 'a');

                if log_file < 0
                    error('cannot open log file');
                end

                fprintf(log_file, '--- Monte Carlo sampling starts ---\n');
            end

            % initialize the 2D histogram if it needs to be computed
            if ~isempty(hist_edge_x) && ~isempty(hist_edge_y)
                pdf_computed = true;

                samp_histpdf = zeros(length(hist_edge_x) - 1, length(hist_edge_y) - 1);
            else
                pdf_computed = false;
            end

            for rep_id = 1:rep_num
                [UB_list(rep_id), samps] = obj.getMMOTUpperBound(samp_num, rand_stream);

                % display output
                if obj.Options.display
                    fprintf('%s: Monte Carlo sampling repetition %3d done\n', class(obj), rep_id);
                end

                % write log
                if ~isempty(obj.Options.log_file)
                    fprintf(log_file, '%s: Monte Carlo sampling repetition %3d done\n', class(obj), rep_id);
                end

                % update the 2D histogram
                if pdf_computed
                    samp_histpdf = samp_histpdf + histcounts2(samps.Barycenter(:, 1), samps.Barycenter(:, 2), ...
                        hist_edge_x, hist_edge_y, 'Normalization', 'pdf');
                end
            end

            % close the log file
            if ~isempty(obj.Options.log_file)
                fprintf(log_file, '--- Monte Carlo sampling ends ---\n\n');
                fclose(log_file);
            end

            % normalize the 2D histogram
            if pdf_computed
                samp_histpdf = samp_histpdf / rep_num;
            end
        end
        
        function [UB, samps] = getMMOTUpperBound(obj, samp_num, rand_stream)
            % Compute an upper bound for the MMOT problem based on Monte Carlo integration. 
            % Inputs: 
            %   samp_num: number of samples for Monte Carlo integration
            %   rand_stream: RandStream object (default is RandStream.getGlobalStream)
            % Output:
            %   UB: the computed upper bound for the MMOT problem based on Monte Carlo integration
            %   samps: struct containg fields Coupling, Barycenter, and Cost where Coupling is a cell array containing coupled samples 
            %   from the marginals, Barycenter is a two-column matrix containing the corresponding samples from the Wasserstein 
            %   barycenter, and Cost is a vector containing the corresponding values of the cost function

            if ~isfield(obj.Runtime, 'DualSolution') || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            if ~exist('rand_stream', 'var') || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            reassembly_samps = obj.randSampleFromReassembly(samp_num, rand_stream);

            [costs, barycenters] = obj.evaluateCostFunction(reassembly_samps);
            costs = costs + obj.QuadraticConstant;

            samps = struct;
            samps.Coupling = reassembly_samps;
            samps.Barycenter = barycenters;
            samps.Cost = costs;

            UB = mean(costs);
        end

        function [UB_list, samps, samp_histpdf] = getMMOTUpperBoundViaW2CouplingWRepetition(obj, samp_num, rep_num, rand_stream, ...
                hist_edge_x, hist_edge_y)
            % Compute an upper bound for the MMOT problem via Wasserstein-2 optimal couplings with repetition of Monte Carlo 
            % integration.
            % Inputs:
            %   samp_num: number of samples for Monte Carlo integration
            %   rep_num: number of repetitions
            %   rand_stream: RandStream object (default is RandStream.getGlobalStream)
            %   hist_edge_x: edges of bins on the x-axis for 2D pdf estimation (default is [])
            %   hist_edge_y: edges of bins on the y-axis for 2D pdf estimation (default is [])
            % Output:
            %   UB_list: vector containing the computed upper bounds for the MMOT problem
            %   samps: one particular set of Monte Carlo samples used in the approximation of the bounds
            %   samp_histpdf: 2D pdf estimation via the histogram of the generated samples from the approximated Wasserstein 
            %   barycenter; only computed if both hist_edge_x and hist_edge_y are set 

            if ~obj.checkIfWasserstein2OTSupported()
                error('Wasserstein-2 optimal transport is not supported by the marginals');
            end

            if ~exist('rand_stream', 'var') || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            if ~exist('hist_edge_x', 'var') || ~exist('hist_edge_y', 'var')
                hist_edge_x = [];
                hist_edge_y = [];
                samp_histpdf = [];
            end

            UB_list = zeros(rep_num, 1);

            % open the log file
            if ~isempty(obj.Options.log_file)
                log_file = fopen(obj.Options.log_file, 'a');

                if log_file < 0
                    error('cannot open log file');
                end

                fprintf(log_file, '--- Monte Carlo sampling starts ---\n');
            end

            % initialize the 2D histogram if it needs to be computed
            if ~isempty(hist_edge_x) && ~isempty(hist_edge_y)
                pdf_computed = true;

                samp_histpdf = zeros(length(hist_edge_x) - 1, length(hist_edge_y) - 1);
            else
                pdf_computed = false;
            end

            for rep_id = 1:rep_num
                [UB_list(rep_id), samps] = obj.getMMOTUpperBoundViaW2Coupling(samp_num, rand_stream);

                % display output
                if obj.Options.display
                    fprintf('%s: Monte Carlo sampling repetition %3d done\n', class(obj), rep_id);
                end

                % write log
                if ~isempty(obj.Options.log_file)
                    fprintf(log_file, '%s: Monte Carlo sampling repetition %3d done\n', class(obj), rep_id);
                end

                % update the 2D histogram
                if pdf_computed
                    samp_histpdf = samp_histpdf + histcounts2(samps.Barycenter(:, 1), samps.Barycenter(:, 2), ...
                        hist_edge_x, hist_edge_y, 'Normalization', 'pdf');
                end
            end

            % close the log file
            if ~isempty(obj.Options.log_file)
                fprintf(log_file, '--- Monte Carlo sampling ends ---\n\n');
                fclose(log_file);
            end

            % normalize the 2D histogram
            if pdf_computed
                samp_histpdf = samp_histpdf / rep_num;
            end
        end
        
        function [UB, samps] = getMMOTUpperBoundViaW2Coupling(obj, samp_num, rand_stream)
            % Compute an upper bound for the MMOT problem via Wasserstein-2 optimal couplings based on Monte Carlo integration. 
            % Inputs: 
            %   samp_num: number of samples for Monte Carlo integration
            %   rand_stream: RandStream object (default is RandStream.getGlobalStream)
            % Output:
            %   UB: the computed upper bound for the MMOT problem based on Monte Carlo integration
            %   samps: struct containg fields Coupling, Barycenter, and Cost where Coupling is a cell array containing coupled samples 
            %   from the marginals, Barycenter is a two-column matrix containing the corresponding samples from the Wasserstein 
            %   barycenter, and Cost is a vector containing the corresponding values of the cost function

            if ~obj.checkIfWasserstein2OTSupported()
                error('Wasserstein-2 optimal transport is not supported by the marginals');
            end

            if ~isfield(obj.Runtime, 'DualSolution') || isempty(obj.Runtime.DualSolution)
                error('need to first execute the cutting-plane algorithm');
            end

            if ~exist('rand_stream', 'var') || isempty(rand_stream)
                rand_stream = RandStream.getGlobalStream;
            end

            coupling_samps = obj.randSampleFromW2Couplings(samp_num, rand_stream);

            [costs, barycenters] = obj.evaluateCostFunction(coupling_samps);
            costs = costs + obj.QuadraticConstant;

            samps = struct;
            samps.Coupling = coupling_samps;
            samps.Barycenter = barycenters;
            samps.Cost = costs;

            UB = mean(costs);
        end

        function EB = getMMOTErrorBoundBasedOnOT(obj, LSIP_tolerance)
            % Compute the error bound for the MMOT problem based on the Lipschitz constant of the cost functions and the optimal 
            % transport distances between the marginals and their discretizations.
            % Input:
            %   LSIP_tolerance: tolerance value used in the computation of the LSIP problem
            % Output:
            %   EB: the computed error bound

            % make sure that the semi-discrete OT problems are solved
            if ~isfield(obj.Storage, 'OTComputed') || ~obj.Storage.OTComputed
                obj.performReassembly();
            end

            marg_num = length(obj.MarginalWeights);
            EB = LSIP_tolerance;

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                EB = EB + obj.Storage.LipschitzConstants(marg_id) * marg.OT.Cost;
            end
        end

        function EB = getMMOTTheoreticalErrorBound(obj, LSIP_tolerance)
            % Compute the theoretical error bound for the objective value of the MMOT problem via the Lipschitz constant of the cost 
            % functions and the mesh sizes of the simplicial covers for the  marginals
            % Input:
            %   LSIP_tolerance: tolerance value used in the computation of the LSIP problem
            % Output:
            %   EB: the computed error bound

            marg_num = length(obj.MarginalWeights);
            EB = LSIP_tolerance;

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                EB = EB + obj.Storage.LipschitzConstants(marg_id) * 2 * marg.SimplicialTestFuncs.MeshSize;
            end
        end
    end

    methods(Access = protected)

        function initializeBeforeRun(obj)
            % Initialize the algorithm by computing some static quantities

            if ~obj.Storage.SimplicialTestFuncsInitialized
                obj.initializeSimplicialTestFuncs();
            end
        end

        function prepareRuntime(obj)
            % Prepare the runtime environment by initializing some variables
            prepareRuntime@LSIPMinCuttingPlaneAlgo(obj);

            % initialize the cuts to be empty
            obj.Runtime.CutIndices = zeros(0, length(obj.MarginalWeights));

            % open the log file for the global minimization oracle
            if ~isempty(obj.GlobalOptions.log_file)
                obj.Runtime.GlobalMinLogFile = fopen(obj.GlobalOptions.log_file, 'a');

                if obj.Runtime.GlobalMinLogFile < 0
                    error('cannot open log file');
                end

                fprintf(obj.Runtime.GlobalMinLogFile, '--- global minimization oracle starts ---\n');
            end
        end

        function cleanUpRuntimeAfterRun(obj)
            % Clean up the runtime environment after finishing the cutting-plane algorithm

            cleanUpRuntimeAfterRun@LSIPMinCuttingPlaneAlgo(obj);

            % close the log file for the global minimization oracle
            if isfield(obj.Runtime, 'GlobalMinLogFile') && ~isempty(obj.Runtime.GlobalMinLogFile)
                fprintf(obj.Runtime.GlobalMinLogFile, '--- global minimization oracle ends ---\n');

                fclose(obj.Runtime.GlobalMinLogFile);
            end
        end

        function [costs, barycenters] = evaluateCostFunction(obj, input_points)
            % Evaluate the cost function at the given tuples of input points. 
            % Inputs: 
            %   input_points: cell array containing coupled points from the marginals; each cell contains a two-column matrix
            % Output:
            %   costs: vector representing the evaluated costs
            %   barycenter: two-column matrix where each row represents the barycenter of a tuple of input points

            input_num = size(input_points{1}, 1);
            marg_num = length(obj.MarginalWeights);

            barycenters = zeros(input_num, 2);

            % matrix for computing the inner product terms directly
            weighted_input_mat = zeros(input_num, 2 * marg_num);

            for marg_id = 1:marg_num
                weighted_inputs = obj.MarginalWeights(marg_id) * input_points{marg_id};
                barycenters = barycenters + weighted_inputs;

                weighted_input_mat(:, 2 * (marg_id - 1) + (1:2)) = weighted_inputs;
            end

            costs = sum(barycenters.^2, 2) - 2 * sum(weighted_input_mat .* repmat(barycenters, 1, marg_num), 2);
        end

        function [costs, vertex_barycenters] = evaluateCostFunctionFromIndices(obj, vertex_indices, batch_size)
            % Evaluate the cost function based on indices in the triangulations of the marginals. Computation is done by first
            % computing the weighted sum of the vertices (i.e., their barycenter) and then evaluate the sum of squares cost.
            % Computation is done in batches if necessary. 
            % Inputs:
            %   vertex_indices: matrix of indices where each row corresponds to an input and each column corresponds to a marginal; 
            %   each index corresponds to the index of a vertex in the triangulation of a marginal
            %   batch_size: the maximum number of inputs to be handled in a vectorized procedure (default is 1e4)
            % Output:
            %   costs: vector representing the evaluated costs
            %   vertex_barycenter: two-column matrix where each row represents the barycenter of a combination of vertices

            if ~exist('batch_size', 'var') || isempty(batch_size)
                batch_size = 1e4;
            end

            input_num = size(vertex_indices, 1);
            marg_num = length(obj.MarginalWeights);

            if input_num <= batch_size
                % if the number of input is less than the batch size, do the computation directly

                % first add the offsets to the indices so that they become row indices in the matrix containing all vertices
                offseted_vertex_indices = vertex_indices' + obj.Storage.MargVertNumOffsets;

                % sparse combination matrix to add up the coordinates of the vertices
                comb_mat = sparse(repelem((1:input_num)', marg_num, 1), (1:input_num * marg_num)', 1, input_num, input_num * marg_num);

                weighted_vertices_mat = obj.Storage.WeightedMargVertices(offseted_vertex_indices(:), :);

                vertex_barycenters = full(comb_mat * weighted_vertices_mat);

                % the cost is given by a quadratic term on vertex_barycenters minus an inner product
                costs = sum(vertex_barycenters.^2, 2) - 2 * full(comb_mat * sum(weighted_vertices_mat ...
                    .* repelem(vertex_barycenters, marg_num, 1), 2));
            else
                % if more than one batch is needed, pre-compute the combination matrix
                batch_num = ceil(input_num / batch_size);
                barycenter_cell = cell(batch_num, 1);
                cost_cell = cell(batch_num, 1);
                comb_mat = sparse(repelem((1:batch_size)', marg_num, 1), (1:batch_size * marg_num)', 1, ...
                    batch_size, batch_size * marg_num);

                for batch_id = 1:batch_num
                    offseted_vertex_indices = vertex_indices(((batch_id - 1) * batch_size + 1):min(batch_id ...
                        * batch_size, input_num), :)' + obj.Storage.MargVertNumOffsets;
                    this_batch_size = size(offseted_vertex_indices, 2);
                    weighted_vertices_mat = obj.Storage.WeightedMargVertices(offseted_vertex_indices(:), :);

                    if batch_id < batch_num
                        batch_comb_mat = comb_mat;
                    else
                        batch_comb_mat = comb_mat(1:this_batch_size, 1:(input_num - (batch_num - 1) * batch_size) * marg_num);
                    end
                    
                    barycenter_cell{batch_id} = full(batch_comb_mat * weighted_vertices_mat);

                    % the cost is given by a quadratic term on vertex_barycenters minus an inner product
                    cost_cell{batch_id} = sum(barycenter_cell{batch_id}.^2, 2) - 2 * full(batch_comb_mat ...
                        * sum(weighted_vertices_mat .* repelem(barycenter_cell{batch_id}, marg_num, 1), 2));
                end

                vertex_barycenters = vertcat(barycenter_cell{:});
                costs = vertcat(cost_cell{:});
            end
        end

        function updateLSIPUB(obj, min_lb, optimizers) %#ok<INUSD> 
            % Update the LSIP upper bound after each call to the global minimization oracle
            % Inputs:
            %   min_lb: the lower bound for the global minimization problem
            %   optimizers: a set of approximate optimizers of the global minimization problem

            obj.Runtime.LSIP_UB = min(obj.Runtime.LSIP_UB, obj.Runtime.LSIP_LB - min_lb);
        end

        function model = generateInitialMinModel(obj)
            % Generate the initial linear programming model for gurobi
            % Output:
            %   model: struct containing the linear programming model in gurobi

            model = struct;
            marg_num = length(obj.MarginalWeights);
            decivar_num = length(obj.Storage.DeciVarIndicesInTestFuncs) + 1;

            integrals_cell = cell(marg_num, 1);

            for marg_id = 1:marg_num
                integrals_cell{marg_id} = obj.Marginals{marg_id}.SimplicialTestFuncs.Integrals;
            end

            integrals_vec = vertcat(integrals_cell{:});
            
            % since the cutting plane algorithm assumes that the problem is a minimization problem, we need to transform our 
            % maximization problem into a minimization problem
            model.modelsense = 'min';

            % the constant in the objective is the sum of the quadratic constants that do not affect the coupling
            model.objcon = -obj.QuadraticConstant;

            % the coefficients corresponding to the first test function of each marginal is not included in the decision variables for
            % identification purposes
            model.obj = [-1; -integrals_vec(obj.Storage.DeciVarIndicesInTestFuncs)];

            model.sense = '>';
            model.lb = -inf(decivar_num, 1);
            model.ub = inf(decivar_num, 1);
            model.A = sparse(0, decivar_num);
            model.rhs = zeros(0, 1);
        end

        function [min_lb, optimizers] = callGlobalMinOracle(obj, vec)
            % Given a decision vector, call the global minimization oracle to approximately determine the "most violated" constraints
            % and return a lower bound for the minimal value
            % Input:
            %   vec: vector corresponding to the current LP solution
            % Outputs:
            %   min_lb: lower bound for the global minimization problem
            %   optimizers: struct containing approximate optimizers of the global minimization problem in the form of vertex indices
            %   in the triangulations for each of the input measures as well as the the corresponding points in the two-dimensional
            %   space

            timer = tic;
            marg_num = length(obj.MarginalWeights);

            % disassemble the vector resulted from solving the LP problem

            % the first component corresponds to the constant term; the second component onwards will be filled into the corresponding 
            % entries of the vector storing the coefficients corresponding to the test functions
            objective_const = vec(1);
            vert_vals = zeros(obj.Storage.TotalVertNum, 1);
            vert_vals(obj.Storage.DeciVarIndicesInTestFuncs) = vec(2:end);

            % prepare the radii of the circles
            circles_cell = obj.Storage.GlobalMin.Circles;
            marg_vert_num_list = obj.Storage.MargVertNumList;
            marg_vert_offsets = obj.Storage.MargVertNumOffsets;

            for marg_id = 1:marg_num
                tf_verts = obj.Marginals{marg_id}.SimplicialTestFuncs.Vertices;

                marg_vert_vals = vert_vals(marg_vert_offsets(marg_id) + (1:marg_vert_num_list(marg_id)));

                % the weight of each cell in the power diagram
                power_diagram_weights = sum(tf_verts.^2, 2) + marg_vert_vals / obj.MarginalWeights(marg_id);

                % subtract the minimum weight then plus 1 to make all weights positive
                power_diagram_weights = power_diagram_weights - min(power_diagram_weights) + 1;

                % the squared radii of the circles are equal to the weights
                circles_cell{marg_id, 2} = power_diagram_weights;
            end

            % call the mex function power_diagram_intersection to determine the relevant cells in the intersection of the power 
            % diagrams that are non-empty
            cell_indices = power_diagram_intersection(circles_cell, ...
                obj.Storage.BoundingBox.BottomLeft, obj.Storage.BoundingBox.TopRight);
            cell_indices = unique(cell_indices, 'rows');

            % evaluate the cost function with respect to the computed indices
            cell_costs = obj.evaluateCostFunctionFromIndices(cell_indices, 1e4);

            objectives = cell_costs - objective_const - sum(vert_vals(cell_indices + marg_vert_offsets'), 2);

            [sorted_objs, sorted_order] = sort(objectives, 'ascend');
            sorted_cell_indices = cell_indices(sorted_order, :);

            pool_size = min(obj.GlobalOptions.pool_size, length(sorted_order));

            pool_vals = sorted_objs(1:pool_size);
            pool_inds = sorted_cell_indices(1:pool_size, :);
            
            min_lb = pool_vals(1);

            % ignore those approximate minimizers whose objective values are non-negative since they do not generate cuts
            pool_neg_list = pool_vals < 0;

            optimizers = struct;
            optimizers.vertex_indices = pool_inds(pool_neg_list, :);

            time_elapsed = toc(timer);

            message_string = sprintf('%s: iteration %4d: found %3d approx minimizers in %6.2fs, min obj = %10.6f\n', class(obj), ...
                obj.Runtime.iter, pool_size, time_elapsed, min_lb);
            
            if obj.GlobalOptions.display
                fprintf(message_string);
            end

            if ~isempty(obj.GlobalOptions.log_file)
                fprintf(obj.Runtime.GlobalMinLogFile, message_string);
            end
        end

        function addConstraints(obj, optimizers)
            % Given a collection of approximate optimizers from the global minimization oracle, generate and add the corresponding
            % linear constraints
            % Inputs:
            %   optimizers: output of the method callGlobalOracle

            constr_num = size(optimizers.vertex_indices, 1);
            marg_num = length(obj.MarginalWeights);
            
            col_indices = optimizers.vertex_indices' + obj.Storage.MargVertNumOffsets;

            % first generate a matrix containing all test functions (each row corresponds to an approximate optimizer, each column
            % corresponds to a test function)
            A_full = sparse(repelem((1:constr_num)', marg_num, 1), col_indices(:), 1, constr_num, obj.Storage.TotalVertNum);
            
            % filter out those test functions whose coefficients are not included in the decision variable, then prepend a column of 1
            A_new = [sparse(ones(constr_num, 1)), A_full(:, obj.Storage.DeciVarIndicesInTestFuncs)];

            rhs_new = obj.evaluateCostFunctionFromIndices(optimizers.vertex_indices);
            
            % add the newly generated constraints to the end
            obj.Runtime.CurrentLPModel.A = [obj.Runtime.CurrentLPModel.A; -A_new];
            obj.Runtime.CurrentLPModel.rhs = [obj.Runtime.CurrentLPModel.rhs; -rhs_new];

            if ~isempty(obj.Runtime.vbasis) && ~isempty(obj.Runtime.cbasis)
                obj.Runtime.CurrentLPModel.vbasis = obj.Runtime.vbasis;
                obj.Runtime.CurrentLPModel.cbasis = [obj.Runtime.cbasis; zeros(constr_num, 1)];
            else
                if isfield(obj.Runtime.CurrentLPModel, 'vbasis')
                    obj.Runtime.CurrentLPModel = rmfield(obj.Runtime.CurrentLPModel, 'vbasis');
                end

                if isfield(obj.Runtime.CurrentLPModel, 'cbasis')
                    obj.Runtime.CurrentLPModel = rmfield(obj.Runtime.CurrentLPModel, 'cbasis');
                end
            end

            % add the added indices to the runtime environment
            obj.Runtime.CutIndices = [obj.Runtime.CutIndices; optimizers.vertex_indices];

            % if obj.Runtime.NumOfInitialConstraints is not set, it means that this is the first call to obj.addConstraints which
            % generates the initial constraints; this number is stored in the runtime environment
            if ~isfield(obj.Runtime, 'NumOfInitialConstraints') || isempty(obj.Runtime.NumOfInitialConstraints)
                obj.Runtime.NumOfInitialConstraints = constr_num;
            end
        end

        function reduceConstraints(obj, result)
            % Remove some of the constraints to speed up the LP solver
            % Input:
            %   result: the output from the gurobi LP solver

            if isinf(obj.Options.reduce.thres) || obj.Runtime.iter <= 0 || obj.Runtime.iter > obj.Options.reduce.max_iter ...
                    || mod(obj.Runtime.iter, obj.Options.reduce.freq) ~= 0
                return;
            end

            flagged_indices = result.slack < -obj.Options.reduce.min_slack;

            if obj.Options.reduce.preserve_init_constr && isfield(obj.Runtime, 'NumOfInitialConstraints') ...
                    && ~isempty(obj.Runtime.NumOfInitialConstraints)
                flagged_indices(1:obj.Runtime.NumOfInitialConstraints) = false;
            end

            cut_slack_thres_quantile = quantile(-result.slack(flagged_indices), obj.Options.reduce.thres_quantile);
            keep_list = ~flagged_indices | (result.slack >= -obj.Options.reduce.thres & result.slack >= -cut_slack_thres_quantile);

            % update all variables
            obj.Runtime.CutIndices = obj.Runtime.CutIndices(keep_list, :);
            obj.Runtime.CurrentLPModel.A = obj.Runtime.CurrentLPModel.A(keep_list, :);
            obj.Runtime.CurrentLPModel.rhs = obj.Runtime.CurrentLPModel.rhs(keep_list);

            if ~isempty(obj.Runtime.cbasis)
                obj.Runtime.cbasis = obj.Runtime.cbasis(keep_list);
            end
        end

        function primal_sol = buildPrimalSolution(obj, result, violation)
            % Given the output from gurobi and a lower bound for the optimal value of the global minimization oracle, build the
            % corresponding primal solution
            % Inputs:
            %   result: output of the gurobi LP solver
            %   violation: a lower bound for the global minimization problem
            % Output:
            %   primal_sol: the constructed potential functions on the support of the input measures

            vec = result.x;
            primal_sol = struct;

            % since violation is non-positive, we shift the first function downwards
            primal_sol.Constant = vec(1) + violation;

            vert_coefs = zeros(obj.Storage.TotalVertNum, 1);
            vert_coefs(obj.Storage.DeciVarIndicesInTestFuncs) = vec(2:end);
            marg_num = length(obj.MarginalWeights);
            primal_sol.Coefficients = cell(marg_num, 1);

            for marg_id = 1:marg_num
                primal_sol.Coefficients{marg_id} = vert_coefs(obj.Storage.MargVertNumOffsets(marg_id) ...
                    + (1:obj.Storage.MargVertNumList(marg_id)));
            end
        end

        function dual_sol = buildDualSolution(obj, result)
            % Given the output from gurobi, build the corresponding dual solution
            % Input:
            %   result: output of the gurobi LP solver
            % Output:
            %   dual_sol: the constructed discrete probability measure for the relaxed MMOT problem

            dual_sol = struct;
            pos_list = result.pi > obj.Options.dual_prob_thres;
            dual_sol.Probabilities = result.pi(pos_list);
            dual_sol.VertexIndices = obj.Runtime.CutIndices(pos_list, :);

            % need to normalize the probabilities to sum to 1; there is sometimes numerical errors significant enough to cause the sum 
            % to be not sufficiently close to 1
            dual_sol.Probabilities = dual_sol.Probabilities ...
                / sum(dual_sol.Probabilities);
        end

        function samps = randSampleFromReassembly(obj, samp_num, rand_stream)
            % Generate independent random samples from a reassembly measure of a discrete measure computed by the cutting-plane
            % algorithm.
            % Inputs: 
            %   samp_num: number of samples to generate
            %   rand_stream: RandStream object used for sampling
            % Outputs:
            %   samps: cell array containing coupled samples from the marginals where each cell corresponds to a marginal

            % make sure that the semi-discrete OT problems are solved
            if ~isfield(obj.Storage, 'OTComputed') || ~obj.Storage.OTComputed
                obj.performReassembly();
            end

            marg_num = length(obj.MarginalWeights);
            n = length(obj.Runtime.DualSolution.Probabilities);
            
            % generate random indices of the atoms according to the probabilities
            disc_atom_index_samps = randsample(rand_stream, n, samp_num, true, obj.Runtime.DualSolution.Probabilities);

            vertex_index_samps = obj.Runtime.DualSolution.VertexIndices(disc_atom_index_samps, :);

            samps = cell(marg_num, 1);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};
                marg_atoms = marg.SimplicialTestFuncs.Vertices;

                samps{marg_id} = zeros(samp_num, 2);

                % count the number of samples coupled with each of the atoms in the discretized marginal
                atom_num_list = accumarray(vertex_index_samps(:, marg_id), 1, [size(marg_atoms, 1), 1]);

                % generate from the conditional distributions
                cond_samp_cell = marg.conditionalRandSample((1:size(marg_atoms, 1))', atom_num_list, rand_stream);

                % fill in the coupled samples from the continuous marginals
                for atom_id = 1:length(atom_num_list)
                    samps{marg_id}(vertex_index_samps(:, marg_id) == atom_id, :) = cond_samp_cell{atom_id};
                end
            end
        end

        function samps = randSampleFromW2Couplings(obj, samp_num, ...
                rand_stream)
            % Generate independent random samples from a probability measure that glues together the Wasserstein-2 optimal couplings 
            % between the discrete measure from the dual LSIP solution and the marginals.
            % Inputs: 
            %   samp_num: number of samples to generate
            %   rand_stream: RandStream object used for sampling
            % Outputs:
            %   samps: cell array containing coupled samples from the marginals where each cell corresponds to a marginal

            % make sure that the semi-discrete Wasserstein-2 OT problems are solved 
            if ~isfield(obj.Runtime, 'W2OTComputed') || ~obj.Runtime.W2OTComputed
                obj.computeW2OptimalCouplings();
            end

            marg_num = length(obj.MarginalWeights);
            n = length(obj.Runtime.W2OT.DiscMeas.Probabilities);
            
            % generate random indices of the atoms according to the probabilities
            disc_atom_index_samps = randsample(rand_stream, n, samp_num, true, obj.Runtime.W2OT.DiscMeas.Probabilities);

            samps = cell(marg_num, 1);

            for marg_id = 1:marg_num
                marg = obj.Marginals{marg_id};

                samps{marg_id} = zeros(samp_num, 2);

                % count the number of samples coupled with each of the atoms in the discretized marginal
                atom_num_list = accumarray(disc_atom_index_samps, 1, [n, 1]);

                % generate from the conditional distributions
                cond_samp_cell = marg.conditionalRandSampleFromW2Coupling((1:n)', atom_num_list, rand_stream);

                % fill in the coupled samples from the continuous marginals
                for atom_id = 1:length(atom_num_list)
                    samps{marg_id}(disc_atom_index_samps == atom_id, :) = cond_samp_cell{atom_id};
                end
            end
        end

        function supported = checkIfWasserstein2OTSupported(obj)
            % Check whether all the marginals support Wasserstein-2 optimal transport.
            % Output:
            %   supported: boolean indicating whether all the marginals support Wasserstein-2 optimal transport

            supported = true;

            for marg_id = 1:length(obj.MarginalWeights)
                if ~isa(obj.Marginals{marg_id}, 'ProbMeas2DConvexPolytopeWithW2OT')
                    supported = false;
                    break;
                end
            end
        end

        function LP_options_runtime_new = handleLPErrors(obj, LP_result, LP_options_runtime, LP_trial_num) %#ok<INUSD>
            % Handle numerical errors that occurred while solving LP
            % Inputs: 
            %   LP_result: struct returned by the gurobi function representing the result from solving LP
            %   LP_options_runtime: struct containing the current options for solving LP
            %   LP_trial_num: integer representing the number of trials so far
            % Output:
            %   LP_options_runtime_new: struct containing the updated options for solving LP

            LP_options_runtime_new = LP_options_runtime;

            if LP_trial_num == 1
                % if the LP solver has failed once (reaching the time limit without converging), retry after  setting higher numeric 
                % focus, turning off presolve, and removing the existing bases
                LP_options_runtime_new.TimeLimit = LP_options_runtime.TimeLimit * 2;
                LP_options_runtime_new.NumericFocus = 3;
                LP_options_runtime_new.Quad = 1;
                LP_options_runtime_new.Presolve = 0;

                if isfield(obj.Runtime.CurrentLPModel, 'cbasis')
                    obj.Runtime.CurrentLPModel = rmfield(obj.Runtime.CurrentLPModel, 'cbasis');
                end

                if isfield(obj.Runtime.CurrentLPModel, 'vbasis')
                    obj.Runtime.CurrentLPModel = rmfield(obj.Runtime.CurrentLPModel, 'vbasis');
                end
            elseif LP_trial_num == 2 && isfield(LP_options_runtime, 'Method') && LP_options_runtime.Method == 2 ...
                && isfield(LP_options_runtime, 'Crossover') && LP_options_runtime.Crossover == 0
                % if the solver is using the barrier algorithm with crossover disabled, switch to the dual-simplex algorithm
                LP_options_runtime_new.Method = 1;
                LP_options_runtime_new = rmfield(LP_options_runtime_new, 'Crossover');
            elseif LP_trial_num == 3 && isfield(LP_options_runtime, 'Method') && LP_options_runtime.Method == 1
                % if the dual-simplex algorithm also fails, switch to the primal-simplex algorithm
                LP_options_runtime_new.Method = 0;
            else
                while true
                    % do nothing
                    warning('waiting for intervention...');
                    pause(10);
                    
                    continue_flag = false;

                    if continue_flag
                        break; %#ok<UNRCH>
                    end
                end
            end
        end
    end
end
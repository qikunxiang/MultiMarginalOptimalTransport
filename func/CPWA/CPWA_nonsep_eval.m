function v = CPWA_nonsep_eval(x, CPWA)
% Evaluate a (non-separable) CPWA function
% Inputs: 
%       x: inputs, each row is an input, columns are the dimensions
%       CPWA: the specification of the CPWA function, which is a struct
%       containing two fields:
%           plus: cell array with two columns
%           minus: cell array with two columns
% Outputs: 
%       v: the function values

[n, ~] = size(x);
v = zeros(n, 1);

if isfield(CPWA, 'plus')
    for s = 1:size(CPWA.plus, 1)
        v = v + max(x * CPWA.plus{s, 1}' + CPWA.plus{s, 2}', [], 2);
    end
end

if isfield(CPWA, 'minus')
    for s = 1:size(CPWA.minus, 1)
        v = v - max(x * CPWA.minus{s, 1}' + CPWA.minus{s, 2}', [], 2);
    end
end

end


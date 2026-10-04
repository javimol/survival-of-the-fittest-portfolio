function frontier = efficient_frontier(mu, Sigma, k, solver, step)
%EFFICIENT_FRONTIER Cardinality-constrained mean-variance frontier.
%   frontier = EFFICIENT_FRONTIER(mu, Sigma, k, solver) sweeps the target
%   return from min(mu) to max(mu) and returns an M-by-2 matrix of
%   [target_return, minimum_variance] rows. solver is @solve_ga or
%   @solve_exhaustive. EFFICIENT_FRONTIER(..., step) sets the spacing of
%   target returns (default 1e-4).

if nargin < 5, step = 1e-4; end

targets  = (min(mu):step:max(mu))';
variance = zeros(size(targets));
for i = 1:numel(targets)
    [~, variance(i)] = solver(mu, Sigma, k, targets(i));
end
frontier = [targets, variance];
end

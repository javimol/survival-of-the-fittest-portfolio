function [w, variance] = solve_ga(mu, Sigma, k, target, maxGenerations, stallGenerations)
%SOLVE_GA Cardinality-constrained portfolio via a genetic algorithm.
%   [w, variance] = SOLVE_GA(mu, Sigma, k, target) searches for the set of
%   exactly k assets whose minimum-variance portfolio (for the given target
%   return) has the lowest variance.
%
%   Hybrid scheme: the GA explores the combinatorial part (which k assets),
%   and each individual is scored by solving the continuous QP for its
%   weights exactly (MIN_VARIANCE_WEIGHTS).
%
%   Encoding: an individual is a row of k distinct asset indices in 1..N.
%   Operators: random initial subsets, R3 (random respectful recombination)
%   crossover and swap mutation.
%
%   SOLVE_GA(..., maxGenerations, stallGenerations) overrides the defaults
%   (100 and 10). w is the full N-by-1 weight vector; variance is Inf if no
%   feasible portfolio was found.

if nargin < 5, maxGenerations   = 100; end
if nargin < 6, stallGenerations = 10;  end

n = numel(mu);
fitness = @(x) min_variance_weights(x, mu, Sigma, target);

options = optimoptions('ga', ...
    'PopulationType',      'custom', ...
    'PopulationSize',      2 * n, ...
    'EliteCount',          2, ...
    'CreationFcn',         @(nvars, ~, o) ga_create_population(nvars, n, o.PopulationSize), ...
    'CrossoverFcn',        @(parents, ~, nvars, ~, ~, pop) ga_crossover_r3(parents, nvars, pop), ...
    'MutationFcn',         @(parents, ~, ~, ~, ~, ~, pop) ga_mutate_swap(parents, n, pop), ...
    'MaxGenerations',      maxGenerations, ...
    'MaxStallGenerations', stallGenerations, ...
    'Display',             'off');

subset = ga(fitness, k, [], [], [], [], [], [], [], options);

w = zeros(n, 1);
[variance, ws] = min_variance_weights(sort(subset), mu, Sigma, target);
if isfinite(variance)
    w(sort(subset)) = ws;
end
end

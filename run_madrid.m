%RUN_MADRID Five Madrid Stock Exchange stocks, cardinality 3.
%   Compares the exact solution (full enumeration of the 10 possible
%   3-stock subsets) with the genetic algorithm on one target return, then
%   draws both frontiers.

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));

[mu, Sigma, names] = load_price_series(fullfile(root, 'data', 'madrid'));
k = 3;
target = 0.0015;

[wExact, vExact] = solve_exhaustive(mu, Sigma, k, target);
[wGA,    vGA]    = solve_ga(mu, Sigma, k, target);

fprintf('Target daily return: %.4f   cardinality: %d\n\n', target, k);
fprintf('%-18s %10s %10s\n', 'Stock', 'Exact', 'GA');
for i = 1:numel(names)
    fprintf('%-18s %10.4f %10.4f\n', names{i}, wExact(i), wGA(i));
end
fprintf('%-18s %10.3g %10.3g\n', 'Variance', vExact, vGA);

step = (max(mu) - min(mu)) / 30;
frontiers = {efficient_frontier(mu, Sigma, k, @solve_exhaustive, step), ...
             efficient_frontier(mu, Sigma, k, @solve_ga, step)};

figure
plot_frontiers(frontiers, {'Exhaustive', 'Genetic algorithm'}, ...
               'Madrid stocks, cardinality 3');

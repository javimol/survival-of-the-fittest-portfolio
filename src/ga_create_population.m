function population = ga_create_population(k, n, populationSize)
%GA_CREATE_POPULATION Random initial population of k-asset subsets.
%   Each row is k distinct asset indices drawn uniformly from 1..n.

population = zeros(populationSize, k);
for i = 1:populationSize
    population(i, :) = randperm(n, k);
end
end

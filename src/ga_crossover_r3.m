function kids = ga_crossover_r3(parents, k, population)
%GA_CROSSOVER_R3 Random Respectful Recombination (R3) for k-asset subsets.
%   The child keeps every asset the two parents share and fills the
%   remaining slots at random from the assets held by only one parent.
%   Parents are paired in order; when two paired parents hold the same
%   assets, the second one is swapped for the next different parent left
%   in the pool, so that crossover is not wasted on clones.

nKids = numel(parents) / 2;
kids  = zeros(nKids, k);
pool  = parents(:)';

for i = 1:nKids
    p1 = population(pool(1), :);

    j = 2;
    while j < numel(pool) && isequal(sort(population(pool(j), :)), sort(p1))
        j = j + 1;
    end
    p2 = population(pool(j), :);
    pool([2 j]) = pool([j 2]);
    pool(1:2) = [];

    common = intersect(p1, p2);
    others = setxor(p1, p2);
    fill   = others(randperm(numel(others), k - numel(common)));
    kids(i, :) = [common, fill];
end
end

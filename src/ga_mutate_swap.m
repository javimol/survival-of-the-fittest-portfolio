function kids = ga_mutate_swap(parents, n, population)
%GA_MUTATE_SWAP Swap one held asset for a random asset not yet held.

kids = population(parents, :);
k = size(kids, 2);

for i = 1:size(kids, 1)
    outside = setdiff(1:n, kids(i, :));
    if ~isempty(outside)
        kids(i, randi(k)) = outside(randi(numel(outside)));
    end
end
end

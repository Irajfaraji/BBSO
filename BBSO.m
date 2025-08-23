function [BestCost, BestPosition, ConvergenceCurve] = BBSO(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction)
% Boxelder Bug Search Optimization (BBSO): Optimization Inspired by the Ecology and Behavior of Boxelder Bugs
% This algorithm simulates the coordinated swarm behavior of Boxelder bugs searching for warm overwintering sites.
% Key concepts: Temperature as inverse of cost, coordinated following of better positions, population reduction over time.
%
% Based on the paper: "Boxelder Bugs Search Optimization: A Novel Reliable Tool for Optimizing Engineering Problems Through Bio-Inspired Ecology of Boxelder Bugs"
% Authors: Iraj Faraji Davoudkhani, Hossein Shayeghi, Abdollah Younesi
% Neural Computing and Applications (2025) ISSN: 0941-0643
% https://doi.org/
% Email: faraji.iraj@gmail.com
%
% Inputs:
% - nPop: Initial population size (recommended: 5*nVar).
% - MaxIt: Maximum number of function evaluations.
% - VarMin: Lower bound for variables.
% - VarMax: Upper bound for variables.
% - nVar: Number of decision variables.
% - CostFunction: Objective function to minimize (@(x) ...).
%
% Outputs:
% - BestCost: Best objective value found.
% - BestPosition: Best position (solution) found.
% - ConvergenceCurve: Convergence curve (best cost over evaluations).

VarSize = [1 nVar];  % Decision Variables Matrix Size

Nw = 0.5;  % Population reduction coefficient in Eq. (12) 

%% Initialization

% Empty Boxelder Bug Structure
BoxelderBug.Position = [];
BoxelderBug.Cost = [];

% Initialize Population Array
pop = repmat(BoxelderBug, nPop, 1);

% Initialize Best Solution
BestSol.Cost = inf;
BestSol.Position = [];

% Create Initial Boxelder Bugs (Random Positions within Bounds)
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    pop(i).Cost = CostFunction(pop(i).Position);
    if pop(i).Cost <= BestSol.Cost
        BestSol = pop(i);
    end
end

% Sort Initial Population by Cost (Ascending: Lower Cost = Higher Temperature)
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);

% Array to Hold Best Cost Values Over Iterations (dynamic size based on estimated iterations)
EstIt = ceil(MaxIt / (nPop * nPop / 2));  % Approximate iterations based on inner loop complexity
ConvergenceCurve = zeros(1, EstIt);
CountIter = 0;

%% BBSO Main Loop

it = nPop;  % Start counting evaluations from initial population
while it < MaxIt
    newpop = repmat(BoxelderBug, nPop, 1);
    
    % Compute Average Cost (Used for Temperature Scaling) (Eq. 9)
    Sfr = sum([pop.Cost]) / nPop;
    
    for i = 1:nPop
        newpop(i).Cost = inf;
        newsol.Position = zeros(1, nVar);
        
        % Coordinated Movement: Model Following Better Bugs Ahead (Eq. 3)
        for j = 2:i
            
            % Scaling Factor Based on Temperature (Inverse Cost Relation) (Eq. 8)
            fr = (pop(i).Cost / Sfr);
            
            % Update Position: Follow j and j-1 (Better Positions), with Decay Term (Eq. 4 – Eq. 7)
            newsol.Position = (pop(j).Position ...
                + rand(1, nVar) .* (pop(j).Position - pop(i).Position) ...
                + rand(1, nVar) .* (pop(j-1).Position - pop(j).Position) ...
                + (fr ^ -(it / nPop)) .* randn(1, nVar) .* (pop(j).Position / VarMax));  % Gaussian perturbation
            
            % Bound the Position
            newsol.Position = max(newsol.Position, VarMin);
            newsol.Position = min(newsol.Position, VarMax);
            
            % Evaluate New Position
            newsol.Cost = CostFunction(newsol.Position);
            it = it + 1;  % Increment Function Evaluation Counter
            
            % Update if Better
            if newsol.Cost <= newpop(i).Cost
                newpop(i) = newsol;
                if newpop(i).Cost <= BestSol.Cost
                    BestSol = newpop(i);
                end
            end
        end
        
    end
    
    % Population Reduction: Simulate Missing/Destruction (Reduce by Factor Proportional to Progress)(Eq. 12)
    nPop_new = nPop - Nw * nPop * (it / MaxIt);
    nPop_new = round(nPop_new);
    if nPop_new < nVar
        nPop_new = nVar;  % Minimum Population Size
    end
    
    % Merge Old and New Swarm (Eq. 2)
    pop = [pop; newpop];
    
    % Sort Combined Swarm by Cost
    [~, SortOrder] = sort([pop.Cost]);
    pop = pop(SortOrder);
    
    % Select Top nPop_new Bugs (Gather at Best Locations) (Eq. 11)
    pop = pop(1:nPop_new);
    nPop = nPop_new;
    
    % Increment Iteration
    CountIter = CountIter +1;
    
    % Store Best Cost
    ConvergenceCurve(CountIter) =  BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(CountIter) ': Best Cost = ' num2str(BestSol.Cost)]);

end

% Outputs
BestCost = BestSol.Cost;
BestPosition = BestSol.Position;

end


%% Test sampling-based methods for solving a class of nonconvex two-stage SP
% the problem takes the form of
% min  c'x + 1/S \sum_i x' Qi yi
% s.t. Ax <= b, Aeq x = beq,
%      Ai_joint [x; yi] <= bi_joint, for i = 1,...,S,
%      Ai_jointeq [x; yi] = bi_jointeq, for i = 1,...,S.
clear
clc

%% dimensions of the problem
I = 5; G = 5; J = 8;
n1 = I + G; n2 = I * J;           % n1: first-stage; n2: second-stage


%% options
clear options
sample_num = 50000;
options.x0 = [15 * ones(I, 1); 1; zeros(G - 1, 1)];

% the stopping tolerance
options.feastol = 1e-4;
options.feastol_abs = 1e-2;
options.opttol = 1e-4;
options.opttol_abs = 1e-2;

options.batch = 100;              % the number of samples in each batch
options.outer_maxiter = 20;       % the maximal number of outer iteration
options.inner_maxiter = 20;       % the maximal number of inner iteration
options.gamma = 15e0;             % (initial value) paramter of partial Moreau envelope
options.epsilon = 1e-1;           % (initial value) tolerance of the inner loop

%% set variance & growth rate of the sample size
varian = [0.5 2 5];
growth_rate = [100 200 400 800 1600]; %[100 400 1600 3200 6400 12800];
v = 1;
options.variance = varian(v);

for m = 1:length(growth_rate)
    options.sample_growth_rate = growth_rate(m);
    for seed = 1:100
        rng(seed,'twister');

        %% generate first-stage coefficient (obj & constraint)
        clear first_stage
        c = 5 * rand(I + G, 1); first_stage.c = c;
        lb_x = [8 * ones(I,1); zeros(G,1)]; ub_x = [15 * ones(I,1); ones(G,1)];
        lb_z = zeros(I * J, 1); ub_z = 5 * ones(I * J, 1);
        B = (c(1:I)' * lb_x(1:I) + min(c(I + 1:I + G)) + c(1:I)' * ub_x(1:I) + max(c(I + 1:I + G)))/2;
        first_stage.b = [B; ub_x; -lb_x];
        first_stage.A = [c'; eye(I + G); -eye(I + G)];
        first_stage.beq = 1;
        first_stage.Aeq = [zeros(1, I), ones(1, G)];


        %% generate second-stage coefficient (training & testing samples)
        clear train
        clear test

        train.num = sample_num; num = train.num;
        pd = makedist('Normal', 'mu', 1, 'sigma', options.variance);
        tq = truncate(pd, 2, 13);
        tpi = truncate(pd, 3, 10);
        td = truncate(pd, 2, ub_z(1,1));
        q = random(tq, I, num);
        pi = random(tpi, J, num);
        p = rand(G,num); p = p./sum(p);
        d = random(td, J, num);

        EI = repmat(speye(I), 1, J);
        temp = repmat({ones(1,I)}, 1, J);
        EJ = blkdiag(temp{:});
        Q = cell(1,num);
        parfor s = 1:num
            temp = q(:,s) - pi(:,s)';
            Q{s} = [zeros(I,n2); p(:,s) * temp(:)'];
        end
        train.Q = Q;
        train.bjoint = repmat([zeros(I,1); ub_z; -lb_z],1,num);
        temp1 = sparse(I+2*n2, n1); temp1(1:I,1:I) = -speye(I);
        train.Ajointx = repmat({temp1},num,1);
        train.Ajointy = repmat({[EI; speye(n2); -speye(n2)]},num,1);
        train.bjointeq = d;
        train.Ajointeqx = repmat({sparse(J,n1)},num,1);
        train.Ajointeqy = repmat({sparse(EJ)},num,1);

        test = train;

        %% deterministic equivalent
        N = size(first_stage.b, 1); Neq = size(first_stage.beq, 1);
        Njoint = size(train.bjoint, 1); Njointeq = size(train.bjointeq, 1);

        % inequality constraints
        A = sparse(N + Njoint * num, n1 + n2 * num);
        A(1:N, 1:n1) = first_stage.A;
        A(N + 1:end, :) = [cell2mat(train.Ajointx), blkdiag(train.Ajointy{:})];

        % equality constraints
        Aeq = sparse(Neq + Njointeq * num, n1 + n2 * num);
        Aeq(1:Neq, 1:n1) = first_stage.Aeq;
        Aeq(Neq + 1:end, :) = [cell2mat(train.Ajointeqx), blkdiag(train.Ajointeqy{:})];


        %% print dimensions of the problem
        fprintf('Dimensions of the test problem\n');
        fprintf('------------------------------\n');
        fprintf('        Stage1_      Stage2      Determinisitc Equivalent\n');
        fprintf('Scen| Rows  Cols | Rows  Cols | Rows  Cols\n');
        fprintf('%4.0f| %4.0f  %4.0f | %4.0f  %4.0f |', num, size(first_stage.b, 1) + size(first_stage.beq, 1),...
                n1, size(train.bjoint, 1) + size(train.bjointeq, 1), n2);
        fprintf(' %3.0f  %3.0f\n\n', size(first_stage.b, 1) + size(first_stage.beq, 1)...
                + num * (size(train.bjoint, 1) + size(train.bjointeq, 1)), n1 + n2 * num);
        fprintf('-------------------------------------------------------\n');

        %% call DPME based on current samples
        [x, info] = DPME(first_stage, train, test, A, Aeq, options);
        fprintf('******************************* variance = %2.1f | growth_rate = %d | seed = %d *********************************\n\n', varian(v), growth_rate(m), seed);


        %% save history data for replications
        his.Outer_iter(m, seed) = info.Outer_iter;
        his.Total_iter(m, seed) = info.Total_iter;
        his.KKT_abs(m, seed) = info.KKT_abs;
        his.KKT_rel(m, seed) = info.KKT_rel;
        his.obj(m, seed) = info.obj;
        his.time(m, seed) = info.time;
    end
end


%% process the results (average & standard deviation)
avg.Outer_iter = mean(his.Outer_iter, 2);
avg.Total_iter = mean(his.Total_iter, 2);
avg.KKT_abs = mean(his.KKT_abs, 2);
avg.KKT_rel = mean(his.KKT_rel, 2);
avg.obj = mean(his.obj, 2);
avg.time = mean(his.time, 2);

sd.Outer_iter = std(his.Outer_iter, 0, 2);
sd.Total_iter = std(his.Total_iter, 0, 2);
sd.KKT_abs = std(his.KKT_abs, 0, 2);
sd.KKT_rel = std(his.KKT_rel, 0, 2);
sd.obj = std(his.obj, 0, 2);
sd.time = std(his.time, 0, 2);

save Avg_sampling.mat avg
save Std_sampling.mat sd
save History_sampling.mat his


%% print the summary table
fprintf('\n ======================= Summary of sampling-based DPME =======================\n');
for m = 1:length(growth_rate)
    fprintf('%d Avg: %1.0f | %1.0f | %1.6f  %1.6f | %2.3f | %4.0f |\n', growth_rate(m), roundn(avg.Outer_iter(m), 0),...
            roundn(avg.Total_iter(m), 0), roundn(avg.KKT_abs(m), -6), roundn(avg.KKT_rel(m), -6), roundn(avg.obj(m), -3), roundn(avg.time(m), 0));
    fprintf('     Std: %1.1f | %1.0f | %1.7f  %1.8f | %2.2f | %4.1f |\n', roundn(sd.Outer_iter(m), 1),...
            roundn(sd.Total_iter(m), 0), roundn(sd.KKT_abs(m), -7), roundn(sd.KKT_rel(m), -8), roundn(sd.obj(m), -2), roundn(sd.time(m), -1));
    fprintf('---------------------------------------------------------------------------\n');
end


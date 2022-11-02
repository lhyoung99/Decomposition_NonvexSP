%% Test different methods for solving a class of nonconvex two-stage SP
% the problem takes the form of
% min  c'x + 1/S \sum_i x' Qi yi
% s.t. Ax <= b, Aeq x = beq,
%      Ai_joint [x; yi] <= bi_joint, for i = 1,...,S,
%      Ai_jointeq [x; yi] = bi_jointeq, for i = 1,...,S.
clear
clc
diary('Output_diary.txt');
diary on;


%% dimensions of the problem
I = 5; G = 5; J = 8;
n1 = I + G; n2 = I * J;           % n1: first-stage; n2: second-stage


%% options
clear options

% specify algorithms/solvers
% 1:Knitro-direct  2:Knitro-CG  3:IPOPT  4:DPME
alg = 4;
sample_num = [1000 5000 10000 30000 80000 120000];
options.x0 = [15 * ones(I, 1); 1; zeros(G - 1, 1)];

% the stopping tolerance (apply to all the algorithms, except IPOPT)
options.feastol = 1e-4;
options.feastol_abs = 1e-2;
options.opttol = 1e-4;
options.opttol_abs = 1e-2;

% the stopping tolerance of IPOPT
options.ipopt_tol = 1e-4;

% extra parameters of DPME
options.batch = 100;              % the number of samples in each batch
options.outer_maxiter = 20;       % the maximal number of outer iteration
options.inner_maxiter = 20;       % the maximal number of inner iteration
options.gamma = 15e0;             % (initial value) paramter of partial Moreau envelope
options.epsilon = 1e-1;           % (initial value) tolerance of the inner loop


for m = 1:length(sample_num)
    for seed = 1:100
        rng(seed,'twister');

        %% generate first-stage coefficients (obj & constraint)
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

        train.num = sample_num(m); num = train.num;
        pd = makedist('Normal', 'mu', 1, 'sigma', 5);
        tq = truncate(pd, 2, 4);
        tpi = truncate(pd, 3, 5);
        td = truncate(pd, 2, ub_z(1, 1));
        q = random(tq, I, num);
        pi = random(tpi, J, num);
        p = rand(G, num); p = p./sum(p);
        d = random(td, J, num);

        EI = repmat(speye(I), 1, J);
        temp = repmat({ones(1, I)}, 1, J);
        EJ = blkdiag(temp{:});
        Q = cell(1, num);
        for s = 1:num
            temp = q(:, s) - pi(:, s)';
            Q{s} = [zeros(I, n2); p(:, s) * temp(:)'];
        end
        train.Q = Q;
        train.bjoint = repmat([zeros(I, 1); ub_z; -lb_z], 1, num);
        temp1 = sparse(I + 2 * n2, n1); temp1(1:I, 1:I) = -speye(I);
        train.Ajointx = repmat({temp1}, num, 1);
        train.Ajointy = repmat({[EI; speye(n2); -speye(n2)]}, num, 1);
        train.bjointeq = d;
        train.Ajointeqx = repmat({sparse(J, n1)}, num, 1);
        train.Ajointeqy = repmat({sparse(EJ)}, num, 1);

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


        %% call algorithms/solvers
        if alg == 1
            [x1, info] = Knitro_QP(first_stage, train, test, A, Aeq, options);
        elseif alg == 2
            [x2, info] = Knitro_QP_CG(first_stage, train,test, A, Aeq, options);
        elseif alg == 3
            [x3, info] = Ipopt(first_stage, train, test, A, Aeq, options);
        elseif alg == 4
            [x4, info] = DPME(first_stage, train, test, A, Aeq, options);
        else
            alg = -1;
            fprintf('Error! Please enter integer 1 - 4 for "alg"\n');
            break;
        end
        % fprintf('Objective value: %4.4f   |   %4.4f   |   %4.4f   |   %4.4f\n',Value(first_stage,test,A,Aeq,x1), Value(first_stage,test,A,Aeq,x2), Value(first_stage,test,A,Aeq,x3), Value(first_stage,test,A,Aeq,x4));
        fprintf('*************************************** sample size = %d || seed = %d ***************************************\n\n', sample_num(m), seed);


        %% save history for replications
        his.iter(m, seed) = info.iter;
        his.KKT_abs(m, seed) = info.KKT_abs;
        his.KKT_rel(m, seed) = info.KKT_rel;
        his.obj(m, seed) = info.obj;
        his.time(m, seed) = info.time;
    end
    if alg == -1
        break;
    end
end

if alg ~= -1
    %% process the results (average & standard deviation)
    avg.iter = mean(his.iter, 2);
    avg.KKT_abs = mean(his.KKT_abs, 2);
    avg.KKT_rel = mean(his.KKT_rel, 2);
    avg.obj = mean(his.obj, 2);
    avg.time = mean(his.time, 2);

    sd.iter = std(his.iter, 0, 2);
    sd.KKT_abs = std(his.KKT_abs, 0, 2);
    sd.KKT_rel = std(his.KKT_rel, 0, 2);
    sd.obj = std(his.obj, 0, 2);
    sd.time = std(his.time, 0, 2);

    save Avg.mat avg
    save Std.mat sd
    save History.mat his


    %% print the summary table
    fprintf('\n ======================= Summary of the current method =======================\n');
    for m = 1:length(sample_num)
        fprintf('%d Avg: %1.0f | %1.6f  %1.6f | %2.3f | %4.0f |\n', sample_num(m), roundn(avg.Total_iter(m), 0),...
                roundn(avg.KKT_abs(m), -6), roundn(avg.KKT_rel(m), -6), roundn(avg.obj(m), -3), roundn(avg.time(m), 0));
        fprintf('     Std: %1.0f | %1.7f  %1.8f | %2.2f | %4.1f |\n', roundn(sd.Total_iter(m), 0),...
                roundn(sd.KKT_abs(m), -7), roundn(sd.KKT_rel(m), -8), roundn(sd.obj(m), -2), roundn(sd.time(m), -1));
        fprintf('---------------------------------------------------------------------------\n');
    end
end
diary off;


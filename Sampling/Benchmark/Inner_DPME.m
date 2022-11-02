%% Inner loop of DPME
function [x_out, FeasErr, OptErr, output] = Inner_DPME(param, first_stage, train, test, sample_size, A, Aeq, x)
clear output
maxiter = param.inner_maxiter;
gamma = param.gamma;
epsilon = param.epsilon;
batch = param.batch;
b = first_stage.b; beq = first_stage.beq;
Q = cell2mat(train.Q);
Ax = first_stage.A; Aeqx = first_stage.Aeq;
Ajointx = train.Ajointx; Ajointy = train.Ajointy;
Ajointeqx = train.Ajointeqx; Ajointeqy = train.Ajointeqy;
bjoint = train.bjoint; bjointeq = train.bjointeq;

% set parameters for "KKT_test" function
KKT_param.rel = param.opttol;
KKT_param.abs = param.opttol_abs;
KKT_param.batch = batch;

% dimension of deicison variables
n1 = size(first_stage.c, 1);
n2 = size(train.Q{1}, 2);

% number of constraints
N = size(b, 1); Neq = size(beq, 1);
Njoint = size(bjoint, 1); Njointeq = size(bjointeq, 1);

%% solve subprolems in batches
sub.Q = sparse(1:n1, 1:n1, 1/(2 * gamma), n1 + n2 * batch, n1 + n2 * batch);
sub.lb = -inf(n1 + n2 * batch, 1);
sub.sense = [repmat('<', N, 1); repmat('=', Neq, 1); repmat('<', Njoint * batch, 1); repmat('=', Njointeq * batch, 1)];
params.outputflag = 0;
params_sub = params;
x_his = zeros(n1, maxiter); x_his(:, 1) = x;
for i = 1:maxiter
    master_obj = zeros(n1, 1);
    prox_x = []; y = [];
    parfor s = 1:sample_size/batch
        sub_local = sub;
        sub_local.obj = [-x'/gamma, x' * Q(:, batch * (s - 1) * n2 + 1:batch * s * n2)/batch];
        sub_local.A = sparse(N + Neq + (Njoint + Njointeq) * batch, n1 + n2 * batch);
        sub_local.A(1:N, 1:n1) = Ax; sub_local.A(N + 1:N + Neq, 1:n1) = Aeqx;
        sub_local.A(N + Neq + 1:end, 1:n1) = [cell2mat(Ajointx(batch * (s-1) + 1:batch * s));
                                cell2mat(Ajointeqx(batch * (s-1) + 1:batch * s))];
        sub_local.A(N + Neq + 1:end, n1 + 1:n1 + n2 * batch) = [blkdiag(Ajointy{batch * (s - 1) + 1:batch * s});...
                                                                blkdiag(Ajointeqy{batch * (s - 1) + 1:batch * s})];
        temp3 = bjoint(:, batch * (s - 1) + 1:batch *s); temp4 = bjointeq(:, batch * (s - 1) + 1:batch * s);
        sub_local.rhs = [b; beq; temp3(:); temp4(:)];
        result_sub = gurobi(sub_local, params_sub);

        % save the solution of each subproblems
        prox_xs = result_sub.x(1:n1); prox_x(:, s) = prox_xs;
        ys = result_sub.x(n1+1:n1+n2*batch); y(:, s) = ys;

        % update subdifferential
        subgrad = - Q(:, batch * (s - 1) * n2 + 1:batch * s * n2) * ys/batch;

        % update coefficient
        master_obj = master_obj - (result_sub.x(1:n1)/gamma + subgrad)/(sample_size/batch);
    end
    

    %% solve the master problem
    master.A = sparse([Ax; Aeqx]);
    master.Q = sparse(1:n1, 1:n1, 1/(2 * gamma), n1, n1);
    master.obj = first_stage.c' + master_obj';
    master.rhs = [b; beq];
    master.lb = -inf(n1, 1);
    master.sense = [repmat('<', size(b, 1), 1); repmat('=', size(beq, 1), 1)];
    result_master = gurobi(master, params);
    x_new = result_master.x;

    
    %% check stopping criteria
    stop_flag = (i > 1 && (norm(x_new - x) < epsilon * gamma)) || (i == maxiter);
    if stop_flag || mod(i,20) == 0
        %% test feasibility & optimality error
        [FeasErr, OptErr, Qy, tau2] = KKT_test(first_stage, test, A, Aeq, x_new, KKT_param);
        output.tau2 = tau2;
        output.i = i;
        output.adj_dis = norm(x_new - x, inf); % the distance between the last two iterations of the master problem
        output.prox_avg = sum(sum(abs(prox_x - x)))/sample_size;
        output.prox_wor = norm(prox_x - x, 1);
        output.obj = first_stage.c' * x_new + x_new' * Qy/test.num;
        
        
        %% break the inner loop or print information (for every 20 inner iterations)
        if stop_flag
            break;
        else
            fprintf(' %17.0f',i);
            fprintf('   %22.2e    %3.2e  %3.2e  ', output.adj_dis/param.gamma, output.prox_avg, output.prox_wor);
            fprintf(' %3.2e  %3.2e       %3.2e     %3.2e ', FeasErr, OptErr, FeasErr/param.tau1, OptErr/tau2);
            fprintf(' %- 8.5e   | %3.2e   %3.2e', output.obj, norm(L_nabla_x, inf), CompleErr);
            fprintf('\n\n');
        end
    end
    x = x_new;
    x_his(:,i+1) = x_new;
end
    x_out = x_new;
end


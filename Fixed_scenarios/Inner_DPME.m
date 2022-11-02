%% Inner loop of DPME
function [x_out, y_out, FeasErr, OptErr, output] = Inner_DPME(param, first_stage, train, sample_size, A, Aeq, x)
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
x_his = zeros(n1, maxiter); x_his(:,1) = x;
for i = 1:maxiter
    master_obj = zeros(n1, 1);
    lambda = []; mu = [];
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
        ys = result_sub.x(n1 + 1:n1 + n2 * batch); y(:, s) = ys;

        % update subdifferential
        subgrad = - Q(:, batch * (s - 1) * n2 + 1:batch * s * n2) * ys/batch;

        % update coefficient
        master_obj = master_obj - (result_sub.x(1:n1)/gamma + subgrad)/(sample_size/batch);

        % update multiplier information
        Alambda_x(:,s) = Ax' * result_sub.pi(1:N) * batch;
        Amu_x(:,s) = Aeqx' * result_sub.pi(N+1:N+Neq) * batch;
        lambda(:,s) = result_sub.pi(N + Neq + 1:N + Neq + Njoint * batch) * batch;
        mu(:,s) = result_sub.pi(N + Neq + Njoint * batch + 1:end) * batch;
    end
    
    % multipliers & second-stage optimal solutions
    lambda = reshape(lambda, Njoint, sample_size);
    mu = reshape(mu, Njointeq, sample_size);
    y = reshape(y, n2, sample_size);

    Alambda = cell2mat(Ajointx)' * lambda(:);
    Amu = cell2mat(Ajointeqx)' * mu(:);

    %% solve the master problem
    master.A = sparse([Ax; Aeqx]);
    master.Q = sparse(1:n1, 1:n1, 1/(2 * gamma), n1, n1);
    master.obj = first_stage.c' + master_obj';
    master.rhs = [b; beq];
    master.lb = -inf(n1, 1);
    master.sense = [repmat('<', size(b,1), 1); repmat('=', size(beq, 1), 1)];
    result_master = gurobi(master, params);
    x_new = result_master.x;
    alpha1 = result_master.pi(1:N);
    alpha2 = result_master.pi(N + 1:N + Neq);


    %% check stopping criteria
    stop_flag = (i > 1 && (norm(x_new - x) < epsilon * gamma)) || (i == maxiter);
    if stop_flag || mod(i,20) == 0
        %% feasibility error
        bAx = b - first_stage.A * x_new;
        bAxeq = first_stage.beq - first_stage.Aeq * x_new;
        bjointAx = bjoint(:) - A(N+1:end,:) * [x_new; y(:)];
        bjointAxeq = bjointeq(:) - Aeq(Neq+1:end,:) * [x_new; y(:)];

        FeasErr = max([0; -bAx; abs(bAxeq)]);
        FeasErr = max([FeasErr; - bjointAx; abs(bjointAxeq)]);

        
        %% optimality error
        Qy = Q * y(:); QTx = Q' * x_new;
        L_nabla_x = first_stage.c + Qy/sample_size - first_stage.A' * alpha1 - first_stage.Aeq' * alpha2...
                    - Alambda/sample_size - Amu/sample_size - sum(Alambda_x,2)/sample_size - sum(Amu_x,2)/sample_size;
        L_nabla_y = QTx/sample_size - cell2mat(arrayfun(@(i)lambda(:,i)' * Ajointy{i}, 1:sample_size, 'un', 0))'/sample_size...
                    - cell2mat(arrayfun(@(i)mu(:,i)' * Ajointeqy{i}, 1:sample_size, 'un', 0))'/sample_size;

        CompleErr1 = max([min([-alpha1' .* bAx'; -alpha1'; bAx']),...
                          min([abs(alpha2' .* bAxeq'); abs(alpha2'); abs(bAxeq)])]);

        CompleErr2 = max([min([-lambda(:)'/sample_size .* bjointAx'; -lambda(:)'/sample_size; bjointAx']),...
                          min([abs(mu(:)'/sample_size .* bjointAxeq'); abs(mu(:)')/sample_size; abs(bjointAxeq')])]);

        OptErr = max([norm([L_nabla_x; L_nabla_y], inf), CompleErr1, CompleErr2]);

        output.i = i;
        output.adj_dis = norm(x_new - x, inf); % the distance between the last two iterations of the master problem
        output.prox_avg = sum(sum(abs(prox_x - x)))/sample_size;
        output.prox_wor = norm(prox_x - x,1);
        output.obj = first_stage.c' * x_new + x_new' * Qy/sample_size;


        %% break the inner loop or print information (for every 20 inner iterations)
        if stop_flag
            break;
        else
            g = [first_stage.c + Qy/train.num; QTx/train.num];
            tau2 = max(1, norm(g, inf));
            fprintf(' %17.0f',i);
            fprintf('   %22.2e    %3.2e  %3.2e  ', output.adj_dis/param.gamma, output.prox_avg, output.prox_wor);
            fprintf(' %3.2e  %3.2e       %3.2e     %3.2e ', FeasErr,OptErr, FeasErr/param.tau1, OptErr/tau2);
            fprintf(' %- 8.5e   | %3.2e   %3.2e   %3.2e   %3.2e', Value(first_stage, train, A, Aeq, x), norm(L_nabla_x, inf), norm(L_nabla_y(:), inf), CompleErr1, CompleErr2);
            fprintf(' %- 8.5e   | %3.2e   %3.2e   %3.2e   %3.2e', output.obj,norm(L_nabla_x, inf), norm(L_nabla_y(:), inf), CompleErr1, CompleErr2);
            fprintf('\n\n');
        end
    end
    x = x_new;
    x_his(:,i+1) = x_new;
end

%% Return candidate solution
x_out = x_new;
y_out = y;
end


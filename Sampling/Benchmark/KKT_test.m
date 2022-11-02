%% estimate feasibility & optimality error via KKT system
function [FeasErr, OptErr, Qy, tau2] = KKT_test(first_stage, test, A, Aeq, x, param)
    sample_size = test.num;
    
    % dimension of deicison variables
    n1 = size(first_stage.c, 1);
    n2 = size(test.Q{1}, 2);
    
    % number of constraints
    N = size(first_stage.b, 1); Neq = size(first_stage.beq, 1);
    Njoint = size(test.bjoint, 1); Njointeq = size(test.bjointeq, 1);
    
    batch = param.batch;
    Q = cell2mat(test.Q);
    Ajointx = test.Ajointx; Ajointy = test.Ajointy;
    Ajointeqx = test.Ajointeqx; Ajointeqy = test.Ajointeqy;
    bjoint = test.bjoint; bjointeq = test.bjointeq;
    
    %% solve the second-stage problem for all scenarios (in batches) at current x
    model.lb = -inf(n2 * batch,1);
    model.sense = [repmat('<', Njoint * batch, 1); repmat('=', Njointeq * batch, 1)];
    params.outputflag = 0;
    parfor s = 1:sample_size/batch
        model_local = model;
        model_local.obj = Q(:, batch * (s-1) * n2 + 1:batch * s * n2)' * x/batch;
        model_local.A = [A(N + batch * (s-1) * Njoint + 1:N + batch * s * Njoint, n1 + batch * (s-1) * n2 + 1:n1 + batch * s * n2);...
                         Aeq(Neq + batch * (s - 1) * Njointeq + 1:Neq + batch * s * Njointeq, n1 + batch * (s-1) * n2 + 1:n1 + batch * s * n2)];
        temp3 = bjoint(:, batch * (s - 1) + 1:batch * s);
        temp4 = bjointeq(:, batch * (s - 1) + 1:batch * s);
        model_local.rhs = [temp3(:) - A(N + batch * (s - 1) * Njoint + 1:N + batch * s * Njoint, 1:n1) * x;...
                           temp4(:) - Aeq(Neq + batch * (s - 1) * Njointeq + 1:Neq + batch * s * Njointeq, 1:n1) * x];
        result = gurobi(model_local, params);
        y(:, s) = result.x;
        lambda(:, s) = result.pi(1:Njoint * batch)/(sample_size/batch);
        mu(:, s) = result.pi(Njoint * batch + 1:end)/(sample_size/batch);
    end
    
    % multipliers & second-stage optimal solutions
    y = y(:);
    lambda = lambda(:);
    mu = mu(:);
    
    %% estimate optimal multipliers by minimizing the KKT residual
    Qy = Q * y; QTx = Q' * x;
    g = [first_stage.c + Qy/test.num; QTx/test.num];
    tau2 = max(1, norm(g, inf));
    
    bAx = first_stage.b - first_stage.A * x;
    bAxeq = first_stage.beq - first_stage.Aeq * x;
    bjointAx = bjoint(:) - cell2mat(Ajointx) * x - cell2mat(arrayfun(@(i)(Ajointy{i} * y((i - 1) * n2 + 1:i * n2))', 1:test.num, 'un', 0))';
    bjointAxeq = bjointeq(:) - cell2mat(Ajointeqx) * x - cell2mat(arrayfun(@(i)(Ajointeqy{i} * y((i - 1) * n2 + 1:i * n2))', 1:test.num, 'un', 0))';
    constant = first_stage.c + Qy/test.num - cell2mat(test.Ajointx)' * lambda - cell2mat(test.Ajointeqx)' * mu;
    
    multip.Q = sparse([first_stage.A; first_stage.Aeq] * [first_stage.A', first_stage.Aeq']);
    multip.lb = -inf(N + Neq,1);
    multip.ub = [zeros(N, 1); inf(Neq, 1)];
    multip.obj = -2 * [first_stage.A; first_stage.Aeq] * constant;
    multip.A = sparse([-bAx', zeros(1, Neq)]);
    multip.rhs = min(param.abs, param.rel * tau2);
    multip.sense = '<';
    result_multip = gurobi(multip, params);
    alpha1 = result_multip.x(1:N);
    alpha2 = result_multip.x(N + 1:end);

    
    %% feasibility error
    FeasErr = max([0; A * [x; y] - [first_stage.b; bjoint(:)]; abs(Aeq * [x; y] - [first_stage.beq; bjointeq(:)])]);

    
    %% optimality error
    L_nabla_x = constant - first_stage.A' * alpha1 - first_stage.Aeq' * alpha2;
    L_nabla_y = QTx/test.num - cell2mat(arrayfun(@(i)lambda((i - 1) * Njoint + 1:i * Njoint)' * Ajointy{i},1:test.num, 'un', 0))'...
                        - cell2mat(arrayfun(@(i)mu((i - 1) * Njointeq + 1:i * Njointeq)' * Ajointeqy{i}, 1:test.num, 'un', 0))';
    CompleErr1 = max([min([-alpha1' .* bAx'; -alpha1'; bAx']),...
                      min([-lambda' .* bjointAx'; -lambda'; bjointAx'])]);
    CompleErr2 = max([min([abs(alpha2' .* bAxeq'); abs(alpha2'); abs(bAxeq')]),...
                      min([abs(mu' .* bjointAxeq'); abs(mu'); abs(bjointAxeq')])]);
    
    OptErr = max([norm([L_nabla_x; L_nabla_y], inf), CompleErr1, CompleErr2]);
end

%% Decomposition via Partial Moreau Envelope
function [x_out, info] = DPME(first_stage, train, test, A, Aeq, options)
%% dimension of first- and second-stage variables
n1 = size(first_stage.c, 1);
n2 = size(train.Q{1}, 2);


%% parameters of the outer loop
feastol = 1e-6; feastol_abs = 1e-3;
opttol = 1e-6; opttol_abs = 1e-3;
outer_maxiter = 2000;
x0 = zeros(n1, 1);
plotyes = 0;
if isfield(options, 'feastol'); feastol = options.feastol; end
if isfield(options, 'feastol_abs'); feastol_abs = options.feastol_abs; end
if isfield(options, 'opttol'); opttol = options.opttol; end
if isfield(options, 'opttol_abs'); opttol_abs = options.opttol_abs; end
if isfield(options, 'outer_maxiter'); outer_maxiter = options.outer_maxiter; end
if isfield(options, 'x0'); x0 = options.x0; end


%% parameters of the inner loop
clear param
param.inner_maxiter = 2000;
param.gamma = 0.1;
param.epsilon = 0.1;
param.batch = 1;
if isfield(options,'inner_maxiter'); param.inner_maxiter = options.inner_maxiter; end
if isfield(options,'gamma'); param.gamma = options.gamma; end
if isfield(options,'epsilon'); param.epsilon = options.epsilon; end
if isfield(options,'batch'); param.batch = options.batch; end


%% compute the scaling factor of the feasibility error
tau1 = max([1; first_stage.b - first_stage.A * x0; abs(first_stage.beq - first_stage.Aeq * x0)]);
tau1 = max([tau1; train.bjoint(:) - [cell2mat(train.Ajointx), cell2mat(train.Ajointy)] * [x0;zeros(n2,1)];...
                abs(train.bjointeq(:) - [cell2mat(train.Ajointeqx), cell2mat(train.Ajointeqy)] * [x0;zeros(n2,1)])]);
param.tau1 = tau1;


%% Outer loop
x = zeros(n1, outer_maxiter); x(:,1) = options.x0;
obj = zeros(1, outer_maxiter); obj(1) = Value(first_stage, test, A, Aeq, x(:,1));
KKT_OptErr = zeros(1, outer_maxiter);

fprintf('=====================================================\n');
fprintf('      Decomposition via Partial Moreau Envelope      \n');
fprintf('=====================================================\n\n');
fprintf(' outer_iter    [gamma   epsilon]  dis/gamma    Prox_avg  Prox_wor    FeasErr    OptErr    FeasErr_rel   OptErr_rel   Objective    inner_iter   time\n');
fprintf('------------  ------------------------------  --------------------  -------------------  --------------------------------------  --------------------\n');
inneriter_total = 0;    % the total number of iterations
flag = 0;               % the number of (outer) iterations since last shrinkage of gamma
tic
for nu = 1:outer_maxiter
    gamma_old = param.gamma;
    sample_size = train.num;
    tstart = clock;
    [x(:, nu + 1), y, FeasErr, OptErr, output] = Inner_DPME(param, first_stage, train, sample_size, A, Aeq , x(:, nu));
    obj(nu + 1) = output.obj;
    KKT_OptErr(nu + 1) = OptErr;
    inneriter_total = inneriter_total + output.i;
    
    %% update the scaling factor of the optimality error
    g = [first_stage.c + cell2mat(train.Q) * y(:)/train.num; cell2mat(train.Q)' * x(:,nu + 1)/train.num];
    tau2 = max(1, norm(g, inf));
    fprintf('  %3.0f       [%3.2e  %3.2e]', nu, param.gamma, param.epsilon);
    fprintf('   %3.2e    %3.2e  %3.2e  ',output.adj_dis/param.gamma, output.prox_avg, output.prox_wor);
    fprintf(' %3.2e  %3.2e       %3.2e     %3.2e   %3.4e', FeasErr, OptErr, FeasErr/tau1, OptErr/tau2, output.obj);
    fprintf(' %11.0f  %5.1f\n', output.i, etime(clock, tstart));

    
    %% check stopping criteria
    feastol_all = min(tau1 * feastol, feastol_abs);
    opttol_all = min(tau2 * opttol, opttol_abs);
    KKTtol = max(feastol_all, opttol_all);
    if FeasErr <= feastol_all && abs(obj(nu) - obj(nu + 1))/max(1, abs(obj(nu))) <= 1e-4
        break
    end
    
    
    %% update gamma & epsilon
    if nu <= 1
        param.gamma = 0.6 * param.gamma;
        param.epsilon = max(0.1 * KKTtol, min(0.1 * param.epsilon, 0.4 * FeasErr/tau1));
    elseif flag > 0
        if FeasErr >= feastol_all
            param.gamma = max([0.8 * param.gamma * feastol_all/FeasErr, 7e-3 * param.gamma,...
                               min([0.1 * param.gamma, 0.1/output.prox_wor, 10000 * output.adj_dis])]);
        else
            param.epsilon = 0.3 * param.epsilon;
        end
    elseif flag == 0
        if param.gamma * param.epsilon > output.adj_dis
            param.epsilon = max(0.1 * KKTtol, min(0.1 * param.epsilon, FeasErr/tau1));
        end
        if FeasErr >= feastol_all
            param.gamma = max([0.8 * param.gamma * feastol_all/FeasErr, 7e-3 * param.gamma,...
                               min([0.02 * param.gamma, 0.1/output.prox_wor, 10000 * output.adj_dis])]);
        end
    elseif flag == -1
        % indicates that we do not solve approximation (for current gamma) accurately enough
        if param.gamma * param.epsilon > output.adj_dis
            param.epsilon = 0.1 * param.epsilon;
        end
        flag = 0;
    end
    
    
    %% update flag
    if gamma_old == param.gamma
        flag = flag + 1;
    elseif gamma_old < param.gamma
        flag = -1;
    else
        flag = 0;
    end
end
toc
x_out = x(:, nu + 1);
info.time = toc;
info.iter = inneriter_total;
info.KKT_abs = max(FeasErr, OptErr);
info.KKT_rel = max(FeasErr/tau1, OptErr/tau2);
info.obj = obj(nu + 1);
fprintf('- CPU time: %4.4f\n', info.time);
fprintf('- Total inner iterates: %4.0f\n', inneriter_total)
if plotyes == 1
    plot(0:nu, obj(1:nu + 1), 'LineWidth',2)
    xlabel('The number of outer iteration')
    ylabel('Value of Objective function')
end
end


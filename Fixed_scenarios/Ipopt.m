function [x_out, info] = Ipopt(first_stage, train, test, A, Aeq, options_outer)

n1 = size(first_stage.c, 1); n2 = size(train.Q{1}, 2);
N = size(A, 1); Neq = size(Aeq, 1);
  
x0 = [options_outer.x0; zeros(n2 * train.num, 1)];  % initial point
options.lb = -inf(1, n1 + n2 * train.num);          % lower bound on the variables
options.ub = inf(1, n1 + n2 * train.num);           % upper bound on the variables

options.cl = [-inf(1,N), zeros(1, Neq)];            % lower bound on the constraints
options.cu = zeros(1, N + Neq);                     % upper bound on the constraints
  
% Set the IPOPT options.
% options.ipopt.print_level      = 3;
% options.ipopt.print_timing_statistics = 'yes';
% options.ipopt.print_user_options = 'yes';
% options.ipopt.hessian_approximation = 'limited-memory';
% options.ipopt.limited_memory_update_type = 'bfgs' ;
% options.ipopt.mu_strategy           = 'adaptive';
% options.ipopt.mu_oracle = 'probing'; % {quality-function}, loqo, [probing]
% options.ipopt.max_iter              = 400;
options.ipopt.jac_c_constant   = 'yes';
options.ipopt.jac_d_constant   = 'yes';
options.ipopt.hessian_constant = 'yes';
options.ipopt.tol              = options_outer.ipopt_tol;
options.ipopt.linear_solver    = 'mumps'; % ma57, pardiso
  
f = [first_stage.c; zeros(n2 * train.num, 1)];
H = sparse(cat(1, cat(2,sparse(n1,n1),cell2mat(train.Q) * 2/train.num), sparse(n2 * train.num,n1 + n2 * train.num)));
H = sparse((H + H')/2);
b = [first_stage.b; train.bjoint(:)];
beq = [first_stage.beq; train.bjointeq(:)];
  
% Set up the auxiliary data.
options.auxdata.H = H;
options.auxdata.f = f;
options.auxdata.A = A;
options.auxdata.Aeq = Aeq;
options.auxdata.b = b;
options.auxdata.beq = beq;
  
% The callback functions.
funcs.objective         = @objective;
funcs.constraints       = @constraints;
funcs.gradient          = @gradient;
funcs.jacobian          = @jacobian;
funcs.jacobianstructure = @(x) sparse([A; Aeq] & 1);
funcs.hessian           = @hessian;
funcs.hessianstructure  = @hessian_pattern;
  
% run IPOPT.
[x, info] = ipopt_auxdata(x0,funcs,options);
lambda1 = info.lambda(1:N); lambda2 = info.lambda(N+1:N+Neq);

% compute the optimality & feasibility error
tau1 = max([1; A * x0 - b; abs(Aeq * x0 - beq)]);
tau2 = max(1, norm(H * x + f, inf));

bAx = b - A * x;
bAx_eq = beq - Aeq * x;
FeasErr = max([0; -bAx; abs(bAx_eq)]);
L_nabla = H * x + f + A' * lambda1 + Aeq' * lambda2;
CompleErr1 = max(min([lambda1' .* bAx'; lambda1'; bAx']));
CompleErr2 = max(min([abs(lambda2' .* bAx_eq'); abs(lambda2'); abs(bAx_eq')]));
OptErr = max([norm(L_nabla, inf), CompleErr1, CompleErr2]);
  
x_out = x(1:n1);

info.time = info.cpu;
info.KKT_abs = max(FeasErr, OptErr);
info.KKT_rel = max(FeasErr/tau1, OptErr/tau2);
info.obj = Value(first_stage, test, A, Aeq, x_out);
end

function obj = objective(x, data)
  obj = x' * data.H * x/2  + data.f' * x;
end

function constraints = constraints(x, data)
  constraints = [data.A * x - data.b; data.Aeq * x - data.beq];
end

function g = gradient(x, data)
  g = x' * data.H + data.f';
end

function jacobian = jacobian(~, data)
  jacobian = sparse([data.A; data.Aeq]);
end
function Hessian = hessian(~, sigma, ~, data)
  Hessian = tril(sigma * data.H);
end

function Hessian = hessian_pattern(data)
  Hessian = tril(data.H & 1);
end
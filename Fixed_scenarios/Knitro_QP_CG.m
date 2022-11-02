function [x_out, info] = Knitro_QP_CG(first_stage, train, test, A, Aeq, options)
%--------------------------------------------------------------------------
%  We solve a QP problem
%
%     minimize      1/2 x' H x  + f'x
%     subject to    Aeq x = beq
%                     A x  <= b
%                 lb <= x  <= ub
%--------------------------------------------------------------------------
feastol = 1e-6; feastol_abs = 1e-3;
opttol = 1e-6; opttol_abs = 1e-3;
if isfield(options,'feastol'); feastol = options.feastol; end
if isfield(options,'feastol_abs'); feastol_abs = options.feastol_abs; end
if isfield(options,'opttol'); opttol = options.opttol; end
if isfield(options,'opttol_abs'); opttol_abs = options.opttol_abs; end

n1 = size(first_stage.c,1);
n2 = size(train.Q{1},2);

f = [first_stage.c; zeros(n2 * train.num,1)];
lb = -inf(n1 + n2 * train.num, 1);
ub = inf(n1 + n2 * train.num, 1);
H = sparse(cat(1, cat(2,sparse(n1, n1), cell2mat(train.Q) * 2/train.num), sparse(n2 * train.num, n1 + n2 * train.num)));
H = sparse((H + H')/2);

b = [first_stage.b; train.bjoint(:)];
beq = [first_stage.beq; train.bjointeq(:)];

% Defining the initial point is optional for QPs. Typically, for
% convex QPs, when using the default interior-point/barrier algorithm,
% it is best not to supply an initial point, and instead to let Knitro
% apply its own initial point strategy (unless using warm-starts).
% Providing a good initial point, however, can be very helpful on
% non-convex QPs or when using the active-set or SQP algorithms.
x0 = [options.x0; zeros(n2 * train.num, 1)];

options = knitro_options('outlev', 1, 'bar_linsys_storage', 1, 'algorithm', 2, 'convex', 0, 'feastol', feastol,...
             'feastol_abs', feastol_abs, 'opttol', opttol, 'opttol_abs', opttol_abs);

% call knitro_qp function to solve the QP.
tic
[sol, ~, ~, output, lambda] = knitro_qp (H, f, A, b, Aeq, beq, lb, ub, x0, [], options);
toc

clear info
info.time = toc;
fprintf('- Time: %.4f\n', info.time);

% compute the optimality & feasibility error
tau1 = max([1; A * x0 - b; abs(Aeq * x0 - beq)]);
tau2 = max(1, norm(H * sol + f, inf));

FeasErr = max([0; A * sol - b; abs(Aeq * sol - beq)]);

bAx = b - A * sol;
bAx_eq = beq - Aeq * sol;
lambda1 = lambda.ineqlin; lambda2 = lambda.eqlin;
L_nabla = H * sol + f + A' * lambda1 + Aeq' * lambda2;
CompleErr1 = max(min([lambda1' .* bAx'; lambda1'; bAx']));
CompleErr2 = max(min([abs(lambda2' .* bAx_eq'); abs(lambda2'); abs(bAx_eq')]));
OptErr = max([norm(L_nabla, inf), CompleErr1, CompleErr2]);

x_out = sol(1:n1);

info.iter = output.iterations;
info.KKT_abs = max(FeasErr, OptErr);
info.KKT_rel = max(FeasErr/tau1, OptErr/tau2);
info.obj = Value(first_stage, test, A, Aeq, x_out);
end


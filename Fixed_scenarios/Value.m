%% Return the value of the original problem for given samples
function obj = Value(first_stage, test, A, Aeq, x)
    n1 = size(first_stage.c, 1);
    n2 = size(test.Q{1}, 2);
    
    N = size(first_stage.b, 1); Neq = size(first_stage.beq, 1);
    Njoint = size(test.bjoint, 1);
    Njointeq = size(test.bjointeq, 1);
    model.lb = -inf(n2*test.num, 1);
    model.sense = [repmat('<', Njoint*test.num, 1); repmat('=', Njointeq*test.num, 1)];
    params.outputflag = 0;
    model.obj = x' * cell2mat(test.Q)/test.num;
    model.A = [A(N+1:end, n1 + 1:n1 + n2 * test.num); Aeq(Neq + 1:end, n1 + 1:n1 + n2 * test.num)];
    model.rhs = [test.bjoint(:) - A(N + 1:end, 1:n1) * x; test.bjointeq(:) - Aeq(Neq + 1:end, 1:n1) * x];
    result = gurobi(model, params);
    
    obj = first_stage.c' * x + result.objval;
end
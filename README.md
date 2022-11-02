# Decomposition_NonvexSP
This repository contains an implementation of the methods presented in "A Decomposition Algorithm for Two-Stage Stochastic Programs with Nonconvex Recourse" [arXiv:2204.01269](https://arxiv.org/abs/2204.01269).

## Dependencies

The proposed decomposition algorithms, named DPME, use [Gurobi](https://www.gurobi.com/downloads/gurobi-optimizer-eula/) to solve the master and subproblems. To run it in parallel, we will also need to install Parallel Computing Toolbox in MATLAB.

For comparison, we consider interior-point-based solvers, including [Knitro](https://www.artelys.com/solvers/knitro/programs/) and [IPOPT](https://coin-or.github.io/Ipopt/), to solve the determinisit equivalent. There is a [MATLAB interface for IPOPT](https://github.com/ebertolazzi/mexIPOPT) that can be installed easily.

## Running the experiments for fixed scenarios

Run `Test_fixed_scenarios.m` in the `Fixed_scenarios/` folder. The results are written to mat files and a text file. By setting the options (lines 18-41) in `Test_fixed_scenarios.m`, we can choose which algorithm to use, specify stopping (feasibility and optimality) tolerance, etc.

## Running the experiments with sampling
Run `Test_sampling_benchmark.m` in the `Sampling/Benchmark/` folder to test the performance of DPME using all of the samples.

To see the efficiency of sampling-based DPME, run `Test_sampling.m` in the `Sampling/Variable_sample_size/` folder with different combination of the variance and the growth rate of the sample size (see lines 33, 34 in `Test_sampling.m`).

**Warning**: The functions `DPME.m` and `Inner_DPME.m` in the `Sampling/` folder are not the same as that of in the `Fixed_scenarios/` folder. This is because a new procedure is needed to estimate the violation of KKT system for the sampling-based algorithm. More detailed explanation can be found in the function `KKT_test.m` and the second to the last paragraph in section 6 of our paper.

## Interpreting the output of DPME

When we call DPME algorithm, a table of iteration stats will be
printed with the following headings for each replication.

-*Runtime:*

`outer_iter` = the current outer loop iteration number

`inner_iter` = the number of inner loop iterations between two consecutive outer loops

`time` = the cumulative run time in seconds

-*Accuracy of the inner loop:*

`gamma` = the parameter of partial Moreau envelope

 `epsilon` = the stopping tolerance of the inner loop

`dis/gamma` = $\|x_{\nu, i} - x_{\nu, i + 1}\|/\gamma_\nu$ = the approximate optimality error of the inner loop

`Prox_avg` = $\sum\limits_{s = 1}^{S}\|x^s_{\nu, i} - x_{\nu, i}\| / S$ = the average progress of proximal subproblems

`Prox_wor` = $\max\limits_{1 \leq s \leq S}\|x^s_{\nu, i} - x_{\nu, i}\|$ = the biggest progress of proximal subproblems

-*Solution quality with respect to the deterministic equivalent:*

`FeasErr` = the feasibility error

`OptErr` = the optimality error

`FeasErr_rel` = the relative feasibility error

`OptErr_rel` = the relative optimality error (based on [Termination criteria in Knitro](https://www.artelys.com/docs/knitro/2_userGuide/termination.html))

`Objective` = the objective value

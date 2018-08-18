%{
Tests correctness of the ADMM and proximal gradient optimization solvers
and gives a basic example of how to use them.

Requires CVX to compute reference solution, and TFOCS to compute the
proximal gradient solver.  ADMM can also benefit from TFOCS being installed
but it is not necessary.

TFOCS:  https://github.com/cvxr/TFOCS/
CVX:    https://github.com/cvxr/CVX

Stephen Becker and Farhad Pourkamali-Anaraki
April 17, 2018

%}

rng(0);
n   = 20;
p   = 10;

X   = randn(p,n);

XtX     = X'*X; % n x n
temp    = abs( XtX - diag(diag(XtX)) );
lambdaE = 10/min(max(temp));


%% Reference solution via CVX
DO_CVX = ( n <= 200 ) && ( p <= 100 ) && (2 == exist('cvx','file'));
affine = false;
if DO_CVX
    cvx_begin
        variable C(n,n)
        cvx_precision best
        minimize norm(C(:),1) + lambdaE/2*sum_square(vec( X - X*C ) )
        subject to
        diag(C) == 0
    cvx_end
	C_ref = C;
else
    C_ref = zeros(n);
end

% === use ADMM solver
% It works, but to get high precision, need to play around with alpha_rho a
% lot, in order to get a good rho value
errFcn  = @(C) norm( C - C_ref, 'fro')/norm(C_ref,'fro');
[C,errHist,resid,objective,parameters] = SSC_viaADMM(X, ...
    'errFcn', errFcn, 'lambda', lambdaE, 'maxIter', 3e4, 'printEvery', 5e2,...
    'tol', 1e-12, 'affine', affine, 'alpha_rho', 10, 'adaptiveRho', true );
errHist_linear_ADMM = errHist(:,2);

% === and use proximal gradient solver
tfocs_opts = struct( 'errFcn', @(objective,C) errFcn(reshape(C,n,n)) );
[C,TFOCSout, TFOCSopts, parameters] = SSC_viaProxGradient(X, 'lambda', lambdaE,...
    'affine', affine, 'tfocs_opts', tfocs_opts , 'tol', 1e-6, 'maxIter', 5e3, ...
    'printEvery',50);
errHist_linear_proxGrad = TFOCSout.err;

% === plot
figure(1); clf;
semilogy( errHist_linear_ADMM );
hold all
semilogy( errHist_linear_proxGrad );
set(gca,'fontsize',18);
legend('Error vs CVX (ADMM)', 'Error vs CVX (Prox Gradient)' );
xlabel('iteration');
ylabel('error');
title('Errors (linear subspace case, no affine constraint)');
%% And repeat, but with affine case
affine = true;
if DO_CVX
    cvx_begin
        variable C(n,n)
        cvx_precision best
        minimize norm(C(:),1) + lambdaE/2*sum_square(vec( X - X*C ) )
        subject to
        diag(C) == 0
        ones(1,n)*C == ones(1,n)
    cvx_end
	C_ref = C;
else
    C_ref = zeros(n);
end
errFcn  = @(C) norm( C - C_ref, 'fro')/norm(C_ref,'fro');

[C,errHist,resid,objective,parameters] = SSC_viaADMM(X, ...
    'errFcn', errFcn, 'lambda', lambdaE, 'maxIter', 3e4, 'printEvery', 5e2,...
    'tol', 1e-12, 'affine', affine, 'alpha_rho',1, 'adaptiveRho', true,...
    'epsCurvature', .2);
errHist_affine_ADMM = errHist(:,2);


% === and use proximal gradient solver
tfocs_opts = struct( 'errFcn', @(objective,C) errFcn(reshape(C,n,n)) );
[C,TFOCSout, TFOCSopts, parameters] = SSC_viaProxGradient(X, 'lambda', lambdaE,...
    'affine', affine, 'tfocs_opts', tfocs_opts , 'tol', 1e-8, 'maxIter', 5e3, ...
    'printEvery',50);
errHist_affine_proxGrad = TFOCSout.err;

% === plot
figure(2); clf;
semilogy( errHist_affine_ADMM );
hold all
semilogy( errHist_affine_proxGrad );
set(gca,'fontsize',18);
legend('Error vs CVX (ADMM)', 'Error vs CVX (Prox Gradient)' );
xlabel('iteration');
ylabel('error');

title('Errors (affine space case)');


%% And do a simple test that tests the scaling with n for the ADMM solver
rng(0);
nList   = round( logspace( 2.2, 3.7 , 8 ) );
p   = 30;

times   = zeros( length(nList), 1 );
for ni  = 1:length( nList )
    n   = nList(ni);
    X   = randn(p,n);
    t1  = tic;
    SSC_viaADMM(X, 'maxIter', 10, 'printEvery', 100, 'affine', false,'errHistEvery',10,'residHistEvery',10 );
    times(ni)   = toc(t1);
    
end

figure(3); clf;
loglog( nList, times, 'o-' )
% for comparison
hold all
loglog( nList, nList.^2 * (times(end)/(nList(end)^2) ), '--' );
set(gca,'fontsize',18);
legend('computational cost of ADMM', 'O(n^2) for comparison','location','northwest');
xlabel('n');
ylabel('time (sec)');

title('Computation scaling for ADMM as function of n');
function [C,TFOCSout, TFOCSopts, parameters] = SSC_viaProxGradient(X, varargin )
% C = SSC_viaProxGradient( X )
%   solves the Sparse Subspace Clustering problem of Vidal et al.
%   using an accelerated proximal gradient algorithm (e.g. FISTA)
%   i.e., solves
%
% min_{C}  lambda/2|| X - X*C ||_F^2 + ||C||_1
%   s.t. diag(C) = 0
%    and (if 'affine' is true), ones(n,1)*C = ones(n,1)
%
%   where X is a p x n data matrix
% 
% C = SSC_viaProxGradient( X, 'parameter', value, ... ) allows for extra parameters
% such as:
%   'lambda'        regularization parameter; by default, alpha_lambda*mu
%       (where mu is a coherence parameter; for alpha_lambda<1, C=0 is
%       optimal );
%   'alpha_lambda'  See above
%   'maxIter'       Max number of iterations (default: 200 )
%   'printEvery'    How often to display to the screen (between 1 and inf)
%   'affine'        Include ones(n,1)*C=ones(n,1) constraint (default: false)
%   'tol'           Stopping tolerance (default: 2e-4 ), based on the 
%       absolute l_inf norm error between the two primal ADMM variables
%   'tfocs_opts'    Additional options (structure) to pass to TFOCS, e.g.,
%       tfocs_opts=struct('errFcn', @(objective,c) norm(reshape(c,n,n)-C_ref,'fro') );
%       In TFOCS, the variable C is handled as a n^2 x 1 vector,
%       so you may need to reshape it back to a n x n matrix sometimes.
%   'initialGuess'  Initial guess (default: zeros(n) )
%
% [C,TFOCSout,TFOCSopts, parameters] = SSC_viaProxGradient( X, ... )
%   returns output information from TFOCS (TFOCSout and TFOCSopts)
%   and parameters are parameter specific to this code
%   e.g., parameters.lambda tells you what lambda was used
%
% This code requires TFOCS (a recent version from github after April 17
% 2018). See https://github.com/cvxr/TFOCS/
%
%
% Stephen Becker and Farhad Pourkamali-Anaraki 2018


param  = inputParser;
addParameter( param, 'maxIter', 200, @(m) (m>=1) );
addParameter( param, 'printEvery', 10 );
addParameter( param, 'tol', 1e-4 ); % stopping tolerance
addParameter( param, 'affine', false );
addParameter( param, 'lambda', [], @(l) (l>0) );
addParameter( param, 'alpha_lambda', 800, @(a) (a>=1) );
addParameter( param, 'tfocs_opts', [] );
addParameter( param, 'initialGuess', [] );
addParameter( param, 'mu_correlation', [] );


parse(param,varargin{:});
parameters  = param.Results;
maxIter     = parameters.maxIter;
printEvery  = parameters.printEvery; if isinf(printEvery), printEvery=0; end
tol         = parameters.tol;
affine      = parameters.affine;
lambda      = parameters.lambda;
alpha_lambda= parameters.alpha_lambda;
tfocs_opts  = parameters.tfocs_opts;
c0          = parameters.initialGuess;
mu_correlation  = parameters.mu_correlation;

[p,n]   = size(X);

% Use conventions of Ehsan Elhamifar and Rene Vidal's 2012 paper
if isempty( lambda )
    if isempty(mu_correlation)
        XtX     = X'*X; % n x n
        temp    = abs( XtX - diag(diag(XtX)) );
        mu_correlation = min(max(temp)); % eq (11) in our arXiv paper
        parameters.mu_correlation   = mu_correlation;
    end
    lambda  = alpha_lambda/mu_correlation; % "lambda_E" in arXiv paper
    parameters.lambda = lambda;
end

%{
 for TFOCS, we'll solve

min_C  lambda_tfocs*||C||_1 + 1/2||X-XC||_F^2

so lambda_tfocs = 1/lambda, and need to scale output objective function

%}
zeroID  = true; b = 1; nCols = n; lambda_tfocs = 1/lambda;
if affine
    %JMF 2 Dec 2018: optimized version is now the default in TFOCS; will use mex if it's compiled
    prox    = prox_l1_and_sum(lambda_tfocs, b, nCols, zeroID);
else
    %JMF 2 Dec 2018: optimized version is now the default in TFOCS; will use mex if it's compiled
    prox    = prox_l1_mat(lambda_tfocs, nCols, zeroID); % no sum(x)=b constraint.
end

tfocs_opts.maxIts       = maxIter;
tfocs_opts.tol          = tol;
tfocs_opts.printEvery   = printEvery;

% Construct the smooth term. For efficiency, TFOCS likes to write
%   a smooth term as f(A*x+offset) because, assuming the linear operator
%   "A" is large, we can be careful and save computation.
smoothF = smooth_quad(); % i.e., .5*norm(x)^2
vc      = linop_vec( [p,n] );  % to see the sizes: celldisp(vc(0,0))
mt      = linop_reshape( [n^2,1], [n,n] );
linX    = linop_matrix( X, 'R2R', n ); % celldisp(linX(0,0))
linearF = linop_compose( vc, linX, mt ); % celldisp(linearF(0,0))
affineF = { linearF, -vec(X) }; % This is in the form of { linear, offset}
if isempty( c0 )
    c0      = zeros(n^2,1);
end
[c, TFOCSout, TFOCSopts ] = tfocs( smoothF, affineF, prox, c0(:), tfocs_opts);
C   = reshape( c, n, n );

% Note: objective function used in TFOCS is divided by lambda,
%   so need to multiply the output objective by lambda if you want
%   to compare (but the iterates C are not affected).
%   Attempting to do that here:
TFOCSout.f = lambda*TFOCSout.f;

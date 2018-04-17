function [C,errHist,resid,objective,parameters] = SSC_viaADMM(X, varargin )
% C = SSC_viaADMM( X )
%   solves the Sparse Subspace Clustering problem of Vidal et al.
%   using the ADMM algorithm. i.e., solves
%
% min_{C}  lambda/2|| X - X*C ||_F^2 + ||C||_1
%   s.t. diag(C) = 0
%    and (if 'affine' is true), ones(n,1)*C = ones(n,1)
%
%   where X is a p x n data matrix
% 
% C = SSC_viaADMM( X, 'parameter', value, ... ) allows for extra parameters
% such as:
%   'lambda'        regularization parameter; by default, alpha_lambda*mu
%       (where mu is a coherence parameter; for alpha_lambda<1, C=0 is
%       optimal );
%   'alpha_mu'      See above
%   'rho'           ADMM parameter; any rho>0 works, but some lead to
%       faster convergence. Default: rho = alpha_rho*alpha_lambda;
%   'alpha_rho'     See above
%   'maxIter'       Max number of iterations (default: 200 )
%   'errFcn'        An optional user-supplied error function, evaluated
%       every iteration
%   'printEvery'    How often to display to the screen (between 1 and inf)
%   'affine'        Include ones(n,1)*C=ones(n,1) constraint (default: false)
%   'tol'           Stopping tolerance (default: 2e-4 )
%
% [C,errHist,resid,objective,parameters] = SSC_viaADMM( X, ... )
%   returns output information. errHist(:,1) is the l_inf norm error
%   between the two primal ADMM variables; errHist(:,2) is the errFcn
%   output (if supplied);  resid(:) is || X - X*C ||_F,
%   while objetive(:) is lambda/2|| X - X*C ||_F^2 + ||C||_1.
%
% Stephen Becker and Farhad Pourkamali-Anaraki 2018
% Based off algorithm of Ehsan Elhamifar and Rene Vidal 2012, but adding
%   in an O(n^2) matrix inversion instead of O(n^3)


param  = inputParser;
addParameter( param, 'maxIter', 200, @(m) (m>=1) );
addParameter( param, 'errFcn', [] ); % e.g., @(C) evalSSR_error( C, true_labels );
addParameter( param, 'printEvery', 10 );
addParameter( param, 'tol', 2e-4 ); % stopping tolerance
addParameter( param, 'affine', false );
addParameter( param, 'lambda', [], @(l) (l>0) );
addParameter( param, 'rho', [], @(rho) (rho>0) );
addParameter( param, 'alpha_lambda', 800, @(a) (a>=1) );
addParameter( param, 'alpha_rho', 1 );


parse(param,varargin{:});
parameters  = param.Results;
maxIter     = parameters.maxIter;
errFcn      = parameters.errFcn;
printEvery  = parameters.printEvery; if isinf(printEvery), printEvery=0; end
tol         = parameters.tol;
affine      = parameters.affine;
lambda      = parameters.lambda;
rho         = parameters.rho;
alpha_lambda= parameters.alpha_lambda;
alpha_rho   = parameters.alpha_rho;

[p,n]   = size(X);

% Use conventions of Ehsan Elhamifar and Rene Vidal's 2012 paper
XtX     = X'*X; % n x n
if isempty( lambda )
    temp    = abs( XtX - diag(diag(XtX)) );
    mu_correlation = min(max(temp)); % eq (11) in our arXiv paper
    lambda  = alpha_lambda/mu_correlation; % "lambda_E" in arXiv paper
    parameters.lambda = lambda;
end
if isempty( rho )  % ADMM parameter, only affects convergence rate
    if isempty( alpha_rho )
        alpha_rho = 1;
    end
    rho     = alpha_rho * alpha_lambda;
    parameters.rho = rho;
    parameters.alpha_rho = alpha_rho;
end
XtX     = lambda*XtX;

% ==== precomputation ====
if affine
    % Make an operator that does the equivalent of
    %   @(x) inv(lambdaE*(X'*X)+rho*eye(n)+rho*ones(n,n)) * x
    % but do it efficiently
    
    mm1     = sqrt(lambda);
    mm2     = sqrt(rho);

    Y1      = sqrt(lambda)*sqrt(rho)*(X*ones(n,1));
    YYt     = [ lambda*(X*X'), Y1; ...
        Y1', rho*n ];
%     iYYt    = inv( eye(p+1) + 1/rho*YYt );
    % make it implicit
%     Afun    = @(RHS) RHS/rho - ([mm1*X',mm2*ones(n,1)]*(iYYt*[mm1*(X*RHS);mm2*(ones(1,n)*RHS)]) )/(rho^2);
    
    % Alternatively, a bit more stable
    rChol   = chol( eye(p+1) + 1/rho*YYt );
    iYYt    = @(RHS) rChol\( rChol'\RHS);
    Afun    = @(RHS) RHS/rho - ([mm1*X',mm2*ones(n,1)]*(iYYt([mm1*(X*RHS);mm2*(ones(1,n)*RHS)])) )/(rho^2);

else
    % Make an operator that does the equivalent of
    %   @(x) inv(lambdaE*(X'*X)+rho*eye(N)) * x
    % but do it efficiently
%     A   = inv(XtX+rho*eye(n));
%     Afun    = @(RHS) A*RHS;
    
%     iYYt    = inv( eye(p) + 1/rho*lambda*(X*X') );
%     Afun    = @(RHS) RHS/rho - (lambda/(rho^2))*(X'*(iYYt*(X*RHS)) );
    
    % Alternatively, a bit more stable, 
    rChol   = chol( eye(p) + 1/rho*lambda*(X*X') );
    iYYt    = @(RHS) rChol\( rChol'\RHS);
    Afun    = @(RHS) RHS/rho - (lambda/(rho^2))*(X'*(iYYt(X*RHS)) );
end

softThresh  = @(X, t) sign(X).*max( 0, abs(X) - t );

C       = zeros(n);
Dual1   = zeros(n);
if affine, Dual2 = zeros(n,1); end
errHist = zeros(maxIter, 1 + isempty( errFcn ) );
[resid,objective]   = deal( zeros(maxIter,1) );
% ==== main ADMM loop ====
for k = 1:maxIter
    if affine
        temp = XtX + rho*( C - Dual1/rho + ones(n,1)*(ones(1,n)-Dual2'/rho) );
    else
        temp = XtX + rho*( C - Dual1/rho );
    end
    Z   = Afun( temp );
    Z   = Z - diag(diag(Z));
    C   = softThresh( Z + Dual1/rho, 1/rho );
    C   = C - diag(diag(Z));
    
    % Update Lagrange Multipliers
    Dual1   = Dual1 + rho*( Z - C );
    if affine, Dual2 = Dual2 + rho*( Z'*ones(n,1) - ones(n,1) ); end
    
    errHist(k,1)    = norm( Z(:) - C(:), Inf );
    if affine, errHist(k,1) = errHist(k,1) + norm( ones(1,n)*Z - ones(1,n), Inf ); end
    
    resid(k)    = norm( X - X*Z, 'fro' );
    objective(k)= norm(Z(:),1) + lambda/2*resid(k)^2;
    
    if ~isempty( errFcn )
        errHist(k,2)    = errFcn( Z );
    end
    
    do_break = ( errHist(k,1) < tol );
    if ~mod( k, printEvery ) || do_break || (printEvery >0 && k==maxIter )
        fprintf('Iter %4d, ADMM residual %.2e, objective %.2e', k, errHist(k,1), objective(k) );
        if ~isempty( errFcn )
            fprintf(', errFcn %.2e', errHist(k,2) );
        end
        fprintf('\n');
    end
    if do_break
        fprintf('Reached tolerance, ending\n');
        errHist     = errHist(1:k,:);
        resid       = resid(1:k,:);
        objective   = objective(1:k,:);
        break
    end
end
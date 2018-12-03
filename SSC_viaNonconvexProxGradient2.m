function [C,errHist] = SSC_viaNonconvexProxGradient( X, sparsity, varargin )
% [C,errHist] = lowMemoryHardThresholdingSSC( X, sparsity )
%   uses a non-convex proximal gradient method, i.e., hard-thresholding,
%   to attempt to solve the l0 subspace clustering problem:
%
%   min_{C} .5|| X - XC||_F^2
%   s.t.
%       diag(C)=0, nnz( C(:,j) ) <= sparsity for j=1:N
%       and if 'affine' is true, also has affine subspace constraint:
%       1'*C = 1
%
%   where X is a p x N matrix of data, and C is a N x N variable
%
%TODO JMF 2 Dec 2018: I don't think this is right...
%     CC is dense and same size as C, which is N x N.  We can do it block-wise,
%     which is more memory-friendly.
%   This code keeps C as a sparse matrix only
%   
% [C,errHist] = lowMemoryHardThresholdingSSC( X, sparsity, 'param', value )
%   for param/value pairs, allows you to specify extra options, e.g.,
%       'maxIter'   (default: 50)
%       'MB_limit'  (default: 2e3 ) maximum variable size, in MB
%       'errFcn'    (default: none) e.g., @(C) evalSSR_error( C, true_labels );
%       'printEvery'(default: 10)   how often to print info
%       'tol'       (default: 1e-3) stopping tolerance, in relative change
%       'stepsize'  (default: none, so will calculate it)
%       'aggressiveness' (default: 1) if this is > 1, will take a larger stepsize than theoretically expected
%       'affine'    (defaut: false) include 1'*C==1 constraint
%
% Note: this code has been designed to stay low-memory, but has not
%   been optimized for speed.
%
% Stephen Becker and Farhad Pourkamali-Anaraki 2018

param  = inputParser;
addParameter( param, 'maxIter', 50 );
addParameter( param, 'MB_limit', 2e3 ); % limit, in MB
addParameter( param, 'errFcn', [] ); % e.g., @(C) evalSSR_error( C, true_labels );
addParameter( param, 'printEvery', 10 );
addParameter( param, 'tol', 1e-3 ); % stopping tolerance
addParameter( param, 'stepsize', [] );
addParameter( param, 'aggressiveness', 1 );
addParameter( param, 'affine', false );
parse(param,varargin{:});
maxIter     = param.Results.maxIter;
MB_limit    = param.Results.MB_limit;
errFcn      = param.Results.errFcn;
printEvery  = param.Results.printEvery;
tol         = param.Results.tol;
step        = param.Results.stepsize;
aggressiveness = param.Results.aggressiveness;
affine      = param.Results.affine;

if isinf(printEvery), printEvery = 0; end % will never print

[p,N]   = size(X);
b       = 1; % affine constraint: sums to b
zeroID  = true; 
nCols   = N;

MB          = 1024^2; % in Bytes. Each double is 8 bytes
blockSize   = ceil( MB_limit*MB/(8*N) );
% blockSize   = 2^floor( log2(blockSize) );

if ~isempty( errFcn )
    errHist     = zeros(maxIter,1);
else
    errHist     = [];
end

if affine
    %proj = @(x,sparsity) GSHP( x, b, sparsity );
    error('not mexified');
else
    %proj = @(x,sparsity) findLargestK( x, sparsity );
    zero_diag = true;
    sparse_output = true;
    proj = @(x,sparsity) proj_largest_k_mex(x, sparsity, zero_diag, sparse_output);
end


if isempty(step)
    if printEvery > 0, disp('Estimating Lipschitz constant for stepsize...'); end
    % Estimate Lipschitz constant:
    c = randn(N,1);
    for k = 1:20
        L = norm(c);
        c = c/L;
        if ~mod(k,printEvery)
            fprintf('Power iteration %2d, spectral norm estimate %e\n', k, L );
        end
        c   = X'*(X*c);
    end
    L   = 1.001*norm(c); % (slight over-)estimate of norm(X)^2
    step= 1/L;
    if printEvery > 0, disp('... done estimating Lipschitz constant.'); end
end

% take a bigger stepsize
step    = aggressiveness * step;


C           = sparse(zeros(N));     % the main variable
for k = 1:maxIter

    if blockSize >= N
        resid   = full(X*C - X);
        CC      = C - step*(X'* resid );    % gradient step; C is sparse but CC may not be sparse
        C = proj(CC, sparsity); % C is sparse

    else
        % Loop, in order to keep it lower memory
        startInd = 1;
        for block = 1:ceil(N/blockSize)
            endInd      = min( N, startInd + blockSize - 1 );
            nCols_block = endInd - startInd + 1;

            resid   = X*C(:,startInd:endInd) - X(:,startInd:endInd);
            CC      = C(:,startInd:endInd) - step*(X'* resid );
            
            CC = proj(CC, sparsity); % CC is sparse
            
            C(:,startInd:endInd)    = CC;
            
            startInd = startInd + blockSize;
        end
    end
    
    breakNext = false;
    if tol > 0
        if k > 1 && norm( C - C_old, 'fro')/max(1e-10,norm(C_old,'fro')) < tol
            if printEvery > 0
                disp('Reached stopping tolerance; quitting');
            end
            breakNext=true;
            %break; % don't break right away, let us print out info first
        end
        C_old = C;
    end   
    
    if ~isempty( errFcn )
        err = errFcn( C );
        errHist(k) = err;
    end
    if ~mod( k, printEvery ) || (printEvery > 0 && breakNext )
        if ~isempty(errFcn)
            fprintf('Iter %4d, residual %.2e, error %.2e\n', k, norm(resid,'fro'),err );
        else
            fprintf('Iter %4d, residual %.2e\n', k, norm(resid,'fro'));
        end
    end
    if breakNext
        break;
    end
    
  
    
end
if ~isempty( errFcn )
    errHist = errHist(1:k);
end


end % end of main routine



% === Subroutines ===
function [x, S] = GSHP(b,lambda, K)
% [x, S] = GSHP(b,lambda, K)
% Computes a minimizer to the problem
%   min_x || x - b ||_2^2
% s.t.
%   x has at most k nonzeros
%   sum(x) == lambda
%
% GSHP stands for "Greedy Selector and Hyperplane Projector"
% This version does *not* allow for a weighted sum, but it is possible
%   to do that.
%
% This version DOES allow "b" to be a matrix
%   and in that case, the output "x" is a matrix, and the output
%   is equivalent to looping over the columns of x and b.
%   ("lambda" is the same for all columns, though this could easily
%    be changed if necessary)
%
% Stephen Becker, 2/217/2018
% Follows code from "Sparse Projections onto the Simplex"
%   by Kyrillidis, Becker, Ceverh, Kock, ICML 2013
%   Available at arXiv.org/abs/1206.1529
%   (In that algorithm, their "w" is our "b")

[~,j] = max( lambda*b ); % automatically vectorized over columns
S     = j;
nCols = size(b,2);
if nCols == 1
    for l = 2:K
        offset = (sum(b(S),1)-lambda)/(l-1);
        resid  = abs( b - offset );
        resid(S) = 0; % make sure we don't select an old index
        [~,j]   = max( resid );
        S       = sort([S;j]);
    end
    % final projection
    xS  = b(S) - ( sum(b(S)) - lambda)/K;
    x   = zeros(size(b,1),1);
    x( S ) = xS;
    
else
    for l = 2:K
        offset = zeros(1,nCols);
        for j = 1:nCols
            offset(j) = (sum(b(S(:,j),j),1)-lambda)/(l-1);
        end
        resid  = abs( b - offset );
        for j = 1:nCols
            resid(S(:,j),j) = 0; 
        end
        [~,j]   = max( resid );
        S       = [S;j];
    end
    % final projection
    x   = zeros(size(b,1),nCols);
    for j = 1:nCols
        SS  = S(:,j);
        xS  = b(SS,j) - ( sum(b(SS,j)) - lambda)/K;
        x( SS, j ) = xS;
    end
end
end % end of GSHP subroutine

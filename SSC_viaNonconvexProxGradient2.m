function [C,errHist] = SSC_viaNonconvexProxGradient( X, sparsity, varargin )
% [C,errHist] = SSC_viaNonConvexProxGradient( X, sparsity )
%   aka 
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
%
% Modification Dec 20, 2019, allows convex case now too. Specify by giving
% a value for "lambda" parameter. Not extensively tested, but at least
% it ought to not break the nonconvex case. Also found bug about
% not zeroing out the diagonal entries

param  = inputParser;
addParameter( param, 'maxIter', 50 );
addParameter( param, 'MB_limit', 2e3 ); % limit, in MB
addParameter( param, 'errFcn', [] ); % e.g., @(C) evalSSR_error( C, true_labels );
addParameter( param, 'printEvery', 10 );
addParameter( param, 'tol', 1e-3 ); % stopping tolerance
addParameter( param, 'stepsize', [] );
addParameter( param, 'aggressiveness', 1 );
addParameter( param, 'affine', false );
addParameter( param, 'lambda', [] ); % Dec 2019, converts to convex mode
parse(param,varargin{:});
maxIter     = param.Results.maxIter;
MB_limit    = param.Results.MB_limit;
errFcn      = param.Results.errFcn;
printEvery  = param.Results.printEvery;
tol         = param.Results.tol;
step        = param.Results.stepsize;
aggressiveness = param.Results.aggressiveness;
affine      = param.Results.affine;
lambda      = param.Results.lambda;

if isinf(printEvery), printEvery = 0; end % will never print

[p,N]   = size(X);
b       = 1; % affine constraint: sums to b
zeroID  = true; 
nCols   = N;

MB          = 1024^2; % in Bytes. Each double is 8 bytes
blockSize   = ceil( MB_limit*MB/(8*N) );
% blockSize   = 2^floor( log2(blockSize) );
if printEvery > 0
    fprintf('Due to memory restrictions, will use %d blocks\n', N/blockSize ); 
end

if ~isempty( errFcn )
    errHist     = zeros(maxIter,1);
else
    errHist     = [];
end

if blockSize >= N
    zero_diag = true; % Dec 2019, before now, this was always true
else
    zero_diag = false; % Take care of it ourselves here
end
if affine
    %proj = @(x,sparsity) GSHP( x, b, sparsity );
    if isempty(lambda)
        % nonconvex projection
        sparse_output = true;
        proj = @(x,ignore) proj_largest_k_affine_mex(x, sparsity, b, zero_diag, sparse_output);
    else
        lambda_tfocs = 1/lambda;
%         proxHelper    = prox_l1_and_sum(lambda_tfocs, b, N, zero_diag);
        proj        = @(x,stepsize) proxWrapper( affine, lambda_tfocs, b,zero_diag,x,stepsize );
    end
else
    %proj = @(x,sparsity) findLargestK( x, sparsity );
    if isempty(lambda)
        sparse_output = true;
        proj = @(x,ignore) proj_largest_k_mex(x, sparsity, zero_diag, sparse_output);
    else
        lambda_tfocs = 1/lambda;
%         proxHelper    = prox_l1_mat(lambda_tfocs, N, zero_diag);
        proj        = @(x,stepsize) proxWrapper( affine, lambda_tfocs,b,zero_diag, x,stepsize );
    end
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


C           = sparse(N,N);     % the main variable
for k = 1:maxIter

    if blockSize >= N
        resid   = full(X*C - X);
        CC      = C - step*(X'* resid );    % gradient step; C is sparse but CC may not be sparse
%         C = proj(CC, sparsity); % C is sparse
        C = proj(CC,step); % Dec 20 2019, new convention

    else
        % Loop, in order to keep it lower memory
        startInd = 1;
        for block = 1:ceil(N/blockSize)
            if ~mod(block,10), fprintf(' Block %d of %d\n', block, ceil(N/blockSize) ); end
            endInd      = min( N, startInd + blockSize - 1 );
            nCols_block = endInd - startInd + 1;

%             resid   = X*C(:,startInd:endInd) - X(:,startInd:endInd);
            resid   = X*C(:,startInd:endInd);
            resid   = resid - X(:,startInd:endInd);
            CC      = C(:,startInd:endInd) - step*(X'* resid );
            
            % Dec 2019, discovered bug: if zero_diag, then the code
            %   wasn't doing that correctly here...
            if ~zero_diag
                % This means proj() doesn't zero out diagonal for us
                %   so do it explicitly
                diag_inds = sub2ind( size(CC), startInd:endInd, 1:size(CC,2) );
                CC(diag_inds) = 0;
            end
            
%             CC = proj(CC, sparsity); % CC is sparse
            CC = proj(CC,step);
            
            C(:,startInd:endInd)    = CC;
            if ~issparse(C), fprintf(2,'WARNING: C is not sparse\n'); end
            
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

function out = proxWrapper( affine, lambda_tfocs, b, zero_diag, X, stepsize )

[p,N]   = size(X);
if affine
    proxHelper    = prox_l1_and_sum(lambda_tfocs, b, N, zero_diag);
else
    useMex        = true;
    proxHelper    = prox_l1_mat(lambda_tfocs, N, zero_diag, useMex);
end

[~,x]     = proxHelper( X(:), stepsize );
out       = sparse(reshape( x, p, N ));
end

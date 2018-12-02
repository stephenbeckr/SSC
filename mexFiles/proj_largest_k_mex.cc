/**
 * Implements the projection
 *      min_y   ||y - x||_2
 *      s.t.    ||y||_0 <= k
 *
 * That is, this finds the closest point to x with only k non-zero entries.
 *
 * When given matrix input, this performs the operation columnwise.
 * It can also enforce diag(Y) == 0.
 *
 *	Inputs:
 *		Mode 1:
 * 			% X is a vector or [nRows x nCols] matrix
 *			Y = proj_largest_k_mex(X, k);
 *
 *		Mode 2 (zero diagonal):
 * 			% X is [nRows x nCols]
 * 			% zeroID = true enforces diag(Y) == 0
 *			Y = proj_largest_k_mex(X, k, zeroID);
 *
 *		Mode 3 (sparse output):
 * 			% X is [nRows x nCols]
 * 			% sparseOutput = true makes Y a sparse matrix instead of a full matrix
 * 			% By default, sparseOutput = false
 *			Y = proj_largest_k_mex(X, k, zeroID, sparseOutput);
 *
 *		Mode 4 (set internal options):
 *			opt = struct('num_threads', 2);
 *			proj_largest_k_mex(opt);
 *
 * Created on 2 Dec 2018 by jamesfolberth@gmail.com
 */

#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#include <omp.h>

#include <mex.h>

struct Options {
    Options() : num_threads(1) {}

    int num_threads;

    void parseStruct(const mxArray *marr);
};

void Options::parseStruct(const mxArray *mstruct) {
    // Check field names
    int n_fields = mxGetNumberOfFields(mstruct);
    for (int i=0; i<n_fields; ++i) {
        std::string name = mxGetFieldNameByNumber(mstruct, i);
        if (name.compare("num_threads") == 0) continue;
        else {
            std::string msg = "Unrecognized options field name: " + name;
            mexErrMsgTxt(msg.c_str());
        }
    }

    mxArray* tmp;
    tmp = mxGetField(mstruct, 0, "num_threads");
    if (nullptr != tmp) {
        double dval = mxGetScalar(tmp);
        if (dval == std::round(dval)) {
            num_threads = static_cast<int>(std::round(dval));
        } else {
            mexErrMsgTxt("opt.num_threads should be an integer.");
        }
    }
}

class FullProjK {
public:
    // Project the vector x onto its largest k values.
    // y is an array with n elements, just like x.  This assumes y is filled with
    // zeros before the call.
    // If zero_ind >= 0, don't use x[zero_ind].  This is used for zeroID mode.
    void run(double *y,
             const double *x,
             size_t n,
             size_t k,
             ptrdiff_t zero_ind=-1);

private:
    std::vector<size_t> inds_;

};

void FullProjK::run(double *y,
                    const double *x,
                    size_t n,
                    size_t k,
                    ptrdiff_t zero_ind) {

    // Gonna do an "argsort"
    // We want to get the inds of the top k largest values of abs(x)
    // We can do this without a sort by instead using nth_element.
    // We make a vector of inds and then sort based on the values of x,
    // which avoids doing the std::pair thing.
    // We also use C++11 lambdas, which are great.  If you don't have
    // a recent, C++11-compatible compiler, you could rewrite this to use
    // a functor instead.
    inds_.resize(n);
    std::iota(inds_.begin(), inds_.end(), 0); // fill with 0:n-1
    if (zero_ind >= 0) {
        std::nth_element(inds_.begin(), inds_.begin() + k, inds_.end(),
                [&](size_t i0, size_t i1) {
                    if (i0 == zero_ind) {
                        return false;
                    } else if (i1 == zero_ind) {
                        return true;
                    } else {
                        return std::abs(x[i0]) > std::abs(x[i1]);
                    }
                }
            );

    } else {
        std::nth_element(inds_.begin(), inds_.begin() + k, inds_.end(),
                [&](size_t i0, size_t i1) {
                    return std::abs(x[i0]) > std::abs(x[i1]);
                }
           );

    }

    // Copy largest k values to y
    for (size_t i=0; i<k; ++i) {
        y[inds_[i]] = x[inds_[i]];
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check inputs
    if (nrhs < 1) {
        mexErrMsgTxt("At least one input is required");
    }

    if (nrhs > 4) {
        mexErrMsgTxt("No more than four inputs");
    }

    if (nlhs > 1) {
        mexErrMsgTxt("Exactly one output");
    }

    // Cached options struct
    static Options opt;

    // First input
    if (mxIsStruct(prhs[0])) {
        opt.parseStruct(prhs[0]);
        return;
    }

    size_t nRows = mxGetM(prhs[0]);
    size_t nCols = mxGetN(prhs[0]);
    const double *X = mxGetPr(prhs[0]);

    // Second input
    if (nrhs < 2) {
        mexErrMsgTxt("Must specify X and k");
    }

    double dk = mxGetScalar(prhs[1]);
    if (std::round(dk) != dk) {
        mexErrMsgTxt("k should be an integer");
    }
    size_t k = std::round(dk);
    if (k > nRows) {
        mexErrMsgTxt("k should be <= nRows");
    }

    // Third input
    bool zero_diag = false;
    if (nrhs >= 3) {
        if (!mxIsLogicalScalar(prhs[2])) {
            mexErrMsgTxt("zeroID should be true or false");
        }
        zero_diag = mxIsLogicalScalarTrue(prhs[2]);

        if (nRows < nCols) {
            mexErrMsgTxt("Cannot zero diagonal if nRows < nCols");
        }
    }

    // Fourth input
    bool sparse_output = false;
    if (nrhs >= 4) {
        if (!mxIsLogicalScalar(prhs[3])) {
            mexErrMsgTxt("sparseOutput should be true or false");
        }
        sparse_output = mxIsLogicalScalarTrue(prhs[3]);
    }

    // Y will be a sparse matrix
    if (sparse_output) {
        mexErrMsgTxt("not implemented");

    // Y is gonna be a full matrix
    } else {
        // Will init all values to zero, which we want
        plhs[0] = mxCreateNumericMatrix(nRows, nCols, mxDOUBLE_CLASS, mxREAL);
        double *Y = mxGetPr(plhs[0]);

        if (nCols == 1) { // don't do zeroID for vectors
            FullProjK proj;
            proj.run(Y, X, nRows, k);

        } else {
            #pragma omp parallel num_threads(opt.num_threads)
            {
                FullProjK proj;

                #pragma omp for schedule(static)
                for (size_t j=0; j<nCols; ++j) {
                    if (zero_diag) {
                        proj.run(Y + nRows*j, X + nRows*j, nRows, k, j);
                    } else {
                        proj.run(Y + nRows*j, X + nRows*j, nRows, k);
                    }
                }
            }
        }
    }
}

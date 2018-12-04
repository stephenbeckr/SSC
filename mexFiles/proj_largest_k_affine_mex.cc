/**
 * Implements the projection
 *      min_y   ||y - x||_2
 *      s.t.    ||y||_0 = k
 *              sum(y) = lambda
 *
 * That is, this finds the closest point to x with exactly k non-zero entries
 * and such that the element-wise sum is equal to lambda.  The algorithm to
 * perform this efficiently is called the "greedy selector and hyperplane projector"
 * (GSHP).  See https://arxiv.org/abs/1206.1529 for derivation and details.
 *
 * When given matrix input, this performs the operation columnwise.
 * It can also enforce diag(Y) == 0.
 *
 *	Inputs:
 *		Mode 1:
 * 			% X is a vector or [nRows x nCols] matrix
 *			Y = proj_largest_k_affine_mex(X, k, lambda);
 *
 *		Mode 2 (zero diagonal):
 * 			% X is [nRows x nCols]
 * 			% zeroID = true enforces diag(Y) == 0
 * 			% Note that k must be less than nRows (k < nRows) when zeroID is true
 *			Y = proj_largest_k_affine_mex(X, k, lambda, zeroID);
 *
 *		Mode 3 (sparse output):
 * 			% X is [nRows x nCols]
 * 			% sparseOutput = true makes Y a sparse matrix instead of a full matrix
 * 			% By default, sparseOutput = false
 *			Y = proj_largest_k_affine_mex(X, k, lambda, zeroID, sparseOutput);
 *
 *		Mode 4 (set internal options):
 *			opt = struct('num_threads', 2);
 *			proj_largest_k_affine_mex(opt);
 *
 * Created on 3 Dec 2018 by jamesfolberth@gmail.com
 */

#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>

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

class GSHP {
public:
    // Project the vector x onto its largest k values.
    // y is an array with n elements, just like x.  This assumes y is filled with
    // zeros before the call.  y can also be null, in which case we find inds
    // but don't set any values of y.  This is useful for the sparseOutput mode.
    // If zero_ind >= 0, don't use x[zero_ind].  This is used for zeroID mode.
    void run(double *y,
             const double *x,
             size_t n,
             size_t k,
             const double lambda,
             ptrdiff_t zero_ind=-1);

    // Raw access to the solution data for sparseOutput mode
    // "Returns" pointer to sorted support inds and y data
    // There are k such inds.
    void get_solution_data(size_t const* &inds, double const* &y) const;

private:
    std::vector<size_t> csupp_; // length n

    // used only for sparse mode; length k each
    std::vector<double> y_;
    std::vector<size_t> supp_;
    std::vector<size_t> argsort_inds_;

};

void GSHP::run(double *y,
               const double *x,
               size_t n,
               size_t k,
               const double lambda,
               ptrdiff_t zero_ind) {

    // Vector of complement/support inds
    // At the end of the algorithm csupp_ = [csupp; supp]
    // with supp grown from the end of the array.
    // In the case zero_ind is specified, the zero_ind is placed at the
    // end of csupp_.  But it doesn't get accessed, since we modify n_csupp appropriately.
    // To move an ind into the support region of csupp_, we do a swap with the last
    // non-support ind of csupp_ (i.e. the last element of csupp).  We manually keep
    // track of where this locaton is in csupp_
    csupp_.resize(n);
    size_t n_csupp = (zero_ind >= 0) ? n-1 : n;
    std::iota(csupp_.begin(), csupp_.end(), 0); // 0:n-1
    if (zero_ind >= 0) { // put the diagonal ind at the very end of csupp; we won't use it again
        std::swap(csupp_[zero_ind], csupp_[n-1]);
    }

    // Get the index of max(lambda * x)
    size_t max_ind = 0;
    if (lambda > 0) {
        double max_val = std::numeric_limits<double>::min();

        for (size_t i=0; i<n_csupp; ++i) {
            if (x[csupp_[i]] > max_val) {
                max_val = x[csupp_[i]];
                max_ind = i;
            }
        }

    } else if (lambda < 0) {
        double min_val = std::numeric_limits<double>::max();

        for (size_t i=0; i<n_csupp; ++i) {
            if (x[csupp_[i]] < min_val) {
                min_val = x[csupp_[i]];
                max_ind = i;
            }
        }

    } else { // lambda == 0
        max_ind = 0;
    }
    //std::cout << "supp[0] = " << csupp_[max_ind] << std::endl;
    std::swap(csupp_[n_csupp-1], csupp_[max_ind]);

    // Now build the rest of the support inds
    for (size_t s=1; s<k; ++s) {
        // Compute offset
        double sum = 0;
        for (size_t i=n_csupp-1; i>n_csupp-1-s; --i) {
            sum += x[csupp_[i]];
        }
        double offset = (sum - lambda) / s;

        // Find the index of the max value of resid = abs(x - offset)
        // But only on the non-support inds
        size_t max_ind = 0;
        double max_val = std::numeric_limits<double>::min();

        for (size_t i=0; i<n_csupp-s; ++i) {
            double val = std::abs(x[csupp_[i]] - offset);

            if (val > max_val) {
                max_val = val;
                max_ind = i;
            }
        }
        //std::cout << "supp[" << s << "] = " << csupp_[max_ind] << std::endl;
        // put supp ind at end of csupp, which we won't access later
        std::swap(csupp_[n_csupp-s-1], csupp_[max_ind]);
    }

    // Now we have the support; do the projection
    double sum = 0;
    for (size_t i=n_csupp-1; i>n_csupp-1-k; --i) {
        sum += x[csupp_[i]];
    }
    double offset = (sum - lambda) / k;

    // Full/Dense output in y; assumed that y is filled with zeros on entry
    if (nullptr != y) {
        for (size_t i=n_csupp-1; i>n_csupp-1-k; --i) {
            y[csupp_[i]] = x[csupp_[i]] - offset;
        }

    // Sparse output
    // To get ready for the caller grabbing the output, put the support inds
    // and y values in order of increasing inds.  This requires an argsort
    // kind of thing.
    } else {
        // Argsort the support inds
        argsort_inds_.resize(k); // inds used for the argsort
        std::iota(argsort_inds_.begin(), argsort_inds_.end(), 0);

        // csupp_offset is the part of csupp_ that stores the support inds
        const size_t *csupp_offset = csupp_.data() + n_csupp - k;
        std::sort(argsort_inds_.begin(), argsort_inds_.end(),
                [&](size_t i0, size_t i1) {
                return csupp_offset[i0] < csupp_offset[i1];
            }
        );

        y_.resize(k);
        supp_.resize(k);

        for (size_t i=0; i<k; ++i) {
            supp_[i] = csupp_offset[argsort_inds_[i]];
            y_[i] = x[supp_[i]] - offset;
        }
    }
}

void GSHP::get_solution_data(size_t const* &inds, double const* &y) const {
    inds = supp_.data();
    y = y_.data();
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check inputs
    if (nrhs < 1) {
        mexErrMsgTxt("At least one input is required");
    }

    if (nrhs > 5) {
        mexErrMsgTxt("No more than five inputs");
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
    // lambda can be a vector or a scalar
    size_t n_lambda = mxGetNumberOfElements(prhs[2]);
    if (n_lambda > 1 && n_lambda != nCols) {
        mexErrMsgTxt("lambda should be a scalar or have number of elements "
                "equal to the number of columns of X.");
    }
    const double *lambda = mxGetPr(prhs[2]);

    // Fourth input
    bool zero_diag = false;
    if (nrhs >= 4) {
        if (!mxIsLogicalScalar(prhs[3])) {
            mexErrMsgTxt("zeroID should be true or false");
        }
        zero_diag = mxIsLogicalScalarTrue(prhs[3]);

        if (nRows < nCols) {
            mexErrMsgTxt("Cannot zero diagonal if nRows < nCols");
        }
    }

    if (k == nRows && zero_diag) {
        mexErrMsgTxt("k == nRows with zero_diag is infeasible.");
    }

    // Fifth input
    bool sparse_output = false;
    if (nrhs >= 5) {
        if (!mxIsLogicalScalar(prhs[4])) {
            mexErrMsgTxt("sparseOutput should be true or false");
        }
        sparse_output = mxIsLogicalScalarTrue(prhs[4]);
    }

    // Y will be a sparse matrix
    if (sparse_output) {
        /*
         * We've got a bit of a special case here.  We know exactly how many
         * non-zeros will be in the array.  And we know exactly how many elements
         * will be in each row.  This makes constructing the compressed sparse
         * matrix a bit easier.
         *
         * There are k non-zeros per column.  So we allocate storage for k*nCols
         * non-zeros. This also means that we know Jc a priori, which means we
         * can parallelize things super nicely.
         */
        size_t nnz = k*nCols; // total number of non-zeros in this matrix
                              // (except for the edge case identified above)
        plhs[0] = mxCreateSparse(nRows, nCols, nnz, mxREAL);
        size_t *Ir = mxGetIr(plhs[0]); // length nnz
        size_t *Jc = mxGetJc(plhs[0]); // length nCols + 1
        double *Pr = mxGetPr(plhs[0]); // length nnz

        if (nCols == 1) { // don't do zeroID for vectors
            GSHP proj;
            proj.run(nullptr, X, nRows, k, lambda[0]);

            const size_t *inds;
            const double *y;
            proj.get_solution_data(inds, y);

            Jc[0] = 0;
            Jc[1] = k;
            for (size_t i=0; i<k; ++i) {
                // Y[inds[i]]
                Ir[i] = inds[i];
                Pr[i] = y[i];
            }

        } else {
            Jc[0] = 0;
            #pragma omp parallel num_threads(opt.num_threads)
            {
                GSHP proj;
                const size_t *inds;
                const double *y;

                #pragma omp for schedule(static)
                for (size_t j=0; j<nCols; ++j) {
                    double lambda_val = (n_lambda > 1) ? lambda[j] : lambda[0];
                    if (zero_diag) {
                        proj.run(nullptr, X + nRows*j, nRows, k, lambda_val, j);
                    } else {
                        proj.run(nullptr, X + nRows*j, nRows, k, lambda_val);
                    }

                    proj.get_solution_data(inds, y);

                    Jc[j+1] = (j+1)*k;
                    for (size_t i=0; i<k; ++i) {
                        // Y[inds[i],j]
                        Ir[k*j + i] = inds[i];
                        Pr[k*j + i] = y[i];
                    }
                }
            }
        }

    // Y is gonna be a full matrix
    } else {
        // Will init all values to zero, which we want
        plhs[0] = mxCreateNumericMatrix(nRows, nCols, mxDOUBLE_CLASS, mxREAL);
        double *Y = mxGetPr(plhs[0]);

        if (nCols == 1) { // don't do zeroID for vectors
            GSHP proj;
            proj.run(Y, X, nRows, k, lambda[0]);

        } else {
            #pragma omp parallel num_threads(opt.num_threads)
            {
                GSHP proj;

                #pragma omp for schedule(static)
                for (size_t j=0; j<nCols; ++j) {
                    double lambda_val = (n_lambda > 1) ? lambda[j] : lambda[0];
                    if (zero_diag) {
                        proj.run(Y + nRows*j, X + nRows*j, nRows, k, lambda_val, j);
                    } else {
                        proj.run(Y + nRows*j, X + nRows*j, nRows, k, lambda_val);
                    }
                }
            }
        }
    }
}

/*
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * Neither the name of the project's author nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __RANDSVD_H
#define __RANDSVD_H

#include <cmath>

#include <iostream>
#include <limits>

#include <boost/random.hpp>

#include <Eigen/Dense>

namespace statismo {
/**
 * TODO comment and add reference to paper
 */
template <typename ScalarType>
class RandSVD {
  public:

    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorType;
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> MatrixType;

    RandSVD(const MatrixType& A, unsigned k) {

        unsigned n = A.rows();


        static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
        static boost::normal_distribution<> dist(0, 1);
        static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

        // create gaussian random amtrix
        MatrixType Omega(n, k);
        for (unsigned i =0; i < n ; i++) {
            for (unsigned j = 0; j < k ; j++) {
                Omega(i,j) = r();
            }
        }


        MatrixType Y = A * A.transpose() * A * Omega;
        Eigen::FullPivHouseholderQR<MatrixType> qr(Y);
        MatrixType Q = qr.matrixQ().leftCols(k + k);

        MatrixType B = Q.transpose() * A;

        typedef Eigen::JacobiSVD<MatrixType> SVDType;
        SVDType SVD(B, Eigen::ComputeThinU);
        MatrixType Uhat = SVD.matrixU();
        m_D = SVD.singularValues();
        m_U = (Q * Uhat).leftCols(k);
    }

    MatrixType matrixU() const {
        return m_U;
    }

    VectorType singularValues() const {
        return m_D;
    }


  private:
    VectorType m_D;
    MatrixType m_U;
};


} // namespace statismo;
#endif // __LANCZOS_H

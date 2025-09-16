#ifndef NYSTROM_H
#define NYSTROM_H

/*
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Statismo is licensed under the BSD licence (3 clause) license
 */

#include <boost/thread.hpp>

#include "CommonTypes.h"
#include "Config.h"
#include "Kernels.h"
#include "RandSVD.h"
#include "Representer.h"

namespace statismo {

/**
 * Computes the Nystrom approximation of a given kernel
 * The type parameter T is the type of the dataset (e.g. Mesh, Image) for which the nystom approximation is computed
 */
template <class T>
class Nystrom {
  public:


    typedef typename Representer<T>::PointType PointType;
    typedef statismo::Domain<typename Representer<T>::PointType> DomainType;
    typedef typename DomainType::DomainPointsListType DomainPointsListType;

    static Nystrom* Create(const Representer<T>* representer, const MatrixValuedKernel<PointType>& kernel, unsigned numEigenfunctions, unsigned numberOfPointsForApproximation) {
        return new Nystrom(representer, kernel, numEigenfunctions, numberOfPointsForApproximation);
    }


    /**
     * Returns a d x n matrix, which holds the d-dimension value of all the n eigenfunctiosn at the given point
     */
    MatrixType computeEigenfunctionsAtPoint(const PointType& pt) const {

        unsigned kernelDim = m_kernel.GetDimension();

        // for every domain point x in the list, we compute the kernel vector
        // kx = (k(x, x1), ... k(x, xm))
        // since the kernel is matrix valued, kx is actually a matrix
        MatrixType kxi = MatrixType::Zero(kernelDim, m_nystromPoints.size() * kernelDim);

        for (unsigned j = 0; j < m_nystromPoints.size(); j++) {
            kxi.block(0, j * kernelDim, kernelDim, kernelDim) = m_kernel(pt, m_nystromPoints[j]);
        }


        MatrixType resMat = MatrixType::Zero(kernelDim, m_numEigenfunctions);
        for (unsigned j = 0; j < m_numEigenfunctions; j++) {
            MatrixType x = (kxi * m_nystromMatrix.col(j));
            resMat.block(0, j, kernelDim, 1) = x;
        }
        return resMat;
    }


    /**
     * Returns a vector of size n, where n is the number of eigenfunctions/eigenvalues that were approximated
     */
    const VectorType& getEigenvalues() const {
        return m_eigenvalues;
    }


  private:


    Nystrom(const Representer<T>* representer, const MatrixValuedKernel<PointType>& kernel, unsigned numEigenfunctions, unsigned numberOfPointsForApproximation)
        : m_representer(representer), m_kernel(kernel), m_numEigenfunctions(numEigenfunctions) {

        DomainType domain = m_representer->GetDomain();
        m_nystromPoints = getNystromPoints(domain, numberOfPointsForApproximation);
        unsigned numDomainPoints = domain.GetNumberOfPoints();

        // compute a eigenvalue decomposition of the kernel matrix, evaluated at the points used for the
        // nystrom approximation

        MatrixType U; // will hold the eigenvectors (principal components)
        VectorType D; // will hold the eigenvalues (variance)
        computeKernelMatrixDecomposition(&kernel, m_nystromPoints, numEigenfunctions, U, D);


        // precompute the part of the nystrom approximation, which is independent of the domain point
        float normFactor = static_cast<float>(m_nystromPoints.size()) / static_cast<float>(numDomainPoints);
        m_nystromMatrix = std::sqrt(normFactor) * (U.leftCols(numEigenfunctions)
                          * D.topRows(numEigenfunctions).asDiagonal().inverse());

        m_eigenvalues = (1.0f / normFactor) * D.topRows(numEigenfunctions);

    }


    /*
     * Returns a random set of points from the domain.
     *
     * @param domain the domain to sample from
     * @param numberOfPoints the size of the sample
     */
    std::vector<PointType> getNystromPoints(DomainType& domain, unsigned numberOfPoints) const {

        numberOfPoints = std::min(numberOfPoints, domain.GetNumberOfPoints());

        std::vector<PointType> shuffledDomainPoints = domain.GetDomainPoints();
        std::random_shuffle ( shuffledDomainPoints.begin(), shuffledDomainPoints.end() );

        return std::vector<PointType>(shuffledDomainPoints.begin(), shuffledDomainPoints.begin() + numberOfPoints);
    }




    /**
     * Compute the kernel matrix for all points given in xs and
     * return a matrix U with the first numComponents eigenvectors and a vector D with
     * the corresponding eigenvalues of this kernel matrix
     */
    void computeKernelMatrixDecomposition(const MatrixValuedKernel<PointType>* kernel,
                                          const std::vector<PointType>& xs, unsigned numComponents,
                                          MatrixType& U, VectorType& D) const {
        unsigned kernelDim = kernel->GetDimension();

        unsigned n = xs.size();
        MatrixTypeDoublePrecision K = MatrixTypeDoublePrecision::Zero(
                                          n * kernelDim, n * kernelDim);
        for (unsigned i = 0; i < n; ++i) {
            for (unsigned j = i; j < n; ++j) {
                MatrixType k_xixj = (*kernel)(xs[i], xs[j]);
                for (unsigned d1 = 0; d1 < kernelDim; d1++) {
                    for (unsigned d2 = 0; d2 < kernelDim; d2++) {
                        double elem_d1d2 = k_xixj(d1, d2);
                        K(i * kernelDim + d1, j * kernelDim + d2) = elem_d1d2;
                        K(j * kernelDim + d2, i * kernelDim + d1) = elem_d1d2;
                    }
                }
            }
        }

        typedef RandSVD<double> SVDType;
        SVDType svd(K, numComponents * kernelDim);
        U = svd.matrixU().cast<ScalarType>();
        D = svd.singularValues().cast<ScalarType>();
    }


    // private, to prevent use
    Nystrom();
    Nystrom<T>& operator=(const Nystrom<T>& rhs);
    Nystrom(const Nystrom<T>& orig);


    //
    // members
    //

    const Representer<T>* m_representer;
    MatrixType m_nystromMatrix;
    VectorType m_eigenvalues;
    std::vector<PointType> m_nystromPoints;
    const MatrixValuedKernel<PointType>& m_kernel;
    unsigned m_numEigenfunctions;


};


} // namespace statismo
#endif // NYSTROM_H

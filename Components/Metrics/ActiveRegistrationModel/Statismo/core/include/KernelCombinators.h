/**
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *         Thomas Gerig  (thomas.gerig@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Statismo is licensed under the BSD licence (3 clause) license
 */

#ifndef KERNELCOMBINATORS_H
#define KERNELCOMBINATORS_H

#include <boost/scoped_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>

#include "CommonTypes.h"
#include "Kernels.h"
#include "Nystrom.h"
#include "Representer.h"

namespace statismo {

/**
 * A (matrix valued) kernel, which represents the sum of two matrix valued kernels.
 */
template<class TPoint>
class SumKernel: public MatrixValuedKernel<TPoint> {
  public:

    typedef MatrixValuedKernel<TPoint> MatrixValuedKernelType;


    SumKernel(const MatrixValuedKernelType* lhs,
              const MatrixValuedKernelType* rhs) :
        MatrixValuedKernelType(lhs->GetDimension()),
        m_lhs(lhs),
        m_rhs(rhs) {
        if (lhs->GetDimension() != rhs->GetDimension()) {
            throw StatisticalModelException(
                "Kernels in SumKernel must have the same dimensionality");
        }
    }

    MatrixType operator()(const TPoint& x, const TPoint& y) const {
        return (*m_lhs)(x, y) + (*m_rhs)(x, y);
    }

    std::string GetKernelInfo() const {
        std::ostringstream os;
        os << m_lhs->GetKernelInfo() << " + " << m_rhs->GetKernelInfo();
        return os.str();
    }

  private:
    const MatrixValuedKernelType* m_lhs;
    const MatrixValuedKernelType* m_rhs;
};



/**
 * A (matrix valued) kernel, which represents the product of two matrix valued kernels.
 */

template<class TPoint>
class ProductKernel: public MatrixValuedKernel<TPoint> {

  public:

    typedef MatrixValuedKernel<TPoint> MatrixValuedKernelType;

    ProductKernel(const MatrixValuedKernelType* lhs,
                  const MatrixValuedKernelType* rhs) :
        MatrixValuedKernelType(lhs->GetDimension()), m_lhs(lhs), m_rhs(
            rhs) {
        if (lhs->GetDimension() != rhs->GetDimension()) {
            throw StatisticalModelException(
                "Kernels in SumKernel must have the same dimensionality");
        }

    }

    MatrixType operator()(const TPoint& x, const TPoint& y) const {
        return (*m_lhs)(x, y) * (*m_rhs)(x, y);
    }

    std::string GetKernelInfo() const {
        std::ostringstream os;
        os << m_lhs->GetKernelInfo() << " * " << m_rhs->GetKernelInfo();
        return os.str();
    }

  private:
    const MatrixValuedKernelType* m_lhs;
    const MatrixValuedKernelType* m_rhs;
};


/**
 * A (matrix valued) kernel, which represents a scalar multiple of a matrix valued kernel.
 */

template<class TPoint>
class ScaledKernel: public MatrixValuedKernel<TPoint> {
  public:


    typedef MatrixValuedKernel<TPoint> MatrixValuedKernelType;


    ScaledKernel(const MatrixValuedKernelType* kernel,
                 double scalingFactor) :
        MatrixValuedKernelType(kernel->GetDimension()), m_kernel(kernel), m_scalingFactor(scalingFactor) {
    }

    MatrixType operator()(const TPoint& x, const TPoint& y) const {
        return (*m_kernel)(x, y) * m_scalingFactor;
    }
    std::string GetKernelInfo() const {
        std::ostringstream os;
        os << (*m_kernel).GetKernelInfo() << " * " << m_scalingFactor;
        return os.str();
    }

  private:
    const MatrixValuedKernelType* m_kernel;
    double m_scalingFactor;
};


/**
 * Takes a scalar valued kernel and creates a matrix valued kernel of the given dimension.
 * The new kernel models the output components as independent, i.e. if K(x,y) is a scalar valued Kernel,
 * the matrix valued kernel becomes Id*K(x,y), where Id is an identity matrix of dimensionality d.
 */
template<class TPoint>
class UncorrelatedMatrixValuedKernel: public MatrixValuedKernel<TPoint> {
  public:

    typedef MatrixValuedKernel<TPoint> MatrixValuedKernelType;

    UncorrelatedMatrixValuedKernel(
        const ScalarValuedKernel<TPoint>* scalarKernel,
        unsigned dimension) :
        MatrixValuedKernelType(	dimension), m_kernel(scalarKernel),
        m_ident(MatrixType::Identity(dimension, dimension)) {
    }

    MatrixType operator()(const TPoint& x, const TPoint& y) const {

        return m_ident * (*m_kernel)(x, y);
    }

    virtual ~UncorrelatedMatrixValuedKernel() {
    }

    std::string GetKernelInfo() const {
        std::ostringstream os;
        os << "UncorrelatedMatrixValuedKernel(" << (*m_kernel).GetKernelInfo()
           << ", " << this->m_dimension << ")";
        return os.str();
    }

  private:

    const ScalarValuedKernel<TPoint>* m_kernel;
    MatrixType m_ident;

};


/**
 * Base class for defining a tempering function for the SpatiallyVaryingKernel
 */
template <class TPoint>
class TemperingFunction {
  public:
    virtual double operator()(const TPoint& pt) const = 0;
    virtual ~TemperingFunction() {}
};

/**
 * spatially-varing kernel, as described in the paper:
 *
 * T. Gerig, K. Shahim, M. Reyes, T. Vetter, M. Luethi
 * Spatially varying registration using gaussian processes
 * Miccai 2014
 */
template<class T>
class SpatiallyVaryingKernel : public MatrixValuedKernel<typename Representer<T>::PointType> {

    typedef boost::unordered_map<statismo::VectorType, statismo::MatrixType> CacheType;

  public:

    typedef Representer<T> RepresenterType;
    typedef typename RepresenterType::PointType PointType;


    /**
     * @brief Make a given kernel spatially varying according to the given tempering function
     * @param representer, A representer which defines the domain over which the approximation is done
     * @param kernel The kernel that is made spatially adaptive
     * @param eta The tempering function that defines the amount of tempering for each point in the domain
     * @param numEigenfunctions The number of eigenfunctions to be used for the approximation
     * @param numberOfPointsForApproximation The number of points used for the nystrom approximation
     * @param cacheValues Cache result of eigenfunction computations. Greatly speeds up the computation.
     */
    SpatiallyVaryingKernel(const RepresenterType* representer, const MatrixValuedKernel<PointType>& kernel, const TemperingFunction<PointType>& eta, unsigned numEigenfunctions, unsigned numberOfPointsForApproximation = 0, bool cacheValues = true)
        : m_representer(representer),
          m_eta(eta),
          m_nystrom(Nystrom<T>::Create(representer, kernel, numEigenfunctions, numberOfPointsForApproximation == 0 ? numEigenfunctions * 2 : numberOfPointsForApproximation)),
          m_eigenvalues(m_nystrom->getEigenvalues()),
          m_cacheValues(cacheValues),
          MatrixValuedKernel<PointType>(kernel.GetDimension()) {
    }

    inline MatrixType operator()(const PointType& x, const PointType& y) const {

        MatrixType sum = MatrixType::Zero(this->m_dimension, this->m_dimension);

        float eta_x = m_eta(x);
        float eta_y = m_eta(y);


        statismo::MatrixType phisAtX = phiAtPoint(x);
        statismo::MatrixType phisAtY = phiAtPoint(y);

        double largestTemperedEigenvalue = std::pow(m_eigenvalues(0), (eta_x + eta_y)/2);

        for (unsigned i = 0; i < m_eigenvalues.size(); ++i) {

            float temperedEigenvalue = std::pow(m_eigenvalues(i), (eta_x + eta_y)/2);

            // ignore too small eigenvalues, as they don't contribute much.
            // (the eigenvalues are ordered, all the following are smaller and can also be ignored)
            if (temperedEigenvalue / largestTemperedEigenvalue < 1e-6)  {
                break;
            } else {
                sum += phisAtX.col(i) * phisAtY.col(i).transpose() * temperedEigenvalue;
            }
        }
        // normalize such that the largest eigenvalue is unaffected by the tempering
        float normalizationFactor = largestTemperedEigenvalue / m_eigenvalues(0);
        sum *= 1.0 / normalizationFactor;
        return sum;
    }


    virtual ~SpatiallyVaryingKernel() {
    }

    std::string GetKernelInfo() const {
        std::ostringstream os;
        os << "SpatiallyVaryingKernel";
        return os.str();
    }




  private:

    // returns a d x n matrix holding the value of all n eigenfunctions evaluated at the given point.
    const statismo::MatrixType phiAtPoint(const PointType& pt) const {

        statismo::MatrixType v;
        if (m_cacheValues) {
            // we need to convert the point to a vector, as the function hash_value (required by boost)
            // is not defined for an arbitrary point.
            const VectorType ptAsVec = this->m_representer->PointToVector(pt);
            _phiCacheLock.lock();
            typename CacheType::const_iterator got = m_phiCache.find (ptAsVec);
            _phiCacheLock.unlock();
            if (got == m_phiCache.end()) {
                v = m_nystrom->computeEigenfunctionsAtPoint(pt);
                _phiCacheLock.lock();
                m_phiCache.insert(std::make_pair(ptAsVec, v));
                _phiCacheLock.unlock();
            } else {
                v = got->second;
            }
        } else {
            v = m_nystrom->computeEigenfunctionsAtPoint(pt);
        }
        return v;
    }


    //
    // members

    const RepresenterType* m_representer;
    boost::scoped_ptr<Nystrom<T> > m_nystrom;
    statismo::VectorType m_eigenvalues;
    const  TemperingFunction<PointType>& m_eta;
    bool m_cacheValues;
    mutable CacheType m_phiCache;
    mutable boost::mutex _phiCacheLock;
};



}

#endif // KERNELCOMBINATORS_H

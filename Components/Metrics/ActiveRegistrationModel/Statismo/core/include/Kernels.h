/**
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Statismo is licensed under the BSD licence (3 clause) license
 */


#ifndef __KERNELS_H
#define __KERNELS_H

#include <cmath>

#include <vector>
#include <memory>
#include <functional>

#include "CommonTypes.h"
#include "Config.h"
#include "ModelInfo.h"
#include "Representer.h"
#include "StatisticalModel.h"


namespace statismo {

/**
 * Base class from which all ScalarValuedKernels derive.
 */
template<class TPoint>
class ScalarValuedKernel {
  public:

    /**
     * Create a new scalar valued kernel.
     */
    ScalarValuedKernel() {	}

    virtual ~ScalarValuedKernel() {
    }

    /**
     * Evaluate the kernel function at the points x and y
     */
    virtual double operator()(const TPoint& x, const TPoint& y) const = 0;

    /**
     * Return a description of this kernel
     */
    virtual std::string GetKernelInfo() const = 0;

};


/**
 * Base class for all matrix valued kernels
 */
template<class TPoint>
class MatrixValuedKernel {
  public:

    /**
     * Create a new MatrixValuedKernel
     */
    MatrixValuedKernel(unsigned dim) :
        m_dimension(dim) {
    }

    /**
     * Evaluate the kernel at the points x and y
     */
    virtual MatrixType operator()(const TPoint& x,
                                  const TPoint& y) const = 0;

    /**
     * Return the dimensionality of the kernel (i.e. the size of the matrix)
     */
    virtual unsigned GetDimension() const {
        return m_dimension;
    }
    ;
    virtual ~MatrixValuedKernel() {
    }

    /**
     * Return a description of this kernel.
     */
    virtual std::string GetKernelInfo() const = 0;

  protected:
    unsigned m_dimension;

};

template<class T>
class StatisticalModelKernel: public MatrixValuedKernel<typename Representer<T>::PointType > {
  public:

    typedef Representer<T> RepresenterType;
    typedef typename RepresenterType::PointType PointType;
    typedef StatisticalModel<T> StatisticalModelType;

    StatisticalModelKernel(const StatisticalModelType* model) :
        MatrixValuedKernel<PointType>(model->GetRepresenter()->GetDimensions()), m_statisticalModel(model) {
    }

    virtual ~StatisticalModelKernel() {
    }

    inline MatrixType operator()(const PointType& x, const PointType& y) const {
        MatrixType m = m_statisticalModel->GetCovarianceAtPoint(x, y);
        return m;
    }

    std::string GetKernelInfo() const {
        return "StatisticalModelKernel";
    }

  private:
    const StatisticalModelType* m_statisticalModel;
};



} // namespace statismo

#endif // __KERNELS_H

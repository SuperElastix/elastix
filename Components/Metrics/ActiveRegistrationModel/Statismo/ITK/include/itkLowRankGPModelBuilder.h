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

#ifndef ITKLOWRANKMODELBUILDER_H_
#define ITKLOWRANKMODELBUILDER_H_

#include <itkObject.h>
#include <itkObjectFactory.h>

#include "itkStatisticalModel.h"

#include "Kernels.h"
#include "LowRankGPModelBuilder.h"
#include "Representer.h"
#include "statismoITKConfig.h"

namespace itk {

/**
 * \brief ITK Wrapper for the statismo::LowRankGPModelBuilder class.
 * \see statismo::LowRankGPModelBuilder for detailed documentation.
 */

template<class T>
class LowRankGPModelBuilder: public Object {
  public:

    typedef LowRankGPModelBuilder Self;
    typedef statismo::Representer<T> RepresenterType;
    typedef Object Superclass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<const Self> ConstPointer;

    itkNewMacro (Self);
    itkTypeMacro( LowRankGPModelBuilder, Object );

    typedef statismo::LowRankGPModelBuilder<T> ImplType;
    typedef itk::StatisticalModel<T> StatisticalModelType;
    typedef statismo::MatrixValuedKernel<typename RepresenterType::PointType> MatrixValuedKernelType;

    LowRankGPModelBuilder() :
        m_impl(0) {
    }


    void SetstatismoImplObj(ImplType* impl) {
        if (m_impl) {
            delete m_impl;
        }
        m_impl = impl;
    }


    void SetRepresenter(const RepresenterType* representer) {
        SetstatismoImplObj(ImplType::Create(representer));
    }

    virtual ~LowRankGPModelBuilder() {
        if (m_impl) {
            delete m_impl;
            m_impl = 0;
        }
    }

    typename StatisticalModelType::Pointer BuildNewZeroMeanModel(
        const MatrixValuedKernelType& kernel, unsigned numComponents,
        unsigned numPointsForNystrom = 500) const {
        if (m_impl == 0) {
            itkExceptionMacro(<< "Model not properly initialized. Maybe you forgot to call SetRepresenter");
        }


        statismo::StatisticalModel<T>* model_statismo = 0;
        try {
            model_statismo = this->m_impl->BuildNewZeroMeanModel(kernel, numComponents, numPointsForNystrom);

        } catch (statismo::StatisticalModelException& s) {
            itkExceptionMacro(<< s.what());
        }

        typename StatisticalModel<T>::Pointer model_itk = StatisticalModel<T>::New();
        model_itk->SetstatismoImplObj(model_statismo);
        return model_itk;

    }

    typename StatisticalModelType::Pointer BuildNewModel(typename RepresenterType::DatasetType* mean, const MatrixValuedKernelType& kernel, unsigned numComponents, unsigned numPointsForNystrom = 500) {
        if (m_impl == 0) {
            itkExceptionMacro(<< "Model not properly initialized. Maybe you forgot to call SetRepresenter");
        }

        statismo::StatisticalModel<T>* model_statismo = 0;
        try {
            model_statismo = this->m_impl->BuildNewModel(mean, kernel, numComponents, numPointsForNystrom);

        } catch (statismo::StatisticalModelException& s) {
            itkExceptionMacro(<< s.what());
        }


        typename StatisticalModel<T>::Pointer model_itk = StatisticalModel<T>::New();
        model_itk->SetstatismoImplObj(model_statismo);
        return model_itk;

    }

  private:
    LowRankGPModelBuilder(const LowRankGPModelBuilder& orig);
    LowRankGPModelBuilder& operator=(const LowRankGPModelBuilder& rhs);

    ImplType* m_impl;
};

}

#endif /* ITKLOWRANKMODELBUILDER_H_ */

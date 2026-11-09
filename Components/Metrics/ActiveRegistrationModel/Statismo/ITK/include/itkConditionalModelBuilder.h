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


#ifndef ITKMODELBUILDER_H_
#define ITKMODELBUILDER_H_

#include <itkObject.h>
#include <itkObjectFactory.h>

#include "itkDataManager.h"
#include "itkStatisticalModel.h"
#include "ConditionalModelBuilder.h"
#include "statismoITKConfig.h"

namespace itk {

/**
 * \brief ITK Wrapper for the statismo::PCAModelBuilder class.
 * \see statismo::PCAModelBuilder for detailed documentation.
 */
template <class Representer>
class ConditionalModelBuilder : public Object {
  public:

    typedef ConditionalModelBuilder            Self;
    typedef Object	Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    itkNewMacro( Self );
    itkTypeMacro( ConditionalModelBuilder, Object );


    typedef statismo::ConditionalModelBuilder<Representer> ImplType;
    typedef statismo::DataManager<Representer> DataManagerType;
    typedef typename DataManagerType::SampleDataStructureListType SampleDataStructureListType;

    ConditionalModelBuilder() : m_impl(ImplType::Create()) {}

    virtual ~ConditionalModelBuilder() {
        if (m_impl) {
            delete m_impl;
            m_impl = 0;
        }
    }

    template <class F>
    typename boost::result_of<F()>::type callstatismoImpl(F f) const {
        try {
            return f();
        } catch (statismo::StatisticalModelException& s) {
            itkExceptionMacro(<< s.what());
        }
    }


    typename StatisticalModel<Representer>::Pointer
    BuildNewModel(SampleDataStructureListType SampleDataStructureList,
                  const typename statismo::ConditionalModelBuilder<Representer>::SurrogateTypeVectorType& surrogateTypes,
                  const typename statismo::ConditionalModelBuilder<Representer>::CondVariableValueVectorType& conditioningInfo,
                  float noiseVariance,
                  double modelVarianceRetained
                 ) {
        statismo::StatisticalModel<Representer>* model_statismo = callstatismoImpl(boost::bind(&ImplType::BuildNewModel, this->m_impl, SampleDataStructureList, surrogateTypes, conditioningInfo, noiseVariance, modelVarianceRetained));
        typename StatisticalModel<Representer>::Pointer model_itk = StatisticalModel<Representer>::New();
        model_itk->SetstatismoImplObj(model_statismo);
        return model_itk;
    }


  private:
    ConditionalModelBuilder(const ConditionalModelBuilder& orig);
    ConditionalModelBuilder& operator=(const ConditionalModelBuilder& rhs);

    ImplType* m_impl;
};


}

#endif /* ITKMODELBUILDER_H_ */

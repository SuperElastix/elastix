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


#ifndef ITK_DATAMANAGER_WITH_SURROGATES_H_
#define ITK_DATAMANAGER_WITH_SURROGATES_H_

#include <boost/bind.hpp>
#include <boost/utility/result_of.hpp>

#include <itkObject.h>
#include <itkObjectFactory.h>

#include "DataManagerWithSurrogates.h"
#include "statismoITKConfig.h"

namespace itk {


/**
 * \brief ITK Wrapper for the statismo::DataManager class.
 * \see statismo::DataManager for detailed documentation.
 */
template <class Representer>
class DataManagerWithSurrogates : public statismo::DataManager<Representer> {
  public:


    typedef DataManagerWithSurrogates            Self;
    typedef statismo::DataManager<Representer>	Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    itkNewMacro( Self );
    itkTypeMacro( DataManagerWithSurrogates, Object );


    typedef statismo::DataManagerWithSurrogates<Representer> ImplType;

    template <class F>
    typename boost::result_of<F()>::type callstatismoImpl(F f) const {
        if (m_impl == 0) {
            itkExceptionMacro(<< "Model not properly initialized. Maybe you forgot to call SetParameters");
        }
        try {
            return f();
        } catch (statismo::StatisticalModelException& s) {
            itkExceptionMacro(<< s.what());
        }
    }


    DataManagerWithSurrogates() : m_impl(0) {}

    virtual ~DataManagerWithSurrogates() {
        if (m_impl) {
            delete m_impl;
            m_impl = 0;
        }
    }


    void SetstatismoImplObj(ImplType* impl) {
        if (m_impl) {
            delete m_impl;
        }
        m_impl = impl;
    }

    void SetRepresenterAndSurrogateFilename(const Representer* representer, const char* surrogTypeFilename) {
        SetstatismoImplObj(ImplType::Create(representer, surrogTypeFilename));
    }

    void SetRepresenter(const Representer* representer) {
        itkExceptionMacro(<< "Please call SetRepresenterAndSurrogateFilename to initialize the object");
    }



    void AddDatasetWithSurrogates(typename Representer::DatasetConstPointerType ds,
                                  const char* datasetURI,
                                  const char* surrogateFilename) {
        callstatismoImpl(boost::bind(&ImplType::AddDatasetWithSurrogates, this->m_impl, ds, datasetURI, surrogateFilename));
    }


  private:

    DataManagerWithSurrogates(const DataManagerWithSurrogates& orig);
    DataManagerWithSurrogates& operator=(const DataManagerWithSurrogates& rhs);

    ImplType* m_impl;
};


}

#endif /* ITK_DATAMANAGER_WITH_SURROGATES_H_ */

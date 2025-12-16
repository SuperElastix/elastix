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


#ifndef ITK_DATAMANAGER_H_
#define ITK_DATAMANAGER_H_

#include <boost/bind.hpp>
#include <boost/utility/result_of.hpp>

#include <itkObject.h>
#include <itkObjectFactory.h>

#include "DataManager.h"
#include "statismoITKConfig.h"

namespace itk {


/**
 * \brief ITK Wrapper for the statismo::DataManager class.
 * \see statismo::DataManager for detailed documentation.
 */
template <class T>
class DataManager : public Object {
  public:


    typedef DataManager            Self;
    typedef Object	Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    itkNewMacro( Self );
    itkTypeMacro( DataManager, Object );


    typedef statismo::DataManager<T> ImplType;
    typedef typename statismo::DataManager<T>::DataItemType     DataItemType;
    typedef typename statismo::DataManager<T>::DataItemListType DataItemListType;
    typedef statismo::Representer<T> RepresenterType;

    template <class F>
    typename boost::result_of<F()>::type callstatismoImpl(F f) const {
        if (m_impl == 0) {
            itkExceptionMacro(<< "Model not properly initialized. Maybe you forgot to call SetRepresenter");
        }
        try {
            return f();
        } catch (statismo::StatisticalModelException& s) {
            itkExceptionMacro(<< s.what());
        }
    }


    DataManager() : m_impl(0) {}

    virtual ~DataManager() {
        if (m_impl) {
            delete m_impl;
            m_impl = 0;
        }
    }

    ImplType* GetstatismoImplObj() const {
        return m_impl;
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

    void AddDataset(typename RepresenterType::DatasetType* ds, const char* filename) {
        callstatismoImpl(boost::bind(&ImplType::AddDataset, this->m_impl, ds, filename));
    }

    void Load(const char* filename) {
        try {
            SetstatismoImplObj(ImplType::Load(filename));
        } catch (statismo::StatisticalModelException& s) {
            itkExceptionMacro(<< s.what());
        }
    }

    void Save(const char* filename) {
        callstatismoImpl(boost::bind(&ImplType::Save, this->m_impl, filename));
    }

    typename statismo::DataManager<T>::DataItemListType GetData() const {
        return callstatismoImpl(boost::bind(&ImplType::GetData, this->m_impl));
    }


  private:
    DataManager(const DataManager& orig);
    DataManager& operator=(const DataManager& rhs);

    ImplType* m_impl;
};


}

#endif /* ITK_DATAMANAGER_H_ */

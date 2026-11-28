/*
 * DataManagerWithSurrogates.h
 *
 * Created by Marcel Luethi and Remi Blanc
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

#ifndef __DATAMANAGERWITHSURROGATES_H_
#define __DATAMANAGERWITHSURROGATES_H_

#include "DataManager.h"

namespace statismo {


/**
 * \brief Manages Training and Test Data for building Statistical Models and provides functionality for Crossvalidation.
 * Manages data together with surrogate information.
 * The surrogate variables are provided through a vector (see DataManager), and can contain both continuous or categorical data.
 * The surrogate data is provided through files. One file for each dataset, and one file describing the types of surrogates. This file is also an ascii file
 * with space or EOL separated values. Those values are either 0 or 1, standing for respectively categorical or continuous variable.
 * This class does not support any missing data, so each dataset must come with a surrogate data file, all of which must contain the same number of entries as the type-file.
 * \sa DataManager
 */
template <typename T>
class DataManagerWithSurrogates : public DataManager<T> {

  public:

    typedef Representer<T> RepresenterType;

    typedef typename RepresenterType::DatasetPointerType DatasetPointerType;
    typedef typename RepresenterType::DatasetConstPointerType DatasetConstPointerType;


    typedef DataItemWithSurrogates<T> DataItemWithSurrogatesType;

    typedef typename DataItemWithSurrogatesType::SurrogateTypeVectorType SurrogateTypeVectorType;

    struct SurrogateTypeInfoType {
        SurrogateTypeVectorType types;
        std::string typeFilename;
    };


    /**
     * Destructor
     */
    virtual ~DataManagerWithSurrogates() {}


    /**
    * Factory method that creates a new instance of a DataManager class
    *
    */
    static DataManagerWithSurrogates<T>* Create(const RepresenterType* representer, const std::string& surrogTypeFilename) {
        return new DataManagerWithSurrogates<T>(representer, surrogTypeFilename);
    }




    /**
     * Add a dataset, together with surrogate information
     * \param datasetFilename
     * \param datasetURI (An URI for the dataset. This info is only added to the metadata).
     * \param surrogateFilename
     */
    void AddDatasetWithSurrogates(DatasetConstPointerType ds,
                                  const std::string& datasetURI,
                                  const std::string& surrogateFilename);

    /**
     * Get a vector indicating the types of surrogates variables (Categorical vs Continuous)
     */
    SurrogateTypeVectorType GetSurrogateTypes() const {
        return m_typeInfo.types;
    }

    /** Returns the source filename defining the surrogate types */
    std::string GetSurrogateTypeFilename() const {
        return m_typeInfo.typeFilename;
    }

    /** Get a structure containing the type info: vector of types, and source filename */
    SurrogateTypeInfoType GetSurrogateTypeInfo() const {
        return m_typeInfo;
    }

  protected:

    /**
     * Loads the information concerning the types of the surrogates variables (categorical=0, continuous=1)
     * => it is assumed to be in a text file with the entries separated by spaces or EOL character
     */
    void LoadSurrogateTypes(const std::string& filename);



    // private - to prevent use
    DataManagerWithSurrogates(const RepresenterType* r, const std::string& filename);

    DataManagerWithSurrogates(const DataManagerWithSurrogates& orig);
    DataManagerWithSurrogates& operator=(const DataManagerWithSurrogates& rhs);

    SurrogateTypeInfoType m_typeInfo;
};

}

#include "DataManagerWithSurrogates.hxx"

#endif /* __DATAMANAGERWITHSURROGATES_H_ */

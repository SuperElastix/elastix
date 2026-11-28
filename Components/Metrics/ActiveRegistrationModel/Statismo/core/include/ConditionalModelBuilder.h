/*
 * ConditionalModelBuilder.h
 *
 * Created by Remi Blanc,
 *
 * Copyright (c) 2011 ETH Zurich
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

#ifndef __CONDITIONALMODELBUILDER_H_
#define __CONDITIONALMODELBUILDER_H_

#include <vector>
#include <memory>

#include "CommonTypes.h"
#include "Config.h"
#include "DataManagerWithSurrogates.h"
#include "ModelBuilder.h"
#include "StatisticalModel.h"

namespace statismo {

/**
 * \brief Creates a StatisticalModel conditioned on some external data
 *
 * The principle of this class is to exploit supplementary information (surrogate variables) describing
 * the samples (e.g. the age and gender of the subject) to generate a conditional statistical model.
 * This class assumes a joint multivariate gaussian distribution of the sample vector and the continuous surrogates
 * Categorical surrogates are taken into account by selecting the subset of samples that fit in the requested categories.
 *
 * For mathematical details and illustrations, see the paper
 * Conditional Variability of Statistical Shape Models Based on Surrogate Variables
 * R. Blanc, M. Reyes, C. Seiler and G. Szekely, In Proc. MICCAI 2009
 *
 * CAVEATS:
 * 	- conditioning on too many categories may lead to small or empty training sets
 * 	- using more surrogate variables than training samples may cause instabilities
 *
 * The class does not implement missing data functionalities.
 *
 * \sa DataManagerWithSurrogates
 */
template <typename T>
class ConditionalModelBuilder : public ModelBuilder<T> {
  public:

    typedef ModelBuilder<T> Superclass;
    typedef typename Superclass::StatisticalModelType StatisticalModelType;

    typedef std::pair<bool, statismo::ScalarType> CondVariableValuePair;		//replace the first element by a bool (indicates whether the variable is in use)
    typedef std::vector<CondVariableValuePair> CondVariableValueVectorType; //replace list by vector, to gain direct access

    typedef DataManagerWithSurrogates<T> DataManagerType;
    typedef typename DataManagerType::DataItemListType DataItemListType;
    typedef typename DataManagerType::DataItemWithSurrogatesType DataItemWithSurrogatesType;
    typedef typename DataManagerType::SurrogateTypeInfoType SurrogateTypeInfoType;

    /**
     * Factory method to create a new ConditionalModelBuilder
     * \param representer The representer
     */
    static ConditionalModelBuilder* Create() {
        return new ConditionalModelBuilder();
    }

    /**
     * Destroy the object.
     * The same effect can be achieved by deleting the object in the usual
     * way using the c++ delete keyword.
     */
    void Delete() {
        delete this;
    }

    /**
     * Builds a new model from the provided data and the requested constraints.
     * \param sampleSet A list training samples with associated surrogate data - typically obtained from a DataManagerWithSurrogates.
     * \param surrogateTypes A vector with length corresponding to the number of surrogate variables, indicating whether a variable is continuous or categorical - typically obtained from a DataManagerWithSurrogates.
     * \param conditioningInfo A vector (length = number of surrogates) indicating which surrogates are used for conditioning, and the conditioning value.
     * \param noiseVariance  The variance of the noise assumed on our data
     * \return a new statistical model
     *
     * \warning The returned model needs to be explicitly deleted by the user of this method.
     */
    StatisticalModelType* BuildNewModel(const DataItemListType& sampleSet,
                                        const SurrogateTypeInfoType& surrogateTypesInfo,
                                        const CondVariableValueVectorType& conditioningInfo,
                                        float noiseVariance,
                                        double modelVarianceRetained = 1) const;

  private:

    unsigned PrepareData(const DataItemListType& DataItemList,
                         const SurrogateTypeInfoType& surrogateTypesInfo,
                         const CondVariableValueVectorType& conditioningInfo,
                         DataItemListType* acceptedSamples,
                         MatrixType* surrogateMatrix,
                         VectorType* conditions) const;

    CondVariableValueVectorType m_conditioningInfo; //keep in storage
};



} // namespace statismo

#include "ConditionalModelBuilder.hxx"

#endif /* __PCAMODELBUILDER_H_ */

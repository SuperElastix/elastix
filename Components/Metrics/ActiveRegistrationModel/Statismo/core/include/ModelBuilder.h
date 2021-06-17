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

#ifndef __MODELBUILDER_H_
#define __MODELBUILDER_H_

#include <vector>
#include <memory>

#include "CommonTypes.h"
#include "DataManager.h"
#include "StatisticalModel.h"

namespace statismo {

/**
 * \brief Common base class for all the model builder classes
 */
template <typename T>
class ModelBuilder {

  public:
    typedef Representer<T> RepresenterType;
    typedef StatisticalModel<T> StatisticalModelType;
    typedef DataManager<T> DataManagerType;
    typedef typename DataManagerType::DataItemListType DataItemListType;

    // Values below this tolerance are treated as 0.
    static const double TOLERANCE;


  protected:

    MatrixType ComputeScores(const MatrixType& X, const StatisticalModelType* model) const {

        MatrixType scores(model->GetNumberOfPrincipalComponents(), X.rows());
        for (unsigned i = 0; i < scores.cols(); i++) {
            scores.col(i) = model->ComputeCoefficientsForSampleVector(X.row(i));
        }
        return scores;
    }


    MatrixType ComputeScores(const DataItemListType& sampleDataList, const StatisticalModelType* model) const {

        unsigned n = sampleDataList.size();
        MatrixType scores(model->GetNumberOfPrincipalComponents(), n);

        unsigned i = 0;
        for (typename DataItemListType::const_iterator it = sampleDataList.begin();
                it != sampleDataList.end(); ++it) {
            // Todo: for sample or for dataset??
            scores.col(i++) = model->ComputeCoefficientsForSampleVector((*it)->GetSampleVector());
        }
        return scores;
    }



    ModelBuilder() {}

    ModelInfo CollectModelInfo() const;

  private:
    // private - to prevent use
    ModelBuilder(const ModelBuilder& orig);
    ModelBuilder& operator=(const ModelBuilder& rhs);

};

template <class Representer>
const double ModelBuilder<Representer>::TOLERANCE = 1e-5;

} // namespace statismo


#endif /* __MODELBUILDER_H_ */

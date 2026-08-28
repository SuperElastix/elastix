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

#ifndef __PCAMODELBUILDER_H_
#define __PCAMODELBUILDER_H_

#include <memory>
#include <vector>

#include "CommonTypes.h"
#include "Config.h"
#include "DataManager.h"
#include "ModelBuilder.h"
#include "ModelInfo.h"
#include "StatisticalModel.h"

namespace statismo {


/**
 * \brief Creates StatisticalModel using Principal Component Analysis.
 *
 * This class implements the classical PCA based approach to Statistical Models.
 */
template <typename T>
class PCAModelBuilder : public ModelBuilder<T> {


  public:

    typedef ModelBuilder<T> Superclass;
    typedef typename Superclass::DataManagerType DataManagerType;
    typedef typename Superclass::StatisticalModelType StatisticalModelType;
    typedef typename DataManagerType::DataItemListType DataItemListType;

    /**
     * @brief The EigenValueMethod enum This type is used to specify which decomposition method resp. eigenvalue solver sould be used. Default is JacobiSVD which is the most accurate but for larger systems quite slow. In this case the SelfAdjointEigensolver is more appropriate (especially, if there are more examples than variables).
     */
    typedef enum { JacobiSVD, SelfAdjointEigenSolver } EigenValueMethod;

    /**
     * Factory method to create a new PCAModelBuilder
     */
    static PCAModelBuilder* Create() {
        return new PCAModelBuilder();
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
     * The desctructor
     */
    virtual ~PCAModelBuilder() {}

    /**
     * Build a new model from the training data provided in the dataManager.
     * \param samples A sampleSet holding the data
     * \param noiseVariance The variance of N(0, noiseVariance) distributed noise on the points.
     * If this parameter is set to 0, we have a standard PCA model. For values > 0 we have a PPCA model.
     * \param computeScores Determines whether the scores (the pca coefficients of the examples) are computed and stored as model info
     * (computing the scores may take a long time for large models).
     * \param method Specifies the method which is used for the decomposition resp. eigenvalue solver.
     *
     * \return A new Statistical model
     * \warning The method allocates a new Statistical Model object, that needs to be deleted by the user.
     */
    StatisticalModelType* BuildNewModel(const DataItemListType& samples, double noiseVariance, bool computeScores = true, EigenValueMethod method = JacobiSVD) const;


  private:
    // to prevent use
    PCAModelBuilder();
    PCAModelBuilder(const PCAModelBuilder& orig);
    PCAModelBuilder& operator=(const PCAModelBuilder& rhs);

    StatisticalModelType* BuildNewModelInternal(const Representer<T>* representer, const MatrixType& X, const VectorType& mu, double noiseVariance, EigenValueMethod method = JacobiSVD) const;


};

} // namespace statismo

#include "PCAModelBuilder.hxx"

#endif /* __PCAMODELBUILDER_H_ */

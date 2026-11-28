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


#ifndef __POSTERIORMODELBUILDER_H_
#define __POSTERIORMODELBUILDER_H_

#include <list>
#include <vector>

#include "CommonTypes.h"
#include "Config.h"
#include "DataManager.h"
#include "ModelBuilder.h"
#include "Representer.h"
#include "StatisticalModel.h"

namespace statismo {


/**
 * \brief Given a statistical model (prior) and a set of point constraints (likelihood), generate a new PCA model (posterior).
 *
 * This class builds a StatisticalModel, just as PCAModelBuilder. However, in addition to the data,
 * this model builder also takes as input a set of point constraints, i.e. known values for points.
 * The resulting model will satisfy these constraints, and thus has a much lower variability than an
 * unconstrained model would have.
 *
 * For mathematical detailes see the paper
 * Posterior Shape Models
 * Thomas Albrecht, Marcel Luethi, Thomas Gerig, Thomas Vetter
 * Medical Image Analysis 2013
 *
 * Add method that allows for the use of the pointId in the constraint.
 */
template <typename T>
class PosteriorModelBuilder : public ModelBuilder<T> {
  public:

    typedef Representer<T> RepresenterType;
    typedef ModelBuilder<T> Superclass;
    typedef typename Superclass::DataManagerType 				DataManagerType;
    typedef typename Superclass::StatisticalModelType 	StatisticalModelType;
    typedef typename RepresenterType::ValueType ValueType;
    typedef typename RepresenterType::PointType PointType;
    typedef typename StatisticalModelType::PointValueListType PointValueListType;
    typedef typename DataManagerType::DataItemListType DataItemListType;


    typedef typename StatisticalModelType::PointValuePairType PointValuePairType;
    typedef typename StatisticalModelType::PointCovarianceMatrixType PointCovarianceMatrixType;
    typedef typename StatisticalModelType::PointValueWithCovariancePairType PointValueWithCovariancePairType;
    typedef typename StatisticalModelType::PointValueWithCovarianceListType PointValueWithCovarianceListType;

    /**
     * Factory method to create a new PosteriorModelBuilder
     * \param representer The representer
     */
    static PosteriorModelBuilder* Create() {
        return new PosteriorModelBuilder();
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
     * destructor
     */
    virtual ~PosteriorModelBuilder() {}

    /**
     * Builds a new model from the data provided in the dataManager, and the given constraints.
     * This version of the function assumes a noise with a uniform uncorrelated variance
     * of the form pointValueNoiseVariance * identityMatrix at every given point.
     * \param DataItemList The list holding the data the model is built from
     * \param pointValues A list of (point, value) pairs with the known values.
     * \param pointValueNoiseVariance The variance of the estimated error at the known points (the pointValues)
     * \param noiseVariance  The variance of the noise assumed on our data
     * \return a new statistical model
     *
     * \warning The returned model needs to be explicitly deleted by the user of this method.
     */
    StatisticalModelType* BuildNewModel(const DataItemListType& dataItemList,
                                        const PointValueListType& pointValues,
                                        double pointValueNoiseVariance,
                                        double noiseVariance) const;


    /**
     * Builds a new model from the data provided in the dataManager, and the given constraints.
     * For this version of the function, the covariance matrix of the noise needs to be specified for
     * every point. These covariance matrices are passed in the pointValuesWithCovariance list.
     *
     * \param DataItemList The list holding the data the model is built from
     * \param pointValuesWithCovariance A list of ((point,value), covarianceMatrix) for each known value.
     * \param noiseVariance  The variance of the noise assumed on our data
     * \return a new statistical model
     *
     * \warning The returned model needs to be explicitly deleted by the user of this method.
     */
    StatisticalModelType* BuildNewModel(const DataItemListType& DataItemList,
                                        const PointValueWithCovarianceListType& pointValuesWithCovariance,
                                        double noiseVariance) const;



    /**
     * Builds a new StatisticalModel given a StatisticalModel and the given constraints.
     * If we interpret the given model as a prior distribution over the modeled objects,
     * the resulting model can (loosely) be interpreted as the posterior distribution,
     * after having observed the data given in the PointValues.
     * This version of the function assumes a noise with a uniform uncorrelated variance
     * of the form pointValueNoiseVariance * identityMatrix at every given point.
     *
     * \param model A statistical model.
     * \param pointValues A list of (point, value) pairs with the known values.
     * \param pointValueNoiseVariance The variance of the estimated error at the known points (the pointValues)
     * \param computeScores Determines whether the scores are computed and stored in the model.
     * \return a new statistical model
     *
     * \warning The returned model needs to be explicitly deleted by the user of this method.
     */
    StatisticalModelType* BuildNewModelFromModel(const StatisticalModelType* model, const PointValueListType& pointValues, double pointValueNoiseVariance, bool computeScores=true) const;


    /**
     * Builds a new StatisticalModel given a StatisticalModel and the given constraints.
     * If we interpret the given model as a prior distribution over the modeled objects,
     * the resulting model can (loosely) be interpreted as the posterior distribution,
     * after having observed the data given in the PointValues.
     * For this version of the function, the covariance matrix of the noise needs to be specified for
     * every point. These covariance matrices are passed in the pointValuesWithCovariance list.
     *
     * \param model A statistical model.
     * \param pointValuesWithCovariance A list of ((point,value), covarianceMatrix) for each known value.
     * \param computeScores Determines whether the scores are computed and stored in the model.
     * \return a new statistical model
     *
     * \warning The returned model needs to be explicitly deleted by the user of this method.
     */
    StatisticalModelType* BuildNewModelFromModel(const StatisticalModelType* model,
            const PointValueWithCovarianceListType& pointValuesWithCovariance,
            bool computeScores=true) const;

    /**
     * A convenience function to create a PointValueWithCovarianceList with uniform variance
     *
     * \param pointValues A list of (point, value) pairs with the known values.
     * \param pointValueNoiseVariance The variance of the estimated error at the known points (the pointValues)
     * \return a PointValueWithCovarianceListType with the given uniform variance
     *
     * \warning The returned model needs to be explicitly deleted by the user of this method.
     */
    PointValueWithCovarianceListType TrivialPointValueWithCovarianceListWithUniformNoise(const PointValueListType& pointValues,
            double pointValueNoiseVariance) const;

  private:
    PosteriorModelBuilder();
    PosteriorModelBuilder(const PosteriorModelBuilder& orig);
    PosteriorModelBuilder& operator=(const PosteriorModelBuilder& rhs);


};

} // namespace statismo

#include "PosteriorModelBuilder.hxx"

#endif /* __POSTERIORMODELBUILDER_H_ */

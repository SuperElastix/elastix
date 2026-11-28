/*
 * Representer.hxx
 *
 * Created by Remi Blanc, Marcel Luethi
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

#ifndef __ConditionalModelBuilder_hxx
#define __ConditionalModelBuilder_hxx

#include "ConditionalModelBuilder.h"

#include <iostream>

#include <Eigen/SVD>

#include "Exceptions.h"
#include "PCAModelBuilder.h"

namespace statismo {

//
// ConditionalModelBuilder
//
//


template <typename T>
unsigned
ConditionalModelBuilder<T>::PrepareData(const DataItemListType& sampleDataList,
                                        const SurrogateTypeInfoType& surrogateTypesInfo,
                                        const CondVariableValueVectorType& conditioningInfo,
                                        DataItemListType *acceptedSamples,
                                        MatrixType *surrogateMatrix,
                                        VectorType *conditions) const {
    bool acceptSample;
    unsigned nbAcceptedSamples = 0;
    unsigned nbContinuousSurrogatesInUse = 0, nbCategoricalSurrogatesInUse = 0;
    std::vector<unsigned> indicesContinuousSurrogatesInUse;
    std::vector<unsigned> indicesCategoricalSurrogatesInUse;

    //first: identify the continuous and categorical variables, which are used for conditioning and which are not
    for (unsigned i=0 ; i<conditioningInfo.size() ; i++) {
        if (conditioningInfo[i].first) { //only variables that are used for conditioning are of interest here
            if (surrogateTypesInfo.types[i] == DataItemWithSurrogatesType::Continuous) {
                nbContinuousSurrogatesInUse++;
                indicesContinuousSurrogatesInUse.push_back(i);
            } else {
                nbCategoricalSurrogatesInUse++;
                indicesCategoricalSurrogatesInUse.push_back(i);
            }
        }
    }
    conditions->resize(nbContinuousSurrogatesInUse);
    for (unsigned i=0 ; i<nbContinuousSurrogatesInUse ; i++) (*conditions)(i) = conditioningInfo[i].second;
    surrogateMatrix->resize(nbContinuousSurrogatesInUse, sampleDataList.size()); //number of variables is now known: nbContinuousSurrogatesInUse ; the number of samples is yet unknown...

    //now, browse all samples to select the ones which fall into the requested categories
    for (typename DataItemListType::const_iterator it = sampleDataList.begin(); it != sampleDataList.end(); ++it) {
        const DataItemWithSurrogatesType* sampleData = dynamic_cast<const DataItemWithSurrogatesType*>(*it);
        if (sampleData == 0)  {
            // this is a normal sample without surrogate information.
            // we simply discard it
            std::cout<<"WARNING: ConditionalModelBuilder, sample data "<< (*it)->GetDatasetURI()<<" has no surrogate data associated, and is ignored"<<std::endl;
            continue;
        }

        VectorType surrogateData = sampleData->GetSurrogateVector();
        acceptSample = true;
        for (unsigned i=0 ; i<nbCategoricalSurrogatesInUse ; i++) { //check that this sample respect the requested categories
            if ( conditioningInfo[indicesCategoricalSurrogatesInUse[i]].second !=
                    surrogateData[indicesCategoricalSurrogatesInUse[i]] ) {
                //if one of the categories does not fit the requested one, then the sample is discarded
                acceptSample = false;
                continue;
            }
        }

        if (acceptSample) { //if the sample is of the right category
            acceptedSamples->push_back(*it);
            //and fill in the matrix of continuous variables
            for (unsigned j=0 ; j<nbContinuousSurrogatesInUse ; j++) {
                (*surrogateMatrix)(j,nbAcceptedSamples) = surrogateData[indicesContinuousSurrogatesInUse[j]];
            }
            nbAcceptedSamples++;
        }
    }
    //resize the matrix of surrogate data to the effective number of accepted samples
    surrogateMatrix->conservativeResize(Eigen::NoChange_t(), nbAcceptedSamples);

    return nbAcceptedSamples;
}

template <typename T>
typename ConditionalModelBuilder<T>::StatisticalModelType*
ConditionalModelBuilder<T>::BuildNewModel(const DataItemListType& sampleDataList,
        const SurrogateTypeInfoType& surrogateTypesInfo,
        const CondVariableValueVectorType& conditioningInfo,
        float noiseVariance,
        double modelVarianceRetained) const {
    DataItemListType acceptedSamples;
    MatrixType X;
    VectorType x0;
    unsigned nSamples = PrepareData(sampleDataList, surrogateTypesInfo, conditioningInfo, &acceptedSamples, &X, &x0);
    assert(nSamples == acceptedSamples.size());

    unsigned nCondVariables = X.rows();

    // build a normal PCA model
    typedef PCAModelBuilder<T> PCAModelBuilderType;
    PCAModelBuilderType* modelBuilder = PCAModelBuilderType::Create();
    StatisticalModelType* pcaModel = modelBuilder->BuildNewModel(acceptedSamples, noiseVariance);

    unsigned nPCAComponents = pcaModel->GetNumberOfPrincipalComponents();

    if ( X.cols() == 0 || X.rows() == 0) {
        return pcaModel;
    } else {
        // the scores in the pca model correspond to the parameters of each sample in the model.
        MatrixType B = pcaModel->GetModelInfo().GetScoresMatrix().transpose();
        assert(B.rows() == nSamples);
        assert(B.cols() == nPCAComponents);

        // A is the joint data matrix B, X, where X contains the conditional information for each sample
        // Thus the i-th row of A contains the PCA parameters b of the i-th sample,
        // together with the conditional information for each sample
        MatrixType A(nSamples, nPCAComponents+nCondVariables);
        A << B,X.transpose();

        // Compute the mean and the covariance of the joint data matrix
        VectorType mu = A.colwise().mean().transpose(); // colwise returns a row vector
        assert(mu.rows() == nPCAComponents + nCondVariables);

        MatrixType A0 = A.rowwise() - mu.transpose(); //
        MatrixType cov = 1.0 / (nSamples-1) * A0.transpose() *  A0;

        assert(cov.rows() == cov.cols());
        assert(cov.rows() == pcaModel->GetNumberOfPrincipalComponents() + nCondVariables);

        // extract the submatrices involving the conditionals x
        // note that since the matrix is symmetric, Sbx = Sxb.transpose(), hence we only store one
        MatrixType Sbx = cov.topRightCorner(nPCAComponents, nCondVariables);
        MatrixType Sxx = cov.bottomRightCorner(nCondVariables, nCondVariables);
        MatrixType Sbb = cov.topLeftCorner(nPCAComponents, nPCAComponents);

        // compute the conditional mean
        VectorType condMean = mu.topRows(nPCAComponents) + Sbx * Sxx.inverse() * (x0 - mu.bottomRows(nCondVariables));

        // compute the conditional covariance
        MatrixType condCov = Sbb - Sbx * Sxx.inverse() * Sbx.transpose();

        // get the sample mean corresponding the the conditional given mean of the parameter vectors
        VectorType condMeanSample = pcaModel->GetRepresenter()->SampleToSampleVector(pcaModel->DrawSample(condMean));


        // so far all the computation have been done in parameter (latent) space. Go back to sample space.
        // (see PartiallyFixedModelBuilder for a detailed documentation)
        // TODO we should factor this out into the base class, as it is the same code as it is used in
        // the partially fixed model builder
        const VectorType& pcaVariance = pcaModel->GetPCAVarianceVector();
        VectorTypeDoublePrecision pcaSdev = pcaVariance.cast<double>().array().sqrt();

        typedef Eigen::JacobiSVD<MatrixTypeDoublePrecision> SVDType;
        MatrixTypeDoublePrecision innerMatrix = pcaSdev.asDiagonal() * condCov.cast<double>() * pcaSdev.asDiagonal();
        SVDType svd(innerMatrix, Eigen::ComputeThinU);
        VectorType singularValues = svd.singularValues().cast<ScalarType>();

        // keep only the necessary number of modes, wrt modelVarianceRetained...
        double totalRemainingVariance = singularValues.sum(); //
        //and count the number of modes required for the model
        double cumulatedVariance = singularValues(0);
        unsigned numComponentsToReachPrescribedVariance = 1;
        while ( cumulatedVariance/totalRemainingVariance < modelVarianceRetained ) {
            numComponentsToReachPrescribedVariance++;
            if (numComponentsToReachPrescribedVariance==singularValues.size()) break;
            cumulatedVariance += singularValues(numComponentsToReachPrescribedVariance-1);
        }

        unsigned numComponentsToKeep = std::min<unsigned>( numComponentsToReachPrescribedVariance, singularValues.size() );

        VectorType newPCAVariance = singularValues.topRows(numComponentsToKeep);
        MatrixType newPCABasisMatrix = (pcaModel->GetOrthonormalPCABasisMatrix() * svd.matrixU().cast<ScalarType>()).leftCols(numComponentsToKeep);

        StatisticalModelType* model = StatisticalModelType::Create(pcaModel->GetRepresenter(),  condMeanSample, newPCABasisMatrix, newPCAVariance, noiseVariance);

        // add builder info and data info to the info list
        MatrixType scores(0,0);
        BuilderInfo::ParameterInfoList bi;

        bi.push_back(BuilderInfo::KeyValuePair("NoiseVariance ", Utils::toString(noiseVariance)));

        //generate a matrix ; first column = boolean (yes/no, this variable is used) ; second: conditioning value.
        MatrixType conditioningInfoMatrix(conditioningInfo.size(), 2);
        for (unsigned i=0 ; i<conditioningInfo.size() ; i++) {
            conditioningInfoMatrix(i,0) = conditioningInfo[i].first;
            conditioningInfoMatrix(i,1) = conditioningInfo[i].second;
        }
        bi.push_back(BuilderInfo::KeyValuePair("ConditioningInfo ", Utils::toString(conditioningInfoMatrix)));

        typename BuilderInfo::DataInfoList di;

        unsigned i = 0;
        for (typename DataItemListType::const_iterator it = sampleDataList.begin();
                it != sampleDataList.end();
                ++it, i++) {
            const DataItemWithSurrogatesType* sampleData = dynamic_cast<const DataItemWithSurrogatesType*>(*it);
            std::ostringstream os;
            os << "URI_" << i;
            di.push_back(BuilderInfo::KeyValuePair(os.str().c_str(),sampleData->GetDatasetURI()));

            os << "_surrogates";
            di.push_back(BuilderInfo::KeyValuePair(os.str().c_str(),sampleData->GetSurrogateFilename()));
        }

        std::ostringstream os;
        os << "surrogates_types";
        di.push_back(BuilderInfo::KeyValuePair(os.str().c_str(),surrogateTypesInfo.typeFilename));


        BuilderInfo builderInfo("ConditionalModelBuilder", di, bi);

        ModelInfo::BuilderInfoList biList;
        biList.push_back(builderInfo);

        ModelInfo info(scores, biList);
        model->SetModelInfo(info);

        delete pcaModel;

        return model;
    }

}

} // namespace statismo

#endif

/**
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Statismo is licensed under the BSD licence (3 clause) license
 */


#ifndef __LOW_RANK_GP_MODEL_BUILDER_H
#define __LOW_RANK_GP_MODEL_BUILDER_H

#include <cmath>

#include <vector>

#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread/future.hpp>

#include "CommonTypes.h"
#include "Config.h"
#include "DataManager.h"
#include "Kernels.h"
#include "ModelInfo.h"
#include "ModelBuilder.h"
#include "Nystrom.h"
#include "Representer.h"
#include "StatisticalModel.h"

namespace statismo {


/**
 * This class holds the result of the eigenfunction computation for
 * the points with index entries (lowerInd to upperInd)
 */
struct EigenfunctionComputationResult {


    EigenfunctionComputationResult(unsigned _lowerInd, unsigned _upperInd,
                                   const MatrixType& _resMat) :
        lowerInd(_lowerInd), upperInd(_upperInd), resultForPoints(_resMat) {
    }

    unsigned lowerInd;
    unsigned upperInd;
    MatrixType resultForPoints;

    // emulate move semantics, as boost::async seems to depend on it.
    EigenfunctionComputationResult& operator=(BOOST_COPY_ASSIGN_REF(EigenfunctionComputationResult) rhs) { // Copy assignment
        if (&rhs != this) {
            copyMembers(rhs);
        }
        return *this;
    }

    EigenfunctionComputationResult(BOOST_RV_REF(EigenfunctionComputationResult) that) { //Move constructor
        copyMembers(that);
    }
    EigenfunctionComputationResult& operator=(BOOST_RV_REF(EigenfunctionComputationResult) rhs) { //Move assignment
        if (&rhs != this) {
            copyMembers(rhs);
        }
        return *this;
    }
  private:
    BOOST_COPYABLE_AND_MOVABLE(EigenfunctionComputationResult)
    void copyMembers(const EigenfunctionComputationResult& that) {
        lowerInd = that.lowerInd;
        upperInd = that.upperInd;
        resultForPoints = that.resultForPoints;
    }
};


/**
 * A model builder for building statistical models that are specified by an arbitrary Gaussian Process.
 * For details on the theoretical basis for this type of model builder, see the paper
 *
 * A unified approach to shape model fitting and non-rigid registration
 * Marcel Lüthi, Christoph Jud and Thomas Vetter
 * IN: Proceedings of the 4th International Workshop on Machine Learning in Medical Imaging,
 * LNCS 8184, pp.66-73 Nagoya, Japan, September 2013
 *
 */

template<typename T>
class LowRankGPModelBuilder: public ModelBuilder<T> {

  public:

    typedef Representer<T>                      RepresenterType;
    typedef typename RepresenterType::PointType PointType;

    typedef ModelBuilder<T>                           Superclass;
    typedef typename Superclass::StatisticalModelType StatisticalModelType;

    typedef Domain<PointType>                         DomainType;
    typedef typename DomainType::DomainPointsListType DomainPointsListType;

    typedef MatrixValuedKernel<PointType> MatrixValuedKernelType;

    /**
     * Factory method to create a new ModelBuilder
     */
    static LowRankGPModelBuilder* Create(const RepresenterType* representer) {
        return new LowRankGPModelBuilder(representer);
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
    virtual ~LowRankGPModelBuilder() {
    }


    /**
    * Build a new model using a zero-mean Gaussian process with given  kernel.
    * \param kernel: A kernel (or covariance) function
    * \param numComponents The number of components used for the low rank approximation.
    * \param numPointsForNystrom  The number of points used for the Nystrom approximation
    *
    * \return a new statistical model representing the given Gaussian process
    */
    StatisticalModelType* BuildNewZeroMeanModel(
        const MatrixValuedKernelType& kernel, unsigned numComponents,
        unsigned numPointsForNystrom = 500) const {

        return BuildNewModel(m_representer->IdentitySample(), kernel, numComponents,
                             numPointsForNystrom);
    }

    /**
     * Build a new model using a Gaussian process with given mean and kernel.
     * \param mean: A dataset that represents the mean (shape or deformation)
     * \param kernel: A kernel (or covariance) function
     * \param numComponents The number of components used for the low rank approximation.
     * \param numPointsForNystrom  The number of points used for the Nystrom approximation
     *
     * \return a new statistical model representing the given Gaussian process
     */
    StatisticalModelType* BuildNewModel(
        typename RepresenterType::DatasetConstPointerType mean,
        const MatrixValuedKernelType& kernel,
        unsigned numComponents,
        unsigned numPointsForNystrom = 500) const {


        std::vector<PointType> domainPoints = m_representer->GetDomain().GetDomainPoints();
        unsigned numDomainPoints = m_representer->GetDomain().GetNumberOfPoints();
        unsigned kernelDim = kernel.GetDimension();


        boost::scoped_ptr<Nystrom<T> > nystrom(Nystrom<T>::Create(m_representer, kernel, numComponents, numPointsForNystrom));

        // we precompute the value of the eigenfunction for each domain point
        // and store it later in the pcaBasis matrix. In this way we obtain
        // a standard statismo model.
        // To save time, we parallelize over the rows
        std::vector<boost::future<EigenfunctionComputationResult>* > futvec;


        unsigned numChunks = boost::thread::hardware_concurrency() + 1;

        for (unsigned i = 0; i <= numChunks; i++) {

            unsigned chunkSize = static_cast< unsigned >( ceil( static_cast< float >( numDomainPoints ) / static_cast< float >( numChunks ) ) );
            unsigned lowerInd = i * chunkSize;
            unsigned upperInd =
                std::min( static_cast< unsigned >(numDomainPoints),
                          (i + 1) * chunkSize);

            if (lowerInd >= upperInd) {
                break;
            }

            boost::future<EigenfunctionComputationResult>* fut = new boost::future<EigenfunctionComputationResult>(
                boost::async(boost::launch::async, boost::bind(&LowRankGPModelBuilder<T>::computeEigenfunctionsForPoints,
                             this, nystrom.get(), &kernel, numComponents, domainPoints,  lowerInd, upperInd)));
            futvec.push_back(fut);
        }

        MatrixType pcaBasis = MatrixType::Zero(numDomainPoints * kernelDim, numComponents);

        // collect the result
        for (unsigned i = 0; i < futvec.size(); i++) {
            EigenfunctionComputationResult res = futvec[i]->get();
            pcaBasis.block(res.lowerInd * kernelDim, 0,
                           (res.upperInd - res.lowerInd) * kernelDim, pcaBasis.cols()) =
                               res.resultForPoints;
            delete futvec[i];
        }


        VectorType pcaVariance = nystrom->getEigenvalues();

        RowVectorType mu = m_representer->SampleToSampleVector(mean);

        StatisticalModelType* model = StatisticalModelType::Create(
                                          m_representer,  mu, pcaBasis, pcaVariance, 0);

        // the model builder does not use any data. Hence the scores and the datainfo is emtpy
        MatrixType scores; // no scores
        typename BuilderInfo::DataInfoList dataInfo;


        typename BuilderInfo::ParameterInfoList bi;
        bi.push_back(BuilderInfo::KeyValuePair("NoiseVariance",   Utils::toString(0)));
        bi.push_back(BuilderInfo::KeyValuePair("KernelInfo", kernel.GetKernelInfo()));

        // finally add meta data to the model info
        BuilderInfo builderInfo("LowRankGPModelBuilder", dataInfo, bi);

        ModelInfo::BuilderInfoList biList( 1, builderInfo );;

        ModelInfo info(scores, biList);
        model->SetModelInfo(info);

        return model;
    }


  private:



    /*
     * Compute the eigenfunction value at the poitns with index lowerInd - upperInd.
     * Return a result object with the given values.
     * This method is used to be able to parallelize the computations.
     */
    EigenfunctionComputationResult computeEigenfunctionsForPoints(
        const Nystrom<T>* nystrom,
        const MatrixValuedKernelType* kernel, unsigned numEigenfunctions,
        const std::vector<PointType> & domainPts,
        unsigned lowerInd, unsigned upperInd) const {

        unsigned kernelDim = kernel->GetDimension();

        assert(upperInd <= domainPts.size());

        // holds the results of the computation
        MatrixType resMat = MatrixType::Zero((upperInd - lowerInd) * kernelDim,
                                             numEigenfunctions);

        // compute the nystrom extension for each point i in domainPts, for which
        // i is in the right range
        for (unsigned i = lowerInd; i < upperInd; i++) {

            PointType pti = domainPts[i];
            resMat.block((i - lowerInd) * kernelDim, 0, kernelDim, resMat.cols()) = nystrom->computeEigenfunctionsAtPoint(pti);

        }
        return EigenfunctionComputationResult(lowerInd, upperInd, resMat);
    }



    /**
     * constructor - only used internally
     */
    LowRankGPModelBuilder(const RepresenterType* representer) :
        m_representer(representer) {
    }

    // purposely not implemented
    LowRankGPModelBuilder(const LowRankGPModelBuilder& orig);
    LowRankGPModelBuilder& operator=(const LowRankGPModelBuilder& rhs);

    const RepresenterType* m_representer;

};

} // namespace statismo

#endif // __LOW_RANK_GP_MODEL_BUILDER_H

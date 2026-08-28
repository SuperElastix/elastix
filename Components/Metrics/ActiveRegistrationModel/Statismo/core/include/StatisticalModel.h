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


#ifndef STATISTICALMODEL_H_
#define STATISTICALMODEL_H_

#include <limits>
#include <vector>

#include "CommonTypes.h"
#include "Config.h"
#include "DataManager.h"
#include "ModelInfo.h"
#include "Representer.h"


namespace statismo {

/**
 * \brief A Point/Value pair that is used to specify a value at a given point.
 */
/*
template <typename Representer>
class PointValuePair {
public:
	typedef typename Representer::PointType PointType;
	typedef typename Representer::ValueType ValueType;

	PointValuePair(const PointType& pt, const ValueType& val) : point(pt), value(val) {}

	PointType point;
	ValueType value;
};
*/

/**
 * \brief Representation of a linear statistical model (PCA Model).
 *
 * The statistical model class represents a statistical (PCA based) model.
 * The implementation is based on the Probabilistic PCA, which includes the Standard PCA as a special case.
 *
 * Mathematically, the statistical model is a generative model, where a sample is given as a linear combination
 * of a \f$n\f$ dimensional latent variable \f$ \alpha \f$ plus additive gaussian noise.
 * \f[ S = \mu + W \alpha + \epsilon. \f]
 * Here, \f$ \mu \in \mathbf{R}^p \f$ is the mean, \f$W \in \mathbf{R}^{p \times n}\f$ is a linear mapping,
 * \f$\alpha \in \mathbf{R}^n \f$ is a vector of latent variables (later also referred to as coefficients)
 * and \f$\epsilon \sim \mathcal{N}(0, \sigma^2)\f$ is a noise component.
 * The linear mapping \f$ W \f$ is referred to as the PCABasis. It is usually given as \f$W = U D \f$ where \f$U\f$ where
 * U is an orthonormal matrix and \f$D\f$ is a scaling matrix, referred to as PCAvariance.
 * Usually, \f$U \in \mathbf{R}^{p \times n}\f$ is the matrix of eigenvectors of the data covariance matrix and
 * \f$D\f$ the corresponding eigenvalues.
 *
 * While all the matrices and vectors defined above could be obtained directly, the main goal of this class
 * is to abstract from these technicalities, by providing a high level interface to shape models.
 * In this high level view, the model represents a multivariate normal distribution over the types defined by the representer
 * (which are typically either, surface meshes, point clouds, deformation fields or images).
 * This class provides the method to sample from this probability distribution, and to compute the probability
 * of given samples directly.
 *
 */
template <typename T>
class StatisticalModel {
  public:
    typedef Representer<T> RepresenterType ;
    typedef typename RepresenterType::DatasetPointerType DatasetPointerType;
    typedef typename RepresenterType::DatasetConstPointerType DatasetConstPointerType;
    typedef typename RepresenterType::ValueType ValueType;
    typedef typename RepresenterType::PointType PointType;


    typedef Domain<PointType> DomainType;

    typedef unsigned PointIdType;


    //typedef  PointValuePair<Representer>  PointValuePairType;
    typedef std::pair<PointType, ValueType> PointValuePairType;
    typedef std::pair<unsigned, ValueType> PointIdValuePairType;
    typedef std::list<PointValuePairType> PointValueListType;
    typedef std::list<PointIdValuePairType> PointIdValueListType;

    // Maybe at some point, we can statically define a 3x3 resp. 2x3 matrix type.
    typedef MatrixType PointCovarianceMatrixType;
    typedef std::pair<PointValuePairType, PointCovarianceMatrixType> PointValueWithCovariancePairType;
    typedef std::list<PointValueWithCovariancePairType> PointValueWithCovarianceListType;




    /**
     * Destructor
     */
    virtual ~StatisticalModel();


    /**
     * @name Creating models
     */
    ///@{

    /**
    * Factory method that creates a new Model.
    *
    * \warning The use of this constructor is discouraged. If possible, use a ModelBuilder to create
    * a new model or call Load to load an existing model
    *
    * \param representer the represener
    * \param m the mean
    * \param orthonormalPCABasis An orthonormal matrix with the principal Axes.
    * \param pcaVariance The Variance for each principal Axis
    * \param noiseVariance The variance of the (N(0,noiseVariance)) noise on each point
    */
    static StatisticalModel* Create(const RepresenterType* representer,
                                    const VectorType& m,
                                    const MatrixType& orthonormalPCABasis,
                                    const VectorType& pcaVariance,
                                    double noiseVariance) {
        return new StatisticalModel(representer, m, orthonormalPCABasis, pcaVariance, noiseVariance);
    }



    /**
     * Destroy the object.
     * The same effect can be achieved by deleting the object in the usual
     * way using the c++ delete keyword.
     */
    void Delete() const {
        delete this;
    }

    ///@}


    /**
     * @name General Info
     */
    ///@{
    /**
     * \return The number of PCA components in the model
     */
    unsigned int GetNumberOfPrincipalComponents() const;

    /**
     * \return A model info object \sa ModelInfo
     */
    const ModelInfo& GetModelInfo() const;
    ///@}

    /**
     * @name Sample from the model
     *
     * \warning Note that these methods return a new Sample. If the representer used returns naked pointers (i.e. not smart pointers),
     * the sample needs to be deleted manually.
     */
    ///@{



    /**
     * Returns the value of the given sample at the point specified with the ptId
     *
     * \param sample A sample
     * \param ptId the point id where to evaluate the sample
     *
     * \returns The value of the sample, at the specified point
     */
    ValueType EvaluateSampleAtPoint(DatasetConstPointerType sample, unsigned ptId) const ;


    /**
     * Returns the value of the given sample corresponding to the given domain point
     *
     * \param sample A sample
     * \param point the (domain) point on which the sample should be evaluated.
     *
     * \returns The value of the sample, at the specified point
     */
    ValueType EvaluateSampleAtPoint(DatasetConstPointerType sample, const PointType& pt) const;


    /**
     * \return A new sample representing the mean of the model
     */
    DatasetPointerType DrawMean() const;

    /**
     * Draws the sample with the given coefficients
     *
     * \param coefficients A coefficient vector. The size of the coefficient vector should be smaller
     * than number of factors in the model. Otherwise an exception is thrown.
     * \param addNoise If true, the Gaussian noise assumed in the model is added to the sample
     *
     * \return A new sample
     * */
    DatasetPointerType DrawSample(const VectorType& coefficients, bool addNoise = false) const ;

    /**
     * As StatisticalModel::DrawSample, but where the coefficients are chosen at random according to a standard normal distribution
     *
     * \param addNoise If true, the Gaussian noise assumed in the model is added to the sample
     *
     * \return A new sample
     * \sa DrawSample
     */
    DatasetPointerType DrawSample(bool addNoise = false) const ;



    /**
    	 * Draws the sample corresponding to the ith pca matrix
    	 *
    	 * \param componentNumber The number of the PCA Basis to be retrieved
    	 *
    	 * \return A new sample
    	 * */
    DatasetPointerType DrawPCABasisSample(unsigned componentNumber) const;


    /**
     * @name Point sampling and point information
     */
    ///@{

    /**
     * Returns the mean of the model, evaluated at the given point.
     *
     * \param point A point on the domain the model is defined
     * \returns The mean Sample evaluated at the point point
     */
    ValueType DrawMeanAtPoint( const PointType& point) const;

    /**
     * Returns the mean of the model, evaluated at the given pointid.
     *
     * \param pointid The pointId of the point where it should be evaluated (as defined by the representer)
     * \returns The mean sample evaluated at the given pointId \see DrawMeanAtPoint
     */
    ValueType DrawMeanAtPoint( unsigned pointId) const;

    /**
     * Returns the value of the sample defined by coefficients at the specified point.
     * This method computes the value of the sample only for the given point, and is thus much more
     * efficient that calling DrawSample, if only a few points are of interest.
     *
     * \param coefficients the coefficients of the sample
     * \param the point of the sample where it is evaluated
     * \param addNoise If true, the Gaussian noise assumed in the model is added to the sample
     */
    ValueType DrawSampleAtPoint(const VectorType& coefficients, const PointType& point, bool addNoise = false) const;

    /**
     * Returns the value of the sample defined by coefficients at the specified pointID.
     * This method computes the value of the sample only for the given point, and is thus much more
     * efficient that calling DrawSample, if only a few points are of interest.
     *
     * \param coefficients the coefficients of the sample
     * \param the point of the sample where it is evaluated
     * \param addNoise If true, the Gaussian noise assumed in the model is added to the sample
     */
    ValueType DrawSampleAtPoint(const VectorType& coefficients, unsigned pointId, bool addNoise = false) const;


    /**
     * Computes the jacobian of the Statistical model at a given point
     * \param pt The point where the Jacobian is computed
     * \param jacobian Output parameter where the jacobian is stored.
     */
    MatrixType GetJacobian(const PointType& pt) const;

    /**
     * Computes the jacobian of the Statistical model at a specified pointID
     * \param ptId The pointID where the Jacobian is computed
     * \param jacobian Output parameter where the jacobian is stored.
     */
    MatrixType GetJacobian(unsigned ptId) const;

    /**
     * Returns the variance in the model for point pt
     * @param pt The point
     * @returns a d x d covariance matrix
     */
    MatrixType GetCovarianceAtPoint(const PointType& pt1, const PointType& pt2) const;

    /**
     * Returns the variance in the model for point pt
     * @param pt The point
     * @returns a d x d covariance matrix
     */
    MatrixType GetCovarianceAtPoint(unsigned ptId1, unsigned ptId2) const;
    ///@}


    /**
     * @name Statistical Information from Dataset
     */
    ///@{

    /**
     * Returns the covariance matrix for the model. If the model is defined on
     * n points, in d dimensions, then this is a \f$nd \times nd\f$ matrix of
     * n \f$d \times d \f$ block matrices corresponding to the covariance at each point.
     * \warning This method is only useful when $n$ is small, since otherwise the matrix
     * becomes huge.
     */
    MatrixType GetCovarianceMatrix() const;

    /**
     * Returns the inverse covariance matrix. Computes if necessary
     */
    MatrixType GetInverseCovarianceMatrix() const;

    /**
     * Returns the probability of observing the given dataset under this model.
     * If the coefficients \f$\alpha \in \mathbf{R}^n\f$ define the dataset, the probability is
     * \f$
     * (2 \pi)^{- \frac{n}{2}} \exp(||\alpha||)
     * \f$
     *
     *
     * \param dataset The dataset
     * \return The probability
     */
    double ComputeProbability(DatasetConstPointerType dataset) const ;

    /**
     * Returns the log probability of observing a given dataset.
     *
     * \param dataset The dataset
     * \return The log probability
     *
     */
    double ComputeLogProbability(DatasetConstPointerType dataset) const ;


    /**
     * Returns the probability of observing the given coefficients under this model.
     * If the coefficients \f$\alpha \in \mathbf{R}^n\f$ define the dataset, the probability is
     * \f$
     * (2 \pi)^{- \frac{n}{2}} \exp(||\alpha||)
     * \f$
     *
     *
     * \param coefficients The coefficients \f$\alpha \in \mathbf{R}^n\f$
     * \return The probability
     */
    double ComputeProbabilityOfCoefficients(const VectorType& coefficients) const ;

    /**
     * Returns the log probability of observing given coefficients.
     *
     * \param dataset The coefficients \f$\alpha \in \mathbf{R}^n\f$
     * \return The log probability
     *
     */
    double ComputeLogProbabilityOfCoefficients(const VectorType& coefficients) const ;


    /**
      * Returns the mahalonoibs distance for the given dataset.
      */
    double ComputeMahalanobisDistance(DatasetConstPointerType dataset) const;


    /**
     * Returns the coefficients of the latent variables for the given dataset, i.e.
     * the vectors of numbers \f$\alpha \f$, such that for the dataset \f$S\f$ it holds that
     * \f$ S = \mu + U \alpha\f$
     *
     * @returns The coefficient vector \f$\alpha\f$
     */
    VectorType ComputeCoefficients(DatasetConstPointerType dataset) const;


    /**
     * Returns the coefficients of the latent variables for the given values provided in the PointValueList.
     * This is useful, when only a part of the dataset is given.
     * The method is described in the paper
     *
     * Probabilistic Modeling and Visualization of the Flexibility in Morphable Models,
     * M. Luethi, T. Albrecht and T. Vetter, Mathematics of Surfaces, 2009
     *
     * \param pointValues A list with PointValuePairs .
     * \param pointValueNoiseVariance The variance of estimated (gaussian) noise at the known points
     *
     */
    VectorType ComputeCoefficientsForPointValues(const PointValueListType&  pointValues, double pointValueNoiseVariance=0.0) const;

    /**
    * Similar to ComputeCoefficientsForPointValues, only here there is no global pointValueNoiseVariance.
    * Instead, a covariance matrix with noise values is specified for each point.
    * The returned coefficients are the mean of the posterior model described in
    *
    * Posterior Shape Models
    * Thomas Albrecht, Marcel Luethi, Thomas Gerig, Thomas Vetter
    * Medical Image Analysis 2013
    *
    * To get the full posterior model, use the PosteriorModelBuilder
    *
    * \param pointValuesWithCovariance A list with PointValuePairs and PointCovarianceMatrices.
    *
    */
    VectorType ComputeCoefficientsForPointValuesWithCovariance(const PointValueWithCovarianceListType&  pointValuesWithCovariance) const;


    /**
     * Same as ComputeCoefficientsForPointValues(const PointValueListType&  pointValues), but used when the
     * point ids, rather than the points are known.
     *
     * \param pointValues A list with (Point,Value) pairs, a list of (PointId, Value) is provided.
     * \param pointValueNoiseVariance The variance of estimated (gaussian) noise at the known points
     */
    //RB: I had to modify the method name, to avoid prototype collisions when the PointType corresponds to unsigned (= type of the point id)
    VectorType ComputeCoefficientsForPointIDValues(const PointIdValueListType&  pointValues, double pointValueNoiseVariance=0.0) const;


    /**
     * @name Low level access
     * These methods provide a low level interface to  the model content. They are of only limited use for
     * an application. Prefer whenever possible the high level functions.
     */
    ///@{

    /**
     * Returns the variance of the noise of the error term, that was set when the model was built.
     */
    float GetNoiseVariance() const;

    /**
     * Returns a vector where each element holds the variance of the corresponding principal component in data space
     * */
    const VectorType& GetPCAVarianceVector() const;

    /**
     * Returns a vector holding the mean. Assume the mean \f$\mu \subset \mathbf{R}^d\f$ is defined on
     * \f$p\f$ points, the returned mean vector \f$m\f$ has dimensionality \f$m \in \mathbf{R}^{dp} \f$, i.e.
     * the \f$d\f$ components are stacked into the vector. The order of the components in the vector is
     * undefined and depends on the representer.
     * */
    const VectorType& GetMeanVector() const;

    /**
      * Returns a matrix with the PCA Basis as its columns.
      * Assume the shapes \f$s \subset \mathbf{R}^d\f$ are defined on
     * \f$n\f$ points, the returned matrix \f$W\f$ has dimensionality \f$W \in \mathbf{R}^{dp \times n} \f$, i.e.
     * the \f$d\f$ components are stacked into the matrix. The order of the components in the matrix is
     * undefined and depends on the representer.
      *
     */
    const MatrixType& GetPCABasisMatrix() const;

    /**
     * Returns the PCA Matrix, but with its principal axis normalized to unit length.
     * \warning This is more expensive than GetPCABasisMatrix as the matrix has to be computed
     * and a copy is returned
     */
    MatrixType GetOrthonormalPCABasisMatrix() const;

    /**
     * Returns an instance for the given coefficients as a vector.
     * \param addNoise If true, the Gaussian noise assumed in the model is added to the sample
     */
    VectorType DrawSampleVector(const VectorType& coefficients, bool addNoise = false) const ;
    ///@}


    ///@{
    /**
     * Sets the model information. This is for library internal use only.
     */
    void SetModelInfo(const ModelInfo& modelInfo);

    /**
     * Computes the coefficients for the given sample vector.
     * This is for library internal use only.
     */
    VectorType ComputeCoefficientsForSampleVector(const VectorType& sample) const;


    /**
     * Return an instance of the representer
     */
    const RepresenterType* GetRepresenter() const {
        return m_representer;
    }


    /**
     * Return the domain of the statistical model
     */
    const DomainType& GetDomain() const {
        return m_representer->GetDomain();
    }

    ///@}

  private:
    // computes the M Matrix for the PPCA Method (see Bishop, PRML, Chapter 12)
    void CheckAndUpdateCachedParameters() const;



    /**
     * Create an instance of the StatisticalModel
     * @param representer An instance of the representer, used to convert the samples to dataset of the represented type.
     */
    StatisticalModel(const RepresenterType* representer, const VectorType& m, const MatrixType& orthonormalPCABasis, const VectorType& pcaVariance, double noiseVariance);

    // to prevent use
    StatisticalModel(const StatisticalModel& rhs);
    StatisticalModel& operator=(const StatisticalModel& rhs);

    const RepresenterType* m_representer;

    VectorType m_mean;
    MatrixType m_pcaBasisMatrix;
    VectorType m_pcaVariance;
    float m_noiseVariance;


    // caching
    mutable bool m_cachedValuesValid;

    //the matrix M^{-1} in Bishops PRML book. This is roughly the Latent Covariance matrix (but not exactly)
    mutable MatrixType m_MInverseMatrix;

    ModelInfo m_modelInfo;
};

} // namespace statismo

#include "StatisticalModel.hxx"

#endif /* STATISTICALMODEL_H_ */

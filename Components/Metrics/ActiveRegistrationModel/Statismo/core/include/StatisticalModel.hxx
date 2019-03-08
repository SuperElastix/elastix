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

#ifndef __StatisticalModel_hxx
#define __StatisticalModel_hxx

#include <cmath>

#include <fstream>
#include <iostream>
#include <string>

#include "Exceptions.h"
#include "HDF5Utils.h"
#include "ModelBuilder.h"
#include "StatisticalModel.h"

namespace statismo {

template <typename T>
StatisticalModel<T>::StatisticalModel(const RepresenterType* representer, const VectorType& m, const MatrixType& orthonormalPCABasis, const VectorType& pcaVariance, double noiseVariance)
    : m_representer(representer->Clone()),
      m_mean(m),
      m_pcaVariance(pcaVariance),
      m_noiseVariance(noiseVariance),
      m_cachedValuesValid(false) {
    VectorType D = pcaVariance.array().sqrt();
    m_pcaBasisMatrix = orthonormalPCABasis * DiagMatrixType(D);
}


template <typename T>
StatisticalModel<T>::~StatisticalModel() {

    if (m_representer != 0) {
//		 not all representers can implement a const correct version of delete.
//		 We therefore simply const cast it. This is save here.
        const_cast<RepresenterType*>(m_representer)->Delete();
        m_representer = 0;
    }

}


template <typename T>
typename StatisticalModel<T>::ValueType
StatisticalModel<T>::EvaluateSampleAtPoint(const DatasetConstPointerType sample, const PointType& point) const {
    unsigned ptid = this->m_representer->GetPointIdForPoint(point);
    return EvaluateSampleAtPoint(sample, ptid);
}


template <typename T>
typename StatisticalModel<T>::ValueType
StatisticalModel<T>::EvaluateSampleAtPoint(const DatasetConstPointerType sample, unsigned ptid) const {
    return this->m_representer->PointSampleFromSample(sample, ptid);
}


template <typename T>
typename StatisticalModel<T>::DatasetPointerType
StatisticalModel<T>::DrawMean() const {
    VectorType coeffs = VectorType::Zero(this->GetNumberOfPrincipalComponents());
    return DrawSample(coeffs, false);
}


template <typename T>
typename StatisticalModel<T>::ValueType
StatisticalModel<T>::DrawMeanAtPoint(const PointType& point) const {
    VectorType coeffs = VectorType::Zero(this->GetNumberOfPrincipalComponents());
    return DrawSampleAtPoint(coeffs, point);

}

template <typename T>
typename StatisticalModel<T>::ValueType
StatisticalModel<T>::DrawMeanAtPoint(unsigned pointId) const {
    VectorType coeffs = VectorType::Zero(this->GetNumberOfPrincipalComponents());
    return DrawSampleAtPoint(coeffs, pointId, false);

}




template <typename T>
typename StatisticalModel<T>::DatasetPointerType
StatisticalModel<T>::DrawSample(bool addNoise) const {

    // we create random coefficients and draw a random sample from the model
    VectorType coeffs = Utils::generateNormalVector(GetNumberOfPrincipalComponents());

    return DrawSample(coeffs, addNoise);
}


template <typename T>
typename StatisticalModel<T>::DatasetPointerType
StatisticalModel<T>::DrawSample(const VectorType& coefficients, bool addNoise) const {
    return m_representer->SampleVectorToSample(DrawSampleVector(coefficients, addNoise));
}


template <typename T>
typename StatisticalModel<T>::DatasetPointerType
StatisticalModel<T>::DrawPCABasisSample(const unsigned pcaComponent) const {
    if (pcaComponent >= this->GetNumberOfPrincipalComponents()) {
        throw StatisticalModelException("Wrong pcaComponent index provided to DrawPCABasisSample!");
    }


    return m_representer->SampleVectorToSample( m_pcaBasisMatrix.col(pcaComponent));
}



template <typename T>
VectorType
StatisticalModel<T>::DrawSampleVector(const VectorType& coefficients, bool addNoise) const {

    if (coefficients.size() != this->GetNumberOfPrincipalComponents()) {
        throw StatisticalModelException("Incorrect number of coefficients provided !");
    }

    unsigned vectorSize = this->m_mean.size();
    assert (vectorSize != 0);

    VectorType epsilon = VectorType::Zero(vectorSize);
    if (addNoise) {
        epsilon = Utils::generateNormalVector(vectorSize) * sqrt(m_noiseVariance);
    }


    return m_mean+ m_pcaBasisMatrix * coefficients + epsilon;
}


template <typename T>
typename StatisticalModel<T>::ValueType
StatisticalModel<T>::DrawSampleAtPoint(const VectorType& coefficients, const PointType& point, bool addNoise) const {

    unsigned ptId = this->m_representer->GetPointIdForPoint(point);

    return DrawSampleAtPoint(coefficients, ptId, addNoise);

}

template <typename T>
typename StatisticalModel<T>::ValueType
StatisticalModel<T>::DrawSampleAtPoint(const VectorType& coefficients, const unsigned ptId, bool addNoise) const {

    unsigned dim = m_representer->GetDimensions();

    VectorType v(dim);
    VectorType epsilon = VectorType::Zero(dim);
    if (addNoise) {
        epsilon = Utils::generateNormalVector(dim) * sqrt(m_noiseVariance);
    }
    for (unsigned d = 0; d < dim; d++) {
        unsigned idx =m_representer->MapPointIdToInternalIdx(ptId, d);

        if (idx >= m_mean.rows()) {
            std::ostringstream os;
            os << "Invalid idx computed in DrawSampleAtPoint. ";
            os << " The most likely cause of this error is that you provided an invalid point id (" << ptId <<")";
            throw StatisticalModelException(os.str().c_str());
        }

        v[d] = m_mean[idx] + m_pcaBasisMatrix.row(idx).dot(coefficients) + epsilon[d];
    }

    return this->m_representer->PointSampleVectorToPointSample(v);
}



template <typename T>
MatrixType
StatisticalModel<T>::GetCovarianceAtPoint(const PointType& pt1, const PointType& pt2) const {
    unsigned ptId1 = this->m_representer->GetPointIdForPoint(pt1);
    unsigned ptId2 = this->m_representer->GetPointIdForPoint(pt2);

    return GetCovarianceAtPoint(ptId1, ptId2);
}

template <typename T>
MatrixType
StatisticalModel<T>::GetCovarianceAtPoint(unsigned ptId1, unsigned ptId2) const {
    unsigned dim = m_representer->GetDimensions();
    MatrixType cov(dim, dim);

    for (unsigned i = 0; i < dim; i++) {
        unsigned idxi = m_representer->MapPointIdToInternalIdx(ptId1, i);
        VectorType vi = m_pcaBasisMatrix.row(idxi);
        for (unsigned j = 0; j < dim; j++) {
            unsigned idxj = m_representer->MapPointIdToInternalIdx(ptId2, j);
            VectorType vj = m_pcaBasisMatrix.row(idxj);
            cov(i,j) = vi.dot(vj);
            if (i == j) cov(i,j) += m_noiseVariance;
        }
    }
    return cov;
}

template <typename T>
MatrixType
StatisticalModel<T>::GetCovarianceMatrix() const {
    MatrixType M = m_pcaBasisMatrix * m_pcaBasisMatrix.transpose();
    M.diagonal() += m_noiseVariance * VectorType::Ones(m_pcaBasisMatrix.rows());
    return M;
}

template <typename T>
MatrixType
StatisticalModel<T>::GetInverseCovarianceMatrix() const {
    CheckAndUpdateCachedParameters();
    return this->m_MInverseMatrix;
}


template <typename T>
VectorType
StatisticalModel<T>::ComputeCoefficients(DatasetConstPointerType ds) const {
    return ComputeCoefficientsForSampleVector(m_representer->SampleToSampleVector(ds));
}

template <typename T>
VectorType
StatisticalModel<T>::ComputeCoefficientsForSampleVector(const VectorType& sample) const {

    CheckAndUpdateCachedParameters();

    const MatrixType& WT = m_pcaBasisMatrix.transpose();

    VectorType coeffs = m_MInverseMatrix * (WT * (sample - m_mean));
    return coeffs;
}



template <typename T>
VectorType
StatisticalModel<T>::ComputeCoefficientsForPointValues(const PointValueListType&  pointValueList, double pointValueNoiseVariance) const {
    PointIdValueListType ptIdValueList;

    for (typename PointValueListType::const_iterator it  = pointValueList.begin();
            it != pointValueList.end();
            ++it) {
        ptIdValueList.push_back(PointIdValuePairType(m_representer->GetPointIdForPoint(it->first), it->second));
    }
    return ComputeCoefficientsForPointIDValues(ptIdValueList, pointValueNoiseVariance);
}

template <typename T>
VectorType
StatisticalModel<T>::ComputeCoefficientsForPointIDValues(const PointIdValueListType&  pointIdValueList, double pointValueNoiseVariance) const {

    unsigned dim = m_representer->GetDimensions();

    double noiseVariance = std::max(pointValueNoiseVariance, (double) m_noiseVariance);

    // build the part matrices with , considering only the points that are fixed
    MatrixType PCABasisPart(pointIdValueList.size()* dim, this->GetNumberOfPrincipalComponents());
    VectorType muPart(pointIdValueList.size() * dim);
    VectorType sample(pointIdValueList.size() * dim);

    unsigned i = 0;
    for (typename PointIdValueListType::const_iterator it = pointIdValueList.begin(); it != pointIdValueList.end(); ++it) {
        VectorType val = this->m_representer->PointSampleToPointSampleVector(it->second);
        unsigned pt_id = it->first;
        for (unsigned d = 0; d < dim; d++) {
            PCABasisPart.row(i * dim + d) = this->GetPCABasisMatrix().row(m_representer->MapPointIdToInternalIdx(pt_id, d));
            muPart[i * dim + d] = this->GetMeanVector()[m_representer->MapPointIdToInternalIdx(pt_id, d)];
            sample[i * dim + d] = val[d];
        }
        i++;
    }

    MatrixType M = PCABasisPart.transpose() * PCABasisPart;
    M.diagonal() += noiseVariance * VectorType::Ones(PCABasisPart.cols());
    VectorType coeffs = M.inverse() * PCABasisPart.transpose() * (sample - muPart);

    return coeffs;
}

template <typename T>
VectorType
StatisticalModel<T>::ComputeCoefficientsForPointValuesWithCovariance(const PointValueWithCovarianceListType&  pointValuesWithCovariance) const {

    // The naming of the variables correspond to those used in the paper
    // Posterior Shape Models,
    // Thomas Albrecht, Marcel Luethi, Thomas Gerig, Thomas Vetter
    //
    const MatrixType& Q = m_pcaBasisMatrix;
    const VectorType& mu = m_mean;

    unsigned dim = m_representer->GetDimensions();

    // build the part matrices with , considering only the points that are fixed
    //
    unsigned numPrincipalComponents = this->GetNumberOfPrincipalComponents();
    MatrixType Q_g(pointValuesWithCovariance.size()* dim, numPrincipalComponents);
    VectorType mu_g(pointValuesWithCovariance.size() * dim);
    VectorType s_g(pointValuesWithCovariance.size() * dim);

    MatrixType LQ_g(pointValuesWithCovariance.size()* dim, numPrincipalComponents);

    unsigned i = 0;
    for (typename PointValueWithCovarianceListType::const_iterator it = pointValuesWithCovariance.begin(); it != pointValuesWithCovariance.end(); ++it) {
        VectorType val = m_representer->PointSampleToPointSampleVector(it->first.second);
        unsigned pt_id = m_representer->GetPointIdForPoint(it->first.first);

        // In the formulas, we actually need the precision matrix, which is the inverse of the covariance.
        const MatrixType pointPrecisionMatrix = it->second.inverse();

        // Get the three rows pertaining to this point:
        const MatrixType Qrows_for_pt_id = Q.block(pt_id * dim, 0, dim, numPrincipalComponents);

        Q_g.block(i * dim, 0, dim, numPrincipalComponents) = Qrows_for_pt_id;
        mu_g.block(i * dim, 0, dim, 1) = mu.block(pt_id * dim, 0, dim, 1);
        s_g.block(i * dim, 0, dim, 1) = val;

        LQ_g.block(i * dim, 0, dim, numPrincipalComponents) = pointPrecisionMatrix * Qrows_for_pt_id;
        i++;
    }

    VectorType D2 = m_pcaVariance.array();

    const MatrixType& Q_gT = Q_g.transpose();

    MatrixType M = Q_gT * LQ_g;
    M.diagonal() += VectorType::Ones(Q_g.cols());

    MatrixTypeDoublePrecision Minv = M.cast<double>().inverse();

    // the MAP solution for the latent variables (coefficients)
    VectorType coeffs = Minv.cast<ScalarType>() * LQ_g.transpose() * (s_g - mu_g);

    return coeffs;

}



template <typename T>
double
StatisticalModel<T>::ComputeLogProbability(DatasetConstPointerType ds) const {
    VectorType alpha = ComputeCoefficients(ds);
    return ComputeLogProbabilityOfCoefficients(alpha);
}

template <typename T>
double
StatisticalModel<T>::ComputeProbability(DatasetConstPointerType ds) const {
    VectorType alpha = ComputeCoefficients(ds);
    return ComputeProbabilityOfCoefficients(alpha);
}


template <typename T>
double
StatisticalModel<T>::ComputeLogProbabilityOfCoefficients(const VectorType& coefficents) const {
    return log(pow(2 * PI, -0.5 * this->GetNumberOfPrincipalComponents())) - 0.5 * coefficents.squaredNorm();
}

template <typename T>
double
StatisticalModel<T>::ComputeProbabilityOfCoefficients(const VectorType& coefficients) const {
    return pow(2 * PI, - 0.5 * this->GetNumberOfPrincipalComponents()) * exp(- 0.5 * coefficients.squaredNorm());
}


template <typename T>
double
StatisticalModel<T>::ComputeMahalanobisDistance(DatasetConstPointerType ds) const {
    VectorType alpha = ComputeCoefficients(ds);
    return std::sqrt(alpha.squaredNorm());
}



template <typename T>
float
StatisticalModel<T>::GetNoiseVariance() const {
    return m_noiseVariance;
}


template <typename T>
const VectorType&
StatisticalModel<T>::GetMeanVector() const {
    return m_mean;
}

template <typename T>
const VectorType&
StatisticalModel<T>::GetPCAVarianceVector() const {
    return m_pcaVariance;
}


template <typename T>
const MatrixType&
StatisticalModel<T>::GetPCABasisMatrix() const {
    return m_pcaBasisMatrix;
}

template <typename T>
MatrixType
StatisticalModel<T>::GetOrthonormalPCABasisMatrix() const {
    // we can recover the orthonormal matrix by undoing the scaling with the pcaVariance
    // (c.f. the method SetParameters)

    assert(m_pcaVariance.maxCoeff() > 1e-8);
    VectorType D = m_pcaVariance.array().sqrt();
    return m_pcaBasisMatrix * DiagMatrixType(D).inverse();
}



template <typename T>
void
StatisticalModel<T>::SetModelInfo(const ModelInfo& modelInfo) {
    m_modelInfo = modelInfo;
}


template <typename T>
const ModelInfo&
StatisticalModel<T>::GetModelInfo() const {
    return m_modelInfo;
}



template <typename T>
unsigned int
StatisticalModel<T>::GetNumberOfPrincipalComponents() const {
    return m_pcaBasisMatrix.cols();
}

template <typename T>
MatrixType
StatisticalModel<T>::GetJacobian(const PointType& pt) const {

    unsigned ptId = m_representer->GetPointIdForPoint(pt);

    return GetJacobian(ptId);
}

template <typename T>
MatrixType
StatisticalModel<T>::GetJacobian(unsigned ptId) const {

    unsigned Dimensions = m_representer->GetDimensions();
    MatrixType J = MatrixType::Zero(Dimensions, GetNumberOfPrincipalComponents());

    for(unsigned i = 0; i < Dimensions; i++) {
        unsigned idx = m_representer->MapPointIdToInternalIdx(ptId, i);
        for(unsigned j = 0; j < GetNumberOfPrincipalComponents(); j++) {
            J(i,j) += m_pcaBasisMatrix(idx,j) ;
        }
    }
    return J;
}

template <typename T>
void
StatisticalModel<T>::CheckAndUpdateCachedParameters() const {

    if (m_cachedValuesValid == false) {
        VectorType I = VectorType::Ones(m_pcaBasisMatrix.cols());
        MatrixType Mmatrix = m_pcaBasisMatrix.transpose() * m_pcaBasisMatrix;
        Mmatrix.diagonal() += m_noiseVariance * I;

        m_MInverseMatrix = Mmatrix.inverse();

    }
    m_cachedValuesValid = true;
}

} // namespace statismo

#endif

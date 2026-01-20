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

#ifndef ITKSTATISTICALMODEL_H_
#define ITKSTATISTICALMODEL_H_

#include <boost/bind.hpp>
#include <boost/utility/result_of.hpp>

#include <itkObject.h>
#include <itkObjectFactory.h>

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>

#include "ModelInfo.h"
#include "Representer.h"
#include "StatisticalModel.h"
#include "statismoITKConfig.h"

namespace itk {

/**
 * \brief ITK Wrapper for the statismo::StatisticalModel class.
 * \see statismo::StatisticalModel for detailed documentation.
 */
template <class T>
class StatisticalModel : public Object {
  public:

    typedef StatisticalModel            Self;
    typedef Object	Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    itkNewMacro( Self );
    itkTypeMacro( StatisticalModel, Object );

    typedef statismo::Representer<T> RepresenterType;

    // statismo stuff
    typedef statismo::StatisticalModel<T> ImplType;

    typedef typename statismo::DataManager<T>::DataItemType     DataItemType;

    typedef vnl_matrix<statismo::ScalarType> MatrixType;
    typedef vnl_vector<statismo::ScalarType> VectorType;


    template <class F>
    typename boost::result_of<F()>::type callstatismoImpl(F f) const {
        try {
            return f();
        } catch (statismo::StatisticalModelException& s) {
            itkExceptionMacro(<< s.what());
        }
    }

    void SetstatismoImplObj(ImplType* impl) {
        if (m_impl) {
            delete m_impl;
        }
        m_impl = impl;
    }

    ImplType* GetstatismoImplObj() const {
        return m_impl;
    }

    StatisticalModel() : m_impl(0) {}

    virtual ~StatisticalModel() {
        if (m_impl) {
            delete m_impl;
        }
    }


    typedef typename RepresenterType::DatasetPointerType DatasetPointerType;
    typedef typename RepresenterType::DatasetConstPointerType DatasetConstPointerType;

    typedef typename RepresenterType::ValueType ValueType;
    typedef typename RepresenterType::PointType PointType;

    typedef typename statismo::StatisticalModel<T>::PointValuePairType PointValuePairType;
    typedef typename statismo::StatisticalModel<T>::PointValueListType PointValueListType;

    typedef typename statismo::StatisticalModel<T>::PointCovarianceMatrixType PointCovarianceMatrixType;
    typedef typename statismo::StatisticalModel<T>::PointValueWithCovariancePairType PointValueWithCovariancePairType;
    typedef typename statismo::StatisticalModel<T>::PointValueWithCovarianceListType PointValueWithCovarianceListType;

    typedef typename statismo::StatisticalModel<T>::DomainType DomainType;

    const RepresenterType* GetRepresenter() const {
        return callstatismoImpl(boost::bind(&ImplType::GetRepresenter, this->m_impl));
    }

    const DomainType& GetDomain() const {
        return callstatismoImpl(boost::bind(&ImplType::GetDomain, this->m_impl));
    }

    DatasetPointerType DrawMean() const {
        return callstatismoImpl(boost::bind(&ImplType::DrawMean, this->m_impl));
    }

    ValueType DrawMeanAtPoint(const PointType& pt) const {
        typedef ValueType (ImplType::*functype)(const PointType&) const;
        return callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::DrawMeanAtPoint), this->m_impl, pt));
    }

    ValueType DrawMeanAtPoint(unsigned ptid) const {
        typedef ValueType (ImplType::*functype)(unsigned) const;
        return callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::DrawMeanAtPoint), this->m_impl, ptid));
    }

    DatasetPointerType DrawSample(const VectorType& coeffs, bool addNoise = false) const {
        typedef DatasetPointerType (ImplType::*functype)(const statismo::VectorType&, bool) const;
        return callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::DrawSample), this->m_impl, fromVnlVector(coeffs), addNoise));
    }

    DatasetPointerType DrawSample(bool addNoise = false) const {
        typedef DatasetPointerType (ImplType::*functype)(bool) const;
        return callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::DrawSample), this->m_impl, addNoise));
    }

    DatasetPointerType DrawPCABasisSample(unsigned componentNumber) const {
        typedef DatasetPointerType (ImplType::*functype)(unsigned) const;
        return callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::DrawPCABasisSample), this->m_impl, componentNumber));
    }

    ValueType DrawSampleAtPoint(const VectorType& coeffs, const PointType& pt, bool addNoise = false) const {
        typedef ValueType (ImplType::*functype)(const statismo::VectorType&, const PointType&, bool) const;
        return callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::DrawSampleAtPoint), this->m_impl, fromVnlVector(coeffs), pt, addNoise));
    }

    ValueType DrawSampleAtPoint(const VectorType& coeffs, unsigned ptid, bool addNoise  = false) const  {
        typedef ValueType (ImplType::*functype)(const statismo::VectorType&, unsigned, bool) const;
        return callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::DrawSampleAtPoint), this->m_impl, fromVnlVector(coeffs), ptid, addNoise));
    }

    VectorType ComputeCoefficients(DatasetConstPointerType ds) const {
        return toVnlVector(callstatismoImpl(boost::bind(&ImplType::ComputeCoefficients, this->m_impl, ds)));
    }

    double ComputeLogProbability(DatasetConstPointerType ds) const {
        return callstatismoImpl(boost::bind(&ImplType::ComputeLogProbability, this->m_impl, ds));
    }

    double ComputeProbability(DatasetConstPointerType ds) const {
        return callstatismoImpl(boost::bind(&ImplType::ComputeProbability, this->m_impl, ds));
    }

    double ComputeLogProbabilityOfCoefficients(const VectorType& coeffs) const {
        return callstatismoImpl(boost::bind(&ImplType::ComputeLogProbabilityOfCoefficients, this->m_impl, fromVnlVector(coeffs)));
    }

    double ComputeProbabilityOfCoefficients(const VectorType& coeffs) const {
        return callstatismoImpl(boost::bind(&ImplType::ComputeProbabilityOfCoefficients, this->m_impl, fromVnlVector(coeffs)));
    }

    double ComputeMahalanobisDistance(DatasetConstPointerType ds) const {
        return callstatismoImpl(boost::bind(&ImplType::ComputeMahalanobisDistance, this->m_impl, ds));
    }

    VectorType ComputeCoefficientsForPointValues(const PointValueListType& pvlist, double variance) const {
        typedef statismo::VectorType (ImplType::*functype)(const PointValueListType&, double) const;
        return toVnlVector(callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::ComputeCoefficientsForPointValues), this->m_impl, pvlist, variance)));
    }

    VectorType ComputeCoefficientsForPointValuesWithCovariance(const PointValueWithCovarianceListType& pvclist) const {
        typedef statismo::VectorType(ImplType::*functype)(const PointValueWithCovarianceListType&) const;
        return toVnlVector(callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::ComputeCoefficientsForPointValuesWithCovariance), this->m_impl, pvclist)));
    }

    unsigned GetNumberOfPrincipalComponents() const {
        return callstatismoImpl(boost::bind(&ImplType::GetNumberOfPrincipalComponents, this->m_impl));
    }

    float GetNoiseVariance() const {
        return callstatismoImpl(boost::bind(&ImplType::GetNoiseVariance, this->m_impl));
    }

    MatrixType GetCovarianceAtPoint(const PointType& pt1, const PointType& pt2) const {
        typedef statismo::MatrixType (ImplType::*functype)(const PointType&, const PointType&) const;
        return  toVnlMatrix(callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::GetCovarianceAtPoint), this->m_impl, pt1, pt2)));
    }

    MatrixType GetCovarianceAtPoint(unsigned ptid1, unsigned  ptid2) const {
        typedef statismo::MatrixType (ImplType::*functype)(unsigned, unsigned ) const;
        return toVnlMatrix(callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::GetCovarianceAtPoint),this->m_impl, ptid1, ptid2)));
    }

    MatrixType GetJacobian(const PointType& pt) const {
        typedef statismo::MatrixType (ImplType::*functype)(const PointType&) const;
        return toVnlMatrix(callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::GetJacobian), this->m_impl, pt)));
    }

    MatrixType GetJacobian(unsigned ptId) const {
        typedef statismo::MatrixType (ImplType::*functype)(unsigned) const;
        return toVnlMatrix(callstatismoImpl(boost::bind(static_cast<functype>(&ImplType::GetJacobian), this->m_impl, ptId)));
    }

    MatrixType GetPCABasisMatrix() const {
        return toVnlMatrix(callstatismoImpl(boost::bind(&ImplType::GetPCABasisMatrix, this->m_impl)));
    }

    MatrixType GetInverseCovarianceMatrix() const {
        return toVnlMatrix(callstatismoImpl(boost::bind(&ImplType::GetInverseCovarianceMatrix, this->m_impl)));
    }

    MatrixType GetOrthonormalPCABasisMatrix() const {
        return toVnlMatrix(callstatismoImpl(boost::bind(&ImplType::GetOrthonormalPCABasisMatrix, this->m_impl)));
    }

    VectorType GetPCAVarianceVector() const {
        return toVnlVector(callstatismoImpl(boost::bind(&ImplType::GetPCAVarianceVector, this->m_impl)));
    }

    VectorType GetMeanVector() const {
        return toVnlVector(callstatismoImpl(boost::bind(&ImplType::GetMeanVector, this->m_impl)));
    }

    const statismo::ModelInfo& GetModelInfo() const {
        return callstatismoImpl(boost::bind(&ImplType::GetModelInfo, this->m_impl));
    }

  private:

    static MatrixType toVnlMatrix(const statismo::MatrixType& M) {
        return MatrixType(M.data(), M.rows(), M.cols());

    }

    static VectorType toVnlVector(const statismo::VectorType& v) {
        return VectorType(v.data(), v.rows());

    }

    static statismo::VectorType fromVnlVector(const VectorType& v) {
        return Eigen::Map<const statismo::VectorType>(v.data_block(), v.size());

    }

    StatisticalModel(const StatisticalModel& orig);
    StatisticalModel& operator=(const StatisticalModel& rhs);

    ImplType* m_impl;
};


}

#endif /* ITKSTATISTICALMODEL_H_ */

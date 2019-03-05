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

#ifndef __ItkInterpolatingStatisticalDeformationModelTransform
#define __ItkInterpolatingStatisticalDeformationModelTransform

#include <iostream>

#include <itkImage.h>
#include <itkVector.h>
#include <itkVectorLinearInterpolateImageFunction.h>

#include "itkStandardImageRepresenter.h"
#include "itkStatisticalModel.h"
#include "itkStatisticalModelTransformBase.h"

#include "Representer.h"

namespace itk {

/**
 *
 * \brief An itk transform that allows for deformations defined by a given Statistical Deformation Model.
 *
 * In contrast to the standard StatisticalDeformationModelTransform, this transform performs a linear interpolation of the
 * PCABasis. This has the advantage that a model can be fitted which has a much lower resolution that the image, that needs to
 * be explained.
 *
 * \ingroup Transforms
 */
template <class TDataset, class TScalarType,  unsigned int TDimension >
class ITK_EXPORT InterpolatingStatisticalDeformationModelTransform :
    public itk::StatisticalModelTransformBase< TDataset, TScalarType , TDimension> {
  public:

    /* Standard class typedefs. */
    typedef InterpolatingStatisticalDeformationModelTransform            Self;
    typedef itk::StatisticalModelTransformBase< TDataset, TScalarType , TDimension>	 Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;


    itkSimpleNewMacro( Self );


    /** Run-time type information (and related methods). */
    itkTypeMacro(InterpolatingStatisticalDeformationModelTransform,  Superclass);


    typedef typename Superclass::InputPointType         InputPointType;
    typedef typename Superclass::OutputPointType        OutputPointType;
    typedef typename Superclass::RepresenterType RepresenterType;
    typedef typename Superclass::StatisticalModelType StatisticalModelType;
    typedef typename Superclass::JacobianType JacobianType;


    typedef typename RepresenterType::DatasetType DeformationFieldType;
    typedef VectorLinearInterpolateImageFunction<DeformationFieldType, TScalarType> InterpolatorType;

    /**
     * Clone the current transform
     */
    virtual ::itk::LightObject::Pointer CreateAnother() const {
        ::itk::LightObject::Pointer smartPtr;
        Pointer another = Self::New().GetPointer();
        this->CopyBaseMembers(another);
        another->m_meanDeformation = this->m_meanDeformation;
        another->m_PCABasisDeformations = this->m_PCABasisDeformations;
        smartPtr = static_cast<Pointer>(another);
        return smartPtr;
    }

    virtual void SetStatisticalModel(const StatisticalModelType* model) {
        this->Superclass::SetStatisticalModel(model);

        m_meanDeformation = InterpolatorType::New();
        typename DeformationFieldType::Pointer meanDf = model->DrawMean();
        m_meanDeformation->SetInputImage(meanDf);
        for (unsigned i = 0; i < model->GetNumberOfPrincipalComponents(); i++) {
            typename DeformationFieldType::Pointer deformationField = model->DrawPCABasisSample(i);
            typename InterpolatorType::Pointer basisI = InterpolatorType::New();
            basisI->SetInputImage(deformationField);
            m_PCABasisDeformations.push_back(basisI);
        }
    }



    void ComputeJacobianWithRespectToParameters(const InputPointType  &pt, JacobianType &jacobian)  const {
        jacobian.SetSize(TDimension, m_PCABasisDeformations.size());
        jacobian.Fill(0);
        if (m_meanDeformation->IsInsideBuffer(pt) == false)
            return;

        for(unsigned j = 0; j < m_PCABasisDeformations.size(); j++) {
            typename RepresenterType::ValueType d = m_PCABasisDeformations[j]->Evaluate(pt);
            for(unsigned i = 0; i < TDimension; i++) {
                jacobian(i,j) += d[i] ;
            }
        }

        itkDebugMacro( << "Jacobian with MM:\n" << jacobian);
        itkDebugMacro( << "After GetMorphableModelJacobian:"
                       << "\nJacobian = \n" << jacobian);
    }


    /**
     * Transform a given point according to the deformation induced by the StatisticalModel,
     * given the current parameters.
     *
     * \param pt The point to tranform
     * \return The transformed point
     */
    virtual OutputPointType  TransformPoint(const InputPointType &pt) const {
        if (m_meanDeformation->IsInsideBuffer(pt) == false) {
            return pt;
        }
        assert(this->m_coeff_vector.size() == m_PCABasisDeformations.size());
        typename RepresenterType::ValueType def = m_meanDeformation->Evaluate(pt);

        for (unsigned i = 0; i < m_PCABasisDeformations.size(); i++) {
            typename RepresenterType::ValueType defBasisI =  m_PCABasisDeformations[i]->Evaluate(pt);
            def += (defBasisI * this->m_coeff_vector[i]);
        }

        OutputPointType transformedPoint;
        for (unsigned i = 0; i < pt.GetPointDimension(); i++) {
            transformedPoint[i] = pt[i] + def[i];
        }

        return transformedPoint;
    }

    virtual ~InterpolatingStatisticalDeformationModelTransform() {}

    InterpolatingStatisticalDeformationModelTransform() {}

  private:


    InterpolatingStatisticalDeformationModelTransform(const InterpolatingStatisticalDeformationModelTransform& orig); // purposely not implemented
    InterpolatingStatisticalDeformationModelTransform& operator=(const InterpolatingStatisticalDeformationModelTransform& rhs); //purposely not implemented


    typename InterpolatorType::Pointer m_meanDeformation;
    std::vector<typename InterpolatorType::Pointer> m_PCABasisDeformations;
};


}  // namespace itk

#endif // __ItkInterpolatingStatisticalDeformationModelTransform

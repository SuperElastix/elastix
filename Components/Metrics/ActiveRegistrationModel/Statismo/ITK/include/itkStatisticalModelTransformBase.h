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

#ifndef __itkStatisticalModelTransform_h
#define __itkStatisticalModelTransform_h

#include <iostream>

#include <itkImage.h>
#include <itkTransform.h>
#include <itkVector.h>

#include "Representer.h"

#include "itkStatisticalModel.h"

namespace itk {

/**
 *
 * \brief Base class that implements an itk transform interface for statistical models.
 *
 * Statistical models (shape or deformation models) are often used to model the typical variations within
 * an object class. The StatisticalModelTransformBase implements the standard Transform interface, and thus allows
 * for the use of statistical models within the ITK registration framework.
 * Subclasses will need to implement the TransformPoint method, as its semantics depends on the type of statistical model.
 *
 * \ingroup Transforms
 */

template <class TDataset, class TScalarType,  unsigned int TInputDimension, unsigned int TOutputDimension = TInputDimension  >
class ITK_EXPORT StatisticalModelTransformBase :
    public itk::Transform< TScalarType , TInputDimension, TOutputDimension> {
  public:
    /* Standard class typedefs. */
    typedef StatisticalModelTransformBase            Self;
    typedef itk::Transform< TScalarType , TInputDimension, TOutputDimension>	Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    typedef vnl_vector<statismo::ScalarType> VectorType;
    typedef vnl_matrix<statismo::ScalarType> MatrixType;


    /**
     * Copy the members of the current transform. To be used by subclasses in the CreateAnother method.
     */
    virtual void CopyBaseMembers(StatisticalModelTransformBase* another) const {
        another->m_StatisticalModel = m_StatisticalModel;
        another->m_coeff_vector = m_coeff_vector;
        another->m_usedNumberCoefficients = m_usedNumberCoefficients;
        another->m_FixedParameters = m_FixedParameters;
        another->m_Parameters = this->m_Parameters;
    }


    /** Run-time type information (and related methods). */
    itkTypeMacro( StatisticalModelTransformBase, Superclass );

    /* Dimension of parameters. */
    itkStaticConstMacro(SpaceDimension, unsigned int, TInputDimension);
    itkStaticConstMacro(InputSpaceDimension, unsigned int, TInputDimension);
    itkStaticConstMacro(OutputSpaceDimension, unsigned int, TOutputDimension);


    /* Parameters Type   */
    typedef typename Superclass::ParametersType         ParametersType;
    typedef typename Superclass::JacobianType           JacobianType;
    typedef typename Superclass::ScalarType             ScalarType;
    typedef typename Superclass::InputPointType         InputPointType;
    typedef typename Superclass::OutputPointType        OutputPointType;
    typedef typename Superclass::InputVectorType        InputVectorType;
    typedef typename Superclass::OutputVectorType       OutputVectorType;
    typedef typename Superclass::InputVnlVectorType     InputVnlVectorType;
    typedef typename Superclass::OutputVnlVectorType    OutputVnlVectorType;
    typedef typename Superclass::InputCovariantVectorType
    InputCovariantVectorType;
    typedef typename Superclass::OutputCovariantVectorType
    OutputCovariantVectorType;

    typedef statismo::Representer<TDataset> RepresenterType;
    typedef itk::StatisticalModel<TDataset> StatisticalModelType;


    virtual void ComputeJacobianWithRespectToParameters(const InputPointType  &pt, JacobianType & jacobian) const;

    /**
     * Transform a given point according to the deformation induced by the StatisticalModel,
     * given the current parameters.
     *
     * \param pt The point to tranform
     * \return The transformed point
     */
    virtual OutputPointType  TransformPoint(const InputPointType &pt) const = 0;

    /**
     *  Set the parameters to the IdentityTransform
     *  */
    virtual void SetIdentity(void);

    /**
     * Set the parameters of the transform
     */
    virtual void SetParameters( const ParametersType & );

    /**
     * Get the parameters of the transform
     */
    virtual const ParametersType& GetParameters(void) const;

    /**
     * Does nothing - as the transform does not have any fixed parameters
     */
    virtual void SetFixedParameters( const ParametersType &p ) {
        // there no fixed parameters

    }

    /**
     * returns an empty Parameter vector, as the tranform does not have any fixed parameters
     */
    virtual const ParametersType& GetFixedParameters(void) const {
        return this->m_FixedParameters;
    };

    /**
     * Convenience method to obtain the current coefficients of the StatisticalModel as a statismo::VectorType.
     * The resulting vector is the same as it would be obtained from GetParameters.
     */
    virtual VectorType GetCoefficients() const {
        return m_coeff_vector;
    }

    /**
     * Convenicne method to set the coefficients of the underlying StatisticalModel from a statismo::VectorType.
     * This has the same effect as calling SetParameters.
     */
    virtual void SetCoefficients( VectorType& coefficients) {
        m_coeff_vector = coefficients;
    }

    /**
     * Set the statistical model that defines the valid transformations.
     */
    void SetStatisticalModel(const StatisticalModelType* model);

    /**
     * Returns the statistical model used.
     */
    typename StatisticalModelType::ConstPointer GetStatisticalModel() const;

    /**
     * Set the number of PCA Coefficients used by the model. This parameters has a
     * regularization effect. Setting it to a small value will restrict the possible tranformations
     * to the main modes of variations.
     */
    void SetUsedNumberOfCoefficients(unsigned n) {
        m_usedNumberCoefficients = n;
    }

    /**
     * returns the number of used model coefficients.
     */
    unsigned GetUsedNumberOfCoefficients() {
        return m_usedNumberCoefficients;
    }

  protected:

    StatisticalModelTransformBase();
    virtual ~StatisticalModelTransformBase() {};

    void PrintSelf(std::ostream &os, Indent indent) const;

    typename StatisticalModelType::ConstPointer m_StatisticalModel;
    VectorType m_coeff_vector;
    unsigned m_usedNumberCoefficients;
    ParametersType m_FixedParameters;

    StatisticalModelTransformBase(const Self& obj);// : Superclass(obj) {} //purposely not implemented
    void operator=(const Self& rhs);// { return Superclass::operator=(rhs); } //purposely not implemented



};


}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
# include "itkStatisticalModelTransformBase.hxx"
#endif

#endif /* __itkStatisticalModelTransform_h */

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

#ifndef __itkStatisticalModelTransformBase_hxx
#define __itkStatisticalModelTransformBase_hxx

#include "itkStatisticalModelTransformBase.h"


namespace itk {

/*!
 * Constructor with default arguments.
 */
template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>
::StatisticalModelTransformBase() :
    Superclass(0), // we don't know the number of parameters at this point.
    m_StatisticalModel(0),
    m_coeff_vector(0),
    m_usedNumberCoefficients(10000) { // something large
    itkDebugMacro( << "Constructor MorphableModelTransform()");

    this->m_FixedParameters.SetSize(0);
}



/*!
 * Set the morphable model and ajust the parameters dimension.
 */
template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
void
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>
::SetStatisticalModel(const StatisticalModelType* model) {
    itkDebugMacro( << "Setting statistical model ");
    m_StatisticalModel = model;


    this->m_Parameters.SetSize(model->GetNumberOfPrincipalComponents());
    this->m_Parameters.Fill(0.0);

    this->m_coeff_vector.set_size(model->GetNumberOfPrincipalComponents());

}

template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
typename StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>::StatisticalModelType::ConstPointer
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>
::GetStatisticalModel() const {
    itkDebugMacro( << "Getting statistical model ");
    return m_StatisticalModel;
}


/*!
 * Set the parameters to the IdentityTransform.
 */
template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
void
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>
::SetIdentity( ) {
    itkDebugMacro( << "Setting Identity");

    for (unsigned i = 0; i  < this->GetNumberOfParameters(); i++)
        this->m_coeff_vector[i] = 0;


    this->Modified();
}

template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
void
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>
::SetParameters( const ParametersType & parameters ) {
    itkDebugMacro( << "Setting parameters " << parameters );

    // Set angle
    for(unsigned int i=0; i < std::min(m_usedNumberCoefficients, (unsigned) this->GetNumberOfParameters()); i++) {
        m_coeff_vector[i] = parameters[i];
    }
    for (unsigned int i = std::min(m_usedNumberCoefficients, (unsigned) this->GetNumberOfParameters()); i <  this->GetNumberOfParameters(); i++) {
        m_coeff_vector[i] = 0;
    }

    // Modified is always called since we just have a pointer to the
    // parameters and cannot know if the parameters have changed.
    this->Modified();

    itkDebugMacro(<<"After setting parameters ");
}





// Get Parameters
template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
const typename StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>::ParametersType &
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>
::GetParameters( void ) const {
    itkDebugMacro( << "Getting parameters ");


    // Get the translation
    for(unsigned int i=0; i < this->GetNumberOfParameters(); i++) {
        this->m_Parameters[i] = this->m_coeff_vector[i];
    }
    itkDebugMacro(<<"After getting parameters " << this->m_Parameters );

    return this->m_Parameters;
}






template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
void
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>
::ComputeJacobianWithRespectToParameters(const InputPointType  &pt, JacobianType &jacobian)  const {
    jacobian.SetSize(OutputSpaceDimension, m_StatisticalModel->GetNumberOfPrincipalComponents());
    jacobian.Fill(0);

    const MatrixType& statModelJacobian = m_StatisticalModel->GetJacobian(pt);

    for (unsigned i = 0; i < statModelJacobian.rows(); i++) {
        for (unsigned j = 0; j <  std::min(m_usedNumberCoefficients, (unsigned) this->GetNumberOfParameters()); j++) {
            jacobian[i][j] = statModelJacobian[i][j];
        }
    }


    itkDebugMacro( << "Jacobian with MM:\n" << jacobian);
    itkDebugMacro( << "After GetMorphableModelJacobian:"
                   << "\nJacobian = \n" << jacobian);
}



// Print self
template <class TRepresenter, class TScalarType, unsigned int TInputDimension, unsigned int TOutputDimension >
void
StatisticalModelTransformBase<TRepresenter,  TScalarType, TInputDimension, TOutputDimension>::
PrintSelf(std::ostream &os, Indent indent) const {
    Superclass::PrintSelf(os,indent);
}

} // namespace

#endif

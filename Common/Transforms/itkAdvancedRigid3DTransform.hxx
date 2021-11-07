/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedRigid3DTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-12-19 16:34:40 $
  Version:   $Revision: 1.36 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkAdvancedRigid3DTransform_hxx
#define _itkAdvancedRigid3DTransform_hxx

#include "itkAdvancedRigid3DTransform.h"

namespace itk
{

// Constructor with default arguments
template <class TScalarType>
AdvancedRigid3DTransform<TScalarType>::AdvancedRigid3DTransform()
  : Superclass(ParametersDimension)
{}

// Constructor with default arguments
template <class TScalarType>
AdvancedRigid3DTransform<TScalarType>::AdvancedRigid3DTransform(unsigned int paramDim)
  : Superclass(paramDim)
{}

// Print self
template <class TScalarType>
void
AdvancedRigid3DTransform<TScalarType>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


// Check if input matrix is orthogonal to within tolerance
template <class TScalarType>
bool
AdvancedRigid3DTransform<TScalarType>::MatrixIsOrthogonal(const MatrixType & matrix, double tolerance)
{
  typename MatrixType::InternalMatrixType test = matrix.GetVnlMatrix() * matrix.GetTranspose();

  if (!test.is_identity(tolerance))
  {
    return false;
  }

  return true;
}


// Directly set the rotation matrix
template <class TScalarType>
void
AdvancedRigid3DTransform<TScalarType>::SetMatrix(const MatrixType & matrix)
{
  const double tolerance = 1e-10;
  if (!this->MatrixIsOrthogonal(matrix, tolerance))
  {
    itkExceptionMacro(<< "Attempting to set a non-orthogonal rotation matrix");
  }

  this->Superclass::SetMatrix(matrix);
}


// Set optimizable parameters from array
template <class TScalarType>
void
AdvancedRigid3DTransform<TScalarType>::SetParameters(const ParametersType & parameters)
{
  this->m_Parameters = parameters;

  unsigned int     par = 0;
  MatrixType       matrix;
  OutputVectorType translation;

  for (unsigned int row = 0; row < 3; ++row)
  {
    for (unsigned int col = 0; col < 3; ++col)
    {
      matrix[row][col] = this->m_Parameters[par];
      ++par;
    }
  }

  for (unsigned int dim = 0; dim < 3; ++dim)
  {
    translation[dim] = this->m_Parameters[par];
    ++par;
  }

  const double tolerance = 1e-10;
  if (!this->MatrixIsOrthogonal(matrix, tolerance))
  {
    itkExceptionMacro(<< "Attempting to set a non-orthogonal rotation matrix");
  }

  this->SetVarMatrix(matrix);
  this->SetVarTranslation(translation);

  // Update matrix and offset.
  // Technically ComputeMatrix() is not require as the parameters are
  // directly the elements of the matrix.
  this->ComputeMatrix();
  this->ComputeOffset();

  this->Modified();
}

} // namespace itk

#endif

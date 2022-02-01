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
  Module:    $RCSfile: itkAdvancedRigid2DTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-12-19 16:34:40 $
  Version:   $Revision: 1.25 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedRigid2DTransform_hxx
#define itkAdvancedRigid2DTransform_hxx

#include "itkAdvancedRigid2DTransform.h"
#include <vnl/algo/vnl_svd_fixed.h>

namespace itk
{

// Default-constructor
template <class TScalarType>
AdvancedRigid2DTransform<TScalarType>::AdvancedRigid2DTransform()
  : Superclass(ParametersDimension)
{
  m_Angle = NumericTraits<TScalarType>::Zero;
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with arguments
template <class TScalarType>
AdvancedRigid2DTransform<TScalarType>::AdvancedRigid2DTransform(unsigned int spaceDimension,
                                                                unsigned int parametersDimension)
  : Superclass(parametersDimension)
{
  m_Angle = NumericTraits<TScalarType>::Zero;
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Print self
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Angle       = " << m_Angle << std::endl;
}


// Set the rotation matrix
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::SetMatrix(const MatrixType & matrix)
{
  itkDebugMacro("setting  m_Matrix  to " << matrix);
  // The matrix must be orthogonal otherwise it is not
  // representing a valid rotaion in 2D space
  typename MatrixType::InternalMatrixType test = matrix.GetVnlMatrix() * matrix.GetTranspose();

  const double tolerance = 1e-10;
  if (!test.is_identity(tolerance))
  {
    itk::ExceptionObject ex(__FILE__, __LINE__, "Attempt to set a Non-Orthogonal matrix", ITK_LOCATION);
    throw ex;
  }

  this->SetVarMatrix(matrix);
  this->ComputeOffset();
  this->ComputeMatrixParameters();
  this->Modified();
}


/** Compute the Angle from the Rotation Matrix */
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::ComputeMatrixParameters()
{
  // Extract the orthogonal part of the matrix
  //
  const vnl_matrix_fixed<TScalarType, 2, 2> p = this->GetMatrix().GetVnlMatrix();
  const vnl_svd_fixed<TScalarType, 2, 2>    svd(p);
  const vnl_matrix_fixed<TScalarType, 2, 2> r = svd.U() * svd.V().transpose();

  m_Angle = std::acos(r[0][0]);

  if (r[1][0] < 0.0)
  {
    m_Angle = -m_Angle;
  }

  if (r[1][0] - sin(m_Angle) > 0.000001)
  {
    itkWarningMacro("Bad Rotation Matrix " << this->GetMatrix());
  }

  /** Update Jacobian of spatial Jacobian */
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Reset the transform to an identity transform
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::SetIdentity()
{
  this->Superclass::SetIdentity();
  m_Angle = NumericTraits<TScalarType>::Zero;
  // make sure to also precompute Jacobian:
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set the angle of rotation
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::SetAngle(TScalarType angle)
{
  m_Angle = angle;
  this->ComputeMatrix();
  this->ComputeOffset();
  this->Modified();
}


// Compute the matrix from the angle
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::ComputeMatrix()
{
  const double ca = std::cos(m_Angle);
  const double sa = std::sin(m_Angle);

  MatrixType rotationMatrix;
  rotationMatrix[0][0] = ca;
  rotationMatrix[0][1] = -sa;
  rotationMatrix[1][0] = sa;
  rotationMatrix[1][1] = ca;

  this->SetVarMatrix(rotationMatrix);
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set Parameters
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::SetParameters(const ParametersType & parameters)
{
  itkDebugMacro(<< "Setting parameters " << parameters);

  // Set angle
  this->SetVarAngle(parameters[0]);

  // Set translation
  OutputVectorType translation;
  for (unsigned int i = 0; i < OutputSpaceDimension; ++i)
  {
    translation[i] = parameters[i + 1];
  }
  this->SetVarTranslation(translation);

  // Update matrix and offset
  this->ComputeMatrix();
  this->ComputeOffset();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

  itkDebugMacro(<< "After setting parameters ");
}


// Get Parameters
template <class TScalarType>
auto
AdvancedRigid2DTransform<TScalarType>::GetParameters() const -> const ParametersType &
{
  itkDebugMacro(<< "Getting parameters ");

  // Get the angle
  this->m_Parameters[0] = m_Angle;

  // Get the translation
  for (unsigned int i = 0; i < OutputSpaceDimension; ++i)
  {
    this->m_Parameters[i + 1] = this->GetTranslation()[i];
  }

  itkDebugMacro(<< "After getting parameters " << this->m_Parameters);

  return this->m_Parameters;
}

// Compute transformation Jacobian
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::GetJacobian(const InputPointType &       p,
                                                   JacobianType &               j,
                                                   NonZeroJacobianIndicesType & nzji) const
{
  // Initialize the Jacobian. Resizing is only performed when needed.
  // Filling with zeros is needed because the lower loops only visit
  // the nonzero positions.
  j.SetSize(OutputSpaceDimension, ParametersDimension);
  j.Fill(0.0);

  // Some helper variables
  const double ca = std::cos(m_Angle);
  const double sa = std::sin(m_Angle);
  const double cx = Superclass::GetCenter()[0];
  const double cy = Superclass::GetCenter()[1];

  // derivatives with respect to the angle
  j[0][0] = -sa * (p[0] - cx) - ca * (p[1] - cy);
  j[1][0] = ca * (p[0] - cx) - sa * (p[1] - cy);

  // compute derivatives for the translation part
  unsigned int blockOffset = 1;
  for (unsigned int dim = 0; dim < OutputSpaceDimension; ++dim)
  {
    j[dim][blockOffset + dim] = 1.0;
  }

  // Copy the constant nonZeroJacobianIndices
  nzji = this->m_NonZeroJacobianIndices;
}


// Precompute Jacobian of Spatial Jacobian
template <class TScalarType>
void
AdvancedRigid2DTransform<TScalarType>::PrecomputeJacobianOfSpatialJacobian()
{
  /** The Jacobian of spatial Jacobian remains constant, so is precomputed */
  const double                    ca = std::cos(m_Angle);
  const double                    sa = std::sin(m_Angle);
  JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
  jsj.resize(ParametersDimension);
  if (ParametersDimension > 1)
  {
    jsj[0](0, 0) = -sa;
    jsj[0](0, 1) = -ca;
    jsj[0](1, 0) = ca;
    jsj[0](1, 1) = -sa;
  }
  for (unsigned int par = 1; par < ParametersDimension; ++par)
  {
    jsj[par].Fill(0.0);
  }
}


} // namespace itk

#endif

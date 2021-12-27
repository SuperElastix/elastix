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
  Module:    $RCSfile: itkAdvancedSimilarity2DTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-01-24 05:12:41 $
  Version:   $Revision: 1.23 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkAdvancedSimilarity2DTransform_hxx
#define _itkAdvancedSimilarity2DTransform_hxx

#include "itkAdvancedSimilarity2DTransform.h"
#include <vnl/vnl_math.h>

namespace itk
{

// Constructor with default arguments
template <class TScalarType>
AdvancedSimilarity2DTransform<TScalarType>::AdvancedSimilarity2DTransform()
  : Superclass(OutputSpaceDimension, ParametersDimension)
{
  m_Scale = 1.0f;
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set Parameters
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::SetParameters(const ParametersType & parameters)
{
  itkDebugMacro(<< "Setting parameters " << parameters);

  // Set scale
  this->SetVarScale(parameters[0]);

  // Set angle
  this->SetVarAngle(parameters[1]);

  // Set translation
  OffsetType translation;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    translation[i] = parameters[i + 2];
  }
  this->SetVarTranslation(translation);

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
AdvancedSimilarity2DTransform<TScalarType>::GetParameters() const -> const ParametersType &
{
  itkDebugMacro(<< "Getting parameters ");

  this->m_Parameters[0] = this->GetScale();
  this->m_Parameters[1] = this->GetAngle();

  // Get the translation
  OffsetType translation = this->GetTranslation();
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    this->m_Parameters[i + 2] = translation[i];
  }

  itkDebugMacro(<< "After getting parameters " << this->m_Parameters);

  return this->m_Parameters;
}

// Set Scale Part
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::SetScale(ScaleType scale)
{
  m_Scale = scale;
  this->ComputeMatrix();
  this->ComputeOffset();
}


// Compute the matrix
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::ComputeMatrix()
{
  const double angle = this->GetAngle();

  const double cc = std::cos(angle);
  const double ss = std::sin(angle);

  const double ca = cc * m_Scale;
  const double sa = ss * m_Scale;

  MatrixType matrix;
  matrix[0][0] = ca;
  matrix[0][1] = -sa;
  matrix[1][0] = sa;
  matrix[1][1] = ca;

  this->SetVarMatrix(matrix);

  this->PrecomputeJacobianOfSpatialJacobian();
}


/** Compute the Angle from the Rotation Matrix */
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::ComputeMatrixParameters()
{
  m_Scale = std::sqrt(vnl_math::sqr(this->GetMatrix()[0][0]) + vnl_math::sqr(this->GetMatrix()[0][1]));

  this->SetVarAngle(std::acos(this->GetMatrix()[0][0] / m_Scale));

  if (this->GetMatrix()[1][0] < 0.0)
  {
    this->SetVarAngle(-this->GetAngle());
  }

  if ((this->GetMatrix()[1][0] / m_Scale) - std::sin(this->GetAngle()) > 0.000001)
  {
    std::cout << "Bad Rotation Matrix" << std::endl;
  }

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Compute the transformation Jacobian
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::GetJacobian(const InputPointType &       p,
                                                        JacobianType &               j,
                                                        NonZeroJacobianIndicesType & nzji) const
{
  // Initialize the Jacobian. Resizing is only performed when needed.
  // Filling with zeros is needed because the lower loops only visit
  // the nonzero positions.
  j.SetSize(OutputSpaceDimension, ParametersDimension);
  j.Fill(0.0);

  // Some helper variables
  const double         angle = this->GetAngle();
  const double         ca = std::cos(angle);
  const double         sa = std::sin(angle);
  const InputPointType center = this->GetCenter();
  const double         cx = center[0];
  const double         cy = center[1];

  // derivatives with respect to the scale
  j[0][0] = ca * (p[0] - cx) - sa * (p[1] - cy);
  j[1][0] = sa * (p[0] - cx) + ca * (p[1] - cy);

  // derivatives with respect to the angle
  j[0][1] = (-sa * (p[0] - cx) - ca * (p[1] - cy)) * m_Scale;
  j[1][1] = (ca * (p[0] - cx) - sa * (p[1] - cy)) * m_Scale;

  // compute derivatives with respect to the translation part
  // first with respect to tx
  j[0][2] = 1.0;
  j[1][2] = 0.0;
  // first with respect to ty
  j[0][3] = 0.0;
  j[1][3] = 1.0;

  // Copy the constant nonZeroJacobianIndices
  nzji = this->m_NonZeroJacobianIndices;
}


// Set Identity
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::SetIdentity()
{
  this->Superclass::SetIdentity();
  m_Scale = static_cast<TScalarType>(1.0f);
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Print self
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Scale =" << m_Scale << std::endl;
}


// Set the similarity matrix
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::SetMatrix(const MatrixType & matrix)
{
  itkDebugMacro("setting  m_Matrix  to " << matrix);

  typename MatrixType::InternalMatrixType test = matrix.GetVnlMatrix() * matrix.GetTranspose();

  test /= test[0][0]; // factor out the scale

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


// Precompute Jacobian of Spatial Jacobian
template <class TScalarType>
void
AdvancedSimilarity2DTransform<TScalarType>::PrecomputeJacobianOfSpatialJacobian()
{
  /** The Jacobian of spatial Jacobian remains constant, so is precomputed */
  const double                    angle = this->GetAngle();
  double                          ca = std::cos(angle);
  double                          sa = std::sin(angle);
  JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
  jsj.resize(ParametersDimension);
  if (ParametersDimension > 1)
  {
    jsj[0](0, 0) = ca;
    jsj[0](0, 1) = -sa;
    jsj[0](1, 0) = sa;
    jsj[0](1, 1) = ca;
    ca *= this->m_Scale;
    sa *= this->m_Scale;
    jsj[1](0, 0) = -sa;
    jsj[1](0, 1) = -ca;
    jsj[1](1, 0) = ca;
    jsj[1](1, 1) = -sa;
  }
  for (unsigned int par = 2; par < ParametersDimension; ++par)
  {
    jsj[par].Fill(0.0);
  }
}


} // namespace itk

#endif

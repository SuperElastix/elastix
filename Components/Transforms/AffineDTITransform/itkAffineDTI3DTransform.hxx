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
  Module:    $RCSfile: itkAffineDTI3DTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-10-13 15:36:31 $
  Version:   $Revision: 1.24 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAffineDTI3DTransform_hxx
#define itkAffineDTI3DTransform_hxx

#include "itkAffineDTI3DTransform.h"
#include <cmath>

namespace itk
{

// Constructor with default arguments
template <class TScalarType>
AffineDTI3DTransform<TScalarType>::AffineDTI3DTransform()
  : Superclass(ParametersDimension)
{
  this->m_Angle.Fill(itk::NumericTraits<ScalarType>::ZeroValue());
  this->m_Shear.Fill(itk::NumericTraits<ScalarType>::ZeroValue());
  this->m_Scale.Fill(itk::NumericTraits<ScalarType>::OneValue());

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with default arguments
template <class TScalarType>
AffineDTI3DTransform<TScalarType>::AffineDTI3DTransform(const MatrixType & matrix, const OutputPointType & offset)
{
  this->SetMatrix(matrix);

  OffsetType off;
  off[0] = offset[0];
  off[1] = offset[1];
  off[2] = offset[2];
  this->SetOffset(off);

  // this->ComputeMatrix?
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with arguments
template <class TScalarType>
AffineDTI3DTransform<TScalarType>::AffineDTI3DTransform(unsigned int spaceDimension, unsigned int parametersDimension)
  : Superclass(spaceDimension, parametersDimension)
{
  this->m_Angle.Fill(itk::NumericTraits<ScalarType>::ZeroValue());
  this->m_Shear.Fill(itk::NumericTraits<ScalarType>::ZeroValue());
  this->m_Scale.Fill(itk::NumericTraits<ScalarType>::OneValue());
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set Angles etc
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::SetVarAngleScaleShear(ScalarArrayType angle,
                                                         ScalarArrayType shear,
                                                         ScalarArrayType scale)
{
  this->m_Angle = angle;
  this->m_Shear = shear;
  this->m_Scale = scale;
}


// Set Parameters
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::SetParameters(const ParametersType & parameters)
{
  itkDebugMacro(<< "Setting parameters " << parameters);

  this->m_Angle[0] = parameters[0];
  this->m_Angle[1] = parameters[1];
  this->m_Angle[2] = parameters[2];
  this->m_Shear[0] = parameters[3];
  this->m_Shear[1] = parameters[4];
  this->m_Shear[2] = parameters[5];
  this->m_Scale[0] = parameters[6];
  this->m_Scale[1] = parameters[7];
  this->m_Scale[2] = parameters[8];
  this->ComputeMatrix();

  // Transfer the translation part
  OutputVectorType newTranslation;
  newTranslation[0] = parameters[9];
  newTranslation[1] = parameters[10];
  newTranslation[2] = parameters[11];
  this->SetVarTranslation(newTranslation);
  this->ComputeOffset();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

  itkDebugMacro(<< "After setting parameters ");
}


// Get Parameters
template <class TScalarType>
auto
AffineDTI3DTransform<TScalarType>::GetParameters() const -> const ParametersType &
{
  this->m_Parameters[0] = this->m_Angle[0];
  this->m_Parameters[1] = this->m_Angle[1];
  this->m_Parameters[2] = this->m_Angle[2];
  this->m_Parameters[3] = this->m_Shear[0];
  this->m_Parameters[4] = this->m_Shear[1];
  this->m_Parameters[5] = this->m_Shear[2];
  this->m_Parameters[6] = this->m_Scale[0];
  this->m_Parameters[7] = this->m_Scale[1];
  this->m_Parameters[8] = this->m_Scale[2];
  this->m_Parameters[9] = this->GetTranslation()[0];
  this->m_Parameters[10] = this->GetTranslation()[1];
  this->m_Parameters[11] = this->GetTranslation()[2];

  return this->m_Parameters;
}

// SetIdentity()
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::SetIdentity()
{
  Superclass::SetIdentity();
  this->m_Angle.Fill(itk::NumericTraits<ScalarType>::ZeroValue());
  this->m_Shear.Fill(itk::NumericTraits<ScalarType>::ZeroValue());
  this->m_Scale.Fill(itk::NumericTraits<ScalarType>::OneValue());
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Compute angles from the rotation matrix
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::ComputeMatrixParameters()
{
  // let's hope we don't need it :)
  itkExceptionMacro(<< "This function has not been implemented yet!");
  this->ComputeMatrix();
}


// Compute the matrix
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::ComputeMatrix()
{
  // need to check if angles are in the right order
  const double cx = std::cos(this->m_Angle[0]);
  const double sx = std::sin(this->m_Angle[0]);
  const double cy = std::cos(this->m_Angle[1]);
  const double sy = std::sin(this->m_Angle[1]);
  const double cz = std::cos(this->m_Angle[2]);
  const double sz = std::sin(this->m_Angle[2]);
  const double gx = this->m_Shear[0];
  const double gy = this->m_Shear[1];
  const double gz = this->m_Shear[2];
  const double ssx = this->m_Scale[0];
  const double ssy = this->m_Scale[1];
  const double ssz = this->m_Scale[2];

  /** NB: opposite definition as in EulerTransform */
  MatrixType RotationX;
  RotationX[0][0] = 1;
  RotationX[0][1] = 0;
  RotationX[0][2] = 0;
  RotationX[1][0] = 0;
  RotationX[1][1] = cx;
  RotationX[1][2] = sx;
  RotationX[2][0] = 0;
  RotationX[2][1] = -sx;
  RotationX[2][2] = cx;

  MatrixType RotationY;
  RotationY[0][0] = cy;
  RotationY[0][1] = 0;
  RotationY[0][2] = -sy;
  RotationY[1][0] = 0;
  RotationY[1][1] = 1;
  RotationY[1][2] = 0;
  RotationY[2][0] = sy;
  RotationY[2][1] = 0;
  RotationY[2][2] = cy;

  MatrixType RotationZ;
  RotationZ[0][0] = cz;
  RotationZ[0][1] = sz;
  RotationZ[0][2] = 0;
  RotationZ[1][0] = -sz;
  RotationZ[1][1] = cz;
  RotationZ[1][2] = 0;
  RotationZ[2][0] = 0;
  RotationZ[2][1] = 0;
  RotationZ[2][2] = 1;

  MatrixType ShearX;
  ShearX[0][0] = 1;
  ShearX[0][1] = 0;
  ShearX[0][2] = gx;
  ShearX[1][0] = 0;
  ShearX[1][1] = 1;
  ShearX[1][2] = 0;
  ShearX[2][0] = 0;
  ShearX[2][1] = 0;
  ShearX[2][2] = 1;

  MatrixType ShearY;
  ShearY[0][0] = 1;
  ShearY[0][1] = 0;
  ShearY[0][2] = 0;
  ShearY[1][0] = gy;
  ShearY[1][1] = 1;
  ShearY[1][2] = 0;
  ShearY[2][0] = 0;
  ShearY[2][1] = 0;
  ShearY[2][2] = 1;

  MatrixType ShearZ;
  ShearZ[0][0] = 1;
  ShearZ[0][1] = 0;
  ShearZ[0][2] = 0;
  ShearZ[1][0] = 0;
  ShearZ[1][1] = 1;
  ShearZ[1][2] = 0;
  ShearZ[2][0] = 0;
  ShearZ[2][1] = gz;
  ShearZ[2][2] = 1;

  MatrixType Scale;
  Scale[0][0] = ssx;
  Scale[0][1] = 0;
  Scale[0][2] = 0;
  Scale[1][0] = 0;
  Scale[1][1] = ssy;
  Scale[1][2] = 0;
  Scale[2][0] = 0;
  Scale[2][1] = 0;
  Scale[2][2] = ssz;

  this->SetVarMatrix(RotationX * RotationY * RotationZ * ShearX * ShearY * ShearZ * Scale);

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set parameters
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::GetJacobian(const InputPointType &       p,
                                               JacobianType &               j,
                                               NonZeroJacobianIndicesType & nzji) const
{
  j.SetSize(OutputSpaceDimension, ParametersDimension);
  j.Fill(0.0);
  const JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;

  /** Compute dR/dmu * (p-c) */
  const InputVectorType pp = p - this->GetCenter();
  for (unsigned int dim = 0; dim < 9; ++dim)
  {
    const InputVectorType column = jsj[dim] * pp;
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      j(i, dim) = column[i];
    }
  }

  // compute derivatives for the translation part
  const unsigned int blockOffset = 9;
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    j[dim][blockOffset + dim] = 1.0;
  }

  nzji = this->m_NonZeroJacobianIndices;
}


// Precompute Jacobian of Spatial Jacobian
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::PrecomputeJacobianOfSpatialJacobian()
{
  if (ParametersDimension < 12)
  {
    /** Some subclass has a different number of parameters */
    return;
  }

  /** The Jacobian of spatial Jacobian is constant over inputspace, so is precomputed */
  JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
  jsj.resize(ParametersDimension);

  // need to check if angles are in the right order
  const double cx = std::cos(this->m_Angle[0]);
  const double sx = std::sin(this->m_Angle[0]);
  const double cy = std::cos(this->m_Angle[1]);
  const double sy = std::sin(this->m_Angle[1]);
  const double cz = std::cos(this->m_Angle[2]);
  const double sz = std::sin(this->m_Angle[2]);
  const double gx = this->m_Shear[0];
  const double gy = this->m_Shear[1];
  const double gz = this->m_Shear[2];
  const double ssx = this->m_Scale[0];
  const double ssy = this->m_Scale[1];
  const double ssz = this->m_Scale[2];

  /** derivatives: */
  const double cxd = -std::sin(this->m_Angle[0]);
  const double sxd = std::cos(this->m_Angle[0]);
  const double cyd = -std::sin(this->m_Angle[1]);
  const double syd = std::cos(this->m_Angle[1]);
  const double czd = -std::sin(this->m_Angle[2]);
  const double szd = std::cos(this->m_Angle[2]);

  /** NB: opposite definition as in EulerTransform */
  MatrixType RotationX;
  RotationX[0][0] = 1;
  RotationX[0][1] = 0;
  RotationX[0][2] = 0;
  RotationX[1][0] = 0;
  RotationX[1][1] = cx;
  RotationX[1][2] = sx;
  RotationX[2][0] = 0;
  RotationX[2][1] = -sx;
  RotationX[2][2] = cx;

  MatrixType RotationY;
  RotationY[0][0] = cy;
  RotationY[0][1] = 0;
  RotationY[0][2] = -sy;
  RotationY[1][0] = 0;
  RotationY[1][1] = 1;
  RotationY[1][2] = 0;
  RotationY[2][0] = sy;
  RotationY[2][1] = 0;
  RotationY[2][2] = cy;

  MatrixType RotationZ;
  RotationZ[0][0] = cz;
  RotationZ[0][1] = sz;
  RotationZ[0][2] = 0;
  RotationZ[1][0] = -sz;
  RotationZ[1][1] = cz;
  RotationZ[1][2] = 0;
  RotationZ[2][0] = 0;
  RotationZ[2][1] = 0;
  RotationZ[2][2] = 1;

  MatrixType ShearX;
  ShearX[0][0] = 1;
  ShearX[0][1] = 0;
  ShearX[0][2] = gx;
  ShearX[1][0] = 0;
  ShearX[1][1] = 1;
  ShearX[1][2] = 0;
  ShearX[2][0] = 0;
  ShearX[2][1] = 0;
  ShearX[2][2] = 1;

  MatrixType ShearY;
  ShearY[0][0] = 1;
  ShearY[0][1] = 0;
  ShearY[0][2] = 0;
  ShearY[1][0] = gy;
  ShearY[1][1] = 1;
  ShearY[1][2] = 0;
  ShearY[2][0] = 0;
  ShearY[2][1] = 0;
  ShearY[2][2] = 1;

  MatrixType ShearZ;
  ShearZ[0][0] = 1;
  ShearZ[0][1] = 0;
  ShearZ[0][2] = 0;
  ShearZ[1][0] = 0;
  ShearZ[1][1] = 1;
  ShearZ[1][2] = 0;
  ShearZ[2][0] = 0;
  ShearZ[2][1] = gz;
  ShearZ[2][2] = 1;

  MatrixType ScaleX;
  ScaleX[0][0] = ssx;
  ScaleX[0][1] = 0;
  ScaleX[0][2] = 0;
  ScaleX[1][0] = 0;
  ScaleX[1][1] = 1;
  ScaleX[1][2] = 0;
  ScaleX[2][0] = 0;
  ScaleX[2][1] = 0;
  ScaleX[2][2] = 1;

  MatrixType ScaleY;
  ScaleY[0][0] = 1;
  ScaleY[0][1] = 0;
  ScaleY[0][2] = 0;
  ScaleY[1][0] = 0;
  ScaleY[1][1] = ssy;
  ScaleY[1][2] = 0;
  ScaleY[2][0] = 0;
  ScaleY[2][1] = 0;
  ScaleY[2][2] = 1;

  MatrixType ScaleZ;
  ScaleZ[0][0] = 1;
  ScaleZ[0][1] = 0;
  ScaleZ[0][2] = 0;
  ScaleZ[1][0] = 0;
  ScaleZ[1][1] = 1;
  ScaleZ[1][2] = 0;
  ScaleZ[2][0] = 0;
  ScaleZ[2][1] = 0;
  ScaleZ[2][2] = ssz;

  /** Derivative matrices: */
  MatrixType RotationXd;
  RotationXd[0][0] = 0;
  RotationXd[0][1] = 0;
  RotationXd[0][2] = 0;
  RotationXd[1][0] = 0;
  RotationXd[1][1] = cxd;
  RotationXd[1][2] = sxd;
  RotationXd[2][0] = 0;
  RotationXd[2][1] = -sxd;
  RotationXd[2][2] = cxd;

  MatrixType RotationYd;
  RotationYd[0][0] = cyd;
  RotationYd[0][1] = 0;
  RotationYd[0][2] = -syd;
  RotationYd[1][0] = 0;
  RotationYd[1][1] = 0;
  RotationYd[1][2] = 0;
  RotationYd[2][0] = syd;
  RotationYd[2][1] = 0;
  RotationYd[2][2] = cyd;

  MatrixType RotationZd;
  RotationZd[0][0] = czd;
  RotationZd[0][1] = szd;
  RotationZd[0][2] = 0;
  RotationZd[1][0] = -szd;
  RotationZd[1][1] = czd;
  RotationZd[1][2] = 0;
  RotationZd[2][0] = 0;
  RotationZd[2][1] = 0;
  RotationZd[2][2] = 0;

  MatrixType ShearXd;
  ShearXd[0][0] = 0;
  ShearXd[0][1] = 0;
  ShearXd[0][2] = 1;
  ShearXd[1][0] = 0;
  ShearXd[1][1] = 0;
  ShearXd[1][2] = 0;
  ShearXd[2][0] = 0;
  ShearXd[2][1] = 0;
  ShearXd[2][2] = 0;

  MatrixType ShearYd;
  ShearYd[0][0] = 0;
  ShearYd[0][1] = 0;
  ShearYd[0][2] = 0;
  ShearYd[1][0] = 1;
  ShearYd[1][1] = 0;
  ShearYd[1][2] = 0;
  ShearYd[2][0] = 0;
  ShearYd[2][1] = 0;
  ShearYd[2][2] = 0;

  MatrixType ShearZd;
  ShearZd[0][0] = 0;
  ShearZd[0][1] = 0;
  ShearZd[0][2] = 0;
  ShearZd[1][0] = 0;
  ShearZd[1][1] = 0;
  ShearZd[1][2] = 0;
  ShearZd[2][0] = 0;
  ShearZd[2][1] = 1;
  ShearZd[2][2] = 0;

  MatrixType ScaleXd;
  ScaleXd[0][0] = 1;
  ScaleXd[0][1] = 0;
  ScaleXd[0][2] = 0;
  ScaleXd[1][0] = 0;
  ScaleXd[1][1] = 0;
  ScaleXd[1][2] = 0;
  ScaleXd[2][0] = 0;
  ScaleXd[2][1] = 0;
  ScaleXd[2][2] = 0;

  MatrixType ScaleYd;
  ScaleYd[0][0] = 0;
  ScaleYd[0][1] = 0;
  ScaleYd[0][2] = 0;
  ScaleYd[1][0] = 0;
  ScaleYd[1][1] = 1;
  ScaleYd[1][2] = 0;
  ScaleYd[2][0] = 0;
  ScaleYd[2][1] = 0;
  ScaleYd[2][2] = 0;

  MatrixType ScaleZd;
  ScaleZd[0][0] = 0;
  ScaleZd[0][1] = 0;
  ScaleZd[0][2] = 0;
  ScaleZd[1][0] = 0;
  ScaleZd[1][1] = 0;
  ScaleZd[1][2] = 0;
  ScaleZd[2][0] = 0;
  ScaleZd[2][1] = 0;
  ScaleZd[2][2] = 1;

  jsj[0] = RotationXd * RotationY * RotationZ * ShearX * ShearY * ShearZ * ScaleX * ScaleY * ScaleZ;
  jsj[1] = RotationX * RotationYd * RotationZ * ShearX * ShearY * ShearZ * ScaleX * ScaleY * ScaleZ;
  jsj[2] = RotationX * RotationY * RotationZd * ShearX * ShearY * ShearZ * ScaleX * ScaleY * ScaleZ;
  jsj[3] = RotationX * RotationY * RotationZ * ShearXd * ShearY * ShearZ * ScaleX * ScaleY * ScaleZ;
  jsj[4] = RotationX * RotationY * RotationZ * ShearX * ShearYd * ShearZ * ScaleX * ScaleY * ScaleZ;
  jsj[5] = RotationX * RotationY * RotationZ * ShearX * ShearY * ShearZd * ScaleX * ScaleY * ScaleZ;
  jsj[6] = RotationX * RotationY * RotationZ * ShearX * ShearY * ShearZ * ScaleXd * ScaleY * ScaleZ;
  jsj[7] = RotationX * RotationY * RotationZ * ShearX * ShearY * ShearZ * ScaleX * ScaleYd * ScaleZ;
  jsj[8] = RotationX * RotationY * RotationZ * ShearX * ShearY * ShearZ * ScaleX * ScaleY * ScaleZd;

  /** Translation parameters: */
  for (unsigned int par = 9; par < 12; ++par)
  {
    jsj[par].Fill(0.0);
  }
}


// Print self
template <class TScalarType>
void
AffineDTI3DTransform<TScalarType>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Angles: " << this->m_Angle << std::endl;
  os << indent << "Shear: " << this->m_Shear << std::endl;
  os << indent << "Scale: " << this->m_Scale << std::endl;
}


} // namespace itk

#endif

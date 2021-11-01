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
  Module:    $RCSfile: itkAdvancedMatrixOffsetTransformBase.txx,v $
  Language:  C++
  Date:      $Date: 2008-06-29 12:58:58 $
  Version:   $Revision: 1.17 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkAdvancedMatrixOffsetTransformBase_hxx
#define _itkAdvancedMatrixOffsetTransformBase_hxx

#include "itkNumericTraits.h"
#include "itkAdvancedMatrixOffsetTransformBase.h"
#include "vnl/algo/vnl_matrix_inverse.h"

namespace itk
{

// Constructor with default arguments
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::AdvancedMatrixOffsetTransformBase()
  : Superclass(ParametersDimension)
{
  this->PrecomputeJacobians(ParametersDimension);
}


// Constructor with default arguments
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::AdvancedMatrixOffsetTransformBase(
  unsigned int paramDims)
  : Superclass(paramDims)
  , m_ItkTransform(paramDims)
{
  this->PrecomputeJacobians(paramDims);
}


// Print self
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::PrecomputeJacobians(
  unsigned int paramDims)
{
  /** Nonzero Jacobian indices, for GetJacobian */
  this->m_NonZeroJacobianIndices.resize(paramDims);
  for (unsigned int par = 0; par < paramDims; ++par)
  {
    this->m_NonZeroJacobianIndices[par] = par;
  }

  /** Set to correct size and fill. This may be different for inheriting classes,
   * such as the RigidTransform. */
  this->m_JacobianOfSpatialJacobian.resize(paramDims);
  unsigned int par = 0;
  for (unsigned int row = 0; row < OutputSpaceDimension; ++row)
  {
    for (unsigned int col = 0; col < InputSpaceDimension; ++col)
    {
      if (par < paramDims)
      {
        SpatialJacobianType sj;
        sj.Fill(0.0);
        sj[row][col] = 1.0;
        this->m_JacobianOfSpatialJacobian[par] = sj;
        ++par;
      }
    }
  }

  /** Set to correct size and initialize to 0 */
  this->m_HasNonZeroJacobianOfSpatialHessian = false;
  this->m_JacobianOfSpatialHessian.resize(paramDims);
  for (par = 0; par < paramDims; ++par)
  {
    for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
    {
      // SK: \todo: how can outputDims ever be different from OutputSpaceDimension?
      this->m_JacobianOfSpatialHessian[par][d].Fill(0.0);
    }
  }

  /** m_SpatialHessian is initialized with zeros */
  this->m_HasNonZeroSpatialHessian = false;
  for (unsigned int d = 0; d < OutputSpaceDimension; ++d)
  {
    // SK: \todo: how can outputDims ever be different from OutputSpaceDimension?
    this->m_SpatialHessian[d].Fill(0.0);
  }

  /** m_SpatialJacobian simply equals m_Matrix */

} // end PrecomputeJacobians


// Print self
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::PrintSelf(std::ostream & os,
                                                                                               Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << m_ItkTransform << std::endl;
}


// Computes matrix - base class does nothing.  In derived classes is
//    used to convert, for example, versor into a matrix
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::ComputeMatrix(void)
{
  // Since parameters explicitly define the matrix in this base class, this
  // function does nothing.  Normally used to compute a matrix when
  // its parameterization (e.g., the class' versor) is modified.
}


// Computes parameters - base class does nothing.  In derived classes is
//    used to convert, for example, matrix into a versor
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::ComputeMatrixParameters(void)
{
  // Since parameters explicitly define the matrix in this base class, this
  // function does nothing.  Normally used to update the parameterization
  // of the matrix (e.g., the class' versor) when the matrix is explicitly
  // set.
}


/**
 * ********************* GetJacobian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobian(
  const InputPointType &       p,
  JacobianType &               j,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  // The Jacobian of the affine transform is composed of
  // subblocks of diagonal matrices, each one of them having
  // a constant value in the diagonal.

  // Initialize the Jacobian. Resizing is only performed when needed.
  // Filling with zeros is needed because the lower loops only visit
  // the nonzero positions.
  j.SetSize(OutputSpaceDimension, ParametersDimension);
  j.Fill(0.0);

  const InputVectorType v = p - this->GetCenter();

  unsigned int blockOffset = 0;
  for (unsigned int block = 0; block < NInputDimensions; ++block)
  {
    for (unsigned int dim = 0; dim < NOutputDimensions; ++dim)
    {
      j(block, blockOffset + dim) = v[dim];
    }
    blockOffset += NInputDimensions;
  }

  for (unsigned int dim = 0; dim < NOutputDimensions; ++dim)
  {
    j(dim, blockOffset + dim) = 1.0;
  }

  // Copy the constant nonZeroJacobianIndices
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;

} // end GetJacobian()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetSpatialJacobian(
  const InputPointType &,
  SpatialJacobianType & sj) const
{
  sj = this->GetMatrix();

} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetSpatialHessian(
  const InputPointType &,
  SpatialHessianType & sh) const
{
  /** The SpatialHessian contains only zeros. */
  sh = this->m_SpatialHessian;

} // end GetSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobianOfSpatialJacobian(
  const InputPointType &,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** The Jacobian of spatial Jacobian remains constant, so was precomputed */
  jsj = this->m_JacobianOfSpatialJacobian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobianOfSpatialJacobian(
  const InputPointType &,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** The Jacobian of spatial Jacobian remains constant, so was precomputed */
  sj = this->GetMatrix();
  jsj = this->m_JacobianOfSpatialJacobian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobianOfSpatialHessian(
  const InputPointType &,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  /** The JacobianOfSpatialHessian contains only zeros.*/
  jsh = this->m_JacobianOfSpatialHessian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobianOfSpatialHessian(
  const InputPointType &,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  /** The Hessian and the JacobianOfSpatialHessian contain only zeros. */
  sh = this->m_SpatialHessian;
  jsh = this->m_JacobianOfSpatialHessian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;

} // end GetJacobianOfSpatialHessian()


} // namespace itk

#endif

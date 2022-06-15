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
  Module:    $RCSfile: itkAdvancedTransform.txx,v $
  Language:  C++
  Date:      $Date: 2007-11-20 20:08:16 $
  Version:   $Revision: 1.27 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkAdvancedTransform_hxx
#define _itkAdvancedTransform_hxx

#include "itkAdvancedTransform.h"

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>::AdvancedTransform()
  : Superclass()
{
  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;

} // end Constructor


/**
 * ********************* Constructor ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>::AdvancedTransform(
  NumberOfParametersType numberOfParameters)
  : Superclass(numberOfParameters)
{
  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;
} // end Constructor


/**
 * ********************* EvaluateJacobianWithImageGradientProduct ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>::EvaluateJacobianWithImageGradientProduct(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** Obtain the Jacobian. Using thread-local storage, so that both the allocation and the deallocation of the internal
   * data occurs only once per thread, as it has appeared as a major performance bottleneck. */
  thread_local JacobianType jacobian;
  this->GetJacobian(inputPoint, jacobian, nonZeroJacobianIndices);

  /** Perform a full multiplication. */
  using JacobianIteratorType = typename JacobianType::const_iterator;
  JacobianIteratorType jac = jacobian.begin();
  imageJacobian.Fill(0.0);

  for (unsigned int dim = 0; dim < InputSpaceDimension; ++dim)
  {
    const double imDeriv = movingImageGradient[dim];

    for (auto & imageJacobianElement : imageJacobian)
    {
      imageJacobianElement += (*jac) * imDeriv;
      ++jac;
    }
  }

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetNumberOfNonZeroJacobianIndices() const
  -> NumberOfParametersType
{
  return this->GetNumberOfParameters();

} // end GetNumberOfNonZeroJacobianIndices()


} // end namespace itk

#endif

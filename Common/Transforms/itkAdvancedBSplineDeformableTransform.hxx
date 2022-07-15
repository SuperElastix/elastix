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

/** version of original itk file on which code is based: */
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkBSplineDeformableTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-05-08 23:22:35 $
  Version:   $Revision: 1.41 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedBSplineDeformableTransform_hxx
#define itkAdvancedBSplineDeformableTransform_hxx

#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkContinuousIndex.h"
#include "itkImageScanlineConstIterator.h"
#include "itkIdentityTransform.h"
#include <vnl/vnl_math.h>
#include <array>
#include <vector>
#include <algorithm> // For std::copy_n.

namespace itk
{

// Constructor with default arguments
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::AdvancedBSplineDeformableTransform()
  : Superclass(VSplineOrder)
{
  // Instantiate weights functions
  this->m_WeightsFunction = WeightsFunctionType::New();
  this->m_DerivativeWeightsFunctions.resize(SpaceDimension);
  this->m_SODerivativeWeightsFunctions.resize(SpaceDimension);
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    this->m_DerivativeWeightsFunctions[i] = DerivativeWeightsFunctionType::New();
    this->m_DerivativeWeightsFunctions[i]->SetDerivativeDirection(i);
    this->m_SODerivativeWeightsFunctions[i].resize(SpaceDimension);
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_SODerivativeWeightsFunctions[i][j] = SODerivativeWeightsFunctionType::New();
      this->m_SODerivativeWeightsFunctions[i][j]->SetDerivativeDirections(i, j);
    }
  }
  this->m_SupportSize = this->m_WeightsFunction->GetSupportSize();

  this->m_InternalParametersBuffer = ParametersType(0);
  // Make sure the parameters pointer is not NULL after construction.
  this->m_InputParametersPointer = &(this->m_InternalParametersBuffer);

  // Initialize coefficient images
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    this->m_WrappedImage[j] = ImageType::New();
    this->m_WrappedImage[j]->SetRegions(this->m_GridRegion);
    this->m_WrappedImage[j]->SetOrigin(this->m_GridOrigin.GetDataPointer());
    this->m_WrappedImage[j]->SetSpacing(this->m_GridSpacing.GetDataPointer());
    this->m_WrappedImage[j]->SetDirection(this->m_GridDirection);
    this->m_CoefficientImages[j] = nullptr;
  }

  // Setup variables for computing interpolation
  this->m_Offset = SplineOrder / 2;
  this->m_ValidRegion = this->m_GridRegion;

  /** Fixed Parameters store the following information:
   *     Grid Size
   *     Grid Origin
   *     Grid Spacing
   *     Grid Direction
   *  The size of these is equal to the  NInputDimensions
   */
  this->m_FixedParameters.SetSize(Superclass::NumberOfFixedParameters);
  this->m_FixedParameters.Fill(0.0);
  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    this->m_FixedParameters[2 * NDimensions + i] = this->m_GridSpacing[i];
  }
  for (unsigned int di = 0; di < NDimensions; ++di)
  {
    for (unsigned int dj = 0; dj < NDimensions; ++dj)
    {
      this->m_FixedParameters[3 * NDimensions + (di * NDimensions + dj)] = this->m_GridDirection[di][dj];
    }
  }

  this->m_LastJacobianIndex = this->m_ValidRegion.GetIndex();

  // needed, there seems to be double functionality compared to base constructor
  this->UpdatePointIndexConversions();

  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;

} // end Constructor


// Set the grid region
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::SetGridRegion(const RegionType & region)
{
  if (this->m_GridRegion != region)
  {

    this->m_GridRegion = region;

    // set regions for each coefficient and Jacobian image
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_WrappedImage[j]->SetRegions(this->m_GridRegion);
    }

    // Set the valid region
    // If the grid spans the interval [start, last].
    // The valid interval for evaluation is [start+offset, last-offset]
    // when spline order is even.
    // The valid interval for evaluation is [start+offset, last-offset)
    // when spline order is odd.
    // Where offset = std::floor(spline / 2 ).
    // Note that the last pixel is not included in the valid region
    // with odd spline orders.
    // For backward compatibility m_ValidRegion is still created.
    typename RegionType::SizeType  size = this->m_GridRegion.GetSize();
    typename RegionType::IndexType index = this->m_GridRegion.GetIndex();
    using CValueType = typename ContinuousIndexType::ValueType;
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_ValidRegionBegin[j] =
        static_cast<CValueType>(index[j]) + (static_cast<CValueType>(SplineOrder) - 1.0) / 2.0;
      this->m_ValidRegionEnd[j] = static_cast<CValueType>(index[j]) + static_cast<CValueType>(size[j] - 1) -
                                  (static_cast<CValueType>(SplineOrder) - 1.0) / 2.0;
      index[j] += static_cast<typename RegionType::IndexValueType>(this->m_Offset);
      size[j] -= static_cast<typename RegionType::SizeValueType>(2 * this->m_Offset);
    }
    this->m_ValidRegion.SetSize(size);
    this->m_ValidRegion.SetIndex(index);

    this->UpdateGridOffsetTable();

    //
    // If we are using the default parameters, update their size and set to identity.
    //

    // Input parameters point to internal buffer => using default parameters.
    if (this->m_InputParametersPointer == &(this->m_InternalParametersBuffer))
    {
      // Check if we need to resize the default parameter buffer.
      if (this->m_InternalParametersBuffer.GetSize() != this->GetNumberOfParameters())
      {
        this->m_InternalParametersBuffer.SetSize(this->GetNumberOfParameters());
        // Fill with zeros for identity.
        this->m_InternalParametersBuffer.Fill(0);
      }
    }

    this->Modified();
  }
}


// Transform a point
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::TransformPoint(
  const InputPointType & point) const -> OutputPointType
{
  /** Allocate memory on the stack: */
  const unsigned long                         numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename ParameterIndexArrayType::ValueType indicesArray[numberOfWeights];
  WeightsType                                 weights;
  ParameterIndexArrayType                     indices(indicesArray, numberOfWeights, false);

  OutputPointType outputPoint;

  InputPointType transformedPoint = point;

  /** Check if the coefficient image has been set. */
  if (!this->m_CoefficientImages[0])
  {
    itkWarningMacro(<< "B-spline coefficients have not been set");
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      outputPoint[j] = transformedPoint[j];
    }
    return outputPoint;
  }

  /***/
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(point);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  if (!this->InsideValidRegion(cindex))
  {
    outputPoint = transformedPoint;
    return outputPoint;
  }

  // Compute interpolation weights
  IndexType supportIndex;
  this->m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  this->m_WeightsFunction->Evaluate(cindex, supportIndex, weights);

  // For each dimension, correlate coefficient with weights
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  outputPoint.Fill(NumericTraits<ScalarType>::ZeroValue());

  /** Create iterators over the coefficient images. */
  using IteratorType = ImageScanlineConstIterator<ImageType>;
  IteratorType      iterators[SpaceDimension];
  unsigned long     counter = 0;
  const PixelType * basePointer = this->m_CoefficientImages[0]->GetBufferPointer();

  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    iterators[j] = IteratorType(this->m_CoefficientImages[j], supportRegion);
  }

  /** Loop over the support region. */
  while (!iterators[0].IsAtEnd())
  {
    while (!iterators[0].IsAtEndOfLine())
    {
      // populate the indices array
      indices[counter] = &(iterators[0].Value()) - basePointer;

      // multiply weight with coefficient to compute displacement
      for (unsigned int j = 0; j < SpaceDimension; ++j)
      {
        outputPoint[j] += static_cast<ScalarType>(weights[counter] * iterators[j].Value());
        ++iterators[j];
      }
      ++counter;
    } // end of scanline

    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      iterators[j].NextLine();
    }

  } // end while

  // The output point is the start point + displacement.
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    outputPoint[j] += transformedPoint[j];
  }

  return outputPoint;
}


/**
 * ********************* GetNumberOfAffectedWeights ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
unsigned int
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetNumberOfAffectedWeights() const
{
  return this->m_WeightsFunction->GetNumberOfWeights();
} // end GetNumberOfAffectedWeights()


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetNumberOfNonZeroJacobianIndices() const
  -> NumberOfParametersType
{
  return this->m_WeightsFunction->GetNumberOfWeights() * SpaceDimension;
} // end GetNumberOfNonZeroJacobianIndices()


/**
 * ********************* GetJacobian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobian(
  const InputPointType &       inputPoint,
  JacobianType &               jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  /** This implements a sparse version of the Jacobian. */

  /** Sanity check. */
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  /** Initialize. */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if ((jacobian.cols() != nnzji) || (jacobian.rows() != SpaceDimension))
  {
    jacobian.SetSize(SpaceDimension, nnzji);
    jacobian.Fill(0.0);
  }

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  if (!this->InsideValidRegion(cindex))
  {
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Compute the number of affected B-spline parameters.
   * Allocate memory on the stack.
   */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  /** Compute the weights. */
  IndexType supportIndex;
  this->m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  this->m_WeightsFunction->Evaluate(cindex, supportIndex, weights);

  /** Setup support region */
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Put at the right positions. */
  ParametersValueType * jacobianPointer = jacobian.data_block();
  for (unsigned int d = 0; d < SpaceDimension; ++d)
  {
    unsigned long offset = d * SpaceDimension * numberOfWeights + d * numberOfWeights;
    std::copy_n(weights.begin(), numberOfWeights, jacobianPointer + offset);
  }

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobian()


/**
 * ********************* EvaluateJacobianAndImageGradientProduct ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::EvaluateJacobianWithImageGradientProduct(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  /** Get sizes. */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();
  const NumberOfParametersType nnzjiPerDimension = nnzji / SpaceDimension;

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  if (!this->InsideValidRegion(cindex))
  {
    nonZeroJacobianIndices.resize(nnzji);
    for (NumberOfParametersType i = 0; i < nnzji; ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    imageJacobian.Fill(0.0);
    return;
  }

  /** Compute the number of affected B-spline parameters.
   * Allocate memory on the stack.
   */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  /** Compute the B-spline derivative weights. */
  IndexType supportIndex;
  this->m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  this->m_WeightsFunction->Evaluate(cindex, supportIndex, weights);

  /** Compute the inner product. */
  NumberOfParametersType counter = 0;
  for (unsigned int d = 0; d < SpaceDimension; ++d)
  {
    const MovingImageGradientValueType mig = movingImageGradient[d];
    for (NumberOfParametersType i = 0; i < nnzjiPerDimension; ++i)
    {
      imageJacobian[counter] = weights[i] * mig;
      ++counter;
    }
  }

  /** Setup support region needed for the nonZeroJacobianIndices. */
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetSpatialJacobian(
  const InputPointType & inputPoint,
  SpatialJacobianType &  sj) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity spatial Jacobian
  if (!this->InsideValidRegion(cindex))
  {
    sj.SetIdentity();
    return;
  }

  /** Compute the number of affected B-spline parameters. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  /** Array for CoefficientImage values */
  std::array<typename WeightsType::ValueType, numberOfWeights * SpaceDimension> coeffs;

  IndexType supportIndex;
  this->m_DerivativeWeightsFunctions[0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Copy values from coefficient image to linear coeffs array. */
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(this->m_CoefficientImages[dim], supportRegion);

    while (!itCoef.IsAtEnd())
    {
      while (!itCoef.IsAtEndOfLine())
      {
        (*itCoeffsLinear) = itCoef.Value();
        ++itCoeffsLinear;
        ++itCoef;
      }
      itCoef.NextLine();
    }
  }

  /** Compute the spatial Jacobian sj:
   *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights * PointToGridIndex.
   */
  sj.Fill(0.0);
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Compute the derivative weights. */
    this->m_DerivativeWeightsFunctions[i]->Evaluate(cindex, supportIndex, weights);

    /** Create an iterator over the coeffs vector.  */
    auto itCoeffs = coeffs.cbegin();

    /** Compute the spatial Jacobian sj:
     *    dT_{dim} / dx_i = \sum coefs_{dim} * weights.
     */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      /** Create an iterator over the correct part of the coefficient
       * image. Create an iterator over the weights vector.
       */
      typename WeightsType::const_iterator itWeights = weights.begin();

      /** Compute the sum for this dimension. */
      for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
      {
        sj(dim, i) += (*itCoeffs) * (*itWeights);
        ++itWeights;
        ++itCoeffs;
      } // end for mu
    }   // end for dim
  }     // end for i

  /** Take into account grid spacing and direction cosines. */
  sj = sj * this->m_PointToIndexMatrix2;

  /** Add contribution of spatial derivative of x. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sj(dim, dim) += 1.0;
  }

} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetSpatialHessian(
  const InputPointType & inputPoint,
  SpatialHessianType &   sh) const
{
  using WeightsValueType = typename WeightsType::ValueType;

  /** Convert the physical point to a continuous index, which
   * is needed for the evaluate functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero spatial Hessian
  if (!this->InsideValidRegion(cindex))
  {
    for (unsigned int i = 0; i < sh.Size(); ++i)
    {
      sh[i].Fill(0.0);
    }
    return;
  }

  /** Helper variables. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  /** Array for CoefficientImage values */
  std::array<WeightsValueType, numberOfWeights * SpaceDimension> coeffs;

  IndexType supportIndex;
  this->m_SODerivativeWeightsFunctions[0][0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Copy values from coefficient image to linear coeffs array. */
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(this->m_CoefficientImages[dim], supportRegion);

    // for( unsigned int mu = 0; mu < numberOfWeights; ++mu )
    while (!itCoef.IsAtEnd())
    {
      while (!itCoef.IsAtEndOfLine())
      {
        (*itCoeffsLinear) = itCoef.Value();
        ++itCoeffsLinear;
        ++itCoef;
      }
      itCoef.NextLine();
    }
  }

  /** For all derivative directions, compute the spatial Hessian.
   * The derivatives are d^2T / dx_i dx_j.
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j <= i; ++j)
    {
      /** Compute the derivative weights. */
      this->m_SODerivativeWeightsFunctions[i][j]->Evaluate(cindex, supportIndex, weights);

      /** Create an iterator over the coeffs vector.  */
      auto itCoeffs = coeffs.cbegin();

      /** Compute d^2T_{dim} / dx_i dx_j = \sum coefs_{dim} * weights. */
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        /** Create an iterator over the weights vector.  */
        typename WeightsType::const_iterator itWeights = weights.begin();

        /** Compute the sum for this dimension. */
        double sum = 0.0;

        for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
        {
          sum += (*itCoeffs) * (*itWeights);
          ++itWeights;
          ++itCoeffs;
        }

        /** Update the spatial Hessian sh. The Hessian is symmetrical. */
        sh[dim](i, j) = sum;
        if (j < i)
        {
          sh[dim](j, i) = sum;
        }
      }
    }
  }

  /** Take into account grid spacing and direction matrix */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sh[dim] = this->m_PointToIndexMatrixTransposed2 * (sh[dim] * this->m_PointToIndexMatrix2);
  }

} // end GetSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  jsj.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero jsj.
  if (!this->InsideValidRegion(cindex))
  {
    for (unsigned int i = 0; i < jsj.size(); ++i)
    {
      jsj[i].Fill(0.0);
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Helper variables. */

  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  IndexType supportIndex;
  this->m_DerivativeWeightsFunctions[0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** On the stack instead of heap is faster. */
  double weightVector[SpaceDimension * numberOfWeights];

  /** For all derivative directions, compute the derivatives of the
   * spatial Jacobian to the transformation parameters mu:
   * d/dmu of dT / dx_i
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Compute the derivative weights. */
    this->m_DerivativeWeightsFunctions[i]->Evaluate(cindex, supportIndex, weights);

    /** Remember the weights. */
    std::copy_n(weights.begin(), numberOfWeights, weightVector + i * numberOfWeights);

  } // end for i

  /** Compute the Jacobian of the spatial Jacobian jsj:
   *    d/dmu dT_{dim} / dx_i = weights.
   */
  SpatialJacobianType * basepointer = &jsj[0];
  for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
  {
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      double tmp = *(weightVector + i * numberOfWeights + mu);
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        (*(basepointer + dim * numberOfWeights + mu))(dim, i) = tmp;
      }
    }
  }

  /** Take into account grid spacing and direction cosines */
  for (unsigned int i = 0; i < jsj.size(); ++i)
  {
    jsj[i] = jsj[i] * this->m_PointToIndexMatrix2;
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  jsj.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity sj and zero jsj.
  if (!this->InsideValidRegion(cindex))
  {
    sj.SetIdentity();
    for (unsigned int i = 0; i < jsj.size(); ++i)
    {
      jsj[i].Fill(0.0);
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Helper variables. */
  IndexType supportIndex;
  this->m_DerivativeWeightsFunctions[0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Allocate weight on the stack. */
  using WeightsValueType = typename WeightsType::ValueType;
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  /** Allocate coefficients on the stack. */
  std::array<WeightsValueType, numberOfWeights * SpaceDimension> coeffs;

  /** Copy values from coefficient image to linear coeffs array. */
  // takes considerable amount of time : 27% of this function. // with old region iterator, check with new
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(this->m_CoefficientImages[dim], supportRegion);

    while (!itCoef.IsAtEnd())
    {
      while (!itCoef.IsAtEndOfLine())
      {
        (*itCoeffsLinear) = itCoef.Value();
        ++itCoeffsLinear;
        ++itCoef;
      }
      itCoef.NextLine();
    }
  }

  /** On the stack instead of heap is faster. */
  const unsigned int d = SpaceDimension * (SpaceDimension + 1) / 2;
  double             weightVector[d * numberOfWeights];

  /** Initialize the spatial Jacobian sj: */
  sj.Fill(0.0);

  /** For all derivative directions, compute the derivatives of the
   * spatial Jacobian to the transformation parameters mu: d/dmu of dT / dx_i
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Compute the derivative weights. */
    this->m_DerivativeWeightsFunctions[i]->Evaluate(cindex, supportIndex, weights);
    /** \todo: we can realise some speedup here to compute the derivative
     * weights at once for all dimensions */

    /** Remember the weights. */
    std::copy_n(weights.begin(), numberOfWeights, weightVector + i * numberOfWeights);

    /** Reset coeffs iterator */
    auto itCoeffs = coeffs.cbegin();

    /** Compute the spatial Jacobian sj:
     *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights.
     */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      /** Reset weights iterator. */
      typename WeightsType::const_iterator itWeights = weights.begin();

      /** Compute the sum for this dimension. */
      for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
      {
        sj(dim, i) += (*itCoeffs) * (*itWeights);
        ++itWeights;
        ++itCoeffs;
      }

    } // end for dim
  }   // end for i

  /** Take into account grid spacing and direction cosines. */
  sj = sj * this->m_PointToIndexMatrix2;

  /** Add contribution of spatial derivative of x. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sj(dim, dim) += 1.0;
  }

  /** Compute the Jacobian of the spatial Jacobian jsj:
   *    d/dmu dT_{dim} / dx_i = weights.
   */
  SpatialJacobianType * basepointer = &jsj[0];
  for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
  {
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      const double tmp = *(weightVector + i * numberOfWeights + mu);
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        (*(basepointer + dim * numberOfWeights + mu))(dim, i) = tmp;
      }
    }
  }

  /** Take into account grid spacing and direction cosines */
  for (unsigned int i = 0; i < jsj.size(); ++i)
  {
    jsj[i] = jsj[i] * this->m_PointToIndexMatrix2;
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  jsh.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity sj and zero jsj.
  if (!this->InsideValidRegion(cindex))
  {
    for (unsigned int i = 0; i < jsh.size(); ++i)
    {
      for (unsigned int j = 0; j < jsh[i].Size(); ++j)
      {
        jsh[i][j].Fill(0.0);
      }
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Compute the number of affected B-spline parameters. */

  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  IndexType supportIndex;
  this->m_SODerivativeWeightsFunctions[0][0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** For all derivative directions, compute the derivatives of the
   * spatial Hessian to the transformation parameters mu:
   * d/dmu of d^2T / dx_i dx_j
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  const unsigned int         d = SpaceDimension * (SpaceDimension + 1) / 2;
  FixedArray<WeightsType, d> weightVector;
  unsigned int               count = 0;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j <= i; ++j)
    {
      /** Compute the derivative weights. */
      this->m_SODerivativeWeightsFunctions[i][j]->Evaluate(cindex, supportIndex, weights);

      /** Remember the weights. */
      weightVector[count] = weights;
      ++count;

    } // end for j
  }   // end for i

  /** Compute d/dmu d^2T_{dim} / dx_i dx_j = weights. */
  for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
  {
    SpatialJacobianType matrix;
    unsigned int        count = 0;
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      for (unsigned int j = 0; j <= i; ++j)
      {
        double tmp = weightVector[count][mu];
        matrix[i][j] = tmp;
        if (i != j)
        {
          matrix[j][i] = tmp;
        }
        ++count;
      }
    }

    /** Take into account grid spacing and direction matrix. */
    matrix = this->m_PointToIndexMatrixTransposed2 * (matrix * this->m_PointToIndexMatrix2);

    /** Copy the matrix to the right locations. */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      jsh[mu + dim * numberOfWeights][dim] = matrix;
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  using WeightsValueType = typename WeightsType::ValueType;

  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  jsh.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity sj and zero jsj.
  if (!this->InsideValidRegion(cindex))
  {
    for (unsigned int i = 0; i < jsh.size(); ++i)
    {
      for (unsigned int j = 0; j < jsh[i].Size(); ++j)
      {
        jsh[i][j].Fill(0.0);
      }
    }
    for (unsigned int i = 0; i < sh.Size(); ++i)
    {
      sh[i].Fill(0.0);
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Get the support region. */
  IndexType supportIndex;
  this->m_SODerivativeWeightsFunctions[0][0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Allocate weight on the stack. */
  using WeightsValueType = typename WeightsType::ValueType;
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  WeightsType         weights;

  /** Allocate coefficients on the stack. */
  std::array<WeightsValueType, numberOfWeights * SpaceDimension> coeffs;

  /** Copy values from coefficient image to linear coeffs array. */
  // takes considerable amount of time : 27% of this function. // with old region iterator, check with new
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(this->m_CoefficientImages[dim], supportRegion);

    while (!itCoef.IsAtEnd())
    {
      while (!itCoef.IsAtEndOfLine())
      {
        (*itCoeffsLinear) = itCoef.Value();
        ++itCoeffsLinear;
        ++itCoef;
      }
      itCoef.NextLine();
    }
  }

  /** On the stack instead of heap is faster. */
  const unsigned int d = SpaceDimension * (SpaceDimension + 1) / 2;
  double             weightVector[d * numberOfWeights];

  /** For all derivative directions, compute the derivatives of the
   * spatial Hessian to the transformation parameters mu:
   * d/dmu of d^2T / dx_i dx_j
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  unsigned int count = 0;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j <= i; ++j)
    {
      /** Compute the derivative weights. */
      this->m_SODerivativeWeightsFunctions[i][j]->Evaluate(cindex, supportIndex, weights);

      /** Remember the weights. */
      std::copy_n(weights.begin(), numberOfWeights, weightVector + count * numberOfWeights);
      ++count;

      /** Reset coeffs iterator */
      auto itCoeffs = coeffs.cbegin();

      /** Compute the spatial Hessian sh:
       *    d^2T_{dim} / dx_i dx_j = \sum coefs_{dim} * weights.
       */
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        /** Reset weights iterator. */
        typename WeightsType::const_iterator itWeights = weights.begin();

        /** Compute the sum for this dimension. */
        double sum = 0.0;
        for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
        {
          sum += (*itCoeffs) * (*itWeights);
          ++itWeights;
          ++itCoeffs;
        }

        /** Update the spatial Hessian sh. The Hessian is symmetrical. */
        sh[dim](i, j) = sum;
        if (j < i)
        {
          sh[dim](j, i) = sum;
        }
      }

    } // end for j
  }   // end for i

  /** Take into account grid spacing and direction matrix. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sh[dim] = this->m_PointToIndexMatrixTransposed2 * (sh[dim] * this->m_PointToIndexMatrix2);
  }

  /** Compute the Jacobian of the spatial Hessian jsh:
   *    d/dmu d^2T_{dim} / dx_i dx_j = weights.
   */
  SpatialJacobianType matrix;
  for (unsigned int mu = 0; mu < numberOfWeights; ++mu)
  {
    unsigned int count = 0;
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      for (unsigned int j = 0; j <= i; ++j)
      {
        const double tmp = *(weightVector + count * numberOfWeights + mu);
        matrix[i][j] = tmp;
        if (i != j)
        {
          matrix[j][i] = tmp;
        }
        ++count;
      }
    }

    /** Take into account grid spacing and direction matrix. */
    if (!this->m_PointToIndexMatrixIsDiagonal)
    {
      matrix = this->m_PointToIndexMatrixTransposed2 * (matrix * this->m_PointToIndexMatrix2);
    }
    else
    {
      for (unsigned int i = 0; i < SpaceDimension; ++i)
      {
        for (unsigned int j = 0; j < SpaceDimension; ++j)
        {
          matrix[i][j] *= this->m_PointToIndexMatrixDiagonalProducts[i + SpaceDimension * j];
        }
      }
    }

    /** Copy the matrix to the right locations. */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      jsh[mu + dim * numberOfWeights][dim] = matrix;
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* ComputeNonZeroJacobianIndices ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType &           supportRegion) const
{
  /** Initialize some helper variables. */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  const unsigned long parametersPerDim = this->GetNumberOfParametersPerDimension();

  nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Compute the first global parameter number. */
  unsigned long globalStartNum = 0;
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    globalStartNum += supportRegion.GetIndex()[dim] * this->m_GridOffsetTable[dim];
  }

  if (SpaceDimension == 2)
  {
    /** Initialize some helper variables. */
    const unsigned int  sx = supportRegion.GetSize()[0];
    const unsigned int  sy = supportRegion.GetSize()[1];
    const unsigned long goy = this->m_GridOffsetTable[1];
    const unsigned long diffxy = goy - sx;

    /** Loop over the support region and compute the nzji. */
    unsigned int  localParNum = 0;
    unsigned long globalParNum = globalStartNum;
    for (unsigned int y = 0; y < sy; ++y)
    {
      for (unsigned int x = 0; x < sx; ++x)
      {
        nonZeroJacobianIndices[localParNum] = globalParNum;
        nonZeroJacobianIndices[localParNum + numberOfWeights] = globalParNum + parametersPerDim;
        ++localParNum;
        ++globalParNum;
      }
      globalParNum += diffxy;
    }
  } // end if SpaceDimension == 2
  else if (SpaceDimension == 3)
  {
    /** Initialize some helper variables. */
    const unsigned int  sx = supportRegion.GetSize()[0];
    const unsigned int  sy = supportRegion.GetSize()[1];
    const unsigned int  sz = supportRegion.GetSize()[2];
    const unsigned long goy = this->m_GridOffsetTable[1];
    const unsigned long goz = this->m_GridOffsetTable[2];
    const unsigned long diffxy = goy - sx;
    const unsigned long diffyz = goz - sy * goy;

    /** Loop over the support region and compute the nzji. */
    unsigned int  localParNum = 0;
    unsigned long globalParNum = globalStartNum;
    for (unsigned int z = 0; z < sz; ++z)
    {
      for (unsigned int y = 0; y < sy; ++y)
      {
        for (unsigned int x = 0; x < sx; ++x)
        {
          nonZeroJacobianIndices[localParNum] = globalParNum;
          nonZeroJacobianIndices[localParNum + numberOfWeights] = globalParNum + parametersPerDim;
          nonZeroJacobianIndices[localParNum + 2 * numberOfWeights] = globalParNum + 2 * parametersPerDim;
          ++localParNum;
          ++globalParNum;
        }
        globalParNum += diffxy;
      }
      globalParNum += diffyz;
    }
  } // end if SpaceDimension == 3
  else
  {
    GridOffsetType supportRegionOffset;
    supportRegionOffset[0] = 1;
    for (unsigned int dim = 1; dim < SpaceDimension; ++dim)
    {
      supportRegionOffset[dim] = supportRegionOffset[dim - 1] * supportRegion.GetSize()[dim - 1];
    }

    /** Loop over the support region and compute the nzji. */
    for (unsigned int localParNum = 0; localParNum < numberOfWeights; ++localParNum)
    // Note that numberOfWeights == supportRegion.GetNumberOfPixels()
    {
      // translate localParNum to a local index
      GridOffsetType localParIndex;
      unsigned int   remainder = localParNum;
      for (int dim = SpaceDimension - 1; dim >= 0; --dim)
      {
        localParIndex[dim] = remainder / supportRegionOffset[dim];
        remainder = remainder % supportRegionOffset[dim];
      }

      // translate local index to global index
      GridOffsetType globalParIndex;
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        globalParIndex[dim] = localParIndex[dim] + supportRegion.GetIndex()[dim];
      }

      // translate global index to global parameter number
      unsigned int globalParNum = 0;
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        globalParNum += globalParIndex[dim] * this->m_GridOffsetTable[dim];
      }

      /** Update the nonZeroJacobianIndices for all directions. */
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        nonZeroJacobianIndices[localParNum + dim * numberOfWeights] = globalParNum + dim * parametersPerDim;
      }
    } // end for
  }   // end general case

} // end ComputeNonZeroJacobianIndices()


/**
 * ********************* PrintSelf ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::PrintSelf(std::ostream & os,
                                                                                      Indent         indent) const
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "WeightsFunction: ";
  os << this->m_WeightsFunction.GetPointer() << std::endl;
}


} // end namespace itk

#endif

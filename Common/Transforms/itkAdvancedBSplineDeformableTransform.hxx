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
#include <numeric> // For iota.
#include <vector>
#include <algorithm> // For std::copy_n.

namespace itk
{

// Constructor with default arguments
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::AdvancedBSplineDeformableTransform()
  : Superclass(VSplineOrder)
{
  // Instantiate weights functions
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    m_DerivativeWeightsFunctions[i] = DerivativeWeightsFunctionType::New();
    m_DerivativeWeightsFunctions[i]->SetDerivativeDirection(i);
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      m_SODerivativeWeightsFunctions[i][j] = SODerivativeWeightsFunctionType::New();
      m_SODerivativeWeightsFunctions[i][j]->SetDerivativeDirections(i, j);
    }
  }

  // Setup variables for computing interpolation
  Superclass::m_HasNonZeroSpatialHessian = true;
  Superclass::m_HasNonZeroJacobianOfSpatialHessian = true;

} // end Constructor


// Set the grid region
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::SetGridRegion(const RegionType & region)
{
  if (Superclass::m_GridRegion != region)
  {

    Superclass::m_GridRegion = region;

    // set regions for each coefficient and Jacobian image
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      Superclass::m_WrappedImage[j]->SetRegions(Superclass::m_GridRegion);
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
    typename RegionType::SizeType  size = Superclass::m_GridRegion.GetSize();
    typename RegionType::IndexType index = Superclass::m_GridRegion.GetIndex();
    using CValueType = typename ContinuousIndexType::ValueType;
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      static constexpr unsigned int offset{ VSplineOrder / 2 };

      Superclass::m_ValidRegionBegin[j] = static_cast<CValueType>(index[j]) + (CValueType{ VSplineOrder } - 1.0) / 2.0;
      Superclass::m_ValidRegionEnd[j] = static_cast<CValueType>(index[j]) + static_cast<CValueType>(size[j] - 1) -
                                        (CValueType{ VSplineOrder } - 1) / 2.0;
      index[j] += IndexValueType{ offset };
      size[j] -= SizeValueType{ 2 * offset };
    }

    this->UpdateGridOffsetTable();

    //
    // If we are using the default parameters, update their size and set to identity.
    //

    // Input parameters point to internal buffer => using default parameters.
    if (Superclass::m_InputParametersPointer == &(Superclass::m_InternalParametersBuffer))
    {
      // Check if we need to resize the default parameter buffer.
      if (Superclass::m_InternalParametersBuffer.GetSize() != this->GetNumberOfParameters())
      {
        Superclass::m_InternalParametersBuffer.SetSize(this->GetNumberOfParameters());
        // Fill with zeros for identity.
        Superclass::m_InternalParametersBuffer.Fill(0);
      }
    }

    this->Modified();
  }
}


// Transform a point
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::TransformPoint(
  const InputPointType & point) const -> OutputPointType
{
  /** Check if the coefficient image has been set. */
  if (!Superclass::m_CoefficientImages[0])
  {
    itkWarningMacro("B-spline coefficients have not been set");
    return point;
  }

  /***/
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(point);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  if (!this->InsideValidRegion(cindex))
  {
    return point;
  }

  // Compute interpolation weights
  IndexType supportIndex;
  m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  const WeightsType weights = m_WeightsFunction->Evaluate(cindex, supportIndex);

  // For each dimension, correlate coefficient with weights
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  OutputPointType outputPoint{};

  /** Create iterators over the coefficient images. */
  using IteratorType = ImageScanlineConstIterator<ImageType>;
  IteratorType  iterators[SpaceDimension];
  unsigned long counter = 0;

  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    iterators[j] = IteratorType(Superclass::m_CoefficientImages[j], supportRegion);
  }

  /** Loop over the support region. */
  while (!iterators[0].IsAtEnd())
  {
    while (!iterators[0].IsAtEndOfLine())
    {
      // multiply weight with coefficient to compute displacement
      for (unsigned int j = 0; j < SpaceDimension; ++j)
      {
        outputPoint[j] += static_cast<ScalarType>(weights[counter] * iterators[j].Value());
        ++iterators[j];
      }
      ++counter;
    } // end of scanline

    for (auto & iterator : iterators)
    {
      iterator.NextLine();
    }

  } // end while

  // The output point is the start point + displacement.
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    outputPoint[j] += point[j];
  }

  return outputPoint;
}


/**
 * ********************* GetNumberOfAffectedWeights ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
unsigned int
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetNumberOfAffectedWeights() const
{
  return NumberOfWeights;
} // end GetNumberOfAffectedWeights()


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetNumberOfNonZeroJacobianIndices() const
  -> NumberOfParametersType
{
  return NumberOfWeights * SpaceDimension;
} // end GetNumberOfNonZeroJacobianIndices()


/**
 * ********************* GetJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobian(
  const InputPointType &       inputPoint,
  JacobianType &               jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  /** This implements a sparse version of the Jacobian. */

  /** Sanity check. */
  if (Superclass::m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro("Cannot compute Jacobian: parameters not set");
  }

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  /** Initialize. */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if ((jacobian.cols() != nnzji) || (jacobian.rows() != SpaceDimension))
  {
    jacobian.set_size(SpaceDimension, nnzji);
    jacobian.fill(0.0);
  }

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  if (!this->InsideValidRegion(cindex))
  {
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    std::iota(nonZeroJacobianIndices.begin(), nonZeroJacobianIndices.end(), 0u);
    return;
  }

  /** Compute the number of affected B-spline parameters.
   */

  /** Compute the weights. */
  IndexType supportIndex;
  m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  const WeightsType weights = m_WeightsFunction->Evaluate(cindex, supportIndex);

  /** Setup support region */
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** Put at the right positions. */
  ParametersValueType * jacobianPointer = jacobian.data_block();
  for (unsigned int d = 0; d < SpaceDimension; ++d)
  {
    unsigned long offset = d * SpaceDimension * NumberOfWeights + d * NumberOfWeights;
    std::copy_n(weights.cbegin(), NumberOfWeights, jacobianPointer + offset);
  }

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobian()


/**
 * ********************* EvaluateJacobianAndImageGradientProduct ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
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
    std::iota(nonZeroJacobianIndices.begin(), nonZeroJacobianIndices.end(), 0u);
    imageJacobian.fill(0.0);
    return;
  }

  /** Compute the number of affected B-spline parameters.
   */

  /** Compute the B-spline derivative weights. */
  IndexType supportIndex;
  m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  const WeightsType weights = m_WeightsFunction->Evaluate(cindex, supportIndex);

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
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
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

  /** Array for CoefficientImage values */
  std::array<typename WeightsType::ValueType, NumberOfWeights * SpaceDimension> coeffs;

  IndexType supportIndex;
  m_DerivativeWeightsFunctions[0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** Copy values from coefficient image to linear coeffs array. */
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(Superclass::m_CoefficientImages[dim], supportRegion);

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
    const WeightsType weights = m_DerivativeWeightsFunctions[i]->Evaluate(cindex, supportIndex);

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
      typename WeightsType::const_iterator itWeights = weights.cbegin();

      /** Compute the sum for this dimension. */
      for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
      {
        sj(dim, i) += (*itCoeffs) * (*itWeights);
        ++itWeights;
        ++itCoeffs;
      } // end for mu
    } // end for dim
  } // end for i

  /** Take into account grid spacing and direction cosines. */
  sj *= Superclass::m_PointToIndexMatrix2;

  /** Add contribution of spatial derivative of x. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sj(dim, dim) += 1.0;
  }

} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
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

  /** Array for CoefficientImage values */
  std::array<WeightsValueType, NumberOfWeights * SpaceDimension> coeffs;

  IndexType supportIndex;
  m_SODerivativeWeightsFunctions[0][0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** Copy values from coefficient image to linear coeffs array. */
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(Superclass::m_CoefficientImages[dim], supportRegion);

    // for( unsigned int mu = 0; mu < NumberOfWeights; ++mu )
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
      const WeightsType weights = m_SODerivativeWeightsFunctions[i][j]->Evaluate(cindex, supportIndex);

      /** Create an iterator over the coeffs vector.  */
      auto itCoeffs = coeffs.cbegin();

      /** Compute d^2T_{dim} / dx_i dx_j = \sum coefs_{dim} * weights. */
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        /** Create an iterator over the weights vector.  */
        typename WeightsType::const_iterator itWeights = weights.cbegin();

        /** Compute the sum for this dimension. */
        double sum = 0.0;

        for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
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
    sh[dim] = Superclass::m_PointToIndexMatrixTransposed2 * (sh[dim] * Superclass::m_PointToIndexMatrix2);
  }

} // end GetSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (Superclass::m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro("Cannot compute Jacobian: parameters not set");
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
    for (auto & matrix : jsj)
    {
      matrix.Fill(0.0);
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    std::iota(nonZeroJacobianIndices.begin(), nonZeroJacobianIndices.end(), 0u);
    return;
  }

  /** Helper variables. */

  IndexType supportIndex;
  m_DerivativeWeightsFunctions[0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** On the stack instead of heap is faster. */
  double weightVector[SpaceDimension * NumberOfWeights];

  /** For all derivative directions, compute the derivatives of the
   * spatial Jacobian to the transformation parameters mu:
   * d/dmu of dT / dx_i
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Compute the derivative weights. */
    const WeightsType weights = m_DerivativeWeightsFunctions[i]->Evaluate(cindex, supportIndex);

    /** Remember the weights. */
    std::copy_n(weights.cbegin(), NumberOfWeights, weightVector + i * NumberOfWeights);

  } // end for i

  /** Compute the Jacobian of the spatial Jacobian jsj:
   *    d/dmu dT_{dim} / dx_i = weights.
   */
  SpatialJacobianType * basepointer = &jsj[0];
  for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
  {
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      double tmp = *(weightVector + i * NumberOfWeights + mu);
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        (*(basepointer + dim * NumberOfWeights + mu))(dim, i) = tmp;
      }
    }
  }

  /** Take into account grid spacing and direction cosines */
  for (auto & matrix : jsj)
  {
    matrix *= Superclass::m_PointToIndexMatrix2;
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (Superclass::m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro("Cannot compute Jacobian: parameters not set");
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
    for (auto & matrix : jsj)
    {
      matrix.Fill(0.0);
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    std::iota(nonZeroJacobianIndices.begin(), nonZeroJacobianIndices.end(), 0u);
    return;
  }

  /** Helper variables. */
  IndexType supportIndex;
  m_DerivativeWeightsFunctions[0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  using WeightsValueType = typename WeightsType::ValueType;

  /** Allocate coefficients on the stack. */
  std::array<WeightsValueType, NumberOfWeights * SpaceDimension> coeffs;

  /** Copy values from coefficient image to linear coeffs array. */
  // takes considerable amount of time : 27% of this function. // with old region iterator, check with new
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(Superclass::m_CoefficientImages[dim], supportRegion);

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
  double             weightVector[d * NumberOfWeights];

  /** Initialize the spatial Jacobian sj: */
  sj.Fill(0.0);

  /** For all derivative directions, compute the derivatives of the
   * spatial Jacobian to the transformation parameters mu: d/dmu of dT / dx_i
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Compute the derivative weights. */
    const WeightsType weights = m_DerivativeWeightsFunctions[i]->Evaluate(cindex, supportIndex);
    /** \todo: we can realise some speedup here to compute the derivative
     * weights at once for all dimensions */

    /** Remember the weights. */
    std::copy_n(weights.cbegin(), NumberOfWeights, weightVector + i * NumberOfWeights);

    /** Reset coeffs iterator */
    auto itCoeffs = coeffs.cbegin();

    /** Compute the spatial Jacobian sj:
     *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights.
     */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      /** Reset weights iterator. */
      typename WeightsType::const_iterator itWeights = weights.cbegin();

      /** Compute the sum for this dimension. */
      for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
      {
        sj(dim, i) += (*itCoeffs) * (*itWeights);
        ++itWeights;
        ++itCoeffs;
      }

    } // end for dim
  } // end for i

  /** Take into account grid spacing and direction cosines. */
  sj *= Superclass::m_PointToIndexMatrix2;

  /** Add contribution of spatial derivative of x. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sj(dim, dim) += 1.0;
  }

  /** Compute the Jacobian of the spatial Jacobian jsj:
   *    d/dmu dT_{dim} / dx_i = weights.
   */
  SpatialJacobianType * basepointer = &jsj[0];
  for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
  {
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      const double tmp = *(weightVector + i * NumberOfWeights + mu);
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        (*(basepointer + dim * NumberOfWeights + mu))(dim, i) = tmp;
      }
    }
  }

  /** Take into account grid spacing and direction cosines */
  for (auto & matrix : jsj)
  {
    matrix *= Superclass::m_PointToIndexMatrix2;
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (Superclass::m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro("Cannot compute Jacobian: parameters not set");
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
    std::iota(nonZeroJacobianIndices.begin(), nonZeroJacobianIndices.end(), 0u);
    return;
  }

  /** Compute the number of affected B-spline parameters. */

  IndexType supportIndex;
  m_SODerivativeWeightsFunctions[0][0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** For all derivative directions, compute the derivatives of the
   * spatial Hessian to the transformation parameters mu:
   * d/dmu of d^2T / dx_i dx_j
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  const unsigned int         d = SpaceDimension * (SpaceDimension + 1) / 2;
  FixedArray<WeightsType, d> weightVector;
  {
    unsigned int count = 0;
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      for (unsigned int j = 0; j <= i; ++j)
      {
        // Compute the derivative weights and remember them.
        weightVector[count] = m_SODerivativeWeightsFunctions[i][j]->Evaluate(cindex, supportIndex);
        ++count;

      } // end for j
    } // end for i
  }

  /** Compute d/dmu d^2T_{dim} / dx_i dx_j = weights. */
  for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
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
    matrix = Superclass::m_PointToIndexMatrixTransposed2 * (matrix * Superclass::m_PointToIndexMatrix2);

    /** Copy the matrix to the right locations. */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      jsh[mu + dim * NumberOfWeights][dim] = matrix;
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
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
  if (Superclass::m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro("Cannot compute Jacobian: parameters not set");
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
    std::iota(nonZeroJacobianIndices.begin(), nonZeroJacobianIndices.end(), 0u);
    return;
  }

  /** Get the support region. */
  IndexType supportIndex;
  m_SODerivativeWeightsFunctions[0][0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** Allocate weight on the stack. */
  using WeightsValueType = typename WeightsType::ValueType;

  /** Allocate coefficients on the stack. */
  std::array<WeightsValueType, NumberOfWeights * SpaceDimension> coeffs;

  /** Copy values from coefficient image to linear coeffs array. */
  // takes considerable amount of time : 27% of this function. // with old region iterator, check with new
  auto itCoeffsLinear = coeffs.begin();
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    ImageScanlineConstIterator<ImageType> itCoef(Superclass::m_CoefficientImages[dim], supportRegion);

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
  double             weightVector[d * NumberOfWeights];

  /** For all derivative directions, compute the derivatives of the
   * spatial Hessian to the transformation parameters mu:
   * d/dmu of d^2T / dx_i dx_j
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  {
    unsigned int count = 0;
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      for (unsigned int j = 0; j <= i; ++j)
      {
        /** Compute the derivative weights. */
        const WeightsType weights = m_SODerivativeWeightsFunctions[i][j]->Evaluate(cindex, supportIndex);

        /** Remember the weights. */
        std::copy_n(weights.cbegin(), NumberOfWeights, weightVector + count * NumberOfWeights);
        ++count;

        /** Reset coeffs iterator */
        auto itCoeffs = coeffs.cbegin();

        /** Compute the spatial Hessian sh:
         *    d^2T_{dim} / dx_i dx_j = \sum coefs_{dim} * weights.
         */
        for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
        {
          /** Reset weights iterator. */
          typename WeightsType::const_iterator itWeights = weights.cbegin();

          /** Compute the sum for this dimension. */
          double sum = 0.0;
          for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
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
    } // end for i
  }

  /** Take into account grid spacing and direction matrix. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sh[dim] = Superclass::m_PointToIndexMatrixTransposed2 * (sh[dim] * Superclass::m_PointToIndexMatrix2);
  }

  /** Compute the Jacobian of the spatial Hessian jsh:
   *    d/dmu d^2T_{dim} / dx_i dx_j = weights.
   */
  SpatialJacobianType matrix;
  for (unsigned int mu = 0; mu < NumberOfWeights; ++mu)
  {
    unsigned int count = 0;
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      for (unsigned int j = 0; j <= i; ++j)
      {
        const double tmp = *(weightVector + count * NumberOfWeights + mu);
        matrix[i][j] = tmp;
        if (i != j)
        {
          matrix[j][i] = tmp;
        }
        ++count;
      }
    }

    /** Take into account grid spacing and direction matrix. */
    if (!Superclass::m_PointToIndexMatrixIsDiagonal)
    {
      matrix = Superclass::m_PointToIndexMatrixTransposed2 * (matrix * Superclass::m_PointToIndexMatrix2);
    }
    else
    {
      for (unsigned int i = 0; i < SpaceDimension; ++i)
      {
        for (unsigned int j = 0; j < SpaceDimension; ++j)
        {
          matrix[i][j] *= Superclass::m_PointToIndexMatrixDiagonalProducts[i + SpaceDimension * j];
        }
      }
    }

    /** Copy the matrix to the right locations. */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      jsh[mu + dim * NumberOfWeights][dim] = matrix;
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* ComputeNonZeroJacobianIndices ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType &           supportRegion) const
{
  /** Initialize some helper variables. */
  const unsigned long parametersPerDim = this->GetNumberOfParametersPerDimension();

  nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Compute the first global parameter number. */
  unsigned long globalStartNum = 0;
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    globalStartNum += supportRegion.GetIndex()[dim] * Superclass::m_GridOffsetTable[dim];
  }

  if constexpr (SpaceDimension == 2)
  {
    /** Initialize some helper variables. */
    const unsigned int  sx = supportRegion.GetSize()[0];
    const unsigned int  sy = supportRegion.GetSize()[1];
    const unsigned long goy = Superclass::m_GridOffsetTable[1];
    const unsigned long diffxy = goy - sx;

    /** Loop over the support region and compute the nzji. */
    unsigned int  localParNum = 0;
    unsigned long globalParNum = globalStartNum;
    for (unsigned int y = 0; y < sy; ++y)
    {
      for (unsigned int x = 0; x < sx; ++x)
      {
        nonZeroJacobianIndices[localParNum] = globalParNum;
        nonZeroJacobianIndices[localParNum + NumberOfWeights] = globalParNum + parametersPerDim;
        ++localParNum;
        ++globalParNum;
      }
      globalParNum += diffxy;
    }
  } // end if SpaceDimension == 2
  else if constexpr (SpaceDimension == 3)
  {
    /** Initialize some helper variables. */
    const unsigned int  sx = supportRegion.GetSize()[0];
    const unsigned int  sy = supportRegion.GetSize()[1];
    const unsigned int  sz = supportRegion.GetSize()[2];
    const unsigned long goy = Superclass::m_GridOffsetTable[1];
    const unsigned long goz = Superclass::m_GridOffsetTable[2];
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
          nonZeroJacobianIndices[localParNum + NumberOfWeights] = globalParNum + parametersPerDim;
          nonZeroJacobianIndices[localParNum + 2 * NumberOfWeights] = globalParNum + 2 * parametersPerDim;
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
    for (unsigned int localParNum = 0; localParNum < NumberOfWeights; ++localParNum)
    // Note that NumberOfWeights == supportRegion.GetNumberOfPixels()
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
        globalParNum += globalParIndex[dim] * Superclass::m_GridOffsetTable[dim];
      }

      /** Update the nonZeroJacobianIndices for all directions. */
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        nonZeroJacobianIndices[localParNum + dim * NumberOfWeights] = globalParNum + dim * parametersPerDim;
      }
    } // end for
  } // end general case

} // end ComputeNonZeroJacobianIndices()


/**
 * ********************* PrintSelf ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::PrintSelf(std::ostream & os,
                                                                                      Indent         indent) const
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "WeightsFunction: ";
  os << m_WeightsFunction.GetPointer() << std::endl;
}


} // end namespace itk

#endif

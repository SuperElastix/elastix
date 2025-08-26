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
#ifndef itkCyclicBSplineDeformableTransform_hxx
#define itkCyclicBSplineDeformableTransform_hxx

#include "itkCyclicBSplineDeformableTransform.h"
#include "itkContinuousIndex.h"
#include "itkImageRegionIterator.h"

namespace itk
{

/** Constructor with default arguments. */
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::CyclicBSplineDeformableTransform()
  : Superclass()
{}

/** Set the grid region. */
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::SetGridRegion(const RegionType & region)
{
  /** Call superclass SetGridRegion. */
  Superclass::SetGridRegion(region);

  /** Check if last dimension of supportregion < last dimension of grid. */
  const int lastDim = NDimensions - 1;
  const int lastDimSize = Superclass::m_GridRegion.GetSize(lastDim);

  // The support size is the same for all dimensions.
  const int supportLastDimSize = VSplineOrder + 1;

  if (supportLastDimSize > lastDimSize)
  {
    itkExceptionMacro("Last dimension (" << lastDim << ") of support size (" << supportLastDimSize
                                         << ") is larger than the number of grid points in the last dimension ("
                                         << lastDimSize << ").");
  }
}


/** Check if the point lies inside a valid region. */
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
bool
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::InsideValidRegion(
  const ContinuousIndexType & index) const
{
  bool inside = true;

  /** Check if index can be evaluated given the current grid. */
  for (unsigned int j = 0; j < SpaceDimension - 1; ++j)
  {
    if (index[j] < Superclass::m_ValidRegionBegin[j] || index[j] >= Superclass::m_ValidRegionEnd[j])
    {
      inside = false;
      break;
    }
  }

  return inside;
}


/** Split region into two parts: 1) The part that reaches from
 * inRegion.index to the border of the inImage in the last dimension and
 * 2) The part that reaches from 0 in the last dimension to the end of the
 * inRegion.
 */
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::SplitRegion(const RegionType & imageRegion,
                                                                                      const RegionType & inRegion,
                                                                                      RegionType &       outRegion1,
                                                                                      RegionType & outRegion2) const
{
  /** Set initial index and sizes of the two regions. */
  IndexType index1 = inRegion.GetIndex();
  IndexType index2 = inRegion.GetIndex();

  SizeType size1 = inRegion.GetSize();
  SizeType size2{};

  /** Get last dimension information. */
  const unsigned int lastDim = NDimensions - 1;
  const unsigned int lastDimSize = imageRegion.GetSize(lastDim);
  const unsigned int supportLastDimSize = inRegion.GetSize(lastDim);

  /** Check if we need to split. */
  const int lastDimIndex = inRegion.GetIndex(lastDim);
  if (lastDimIndex < 0)
  {
    /** Set new index and size for supportRegion1. */
    index1.SetElement(lastDim, lastDimSize + lastDimIndex);
    size1.SetElement(lastDim, abs(lastDimIndex));

    /** Set new index and size for supportRegion2. */
    index2.SetElement(lastDim, 0);
    size2 = inRegion.GetSize();
    size2.SetElement(lastDim, supportLastDimSize + lastDimIndex);
  }
  else if (lastDimIndex + supportLastDimSize > lastDimSize)
  {
    /** Set last dimension item of index2 to zero. */
    index2.SetElement(lastDim, 0);

    /** Set new size of supportRegion1. */
    size1.SetElement(lastDim, lastDimSize - lastDimIndex);

    /** Set size and index of supportRegion2. */
    size2 = inRegion.GetSize();
    size2.SetElement(lastDim, supportLastDimSize - size1.GetElement(lastDim));
  }

  /** Set region indices and sizes. */
  outRegion1.SetIndex(index1);
  outRegion1.SetSize(size1);
  outRegion2.SetIndex(index2);
  outRegion2.SetSize(size2);
}


// Transform a point
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::TransformPoint(
  const InputPointType & point) const -> OutputPointType
{
  /** Check if the coefficient image has been set. */
  if (!Superclass::m_CoefficientImages[0])
  {
    itkWarningMacro("B-spline coefficients have not been set");
    return point;
  }

  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(point);

  /** NOTE: if the support region does not lie totally within the grid
   * (except for the last dimension, which wraps around) we assume
   * zero displacement and return the input point.
   */
  if (!this->InsideValidRegion(cindex))
  {
    return point;
  }

  /** Compute interpolation weights. */
  IndexType   supportIndex;
  WeightsType weights;
  Superclass::m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  weights = Superclass::m_WeightsFunction->Evaluate(cindex, supportIndex);

  /** For each dimension, correlate coefficient with weights. */
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** Split support region into two parts. */
  RegionType supportRegions[2];
  this->SplitRegion(Superclass::m_CoefficientImages[0]->GetLargestPossibleRegion(),
                    supportRegion,
                    supportRegions[0],
                    supportRegions[1]);

  /** Zero output point elements. */
  OutputPointType outputPoint{};

  unsigned long counter = 0;
  for (const auto & region : supportRegions)
  {
    /** Create iterators over the coefficient images
     * (for both supportRegion1 and supportRegion2.
     */
    using IteratorType = ImageRegionConstIterator<ImageType>;
    IteratorType iterators[SpaceDimension];

    for (unsigned int j = 0; j < SpaceDimension - 1; ++j)
    {
      iterators[j] = IteratorType(Superclass::m_CoefficientImages[j], region);
    }

    /** Loop over this support region. */
    while (!iterators[0].IsAtEnd())
    {
      /** Multiply weigth with coefficient to compute displacement. */
      for (unsigned int j = 0; j < SpaceDimension - 1; ++j)
      {
        outputPoint[j] += static_cast<ScalarType>(weights[counter] * iterators[j].Value());
        ++iterators[j];
      }
      ++counter;

    } // end while
  }

  /** The output point is the start point + displacement. */
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    outputPoint[j] += point[j];
  }

  return outputPoint;
}


/** Compute the Jacobian in one position. */
template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetJacobian(
  const InputPointType &    point,
  WeightsType &             weights,
  ParameterIndexArrayType & indexes) const
{
  RegionType supportRegion;
  supportRegion.SetSize(WeightsFunctionType::SupportSize);
  const PixelType * basePointer = Superclass::m_CoefficientImages[0]->GetBufferPointer();

  /** Tranform from world coordinates to grid coordinates. */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(point);

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and return the input point.
   */
  if (!this->InsideValidRegion(cindex))
  {
    weights.Fill(0.0);
    indexes.Fill(0);
    return;
  }

  /** Compute interpolation weights. */
  IndexType supportIndex;
  Superclass::m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
  weights = Superclass::m_WeightsFunction->Evaluate(cindex, supportIndex);

  supportRegion.SetIndex(supportIndex);
  /** Split support region into two parts. */
  RegionType supportRegions[2];
  this->SplitRegion(Superclass::m_CoefficientImages[0]->GetLargestPossibleRegion(),
                    supportRegion,
                    supportRegions[0],
                    supportRegions[1]);

  /** For each dimension, copy the weight to the support region. */
  unsigned long counter = 0;
  for (const auto & region : supportRegions)
  {
    ImageRegionIterator<JacobianImageType> iterator(Superclass::m_CoefficientImages[0], region);

    while (!iterator.IsAtEnd())
    {
      indexes[counter] = &(iterator.Value()) - basePointer;

      /** Go to next coefficient in the support region. */
      ++counter;
      ++iterator;
    }
  }
}


/**
 * ********************* GetSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::GetSpatialJacobian(
  const InputPointType & inputPoint,
  SpatialJacobianType &  sj) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (Superclass::m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro("Cannot compute Jacobian: parameters not set");
  }

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
  WeightsType weights;

  IndexType supportIndex;
  Superclass::m_DerivativeWeightsFunctions[0]->ComputeStartIndex(cindex, supportIndex);
  const RegionType supportRegion(supportIndex, WeightsFunctionType::SupportSize);

  /** Split support region into two parts. */
  RegionType supportRegions[2];
  this->SplitRegion(Superclass::m_CoefficientImages[0]->GetLargestPossibleRegion(),
                    supportRegion,
                    supportRegions[0],
                    supportRegions[1]);

  sj.Fill(0.0);

  /** Compute the spatial Jacobian sj:
   *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights.
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Compute the derivative weights. */
    weights = Superclass::m_DerivativeWeightsFunctions[i]->Evaluate(cindex, supportIndex);

    /** Compute the spatial Jacobian sj:
     *    dT_{dim} / dx_i = \sum coefs_{dim} * weights.
     */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      /** Compute the sum for this dimension. */
      double sum = 0.0;

      typename WeightsType::const_iterator itWeights = weights.begin();

      for (const auto & region : supportRegions)
      {
        /** Create an iterator over the correct part of the coefficient
         * image. Create an iterator over the weights vector.
         */
        ImageRegionConstIterator<ImageType> itCoef(Superclass::m_CoefficientImages[dim], region);

        while (!itCoef.IsAtEnd())
        {
          sum += itCoef.Value() * (*itWeights);
          ++itWeights;
          ++itCoef;
        }
      } // end for r

      /** Update the spatial Jacobian sj. */
      sj(dim, i) += sum;
    } // end for dim
  } // end for i

  /** Take into account grid spacing and direction cosines. */
  sj *= Superclass::m_PointToIndexMatrix;

  /** Add identity. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sj(dim, dim) += 1.0;
  }

} // end GetSpatialJacobian()


/**
 * ********************* ComputeNonZeroJacobianIndices ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType &           supportRegion) const
{
  nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Split support region into two parts. */
  RegionType supportRegions[2];
  this->SplitRegion(Superclass::m_CoefficientImages[0]->GetLargestPossibleRegion(),
                    supportRegion,
                    supportRegions[0],
                    supportRegions[1]);

  /** Initialize some helper variables. */
  const SizeValueType numberOfWeights = WeightsFunctionType::NumberOfWeights;
  const SizeValueType parametersPerDim = this->GetNumberOfParametersPerDimension();
  unsigned long       mu = 0;

  for (const auto & region : supportRegions)
  {
    /** Create iterator over the coefficient image (for current supportRegion). */
    ImageRegionConstIteratorWithIndex<ImageType> iterator(Superclass::m_CoefficientImages[0], region);

    /** For all control points in the support region, set which of the
     * indices in the parameter array are non-zero.
     */
    const PixelType * basePointer = Superclass::m_CoefficientImages[0]->GetBufferPointer();
    while (!iterator.IsAtEnd())
    {
      /** Translate the index into a parameter number for the x-direction. */
      const IdentifierType parameterNumber = &(iterator.Value()) - basePointer;

      /** Update the nonZeroJacobianIndices for all directions. */
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        nonZeroJacobianIndices[mu + dim * numberOfWeights] = parameterNumber + dim * parametersPerDim;
      }

      /** Increase the iterators. */
      ++iterator;
      ++mu;
    } // end while
  } // end for (supportregions)

} // end ComputeNonZeroJacobianIndices()


} // namespace itk

#endif

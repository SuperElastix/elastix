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
  Module:    $RCSfile: itkGradientDifferenceImageToImageMetric2.hxx,v $
  Language:  C++
  Date:      $Date: 2011-29-04 14:33 $
  Version:   $Revision: 2.0 $

  Copyright (c) 2002 Insight Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkGradientDifferenceImageToImageMetric2_hxx
#define itkGradientDifferenceImageToImageMetric2_hxx

#include "itkGradientDifferenceImageToImageMetric2.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileWriter.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "itkSimpleFilterWatcher.h"

namespace itk
{

/**
 * ********************* Constructor ******************************
 */

template <class TFixedImage, class TMovingImage>
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GradientDifferenceImageToImageMetric()
{
  unsigned int iDimension;
  this->m_CastMovedImageFilter = CastMovedImageFilterType::New();
  this->m_CastFixedImageFilter = CastFixedImageFilterType::New();
  this->m_CombinationTransform = CombinationTransformType::New();
  this->m_TransformMovingImageFilter = TransformMovingImageFilterType::New();

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    this->m_MinFixedGradient[iDimension] = 0;
    this->m_MaxFixedGradient[iDimension] = 0;
    this->m_Variance[iDimension] = 0;
  }

  for (iDimension = 0; iDimension < MovedImageDimension; ++iDimension)
  {
    this->m_MinMovedGradient[iDimension] = 0;
    this->m_MaxMovedGradient[iDimension] = 0;
  }

  this->m_DerivativeDelta = 0.001;
  this->m_Rescalingfactor = 1.0;
}


/**
 * ********************* Initialize ******************************
 */

template <class TFixedImage, class TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialise the base class */
  Superclass::Initialize();

  unsigned int iFilter;

  /** Compute the gradient of the fixed images */
  this->m_CastFixedImageFilter->SetInput(this->m_FixedImage);
  this->m_CastFixedImageFilter->Update();

  for (iFilter = 0; iFilter < FixedImageDimension; ++iFilter)
  {
    this->m_FixedSobelOperators[iFilter].SetDirection(iFilter);
    this->m_FixedSobelOperators[iFilter].CreateDirectional();
    this->m_FixedSobelFilters[iFilter] = FixedSobelFilter::New();
    this->m_FixedSobelFilters[iFilter]->OverrideBoundaryCondition(&this->m_FixedBoundCond);
    this->m_FixedSobelFilters[iFilter]->SetOperator(this->m_FixedSobelOperators[iFilter]);
    this->m_FixedSobelFilters[iFilter]->SetInput(this->m_CastFixedImageFilter->GetOutput());
    this->m_FixedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
  }

  /** Resampling for 3D->2D */
  RayCastInterpolatorType * rayCaster = dynamic_cast<RayCastInterpolatorType *>(this->GetInterpolator());
  if (rayCaster != nullptr)
  {
    this->m_TransformMovingImageFilter->SetTransform(rayCaster->GetTransform());
  }
  else
  {
    itkExceptionMacro(<< "ERROR: the NormalizedGradientCorrelationImageToImageMetric is currently only suitable for "
                         "2D-3D registration.\n"
                      << "  Therefore it expects an interpolator of type RayCastInterpolator.");
  }
  this->m_TransformMovingImageFilter->SetInterpolator(this->m_Interpolator);
  this->m_TransformMovingImageFilter->SetInput(this->m_MovingImage);
  this->m_TransformMovingImageFilter->SetDefaultPixelValue(0);
  this->m_TransformMovingImageFilter->SetSize(this->m_FixedImage->GetLargestPossibleRegion().GetSize());
  this->m_TransformMovingImageFilter->SetOutputOrigin(this->m_FixedImage->GetOrigin());
  this->m_TransformMovingImageFilter->SetOutputSpacing(this->m_FixedImage->GetSpacing());
  this->m_TransformMovingImageFilter->SetOutputDirection(this->m_FixedImage->GetDirection());
  this->m_TransformMovingImageFilter->Update();

  this->m_CastMovedImageFilter->SetInput(this->m_TransformMovingImageFilter->GetOutput());

  for (iFilter = 0; iFilter < MovedImageDimension; ++iFilter)
  {
    this->m_MovedSobelOperators[iFilter].SetDirection(iFilter);
    this->m_MovedSobelOperators[iFilter].CreateDirectional();
    this->m_MovedSobelFilters[iFilter] = MovedSobelFilter::New();
    this->m_MovedSobelFilters[iFilter]->OverrideBoundaryCondition(&this->m_MovedBoundCond);
    this->m_MovedSobelFilters[iFilter]->SetOperator(this->m_MovedSobelOperators[iFilter]);
    this->m_MovedSobelFilters[iFilter]->SetInput(this->m_CastMovedImageFilter->GetOutput());
    this->m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
  }

  /** Compute the variance */
  ComputeVariance();

  /* Rescale the similarity measure between 0-1; */
  MeasureType tmpmeasure = this->GetValue(this->m_Transform->GetParameters());

  while ((fabs(tmpmeasure) / m_Rescalingfactor) > 1)
  {
    this->m_Rescalingfactor *= 10;
  }

} // end Initialize()


/**
 * ********************* PrintSelf ******************************
 */

template <class TFixedImage, class TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "DerivativeDelta: " << this->m_DerivativeDelta << std::endl;
}


/**
 * ******************** ComputeMovedGradientRange ******************************
 */

template <class TFixedImage, class TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ComputeMovedGradientRange() const
{
  unsigned int           iDimension;
  MovedGradientPixelType gradient;

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    using IteratorType = itk::ImageRegionConstIteratorWithIndex<MovedGradientImageType>;

    IteratorType iterate(m_MovedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    gradient = iterate.Get();

    m_MinMovedGradient[iDimension] = gradient;
    m_MaxMovedGradient[iDimension] = gradient;

    while (!iterate.IsAtEnd())
    {
      gradient = iterate.Get();

      if (gradient > m_MaxMovedGradient[iDimension])
      {
        m_MaxMovedGradient[iDimension] = gradient;
      }

      if (gradient < m_MinMovedGradient[iDimension])
      {
        m_MinMovedGradient[iDimension] = gradient;
      }

      ++iterate;
    }
  }
}


/**
 * ******************** ComputeVariance ******************************
 */
template <class TFixedImage, class TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ComputeVariance() const
{
  unsigned int           iDimension;
  unsigned long          nPixels;
  FixedGradientPixelType mean[FixedImageDimension];
  FixedGradientPixelType gradient;

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    using IteratorType = itk::ImageRegionConstIteratorWithIndex<FixedGradientImageType>;

    IteratorType iterate(this->m_FixedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    /** Calculate the mean gradients */
    nPixels = 0;
    gradient = iterate.Get();
    mean[iDimension] = 0;

    this->m_MinMovedGradient[iDimension] = gradient;
    this->m_MaxMovedGradient[iDimension] = gradient;

    typename FixedImageType::IndexType currentIndex;
    typename FixedImageType::PointType point;
    bool                               sampleOK = false;

    if (this->m_FixedImageMask.IsNull())
    {
      sampleOK = true;
    }

    while (!iterate.IsAtEnd())
    {
      /** Get current index */
      currentIndex = iterate.GetIndex();
      this->m_FixedImage->TransformIndexToPhysicalPoint(currentIndex, point);

      /** if fixedMask is given */
      if (!this->m_FixedImageMask.IsNull())
      {
        if (this->m_FixedImageMask->IsInsideInWorldSpace(point))
        {
          sampleOK = true;
        }
        else
        {
          sampleOK = false;
        }
      }

      if (sampleOK)
      {
        gradient = iterate.Get();
        mean[iDimension] += gradient;

        if (gradient > this->m_MaxFixedGradient[iDimension])
        {
          this->m_MaxFixedGradient[iDimension] = gradient;
        }

        if (gradient < this->m_MinFixedGradient[iDimension])
        {
          this->m_MinFixedGradient[iDimension] = gradient;
        }

        ++nPixels;

      } // end if sampleOK

      ++iterate;
    } // end while iterate

    if (nPixels > 0)
    {
      mean[iDimension] /= nPixels;
    }

    /** Calculate the variance */
    iterate.GoToBegin();
    this->m_Variance[iDimension] = 0;

    while (!iterate.IsAtEnd())
    {
      currentIndex = iterate.GetIndex();
      this->m_FixedImage->TransformIndexToPhysicalPoint(currentIndex, point);

      /** if fixedMask is given */
      if (!this->m_FixedImageMask.IsNull())
      {
        if (this->m_FixedImageMask->IsInsideInWorldSpace(point))
        {
          sampleOK = true;
        }
        else
        {
          sampleOK = false;
        }
      }

      if (sampleOK)
      {
        gradient = iterate.Get();
        gradient -= mean[iDimension];
        this->m_Variance[iDimension] += gradient * gradient;

      } // end sampleOK

      ++iterate;
    }

    this->m_Variance[iDimension] /= nPixels;
  } // end for iDimension

} // end ComputeVariance()


/**
 * ******************** ComputeMeasure ******************************
 */

template <class TFixedImage, class TMovingImage>
auto
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ComputeMeasure(
  const TransformParametersType & parameters,
  const double *                  subtractionFactor) const -> MeasureType
{
  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);
  // this->SetTransformParameters( parameters );

  unsigned int iDimension;
  this->m_TransformMovingImageFilter->Modified();
  this->m_TransformMovingImageFilter->UpdateLargestPossibleRegion();
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  typename FixedImageType::IndexType currentIndex;
  typename FixedImageType::PointType point;

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {

    if (this->m_Variance[iDimension] == NumericTraits<MovedGradientPixelType>::ZeroValue())
    {
      continue;
    }

    /** Iterate over the fixed and moving gradient images
     *  calculating the similarity measure
     */

    MovedGradientPixelType movedGradient;
    FixedGradientPixelType fixedGradient;
    MovedGradientPixelType diff;

    using FixedIteratorType = itk::ImageRegionConstIteratorWithIndex<FixedGradientImageType>;

    FixedIteratorType fixedIterator(this->m_FixedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    using MovedIteratorType = itk::ImageRegionConstIteratorWithIndex<MovedGradientImageType>;

    MovedIteratorType movedIterator(this->m_MovedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    this->m_FixedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
    this->m_MovedSobelFilters[iDimension]->UpdateLargestPossibleRegion();

    bool sampleOK = false;

    if (this->m_FixedImageMask.IsNull())
    {
      sampleOK = true;
    }

    while (!fixedIterator.IsAtEnd())
    {
      /** Get current index */
      currentIndex = fixedIterator.GetIndex();
      this->m_FixedImage->TransformIndexToPhysicalPoint(currentIndex, point);

      /** if fixedMask is given */
      if (!this->m_FixedImageMask.IsNull())
      {

        if (this->m_FixedImageMask->IsInsideInWorldSpace(point)) // sample is good
        {
          sampleOK = true;
        }
        else // sample no good
        {
          sampleOK = false;
        }
      }

      if (sampleOK)
      {
        movedGradient = movedIterator.Get();
        fixedGradient = fixedIterator.Get();
        diff = fixedGradient - subtractionFactor[iDimension] * movedGradient;
        measure += this->m_Variance[iDimension] / (this->m_Variance[iDimension] + diff * diff);

      } // end if sampleOK

      ++fixedIterator;
      ++movedIterator;
    } // end while fixedIterator

  } // end for iDimension

  return measure /= -this->m_Rescalingfactor; // negative for minimization

} // end ComputeMeasure()


/**
 * ******************** GetValue ******************************
 */

template <class TFixedImage, class TMovingImage>
auto
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
{
  unsigned int iFilter;
  unsigned int iDimension;
  this->SetTransformParameters(parameters);
  this->m_TransformMovingImageFilter->Modified();
  this->m_TransformMovingImageFilter->UpdateLargestPossibleRegion();

  /** Update the gradient images */
  for (iFilter = 0; iFilter < MovedImageDimension; ++iFilter)
  {
    this->m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
  }

  /** Compute the range of the moved image gradients */
  this->ComputeMovedGradientRange();

  MovedGradientPixelType subtractionFactor[FixedImageDimension];
  MeasureType            currentMeasure;

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    subtractionFactor[iDimension] = this->m_MaxFixedGradient[iDimension] / this->m_MaxMovedGradient[iDimension];
  }

  currentMeasure = this->ComputeMeasure(parameters, subtractionFactor);

  return currentMeasure;

} // end GetValue()


/**
 * ******************** GetDerivative ******************************
 */

template <class TFixedImage, class TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType &                derivative) const
{
  TransformParametersType testPoint;
  testPoint = parameters;
  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  derivative = DerivativeType(numberOfParameters);

  for (unsigned int i = 0; i < numberOfParameters; ++i)
  {
    testPoint[i] -= this->m_DerivativeDelta / std::sqrt(this->m_Scales[i]);
    const MeasureType valuep0 = this->GetValue(testPoint);
    testPoint[i] += 2 * this->m_DerivativeDelta / std::sqrt(this->m_Scales[i]);
    const MeasureType valuep1 = this->GetValue(testPoint);
    derivative[i] = (valuep1 - valuep0) / (2 * this->m_DerivativeDelta / std::sqrt(this->m_Scales[i]));
    testPoint[i] = parameters[i];
  }

} // end GetDerivative()


/**
 * ******************** GetValueAndDerivative ******************************
 */

template <class TFixedImage, class TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   Value,
  DerivativeType &                derivative) const
{
  Value = this->GetValue(parameters);
  this->GetDerivative(parameters, derivative);

} // end GetValueAndDerivative()


} // end namespace itk

#endif // end #ifndef itkGradientDifferenceImageToImageMetric2_hxx

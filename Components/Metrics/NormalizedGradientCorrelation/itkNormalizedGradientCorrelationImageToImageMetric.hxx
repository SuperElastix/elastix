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
#ifndef itkNormalizedGradientCorrelationImageToImageMetric_hxx
#define itkNormalizedGradientCorrelationImageToImageMetric_hxx

#include "itkNormalizedGradientCorrelationImageToImageMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkSimpleFilterWatcher.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>

namespace itk
{

/**
 * ***************** Constructor *****************
 */

template <class TFixedImage, class TMovingImage>
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,
                                                TMovingImage>::NormalizedGradientCorrelationImageToImageMetric()
{
  this->m_CastFixedImageFilter = CastFixedImageFilterType::New();
  this->m_CastMovedImageFilter = CastMovedImageFilterType::New();
  this->m_CombinationTransform = CombinationTransformType::New();
  this->m_TransformMovingImageFilter = TransformMovingImageFilterType::New();
  this->m_DerivativeDelta = 0.001;

  for (unsigned int iDimension = 0; iDimension < MovedImageDimension; ++iDimension)
  {
    this->m_MeanFixedGradient[iDimension] = 0;
    this->m_MeanMovedGradient[iDimension] = 0;
  }

} // end Constructor


/**
 * ***************** Initialize *****************
 */

template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize the base class */
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

  this->ComputeMeanFixedGradient();

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

} // end Initialize()


/**
 * ***************** PrintSelf *****************
 */

template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os,
                                                                                      Indent         indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "DerivativeDelta: " << this->m_DerivativeDelta << std::endl;
} // end PrintSelf()


/**
 * ***************** ComputeMeanFixedGradient *****************
 */

template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::ComputeMeanFixedGradient() const
{
  typename FixedGradientImageType::IndexType currentIndex;
  typename FixedGradientImageType::PointType point;

  for (int iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    this->m_FixedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
  }

  using FixedIteratorType = itk::ImageRegionConstIteratorWithIndex<FixedGradientImageType>;
  FixedIteratorType fixedIteratorx(this->m_FixedSobelFilters[0]->GetOutput(), this->GetFixedImageRegion());
  FixedIteratorType fixedIteratory(this->m_FixedSobelFilters[1]->GetOutput(), this->GetFixedImageRegion());

  fixedIteratorx.GoToBegin();
  fixedIteratory.GoToBegin();

  bool                   sampleOK = false;
  FixedGradientPixelType fixedGradient[FixedImageDimension];
  for (int i = 0; i < FixedImageDimension; ++i)
  {
    fixedGradient[i] = 0.0;
  }

  unsigned long nPixels = 0;

  if (this->m_FixedImageMask.IsNull())
  {
    sampleOK = true;
  }

  while (!fixedIteratorx.IsAtEnd())
  {
    /** Get current index */
    currentIndex = fixedIteratorx.GetIndex();
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
      fixedGradient[0] += fixedIteratorx.Get();
      fixedGradient[1] += fixedIteratory.Get();
      ++nPixels;
    }

    ++fixedIteratorx;
    ++fixedIteratory;
  } // end while

  this->m_MeanFixedGradient[0] = fixedGradient[0] / nPixels;
  this->m_MeanFixedGradient[1] = fixedGradient[1] / nPixels;

} // end ComputeMeanFixedGradient()


/**
 * ***************** ComputeMeanMovedGradient *****************
 */

template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::ComputeMeanMovedGradient() const
{
  typename MovedGradientImageType::IndexType currentIndex;
  typename MovedGradientImageType::PointType point;

  for (int iDimension = 0; iDimension < MovedImageDimension; ++iDimension)
  {
    this->m_MovedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
  }

  using MovedIteratorType = itk::ImageRegionConstIteratorWithIndex<MovedGradientImageType>;

  MovedIteratorType movedIteratorx(this->m_MovedSobelFilters[0]->GetOutput(), this->GetFixedImageRegion());
  MovedIteratorType movedIteratory(this->m_MovedSobelFilters[1]->GetOutput(), this->GetFixedImageRegion());

  movedIteratorx.GoToBegin();
  movedIteratory.GoToBegin();

  bool sampleOK = false;

  if (this->m_FixedImageMask.IsNull())
  {
    sampleOK = true;
  }

  MovedGradientPixelType movedGradient[MovedImageDimension];

  for (int i = 0; i < MovedImageDimension; ++i)
  {
    movedGradient[i] = 0.0;
  }

  unsigned long nPixels = 0;

  while (!movedIteratorx.IsAtEnd())
  {
    /** Get current index */
    currentIndex = movedIteratorx.GetIndex();
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
      movedGradient[0] += movedIteratorx.Get();
      movedGradient[1] += movedIteratory.Get();
      ++nPixels;
    } // end if sampleOK

    ++movedIteratorx;
    ++movedIteratory;
  } // end while

  this->m_MeanMovedGradient[0] = movedGradient[0] / nPixels;
  this->m_MeanMovedGradient[1] = movedGradient[1] / nPixels;

} // end ComputeMeanMovedGradient()


/**
 * ***************** ComputeMeasure *****************
 */

template <class TFixedImage, class TMovingImage>
auto
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::ComputeMeasure(
  const TransformParametersType & parameters) const -> MeasureType
{
  this->SetTransformParameters(parameters);
  this->m_TransformMovingImageFilter->Modified();
  this->m_TransformMovingImageFilter->UpdateLargestPossibleRegion();

  typename FixedImageType::IndexType currentIndex;
  typename FixedImageType::PointType point;

  MeasureType measure = NumericTraits<MeasureType>::Zero;

  MovedGradientPixelType NmovedGradient[FixedImageDimension];
  FixedGradientPixelType NfixedGradient[FixedImageDimension];

  MeasureType NGcrosscorrelation = NumericTraits<MeasureType>::Zero;
  MeasureType NGautocorrelationfixed = NumericTraits<MeasureType>::Zero;
  MeasureType NGautocorrelationmoving = NumericTraits<MeasureType>::Zero;

  /** Make sure all is updated */
  for (int iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    this->m_FixedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
    this->m_MovedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
  }

  using FixedIteratorType = itk::ImageRegionConstIteratorWithIndex<FixedGradientImageType>;

  FixedIteratorType fixedIteratorx(this->m_FixedSobelFilters[0]->GetOutput(), this->GetFixedImageRegion());
  FixedIteratorType fixedIteratory(this->m_FixedSobelFilters[1]->GetOutput(), this->GetFixedImageRegion());

  fixedIteratorx.GoToBegin();
  fixedIteratory.GoToBegin();

  using MovedIteratorType = itk::ImageRegionConstIteratorWithIndex<MovedGradientImageType>;

  MovedIteratorType movedIteratorx(this->m_MovedSobelFilters[0]->GetOutput(), this->GetFixedImageRegion());
  MovedIteratorType movedIteratory(this->m_MovedSobelFilters[1]->GetOutput(), this->GetFixedImageRegion());

  movedIteratorx.GoToBegin();
  movedIteratory.GoToBegin();

  this->m_NumberOfPixelsCounted = 0;
  bool sampleOK = false;

  if (this->m_FixedImageMask.IsNull())
  {
    sampleOK = true;
  }

  while (!fixedIteratorx.IsAtEnd())
  {
    currentIndex = fixedIteratorx.GetIndex();
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
      NmovedGradient[0] = movedIteratorx.Get() - this->m_MeanMovedGradient[0];
      NfixedGradient[0] = fixedIteratorx.Get() - this->m_MeanFixedGradient[0];
      NmovedGradient[1] = movedIteratory.Get() - this->m_MeanMovedGradient[1];
      NfixedGradient[1] = fixedIteratory.Get() - this->m_MeanFixedGradient[1];
      NGcrosscorrelation += NmovedGradient[0] * NfixedGradient[0] + NmovedGradient[1] * NfixedGradient[1];
      NGautocorrelationmoving += NmovedGradient[0] * NmovedGradient[0] + NmovedGradient[1] * NmovedGradient[1];
      NGautocorrelationfixed += NfixedGradient[0] * NfixedGradient[0] + NfixedGradient[1] * NfixedGradient[1];

    } // end if sampleOK

    ++fixedIteratorx;
    ++fixedIteratory;
    ++movedIteratorx;
    ++movedIteratory;

  } // end while

  measure = -1.0 * (NGcrosscorrelation / (std::sqrt(NGautocorrelationfixed) * std::sqrt(NGautocorrelationmoving)));
  return measure;

} // end ComputeMeasure()


/**
 * ***************** GetValue *****************
 */

template <class TFixedImage, class TMovingImage>
auto
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
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

  unsigned int iFilter;
  this->m_TransformMovingImageFilter->Modified();
  this->m_TransformMovingImageFilter->UpdateLargestPossibleRegion();

  for (iFilter = 0; iFilter < MovedImageDimension; ++iFilter)
  {
    this->m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
  }

  this->ComputeMeanMovedGradient();
  MeasureType currentMeasure = this->ComputeMeasure(parameters);

  return currentMeasure;

} // end GetValue()


/**
 * ***************** SetTransformParameters *****************
 */

template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::SetTransformParameters(
  const TransformParametersType & parameters) const
{
  if (!this->m_Transform)
  {
    itkExceptionMacro(<< "Transform has not been assigned");
  }
  this->m_Transform->SetParameters(parameters);

} // end SetTransformParameters()


/**
 * ***************** GetDerivative *****************
 */

template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(
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
 * ***************** GetValueAndDerivative *****************
 */

template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  value = this->GetValue(parameters);
  this->GetDerivative(parameters, derivative);

} // end GetValueAndDerivative()


} // end namespace itk

#endif

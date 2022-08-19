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
#ifndef _itkMultiMetricMultiResolutionImageRegistrationMethod_hxx
#define _itkMultiMetricMultiResolutionImageRegistrationMethod_hxx

#include "itkMultiMetricMultiResolutionImageRegistrationMethod.h"

#include "itkContinuousIndex.h"
#include <vnl/vnl_math.h>

/** Macro that implements the set methods. */
#define itkImplementationSetMacro(_name, _type)                                                                        \
  template <typename TFixedImage, typename TMovingImage>                                                               \
  void MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::Set##_name(_type        _arg,     \
                                                                                                unsigned int pos)      \
  {                                                                                                                    \
    if (pos == 0)                                                                                                      \
    {                                                                                                                  \
      this->Superclass::Set##_name(_arg);                                                                              \
    }                                                                                                                  \
    if (pos >= this->GetNumberOf##_name##s())                                                                          \
    {                                                                                                                  \
      this->SetNumberOf##_name##s(pos + 1);                                                                            \
    }                                                                                                                  \
    if (this->m_##_name##s[pos] != _arg)                                                                               \
    {                                                                                                                  \
      this->m_##_name##s[pos] = _arg;                                                                                  \
      this->Modified();                                                                                                \
    }                                                                                                                  \
  } // comment to allow ; after calling macro

/** Macro that implements the get methods. */
#define itkImplementationGetMacro(_name, _type1, _type2)                                                               \
  template <typename TFixedImage, typename TMovingImage>                                                               \
  auto MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::Get##_name(unsigned int pos)      \
    const->_type1 _type2                                                                                               \
  {                                                                                                                    \
    if (pos >= this->GetNumberOf##_name##s())                                                                          \
    {                                                                                                                  \
      return 0;                                                                                                        \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
      return this->m_##_name##s[pos].GetPointer();                                                                     \
    }                                                                                                                  \
  } // comment to allow ; after calling macro

namespace itk
{

itkImplementationSetMacro(FixedImage, const FixedImageType *);
itkImplementationSetMacro(MovingImage, const MovingImageType *);
itkImplementationSetMacro(FixedImageRegion, const FixedImageRegionType);
itkImplementationSetMacro(Interpolator, InterpolatorType *);
itkImplementationSetMacro(FixedImagePyramid, FixedImagePyramidType *);
itkImplementationSetMacro(MovingImagePyramid, MovingImagePyramidType *);

itkImplementationGetMacro(FixedImage, const, FixedImageType *);
itkImplementationGetMacro(MovingImage, const, MovingImageType *);
itkImplementationGetMacro(Interpolator, , InterpolatorType *);
itkImplementationGetMacro(FixedImagePyramid, , FixedImagePyramidType *);
itkImplementationGetMacro(MovingImagePyramid, , MovingImagePyramidType *);


/**
 * ****************** Constructor ******************
 */

template <typename TFixedImage, typename TMovingImage>
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage,
                                                  TMovingImage>::MultiMetricMultiResolutionImageRegistrationMethod()
{
  this->SetMetric(CombinationMetricType::New());
  this->m_Stop = false;
  this->m_LastTransformParameters = ParametersType(1);
  this->m_LastTransformParameters.Fill(0.0f);

} // end Constructor


/**
 * **************** GetFixedImageRegion **********************************
 */

template <typename TFixedImage, typename TMovingImage>
auto
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::GetFixedImageRegion(
  unsigned int pos) const -> const FixedImageRegionType &
{
  if (pos >= this->GetNumberOfFixedImageRegions())
  {
    /** Return a dummy fixed image region. */
    return this->m_NullFixedImageRegion;
  }
  else
  {
    return this->m_FixedImageRegions[pos];
  }

} // end GetFixedImageRegion()


/**
 * ********************** SetMetric *******************************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::SetMetric(MetricType * _arg)
{
  CombinationMetricType * testPtr = dynamic_cast<CombinationMetricType *>(_arg);
  if (testPtr)
  {
    if (this->m_CombinationMetric != testPtr)
    {
      this->m_CombinationMetric = testPtr;
      this->Superclass::SetMetric(this->m_CombinationMetric);
      this->Modified();
    }
  }
  else
  {
    itkExceptionMacro(<< "The metric must of type CombinationImageToImageMetric!");
  }

} // end SetMetric()


/**
 * ****************** Initialize *******************************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::Initialize()
{
  this->CheckOnInitialize();

  /** Setup the metric. */
  this->GetCombinationMetric()->SetTransform(this->GetModifiableTransform());

  this->GetCombinationMetric()->SetFixedImage(this->GetFixedImagePyramid()->GetOutput(this->GetCurrentLevel()));
  for (unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i)
  {
    this->GetCombinationMetric()->SetFixedImage(this->GetFixedImagePyramid(i)->GetOutput(this->GetCurrentLevel()), i);
  }

  this->GetCombinationMetric()->SetMovingImage(this->GetMovingImagePyramid()->GetOutput(this->GetCurrentLevel()));
  for (unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i)
  {
    this->GetCombinationMetric()->SetMovingImage(this->GetMovingImagePyramid(i)->GetOutput(this->GetCurrentLevel()), i);
  }

  this->GetCombinationMetric()->SetInterpolator(this->GetInterpolator());
  for (unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i)
  {
    this->GetCombinationMetric()->SetInterpolator(this->GetInterpolator(i), i);
  }

  this->GetCombinationMetric()->SetFixedImageRegion(this->m_FixedImageRegionPyramids[0][this->GetCurrentLevel()]);
  for (unsigned int i = 0; i < this->m_FixedImageRegionPyramids.size(); ++i)
  {
    this->GetCombinationMetric()->SetFixedImageRegion(this->m_FixedImageRegionPyramids[i][this->GetCurrentLevel()], i);
  }

  // this->GetMetric()->Initialize();
  this->GetCombinationMetric()->Initialize();

  /** Setup the optimizer. */
  this->GetModifiableOptimizer()->SetCostFunction(this->GetModifiableMetric());
  this->GetModifiableOptimizer()->SetInitialPosition(this->GetInitialTransformParametersOfNextLevel());

  /** Connect the transform to the Decorator. */
  TransformOutputType * transformOutput = static_cast<TransformOutputType *>(this->ProcessObject::GetOutput(0));

  transformOutput->Set(this->GetTransform());

} // end Initialize()


/**
 * ****************** PrepareAllPyramids ******************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::PrepareAllPyramids()
{
  this->CheckPyramids();

  /** Set up the fixed image pyramids and the fixed image region pyramids. */
  using SizeType = typename FixedImageRegionType::SizeType;
  using IndexType = typename FixedImageRegionType::IndexType;
  using ScheduleType = typename FixedImagePyramidType::ScheduleType;

  this->m_FixedImageRegionPyramids.resize(this->GetNumberOfFixedImagePyramids());
  for (unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i)
  {
    // Setup the fixed image pyramid
    FixedImagePyramidPointer fixpyr = this->GetFixedImagePyramid(i);
    if (fixpyr.IsNotNull())
    {
      fixpyr->SetNumberOfLevels(this->GetNumberOfLevels());
      if (this->GetNumberOfFixedImages() > 1)
      {
        fixpyr->SetInput(this->GetFixedImage(i));
      }
      else
      {
        fixpyr->SetInput(this->GetFixedImage());
      }
      fixpyr->UpdateLargestPossibleRegion();

      ScheduleType schedule = fixpyr->GetSchedule();

      FixedImageRegionType fixedImageRegion;
      if (this->GetNumberOfFixedImageRegions() > 1)
      {
        fixedImageRegion = this->GetFixedImageRegion(i);
      }
      else
      {
        fixedImageRegion = this->GetFixedImageRegion();
      }
      SizeType  inputSize = fixedImageRegion.GetSize();
      IndexType inputStart = fixedImageRegion.GetIndex();
      IndexType inputEnd = inputStart;
      for (unsigned int dim = 0; dim < TFixedImage::ImageDimension; ++dim)
      {
        inputEnd[dim] += (inputSize[dim] - 1);
      }

      this->m_FixedImageRegionPyramids[i].reserve(this->GetNumberOfLevels());
      this->m_FixedImageRegionPyramids[i].resize(this->GetNumberOfLevels());

      // Compute the FixedImageRegion corresponding to each level of the
      // pyramid.
      //
      // In the ITK implementation this uses the same algorithm of the ShrinkImageFilter
      // since the regions should be compatible. However, we inherited another
      // Multiresolution pyramid, which does not use the same shrinking pattern.
      // Instead of copying the shrinking code, we compute image regions from
      // the result of the fixed image pyramid.
      using PointType = typename FixedImageType::PointType;
      using CoordRepType = typename PointType::CoordRepType;
      using IndexValueType = typename IndexType::IndexValueType;
      using SizeValueType = typename SizeType::SizeValueType;

      PointType inputStartPoint;
      PointType inputEndPoint;
      fixpyr->GetInput()->TransformIndexToPhysicalPoint(inputStart, inputStartPoint);
      fixpyr->GetInput()->TransformIndexToPhysicalPoint(inputEnd, inputEndPoint);

      for (unsigned int level = 0; level < this->GetNumberOfLevels(); ++level)
      {
        SizeType         size;
        IndexType        start;
        FixedImageType * fixedImageAtLevel = fixpyr->GetOutput(level);
        /** map the original fixed image region to the image resulting from the
         * FixedImagePyramid at level l.
         * To be on the safe side, the start point is ceiled, and the end point
         * is floored. To see why, consider an image of 4 by 4, and its
         * downsampled version of 2 by 2.
         */
        const auto startcindex =
          fixedImageAtLevel->template TransformPhysicalPointToContinuousIndex<CoordRepType>(inputStartPoint);
        const auto endcindex =
          fixedImageAtLevel->template TransformPhysicalPointToContinuousIndex<CoordRepType>(inputEndPoint);
        for (unsigned int dim = 0; dim < TFixedImage::ImageDimension; ++dim)
        {
          start[dim] = static_cast<IndexValueType>(std::ceil(startcindex[dim]));
          size[dim] = std::max(
            NumericTraits<SizeValueType>::One,
            static_cast<SizeValueType>(static_cast<SizeValueType>(std::floor(endcindex[dim])) - start[dim] + 1));
        }

        this->m_FixedImageRegionPyramids[i][level].SetSize(size);
        this->m_FixedImageRegionPyramids[i][level].SetIndex(start);

      } // end for loop over res levels

    } // end if fixpyr!=0

  } // end for loop over fixed pyramids

  /** Setup the moving image pyramids. */
  for (unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i)
  {
    MovingImagePyramidPointer movpyr = this->GetMovingImagePyramid(i);
    if (movpyr.IsNotNull())
    {
      movpyr->SetNumberOfLevels(this->GetNumberOfLevels());
      if (this->GetNumberOfMovingImages() > 1)
      {
        movpyr->SetInput(this->GetMovingImage(i));
      }
      else
      {
        movpyr->SetInput(this->GetMovingImage());
      }
      movpyr->UpdateLargestPossibleRegion();
    }
  }

} // end PrepareAllPyramids()


/**
 * ****************** PrintSelf ******************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os,
                                                                                        Indent         indent) const
{
  Superclass::PrintSelf(os, indent);

} // end PrintSelf()


/**
 * ********************* GenerateData ***********************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::GenerateData()
{
  this->m_Stop = false;

  /** Check the transform and set the initial parameters. */
  if (this->GetTransform() == nullptr)
  {
    itkExceptionMacro(<< "Transform is not present");
  }

  this->SetInitialTransformParametersOfNextLevel(this->GetInitialTransformParameters());

  if (this->GetInitialTransformParametersOfNextLevel().Size() != this->GetTransform()->GetNumberOfParameters())
  {
    itkExceptionMacro(<< "Size mismatch between initial parameter and transform");
  }

  /** Prepare the fixed and moving pyramids. */
  this->PrepareAllPyramids();

  /** Loop over the resolution levels. */
  for (unsigned int currentLevel = 0; currentLevel < this->GetNumberOfLevels(); ++currentLevel)
  {
    this->SetCurrentLevel(currentLevel);

    // Invoke an iteration event.
    // This allows a UI to reset any of the components between
    // resolution level.
    this->InvokeEvent(IterationEvent());

    // Check if there has been a stop request
    if (this->m_Stop)
    {
      break;
    }

    try
    {
      // initialize the interconnects between components
      this->Initialize();
    }
    catch (const ExceptionObject &)
    {
      this->m_LastTransformParameters = ParametersType(1);
      this->m_LastTransformParameters.Fill(0.0f);

      // pass exception to caller
      throw;
    }

    try
    {
      // do the optimization
      this->GetModifiableOptimizer()->StartOptimization();
    }
    catch (const ExceptionObject &)
    {
      // An error has occurred in the optimization.
      // Update the parameters
      this->m_LastTransformParameters = this->GetOptimizer()->GetCurrentPosition();

      // Pass exception to caller
      throw;
    }

    // get the results
    this->m_LastTransformParameters = this->GetOptimizer()->GetCurrentPosition();
    this->GetModifiableTransform()->SetParameters(this->m_LastTransformParameters);

    // setup the initial parameters for next level
    if (this->GetCurrentLevel() < this->GetNumberOfLevels() - 1)
    {
      this->SetInitialTransformParametersOfNextLevel(this->m_LastTransformParameters);
    }

  } // end for loop over res levels

} // end GenerateData()


/**
 * ***************** GetMTime ******************
 */

template <typename TFixedImage, typename TMovingImage>
ModifiedTimeType
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::GetMTime() const
{
  ModifiedTimeType mtime = Superclass::GetMTime();
  ModifiedTimeType m;

  // Some of the following should be removed once ivars are put in the
  // input and output lists

  for (unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i)
  {
    InterpolatorPointer interpolator = this->GetInterpolator(i);
    if (interpolator)
    {
      m = interpolator->GetMTime();
      mtime = (m > mtime ? m : mtime);
    }
  }

  for (unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i)
  {
    FixedImageConstPointer fixedImage = this->GetFixedImage(i);
    if (fixedImage)
    {
      m = fixedImage->GetMTime();
      mtime = (m > mtime ? m : mtime);
    }
  }

  for (unsigned int i = 0; i < this->GetNumberOfMovingImages(); ++i)
  {
    MovingImageConstPointer movingImage = this->GetMovingImage(i);
    if (movingImage)
    {
      m = movingImage->GetMTime();
      mtime = (m > mtime ? m : mtime);
    }
  }

  for (unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i)
  {
    FixedImagePyramidPointer fixedImagePyramid = this->GetFixedImagePyramid(i);
    if (fixedImagePyramid)
    {
      m = fixedImagePyramid->GetMTime();
      mtime = (m > mtime ? m : mtime);
    }
  }

  for (unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i)
  {
    MovingImagePyramidPointer movingImagePyramid = this->GetMovingImagePyramid(i);
    if (movingImagePyramid)
    {
      m = movingImagePyramid->GetMTime();
      mtime = (m > mtime ? m : mtime);
    }
  }

  return mtime;

} // end GetMTime()


/**
 * ****************** CheckPyramids ******************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::CheckPyramids()
{
  /** Check if at least one of the following are provided. */
  if (this->GetFixedImage() == nullptr)
  {
    itkExceptionMacro(<< "FixedImage is not present");
  }
  if (this->GetMovingImage() == nullptr)
  {
    itkExceptionMacro(<< "MovingImage is not present");
  }
  if (this->GetFixedImagePyramid() == nullptr)
  {
    itkExceptionMacro(<< "Fixed image pyramid is not present");
  }
  if (this->GetMovingImagePyramid() == nullptr)
  {
    itkExceptionMacro(<< "Moving image pyramid is not present");
  }

  /** Check if the number if fixed/moving pyramids >= nr of fixed/moving images,
   * and whether the number of fixed image regions == the number of fixed images.
   */
  if (this->GetNumberOfFixedImagePyramids() < this->GetNumberOfFixedImages())
  {
    itkExceptionMacro(<< "The number of fixed image pyramids should be >= the number of fixed images");
  }
  if (this->GetNumberOfMovingImagePyramids() < this->GetNumberOfMovingImages())
  {
    itkExceptionMacro(<< "The number of moving image pyramids should be >= the number of moving images");
  }
  if (this->GetNumberOfFixedImageRegions() != this->GetNumberOfFixedImages())
  {
    itkExceptionMacro(<< "The number of fixed image regions should equal the number of fixed images");
  }

} // end CheckPyramids()


/**
 * ****************** CheckOnInitialize ******************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiMetricMultiResolutionImageRegistrationMethod<TFixedImage, TMovingImage>::CheckOnInitialize()
{
  /** Check if at least one of the following is present. */
  if (this->GetMetric() == nullptr)
  {
    itkExceptionMacro(<< "Metric is not present");
  }
  if (this->GetOptimizer() == nullptr)
  {
    itkExceptionMacro(<< "Optimizer is not present");
  }
  if (this->GetTransform() == nullptr)
  {
    itkExceptionMacro(<< "Transform is not present");
  }
  if (this->GetInterpolator() == nullptr)
  {
    itkExceptionMacro(<< "Interpolator is not present");
  }

  /** nrofmetrics >= nrofinterpolators >= nrofpyramids >= nofimages */
  unsigned int nrOfMetrics = this->GetCombinationMetric()->GetNumberOfMetrics();
  if (this->GetNumberOfInterpolators() > nrOfMetrics)
  {
    itkExceptionMacro(<< "NumberOfInterpolators can not exceed the NumberOfMetrics in the CombinationMetric!");
  }
  if (this->GetNumberOfFixedImagePyramids() > nrOfMetrics)
  {
    itkExceptionMacro(<< "NumberOfFixedImagePyramids can not exceed the NumberOfMetrics in the CombinationMetric!");
  }
  if (this->GetNumberOfMovingImagePyramids() > nrOfMetrics)
  {
    itkExceptionMacro(<< "NumberOfMovingImagePyramids can not exceed the NumberOfMetrics in the CombinationMetric!");
  }
  if (this->GetNumberOfMovingImagePyramids() > this->GetNumberOfInterpolators())
  {
    itkExceptionMacro(<< "NumberOfMovingImagePyramids can not exceed the NumberOfInterpolators!");
  }

  /** For all components: ==nrofmetrics of ==1. */
  if ((this->GetNumberOfInterpolators() != 1) && (this->GetNumberOfInterpolators() != nrOfMetrics))
  {
    itkExceptionMacro(<< "The NumberOfInterpolators should equal 1 or equal the NumberOfMetrics");
  }
  if ((this->GetNumberOfFixedImagePyramids() != 1) && (this->GetNumberOfFixedImagePyramids() != nrOfMetrics))
  {
    itkExceptionMacro(<< "The NumberOfFixedImagePyramids should equal 1 or equal the NumberOfMetrics");
  }
  if ((this->GetNumberOfMovingImagePyramids() != 1) && (this->GetNumberOfMovingImagePyramids() != nrOfMetrics))
  {
    itkExceptionMacro(<< "The NumberOfMovingImagePyramids should equal 1 or equal the NumberOfMetrics");
  }
  if ((this->GetNumberOfFixedImages() != 1) && (this->GetNumberOfFixedImages() != nrOfMetrics))
  {
    itkExceptionMacro(<< "The NumberOfFixedImages should equal 1 or equal the NumberOfMetrics");
  }
  if ((this->GetNumberOfMovingImages() != 1) && (this->GetNumberOfMovingImages() != nrOfMetrics))
  {
    itkExceptionMacro(<< "The NumberOfMovingImages should equal 1 or equal the NumberOfMetrics");
  }

} // end CheckOnInitialize()


} // end namespace itk

#undef itkImplementationSetMacro
#undef itkImplementationGetMacro

#endif

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

/** This class is a slight modification of the original ITK class:
 * MultiResolutionImageRegistrationMethod.
 * The original copyright message is pasted here, which includes also
 * the version information: */
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Date:      $Date: 2008-11-06 16:31:54 +0100 (Thu, 06 Nov 2008) $
  Version:   $Revision: 2637 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkMultiResolutionImageRegistrationMethod2_hxx
#define _itkMultiResolutionImageRegistrationMethod2_hxx

#include "itkMultiResolutionImageRegistrationMethod2.h"
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkContinuousIndex.h"
#include <vnl/vnl_math.h>

namespace itk
{

/*
 * Constructor
 */
template <typename TFixedImage, typename TMovingImage>
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::MultiResolutionImageRegistrationMethod2()
{
  this->SetNumberOfRequiredOutputs(1); // for the Transform

  m_FixedImage = nullptr;   // has to be provided by the user.
  m_MovingImage = nullptr;  // has to be provided by the user.
  m_Transform = nullptr;    // has to be provided by the user.
  m_Interpolator = nullptr; // has to be provided by the user.
  m_Metric = nullptr;       // has to be provided by the user.
  m_Optimizer = nullptr;    // has to be provided by the user.

  // Use MultiResolutionPyramidImageFilter as the default
  // image pyramids.
  m_FixedImagePyramid = FixedImagePyramidType::New();
  m_MovingImagePyramid = MovingImagePyramidType::New();

  m_NumberOfLevels = 1;
  m_CurrentLevel = 0;

  m_Stop = false;

  this->ProcessObject::SetNthOutput(0, TransformOutputType::New().GetPointer());

} // end Constructor


/*
 * Initialize by setting the interconnects between components.
 */
template <typename TFixedImage, typename TMovingImage>
void
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::Initialize()
{

  // Sanity checks
  if (!m_Metric)
  {
    itkExceptionMacro("Metric is not present");
  }

  if (!m_Optimizer)
  {
    itkExceptionMacro("Optimizer is not present");
  }

  if (!m_Transform)
  {
    itkExceptionMacro("Transform is not present");
  }

  if (!m_Interpolator)
  {
    itkExceptionMacro("Interpolator is not present");
  }

  // The transform parameters must be set before initializing the metric, in order to support the IMPACT metric in
  // "Static" mode. The IMPACT metric is written by Valentin Boussot, pull request #1311.
  m_Transform->SetParameters(m_InitialTransformParametersOfNextLevel);

  // Setup the metric
  m_Metric->SetMovingImage(m_MovingImagePyramid->GetOutput(m_CurrentLevel));
  m_Metric->SetFixedImage(m_FixedImagePyramid->GetOutput(m_CurrentLevel));
  m_Metric->SetTransform(m_Transform);
  m_Metric->SetInterpolator(m_Interpolator);
  m_Metric->SetFixedImageRegion(m_FixedImageRegionPyramid[m_CurrentLevel]);
  m_Metric->Initialize();

  // Setup the optimizer
  m_Optimizer->SetCostFunction(m_Metric);
  m_Optimizer->SetInitialPosition(m_InitialTransformParametersOfNextLevel);

  //
  // Connect the transform to the Decorator.
  //
  auto * transformOutput = static_cast<TransformOutputType *>(this->ProcessObject::GetOutput(0));

  transformOutput->Set(m_Transform.GetPointer());

} // end Initialize()


/*
 * Stop the Registration Process
 */
template <typename TFixedImage, typename TMovingImage>
void
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::StopRegistration()
{
  m_Stop = true;
}


/*
 * Stop the Registration Process
 */
template <typename TFixedImage, typename TMovingImage>
void
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::PreparePyramids()
{
  if (!m_Transform)
  {
    itkExceptionMacro("Transform is not present");
  }

  m_InitialTransformParametersOfNextLevel = m_InitialTransformParameters;

  if (const auto numberOfInitialTransformParameters = m_InitialTransformParameters.size();
      numberOfInitialTransformParameters != m_Transform->GetNumberOfParameters())
  {
    itkExceptionMacro("Size mismatch between initial parameters (" << numberOfInitialTransformParameters
                                                                   << ") and transform ("
                                                                   << m_Transform->GetNumberOfParameters() << ")");
  }

  // Sanity checks
  if (!m_FixedImage)
  {
    itkExceptionMacro("FixedImage is not present");
  }

  if (!m_MovingImage)
  {
    itkExceptionMacro("MovingImage is not present");
  }

  if (!m_FixedImagePyramid)
  {
    itkExceptionMacro("Fixed image pyramid is not present");
  }

  if (!m_MovingImagePyramid)
  {
    itkExceptionMacro("Moving image pyramid is not present");
  }

  // Setup the fixed image pyramid
  m_FixedImagePyramid->SetNumberOfLevels(m_NumberOfLevels);
  m_FixedImagePyramid->SetInput(m_FixedImage);
  m_FixedImagePyramid->UpdateLargestPossibleRegion();

  // Setup the moving image pyramid
  m_MovingImagePyramid->SetNumberOfLevels(m_NumberOfLevels);
  m_MovingImagePyramid->SetInput(m_MovingImage);
  m_MovingImagePyramid->UpdateLargestPossibleRegion();

  using SizeType = typename FixedImageRegionType::SizeType;
  using IndexType = typename FixedImageRegionType::IndexType;
  using ScheduleType = typename FixedImagePyramidType::ScheduleType;

  ScheduleType schedule = m_FixedImagePyramid->GetSchedule();

  SizeType  inputSize = m_FixedImageRegion.GetSize();
  IndexType inputStart = m_FixedImageRegion.GetIndex();
  IndexType inputEnd = inputStart;
  for (unsigned int dim = 0; dim < TFixedImage::ImageDimension; ++dim)
  {
    inputEnd[dim] += (inputSize[dim] - 1);
  }

  m_FixedImageRegionPyramid.resize(m_NumberOfLevels);

  // Compute the FixedImageRegion corresponding to each level of the
  // pyramid.
  //
  // In the ITK implementation this uses the same algorithm of the ShrinkImageFilter
  // since the regions should be compatible. However, we inherited another
  // Multiresolution pyramid, which does not use the same shrinking pattern.
  // Instead of copying the shrinking code, we compute image regions from
  // the result of the fixed image pyramid.
  using PointType = typename FixedImageType::PointType;
  using CoordinateType = typename PointType::CoordinateType;

  PointType inputStartPoint;
  PointType inputEndPoint;
  m_FixedImage->TransformIndexToPhysicalPoint(inputStart, inputStartPoint);
  m_FixedImage->TransformIndexToPhysicalPoint(inputEnd, inputEndPoint);

  for (unsigned int level = 0; level < m_NumberOfLevels; ++level)
  {
    SizeType         size;
    IndexType        start;
    FixedImageType * fixedImageAtLevel = m_FixedImagePyramid->GetOutput(level);
    /** map the original fixed image region to the image resulting from the
     * FixedImagePyramid at level l.
     * To be on the safe side, the start point is ceiled, and the end point is
     * floored. To see why, consider an image of 4 by 4, and its downsampled version of 2 by 2. */
    const auto startcindex =
      fixedImageAtLevel->template TransformPhysicalPointToContinuousIndex<CoordinateType>(inputStartPoint);
    const auto endcindex =
      fixedImageAtLevel->template TransformPhysicalPointToContinuousIndex<CoordinateType>(inputEndPoint);
    for (unsigned int dim = 0; dim < TFixedImage::ImageDimension; ++dim)
    {
      start[dim] = static_cast<IndexValueType>(std::ceil(startcindex[dim]));
      size[dim] =
        std::max(NumericTraits<SizeValueType>::One,
                 static_cast<SizeValueType>(static_cast<SizeValueType>(std::floor(endcindex[dim])) - start[dim] + 1));
    }

    m_FixedImageRegionPyramid[level].SetSize(size);
    m_FixedImageRegionPyramid[level].SetIndex(start);
  }

} // end PreparePyramids()


/*
 * PrintSelf
 */
template <typename TFixedImage, typename TMovingImage>
void
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Metric: " << m_Metric.GetPointer() << std::endl;
  os << indent << "Optimizer: " << m_Optimizer.GetPointer() << std::endl;
  os << indent << "Transform: " << m_Transform.GetPointer() << std::endl;
  os << indent << "Interpolator: " << m_Interpolator.GetPointer() << std::endl;
  os << indent << "FixedImage: " << m_FixedImage.GetPointer() << std::endl;
  os << indent << "MovingImage: " << m_MovingImage.GetPointer() << std::endl;
  os << indent << "FixedImagePyramid: " << m_FixedImagePyramid.GetPointer() << std::endl;
  os << indent << "MovingImagePyramid: " << m_MovingImagePyramid.GetPointer() << std::endl;

  os << indent << "NumberOfLevels: " << m_NumberOfLevels << std::endl;
  os << indent << "CurrentLevel: " << m_CurrentLevel << std::endl;

  os << indent << "InitialTransformParameters: " << m_InitialTransformParameters << std::endl;
  os << indent << "InitialTransformParametersOfNextLevel: " << m_InitialTransformParametersOfNextLevel << std::endl;
  os << indent << "LastTransformParameters: " << m_LastTransformParameters << std::endl;
  os << indent << "FixedImageRegion: " << m_FixedImageRegion << std::endl;

  for (unsigned int level = 0; level < m_FixedImageRegionPyramid.size(); ++level)
  {
    os << indent << "FixedImageRegion at level " << level << ": " << m_FixedImageRegionPyramid[level] << std::endl;
  }

} // end PrintSelf()


/*
 * Generate Data
 */
template <typename TFixedImage, typename TMovingImage>
void
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::GenerateData()
{
  m_Stop = false;

  this->PreparePyramids();

  for (m_CurrentLevel = 0; m_CurrentLevel < m_NumberOfLevels; m_CurrentLevel++)
  {

    // Invoke an iteration event.
    // This allows a UI to reset any of the components between
    // resolution level.
    this->InvokeEvent(IterationEvent());

    // Check if there has been a stop request
    if (m_Stop)
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
      m_LastTransformParameters = ParametersType(1);
      m_LastTransformParameters.Fill(0.0f);

      // pass exception to caller
      throw;
    }

    try
    {
      // do the optimization
      m_Optimizer->StartOptimization();
    }
    catch (const ExceptionObject &)
    {
      // An error has occurred in the optimization.
      // Update the parameters
      m_LastTransformParameters = m_Optimizer->GetCurrentPosition();

      // Pass exception to caller
      throw;
    }

    // get the results
    m_LastTransformParameters = m_Optimizer->GetCurrentPosition();
    m_Transform->SetParameters(m_LastTransformParameters);

    // setup the initial parameters for next level
    if (m_CurrentLevel < m_NumberOfLevels - 1)
    {
      m_InitialTransformParametersOfNextLevel = m_LastTransformParameters;
    }
  }
}


template <typename TFixedImage, typename TMovingImage>
ModifiedTimeType
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::GetMTime() const
{
  ModifiedTimeType mtime = Superclass::GetMTime();
  ModifiedTimeType m;

  // Some of the following should be removed once ivars are put in the
  // input and output lists

  if (m_Transform)
  {
    m = m_Transform->GetMTime();
    mtime = (m > mtime ? m : mtime);
  }

  if (m_Interpolator)
  {
    m = m_Interpolator->GetMTime();
    mtime = (m > mtime ? m : mtime);
  }

  if (m_Metric)
  {
    m = m_Metric->GetMTime();
    mtime = (m > mtime ? m : mtime);
  }

  if (m_Optimizer)
  {
    m = m_Optimizer->GetMTime();
    mtime = (m > mtime ? m : mtime);
  }

  if (m_FixedImage)
  {
    m = m_FixedImage->GetMTime();
    mtime = (m > mtime ? m : mtime);
  }

  if (m_MovingImage)
  {
    m = m_MovingImage->GetMTime();
    mtime = (m > mtime ? m : mtime);
  }

  return mtime;

} // end GetMTime()


/*
 *  Get Output
 */
template <typename TFixedImage, typename TMovingImage>
auto
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::GetOutput() const -> const TransformOutputType *
{
  return static_cast<const TransformOutputType *>(this->ProcessObject::GetOutput(0));
}

template <typename TFixedImage, typename TMovingImage>
DataObject::Pointer
MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>::MakeOutput(
  ProcessObject::DataObjectPointerArraySizeType output)
{
  if (output > 0)
  {
    itkExceptionMacro("MakeOutput request for an output number larger than the expected number of outputs.");
  }
  return TransformOutputType::New().GetPointer();
}


} // end namespace itk

#endif

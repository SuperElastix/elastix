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
  Module:    $RCSfile: itkTransformToDeterminantOfSpatialJacobianSource.txx,v $
  Language:  C++
  Date:      $Date: 2008-08-01 13:42:00 $
  Version:   $Revision: 1.2 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkTransformToDeterminantOfSpatialJacobianSource_hxx
#define itkTransformToDeterminantOfSpatialJacobianSource_hxx

#include "itkTransformToDeterminantOfSpatialJacobianSource.h"

#include "itkProgressReporter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <vnl/vnl_det.h>

namespace itk
{

/**
 * Constructor
 */
template <class TOutputImage, class TTransformPrecisionType>
TransformToDeterminantOfSpatialJacobianSource<TOutputImage,
                                              TTransformPrecisionType>::TransformToDeterminantOfSpatialJacobianSource()
{
  // Use the classic (ITK4) threading model, to ensure ThreadedGenerateData is being called.
  this->itk::ImageSource<TOutputImage>::DynamicMultiThreadingOff();

} // end Constructor


/**
 * Print out a description of self
 *
 * \todo Add details about this class
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::PrintSelf(std::ostream & os,
                                                                                                Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "OutputRegion: " << this->m_OutputRegion << std::endl;
  os << indent << "OutputSpacing: " << this->m_OutputSpacing << std::endl;
  os << indent << "OutputOrigin: " << this->m_OutputOrigin << std::endl;
  os << indent << "OutputDirection: " << this->m_OutputDirection << std::endl;
  os << indent << "Transform: " << this->m_Transform.GetPointer() << std::endl;

} // end PrintSelf()


/**
 * Set the output image size.
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::SetOutputSize(
  const SizeType & size)
{
  this->m_OutputRegion.SetSize(size);
}


/**
 * Get the output image size.
 */
template <class TOutputImage, class TTransformPrecisionType>
auto
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::GetOutputSize()
  -> const SizeType &
{
  return this->m_OutputRegion.GetSize();
}

/**
 * Set the output image index.
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::SetOutputIndex(
  const IndexType & index)
{
  this->m_OutputRegion.SetIndex(index);
}


/**
 * Get the output image index.
 */
template <class TOutputImage, class TTransformPrecisionType>
auto
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::GetOutputIndex()
  -> const IndexType &
{
  return this->m_OutputRegion.GetIndex();
}

/**
 * Set the output image spacing.
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::SetOutputSpacing(
  const double * spacing)
{
  SpacingType s(spacing);
  this->SetOutputSpacing(s);

} // end SetOutputSpacing()


/**
 * Set the output image origin.
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::SetOutputOrigin(
  const double * origin)
{
  OriginType p(origin);
  this->SetOutputOrigin(p);
}


/** Helper method to set the output parameters based on this image */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::SetOutputParametersFromImage(
  const ImageBaseType * image)
{
  if (!image)
  {
    itkExceptionMacro(<< "Cannot use a null image reference");
  }

  this->SetOutputOrigin(image->GetOrigin());
  this->SetOutputSpacing(image->GetSpacing());
  this->SetOutputDirection(image->GetDirection());
  this->SetOutputRegion(image->GetLargestPossibleRegion());

} // end SetOutputParametersFromImage()


/**
 * Set up state of filter before multi-threading.
 * InterpolatorType::SetInputImage is not thread-safe and hence
 * has to be set up before ThreadedGenerateData
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::BeforeThreadedGenerateData()
{
  if (!this->m_Transform)
  {
    itkExceptionMacro(<< "Transform not set");
  }

  // Check whether we can use a fast path for resampling. Fast path
  // can be used if the transformation is linear. Transform respond
  // to the IsLinear() call.
  if (this->m_Transform->IsLinear())
  {
    this->LinearGenerateData();
  }

} // end BeforeThreadedGenerateData()


/**
 * ThreadedGenerateData
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::ThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  ThreadIdType                  threadId)
{
  // In case of linear transforms, the computation has already been
  // completed in the BeforeThreadedGenerateData
  if (this->m_Transform->IsLinear())
  {
    return;
  }

  // Otherwise, we use the normal method where the transform is called
  // for computing the transformation of every point.
  this->NonlinearThreadedGenerateData(outputRegionForThread, threadId);

} // end ThreadedGenerateData()


template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::NonlinearThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  ThreadIdType                  threadId)
{
  // Get the output pointer
  OutputImagePointer outputPtr = this->GetOutput();

  // Create an iterator that will walk the output region for this thread.
  using OutputIteratorType = ImageRegionIteratorWithIndex<TOutputImage>;
  OutputIteratorType it(outputPtr, outputRegionForThread);
  it.GoToBegin();

  // pixel coordinates
  PointType point;

  // Support for progress methods/callbacks
  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  // Walk the output region
  while (!it.IsAtEnd())
  {
    // Determine the coordinates of the current voxel
    outputPtr->TransformIndexToPhysicalPoint(it.GetIndex(), point);

    SpatialJacobianType sj;
    this->m_Transform->GetSpatialJacobian(point, sj);
    const PixelType detjac = static_cast<PixelType>(vnl_det(sj.GetVnlMatrix()));

    // Set it
    it.Set(detjac);

    // Update progress and iterator
    progress.CompletedPixel();
    ++it;
  }

} // end NonlinearThreadedGenerateData()


template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::LinearGenerateData()
{
  // Use an unthreaded implementation here, since the FillBuffer method
  // is used.

  // Get the output pointer
  OutputImagePointer outputPtr = this->GetOutput();

  // For linear transformation the spatial derivative is a constant,
  // i.e. it is independent of the spatial position.
  IndexType index;
  index.Fill(1);
  PointType point;
  outputPtr->TransformIndexToPhysicalPoint(index, point);
  SpatialJacobianType sj;
  this->m_Transform->GetSpatialJacobian(point, sj);
  const PixelType detjac = static_cast<PixelType>(vnl_det(sj.GetVnlMatrix()));

  outputPtr->FillBuffer(detjac);

} // end LinearThreadedGenerateData()


/**
 * Inform pipeline of required output region
 */
template <class TOutputImage, class TTransformPrecisionType>
void
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::GenerateOutputInformation()
{
  // call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointer to the output
  OutputImagePointer outputPtr = this->GetOutput();
  if (!outputPtr)
  {
    return;
  }

  outputPtr->SetLargestPossibleRegion(m_OutputRegion);
  outputPtr->SetSpacing(m_OutputSpacing);
  outputPtr->SetOrigin(m_OutputOrigin);
  outputPtr->SetDirection(m_OutputDirection);
  outputPtr->Allocate();

} // end GenerateOutputInformation()


/**
 * Verify if any of the components has been modified.
 */
template <class TOutputImage, class TTransformPrecisionType>
ModifiedTimeType
TransformToDeterminantOfSpatialJacobianSource<TOutputImage, TTransformPrecisionType>::GetMTime() const
{
  ModifiedTimeType latestTime = Object::GetMTime();

  if (this->m_Transform)
  {
    if (latestTime < this->m_Transform->GetMTime())
    {
      latestTime = this->m_Transform->GetMTime();
    }
  }

  return latestTime;
} // end GetMTime()


} // end namespace itk

#endif // end #ifndef _itkTransformToDeterminantOfSpatialJacobianSource_hxx

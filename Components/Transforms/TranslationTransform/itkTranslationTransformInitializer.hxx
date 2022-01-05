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

#ifndef itkTranslationTransformInitializer_hxx
#define itkTranslationTransformInitializer_hxx

#include "itkTranslationTransformInitializer.h"
#include "itkImageMaskSpatialObject.h"

namespace itk
{

/**
 * ************************* Constructor *********************
 */

template <class TTransform, class TFixedImage, class TMovingImage>
TranslationTransformInitializer<TTransform, TFixedImage, TMovingImage>::TranslationTransformInitializer()
{
  this->m_FixedCalculator = FixedImageCalculatorType::New();
  this->m_MovingCalculator = MovingImageCalculatorType::New();
}


/**
 * ************************* InitializeTransform *********************
 */

template <class TTransform, class TFixedImage, class TMovingImage>
void
TranslationTransformInitializer<TTransform, TFixedImage, TMovingImage>::InitializeTransform() const
{
  // Sanity check
  if (!this->m_FixedImage)
  {
    itkExceptionMacro("Fixed Image has not been set");
    return;
  }
  if (!this->m_MovingImage)
  {
    itkExceptionMacro("Moving Image has not been set");
    return;
  }
  if (!this->m_Transform)
  {
    itkExceptionMacro("Transform has not been set");
    return;
  }

  // If images come from filters, then update those filters.
  if (this->m_FixedImage->GetSource())
  {
    this->m_FixedImage->GetSource()->Update();
  }
  if (this->m_MovingImage->GetSource())
  {
    this->m_MovingImage->GetSource()->Update();
  }

  OutputVectorType translationVector;

  using FixedMaskSpatialObjectType = ImageMaskSpatialObject<InputSpaceDimension>;
  using MovingMaskSpatialObjectType = ImageMaskSpatialObject<OutputSpaceDimension>;

  if (this->m_UseMoments)
  {
    // Convert the masks to spatial objects
    typename FixedMaskSpatialObjectType::Pointer fixedMaskAsSpatialObject; // default-constructed (null)
    if (this->m_FixedMask)
    {
      fixedMaskAsSpatialObject = FixedMaskSpatialObjectType::New();
      fixedMaskAsSpatialObject->SetImage(this->m_FixedMask);
      fixedMaskAsSpatialObject->Update();
    }

    typename MovingMaskSpatialObjectType::Pointer movingMaskAsSpatialObject; // default-constructed (null)
    if (this->m_MovingMask)
    {
      movingMaskAsSpatialObject = MovingMaskSpatialObjectType::New();
      movingMaskAsSpatialObject->SetImage(this->m_MovingMask);
      movingMaskAsSpatialObject->Update();
    }

    // Compute the image moments
    this->m_FixedCalculator->SetImage(this->m_FixedImage);
    this->m_FixedCalculator->SetSpatialObjectMask(fixedMaskAsSpatialObject);
    this->m_FixedCalculator->Compute();

    this->m_MovingCalculator->SetImage(this->m_MovingImage);
    this->m_MovingCalculator->SetSpatialObjectMask(movingMaskAsSpatialObject);
    this->m_MovingCalculator->Compute();

    // Get the center of gravities
    typename FixedImageCalculatorType::VectorType fixedCenter = this->m_FixedCalculator->GetCenterOfGravity();

    typename MovingImageCalculatorType::VectorType movingCenter = this->m_MovingCalculator->GetCenterOfGravity();

    // Compute the difference between the centers
    for (unsigned int i = 0; i < InputSpaceDimension; ++i)
    {
      translationVector[i] = movingCenter[i] - fixedCenter[i];
    }
  }
  else
  {
    // Align the geometrical centers of the fixed and moving image.
    // When masks are used the geometrical centers of the bounding box
    // of the masks are used.

    // Get fixed image (mask) information
    using FixedRegionType = typename FixedImageType::RegionType;
    FixedRegionType fixedRegion = this->m_FixedImage->GetLargestPossibleRegion();
    if (this->m_FixedMask)
    {
      auto fixedMaskAsSpatialObject = FixedMaskSpatialObjectType::New();
      fixedMaskAsSpatialObject->SetImage(this->m_FixedMask);
      fixedRegion = fixedMaskAsSpatialObject->ComputeMyBoundingBoxInIndexSpace();
    }

    // Compute center of the fixed image (mask bounding box) in physical units
    ContinuousIndex<double, InputSpaceDimension> fixedCenterCI;
    for (unsigned int k = 0; k < InputSpaceDimension; ++k)
    {
      fixedCenterCI[k] = fixedRegion.GetIndex()[k] + fixedRegion.GetSize()[k] / 2.0;
    }
    typename TransformType::InputPointType centerFixed;
    this->m_FixedImage->TransformContinuousIndexToPhysicalPoint(fixedCenterCI, centerFixed);

    // Get moving image (mask) information
    using MovingRegionType = typename MovingImageType::RegionType;
    MovingRegionType movingRegion = this->m_MovingImage->GetLargestPossibleRegion();
    if (this->m_MovingMask)
    {
      auto movingMaskAsSpatialObject = MovingMaskSpatialObjectType::New();
      movingMaskAsSpatialObject->SetImage(this->m_MovingMask);
      movingRegion = movingMaskAsSpatialObject->ComputeMyBoundingBoxInIndexSpace();
    }

    // Compute center of the moving image (mask bounding box) in physical units
    ContinuousIndex<double, InputSpaceDimension> movingCenterCI;
    for (unsigned int k = 0; k < InputSpaceDimension; ++k)
    {
      movingCenterCI[k] = movingRegion.GetIndex()[k] + movingRegion.GetSize()[k] / 2.0;
    }
    typename TransformType::InputPointType centerMoving;
    this->m_MovingImage->TransformContinuousIndexToPhysicalPoint(movingCenterCI, centerMoving);

    // Compute the difference between the centers
    for (unsigned int i = 0; i < InputSpaceDimension; ++i)
    {
      translationVector[i] = centerMoving[i] - centerFixed[i];
    }
  }

  // Initialize the transform
  this->m_Transform->SetOffset(translationVector);

} // end InitializeTransform()


/**
 * ************************* PrintSelf *********************
 */

template <class TTransform, class TFixedImage, class TMovingImage>
void
TranslationTransformInitializer<TTransform, TFixedImage, TMovingImage>::PrintSelf(std::ostream & os,
                                                                                  Indent         indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Transform   = " << std::endl;
  if (this->m_Transform)
  {
    os << indent << this->m_Transform << std::endl;
  }
  else
  {
    os << indent << "None" << std::endl;
  }

  os << indent << "FixedImage   = " << std::endl;
  if (this->m_FixedImage)
  {
    os << indent << this->m_FixedImage << std::endl;
  }
  else
  {
    os << indent << "None" << std::endl;
  }

  os << indent << "MovingImage   = " << std::endl;
  if (this->m_MovingImage)
  {
    os << indent << this->m_MovingImage << std::endl;
  }
  else
  {
    os << indent << "None" << std::endl;
  }

  os << indent << "MovingMomentCalculator   = " << std::endl;
  if (this->m_MovingCalculator)
  {
    os << indent << this->m_MovingCalculator << std::endl;
  }
  else
  {
    os << indent << "None" << std::endl;
  }

  os << indent << "FixedMomentCalculator   = " << std::endl;
  if (this->m_FixedCalculator)
  {
    os << indent << this->m_FixedCalculator << std::endl;
  }
  else
  {
    os << indent << "None" << std::endl;
  }

} // end PrintSelf()


} // namespace itk

#endif /* itkTranslationTransformInitializer_hxx */

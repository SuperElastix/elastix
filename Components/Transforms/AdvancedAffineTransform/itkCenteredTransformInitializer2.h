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
  Module:    $RCSfile: itkCenteredTransformInitializer2.h,v $
  Language:  C++
  Date:      $Date: 2010-07-04 10:30:49 $
  Version:   $Revision: 1.11 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkCenteredTransformInitializer2_h
#define itkCenteredTransformInitializer2_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkSpatialObject.h"
#include "itkAdvancedImageMomentsCalculator.h"

#include <iostream>

namespace itk
{

/** \class CenteredTransformInitializer2
 * \brief CenteredTransformInitializer2 is a helper class intended to
 * initialize the center of rotation and the translation of Transforms having
 * the center of rotation among their parameters.
 *
 * This class is connected to the fixed image, moving image and transform
 * involved in the registration. Three modes of operation are possible:
 *
 * - Geometrical,
 * - Center of mass,
 * - Origins,
 * - GeometryTop
 *
 * In the first mode, the geometrical center of the fixed image is passed as
 * initial center of rotation to the transform and the vector from the center
 * of the  fixed image to the center of the moving image is passed as the
 * initial translation. This mode basically assumes that the anatomical objects
 * to be registered are centered in their respective images. Hence the best
 * initial guess for the registration is the one that superimposes those two
 * centers.
 *
 * In the second mode, the moments of gray level values are computed
 * for both images. The center of mass of the moving image is then
 * used as center of rotation. The vector between the two centers of
 * mass is passes as the initial translation to the transform. This
 * second approach assumes that the moments of the anatomical objects
 * are similar for both images and hence the best initial guess for
 * registration is to superimpose both mass centers.  Note that this
 * assumption will probably not hold in multi-modality registration.
 *
 * In the third mode, the vector from the coordinates (0,0,0) of the
 * fixed image to the coordinates (0,0,0) of the moving image is passed as the
 * initial translation T and the geometrical center of the
 * moving image, translated by inv(T), is passed as initial center of rotation to the transform.
 *
 * In the fourth mode, the world coordinates of the eight corner points of both
 * images are determined. For both images, the minimum of the elements of world coordinates
 * is taken and the initial translation is taken to be the vector pointing from the minimum
 * coordinates of the fixed image to the minimum coordinates of the moving image. The
 * rotation point is set to the center of the fixed image. Note that this method does not make
 * sense for 2D images, in that case the result will be equal to Geometrical method.
 *
 * \ingroup Transforms
 */
template <class TTransform, class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT CenteredTransformInitializer2 : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CenteredTransformInitializer2);

  /** Standard class typedefs. */
  using Self = CenteredTransformInitializer2;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CenteredTransformInitializer2, Object);

  /** Type of the transform to initialize */
  using TransformType = TTransform;
  using TransformPointer = typename TransformType::Pointer;

  /** Dimension of parameters. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, TransformType::InputSpaceDimension);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, TransformType::OutputSpaceDimension);

  /** Image Types to use in the initialization of the transform */
  using FixedImageType = TFixedImage;
  using MovingImageType = TMovingImage;

  using FixedImagePointer = typename FixedImageType::ConstPointer;
  using MovingImagePointer = typename MovingImageType::ConstPointer;

  using FixedImageMaskType = Image<unsigned char, InputSpaceDimension>;
  using MovingImageMaskType = Image<unsigned char, OutputSpaceDimension>;
  using FixedImageMaskPointer = typename FixedImageMaskType::ConstPointer;
  using MovingImageMaskPointer = typename MovingImageMaskType::ConstPointer;

  /** Moment calculators */
  using FixedImageCalculatorType = AdvancedImageMomentsCalculator<FixedImageType>;
  using MovingImageCalculatorType = AdvancedImageMomentsCalculator<MovingImageType>;

  using FixedImageCalculatorPointer = typename FixedImageCalculatorType::Pointer;
  using MovingImageCalculatorPointer = typename MovingImageCalculatorType::Pointer;

  /** Offset type. */
  using OffsetType = typename TransformType::OffsetType;

  /** Point type. */
  using InputPointType = typename TransformType::InputPointType;

  /** Vector type. */
  using OutputVectorType = typename TransformType::OutputVectorType;

  using InputPixelType = typename FixedImageType::PixelType;

  /** Set the transform to be initialized */
  itkSetObjectMacro(Transform, TransformType);

  /** Set the fixed image used in the registration process */
  itkSetConstObjectMacro(FixedImage, FixedImageType);

  /** Set the moving image used in the registration process */
  itkSetConstObjectMacro(MovingImage, MovingImageType);

  /** Mask support. */
  itkSetConstObjectMacro(FixedImageMask, FixedImageMaskType);
  itkSetConstObjectMacro(MovingImageMask, MovingImageMaskType);

  /** Settings for MomentsCalculator. */
  itkSetMacro(NumberOfSamplesForCenteredTransformInitialization, SizeValueType);
  itkSetMacro(LowerThresholdForCenterGravity, InputPixelType);
  itkSetMacro(CenterOfGravityUsesLowerThreshold, bool);

  /** Initialize the transform using data from the images */
  virtual void
  InitializeTransform();

  /** Select between using the geometrical center of the images or
      using the center of mass given by the image intensities. */
  void
  GeometryOn()
  {
    m_UseMoments = false;
    m_UseOrigins = false;
    m_UseTop = false;
  }
  void
  MomentsOn()
  {
    m_UseMoments = true;
    m_UseOrigins = false;
    m_UseTop = false;
  }
  void
  OriginsOn()
  {
    m_UseMoments = false;
    m_UseOrigins = true;
    m_UseTop = false;
  }
  void
  GeometryTopOn()
  {
    m_UseMoments = false;
    m_UseOrigins = false;
    m_UseTop = true;
  }

  /** Get() access to the moments calculators */
  itkGetConstObjectMacro(FixedCalculator, FixedImageCalculatorType);
  itkGetConstObjectMacro(MovingCalculator, MovingImageCalculatorType);

protected:
  CenteredTransformInitializer2();
  ~CenteredTransformInitializer2() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  itkGetModifiableObjectMacro(Transform, TransformType);

  /** Settings for MomentsCalculator. */
  SizeValueType  m_NumberOfSamplesForCenteredTransformInitialization;
  InputPixelType m_LowerThresholdForCenterGravity;
  bool           m_CenterOfGravityUsesLowerThreshold;

private:
  TransformPointer m_Transform;

  FixedImagePointer      m_FixedImage;
  MovingImagePointer     m_MovingImage;
  FixedImageMaskPointer  m_FixedImageMask;
  MovingImageMaskPointer m_MovingImageMask;

  bool m_UseMoments;
  bool m_UseOrigins;
  bool m_UseTop;

  FixedImageCalculatorPointer  m_FixedCalculator;
  MovingImageCalculatorPointer m_MovingCalculator;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCenteredTransformInitializer2.hxx"
#endif

#endif /* itkCenteredTransformInitializer2_h */

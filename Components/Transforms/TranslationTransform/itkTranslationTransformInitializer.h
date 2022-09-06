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

#ifndef itkTranslationTransformInitializer_h
#define itkTranslationTransformInitializer_h

// ITK header files:
#include <itkImageMomentsCalculator.h>
#include <itkObject.h>

#include <iostream>

namespace itk
{

/**
 * \class TranslationTransformInitializer
 *
 * \brief TranslationTransformInitializer is a helper class intended to
 * initialize the translation of a TranslationTransforms
 *
 * This class is connected to the fixed image, moving image and transform
 * involved in the registration. Two modes of operation are possible:
 *
 * - Geometrical,
 * - Center of mass
 *
 * In the first mode, the geometrical centers of the images are computed.
 * The distance between them is set as an initial translation. This mode
 * basically assumes that the anatomical objects
 * to be registered are centered in their respective images. Hence the best
 * initial guess for the registration is the one that superimposes those two
 * centers.
 *
 * In the second mode, the moments of gray level values are computed
 * for both images. The vector between the two centers of
 * mass is passes as the initial translation to the transform. This
 * second approach assumes that the moments of the anatomical objects
 * are similar for both images and hence the best initial guess for
 * registration is to superimpose both mass centers.  Note that this
 * assumption will probably not hold in multi-modality registration.
 *
 * \ingroup Transforms
 */
template <class TTransform, class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT TranslationTransformInitializer : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TranslationTransformInitializer);

  /** Standard class typedefs. */
  using Self = TranslationTransformInitializer;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TranslationTransformInitializer, Object);

  /** Type of the transform to initialize */
  using TransformType = TTransform;
  using TransformPointer = typename TransformType::Pointer;

  /** Dimension of parameters. */
  itkStaticConstMacro(SpaceDimension, unsigned int, TransformType::SpaceDimension);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, TransformType::InputSpaceDimension);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, TransformType::OutputSpaceDimension);

  /** Image Types to use in the initialization of the transform */
  using FixedImageType = TFixedImage;
  using MovingImageType = TMovingImage;
  using FixedImagePointer = typename FixedImageType::ConstPointer;
  using MovingImagePointer = typename MovingImageType::ConstPointer;
  using FixedMaskType = Image<unsigned char, InputSpaceDimension>;
  using MovingMaskType = Image<unsigned char, OutputSpaceDimension>;
  using FixedMaskPointer = typename FixedMaskType::ConstPointer;
  using MovingMaskPointer = typename MovingMaskType::ConstPointer;

  /** Moment calculators */
  using FixedImageCalculatorType = ImageMomentsCalculator<FixedImageType>;
  using MovingImageCalculatorType = ImageMomentsCalculator<MovingImageType>;

  using FixedImageCalculatorPointer = typename FixedImageCalculatorType::Pointer;
  using MovingImageCalculatorPointer = typename MovingImageCalculatorType::Pointer;

  /** Point type. */
  using InputPointType = typename TransformType::InputPointType;

  /** Vector type. */
  using OutputVectorType = typename TransformType::OutputVectorType;

  /** Set the transform to be initialized */
  itkSetObjectMacro(Transform, TransformType);

  /** Set the fixed image used in the registration process */
  itkSetConstObjectMacro(FixedImage, FixedImageType);

  /** Set the moving image used in the registration process */
  itkSetConstObjectMacro(MovingImage, MovingImageType);

  /** Set the fixed image mask used in the registration process */
  itkSetConstObjectMacro(FixedMask, FixedMaskType);

  /** Set the moving image mask used in the registration process */
  itkSetConstObjectMacro(MovingMask, MovingMaskType);

  /** Initialize the transform using data from the images */
  virtual void
  InitializeTransform() const;

  /** Select between using the geometrical center of the images or
      using the center of mass given by the image intensities. */
  void
  GeometryOn()
  {
    m_UseMoments = false;
  }
  void
  MomentsOn()
  {
    m_UseMoments = true;
  }

  /** Get() access to the moments calculators */
  itkGetConstObjectMacro(FixedCalculator, FixedImageCalculatorType);
  itkGetConstObjectMacro(MovingCalculator, MovingImageCalculatorType);

protected:
  TranslationTransformInitializer();
  ~TranslationTransformInitializer() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  TransformPointer   m_Transform;
  FixedImagePointer  m_FixedImage;
  MovingImagePointer m_MovingImage;
  FixedMaskPointer   m_FixedMask;
  MovingMaskPointer  m_MovingMask;
  bool               m_UseMoments;

  FixedImageCalculatorPointer  m_FixedCalculator;
  MovingImageCalculatorPointer m_MovingCalculator;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTranslationTransformInitializer.hxx"
#endif

#endif /* itkTranslationTransformInitializer_h */

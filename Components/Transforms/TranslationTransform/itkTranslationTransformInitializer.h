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

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkImageMomentsCalculator.h"

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
  /** Standard class typedefs. */
  typedef TranslationTransformInitializer Self;
  typedef Object                          Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TranslationTransformInitializer, Object);

  /** Type of the transform to initialize */
  typedef TTransform                      TransformType;
  typedef typename TransformType::Pointer TransformPointer;

  /** Dimension of parameters. */
  itkStaticConstMacro(SpaceDimension, unsigned int, TransformType::SpaceDimension);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, TransformType::InputSpaceDimension);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, TransformType::OutputSpaceDimension);

  /** Image Types to use in the initialization of the transform */
  typedef TFixedImage                                FixedImageType;
  typedef TMovingImage                               MovingImageType;
  typedef typename FixedImageType::ConstPointer      FixedImagePointer;
  typedef typename MovingImageType::ConstPointer     MovingImagePointer;
  typedef Image<unsigned char, InputSpaceDimension>  FixedMaskType;
  typedef Image<unsigned char, OutputSpaceDimension> MovingMaskType;
  typedef typename FixedMaskType::ConstPointer       FixedMaskPointer;
  typedef typename MovingMaskType::ConstPointer      MovingMaskPointer;

  /** Moment calculators */
  typedef ImageMomentsCalculator<FixedImageType>  FixedImageCalculatorType;
  typedef ImageMomentsCalculator<MovingImageType> MovingImageCalculatorType;

  typedef typename FixedImageCalculatorType::Pointer  FixedImageCalculatorPointer;
  typedef typename MovingImageCalculatorType::Pointer MovingImageCalculatorPointer;

  /** Point type. */
  typedef typename TransformType::InputPointType InputPointType;

  /** Vector type. */
  typedef typename TransformType::OutputVectorType OutputVectorType;

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
  TranslationTransformInitializer(const Self &) = delete;
  void
  operator=(const Self &) = delete;

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

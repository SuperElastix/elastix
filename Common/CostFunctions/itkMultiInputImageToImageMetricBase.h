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
#ifndef itkMultiInputImageToImageMetricBase_h
#define itkMultiInputImageToImageMetricBase_h

#include "itkAdvancedImageToImageMetric.h"
#include <vector>

/** Macro for setting the number of objects. */
#define itkSetNumberOfMacro(name)                                                                                      \
  virtual void SetNumberOf##name##s(const unsigned int _arg)                                                           \
  {                                                                                                                    \
    if (this->m_NumberOf##name##s != _arg)                                                                             \
    {                                                                                                                  \
      this->m_##name##Vector.resize(_arg);                                                                             \
      this->m_NumberOf##name##s = _arg;                                                                                \
      this->Modified();                                                                                                \
    }                                                                                                                  \
  } // comments for allowing ; after calling the macro

namespace itk
{

/** \class MultiInputImageToImageMetricBase
 *
 * \brief Implements a metric base class that takes multiple inputs.
 *
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT MultiInputImageToImageMetricBase
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiInputImageToImageMetricBase);

  /** Standard class typedefs. */
  using Self = MultiInputImageToImageMetricBase;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiInputImageToImageMetricBase, AdvancedImageToImageMetric);

  /** Constants for the image dimensions */
  itkStaticConstMacro(MovingImageDimension, unsigned int, TMovingImage::ImageDimension);
  itkStaticConstMacro(FixedImageDimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImagePointer;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImagePointer;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::GradientImageFilterType;
  using typename Superclass::GradientImageFilterPointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::ParametersType;

  using FixedImageInterpolatorType = InterpolateImageFunction<FixedImageType, CoordinateRepresentationType>;
  using FixedImageInterpolatorPointer = typename FixedImageInterpolatorType::Pointer;

  /** Typedef's for storing multiple inputs. */
  using FixedImageVectorType = std::vector<FixedImageConstPointer>;
  using FixedImageMaskVectorType = std::vector<FixedImageMaskPointer>;
  using FixedImageRegionVectorType = std::vector<FixedImageRegionType>;
  using MovingImageVectorType = std::vector<MovingImageConstPointer>;
  using MovingImageMaskVectorType = std::vector<MovingImageMaskPointer>;
  using InterpolatorVectorType = std::vector<InterpolatorPointer>;
  using FixedImageInterpolatorVectorType = std::vector<FixedImageInterpolatorPointer>;

  /** ******************** Fixed images ******************** */

  /** Set the fixed images. */
  virtual void
  SetFixedImage(const FixedImageType * _arg, unsigned int pos);

  /** Set the first fixed image. */
  void
  SetFixedImage(const FixedImageType * _arg) override
  {
    this->SetFixedImage(_arg, 0);
  }


  /** Get the fixed images. */
  virtual const FixedImageType *
  GetFixedImage(unsigned int pos) const;

  /** Get the first fixed image. */
  const FixedImageType *
  GetFixedImage() const override
  {
    return this->GetFixedImage(0);
  }


  /** Set the number of fixed images. */
  itkSetNumberOfMacro(FixedImage);

  /** Get the number of fixed images. */
  itkGetConstMacro(NumberOfFixedImages, unsigned int);

  /** ******************** Fixed image masks ******************** */

  /** Set the fixed image masks. */
  virtual void
  SetFixedImageMask(FixedImageMaskType * _arg, unsigned int pos);

  /** Set the first fixed image mask. */
  void
  SetFixedImageMask(FixedImageMaskType * _arg) override
  {
    this->SetFixedImageMask(_arg, 0);
  }


  /** Get the fixed image masks. */
  virtual FixedImageMaskType *
  GetFixedImageMask(unsigned int pos) const;

  /** Get the first fixed image mask. */
  FixedImageMaskType *
  GetFixedImageMask() const override
  {
    return this->GetFixedImageMask(0);
  }


  /** Set the number of fixed image masks. */
  itkSetNumberOfMacro(FixedImageMask);

  /** Get the number of fixed image masks. */
  itkGetConstMacro(NumberOfFixedImageMasks, unsigned int);

  /** ******************** Fixed image regions ******************** */

  /** Set the fixed image regions. */
  virtual void
  SetFixedImageRegion(const FixedImageRegionType _arg, unsigned int pos);

  /** Set the first fixed image region. */
  void
  SetFixedImageRegion(const FixedImageRegionType _arg) override
  {
    this->SetFixedImageRegion(_arg, 0);
  }


  /** Get the fixed image regions. */
  virtual const FixedImageRegionType &
  GetFixedImageRegion(unsigned int pos) const;

  /** Get the first fixed image region. */
  const FixedImageRegionType &
  GetFixedImageRegion() const override
  {
    return this->GetFixedImageRegion(0);
  }


  /** Set the number of fixed image regions. */
  itkSetNumberOfMacro(FixedImageRegion);

  /** Get the number of fixed image regions. */
  itkGetConstMacro(NumberOfFixedImageRegions, unsigned int);

  /** ******************** Moving images ******************** */

  /** Set the moving images. */
  virtual void
  SetMovingImage(const MovingImageType * _arg, unsigned int pos);

  /** Set the first moving image. */
  void
  SetMovingImage(const MovingImageType * _arg) override
  {
    this->SetMovingImage(_arg, 0);
  }


  /** Get the moving images. */
  virtual const MovingImageType *
  GetMovingImage(unsigned int pos) const;

  /** Get the first moving image. */
  const MovingImageType *
  GetMovingImage() const override
  {
    return this->GetMovingImage(0);
  }


  /** Set the number of moving images. */
  itkSetNumberOfMacro(MovingImage);

  /** Get the number of moving images. */
  itkGetConstMacro(NumberOfMovingImages, unsigned int);

  /** ******************** Moving image masks ******************** */

  /** Set the moving image masks. */
  virtual void
  SetMovingImageMask(MovingImageMaskType * _arg, unsigned int pos);

  /** Set the first moving image mask. */
  void
  SetMovingImageMask(MovingImageMaskType * _arg) override
  {
    this->SetMovingImageMask(_arg, 0);
  }


  /** Get the moving image masks. */
  virtual MovingImageMaskType *
  GetMovingImageMask(unsigned int pos) const;

  /** Get the first moving image mask. */
  MovingImageMaskType *
  GetMovingImageMask() const override
  {
    return this->GetMovingImageMask(0);
  }


  /** Set the number of moving image masks. */
  itkSetNumberOfMacro(MovingImageMask);

  /** Get the number of moving image masks. */
  itkGetConstMacro(NumberOfMovingImageMasks, unsigned int);

  /** ******************** Interpolators ********************
   * These interpolators are used for the moving images.
   */

  /** Set the interpolators. */
  virtual void
  SetInterpolator(InterpolatorType * _arg, unsigned int pos);

  /** Set the first interpolator. */
  void
  SetInterpolator(InterpolatorType * _arg) override
  {
    return this->SetInterpolator(_arg, 0);
  }


  /** Get the interpolators. */
  virtual InterpolatorType *
  GetInterpolator(unsigned int pos) const;

  /** Get the first interpolator. */
  InterpolatorType *
  GetInterpolator() const override
  {
    return this->GetInterpolator(0);
  }


  /** Set the number of interpolators. */
  itkSetNumberOfMacro(Interpolator);

  /** Get the number of interpolators. */
  itkGetConstMacro(NumberOfInterpolators, unsigned int);

  /** A function to check if all moving image interpolators are of type B-spline. */
  itkGetConstMacro(InterpolatorsAreBSpline, bool);

  /** ******************** FixedImageInterpolators ********************
   * These interpolators are used for the fixed images.
   */

  /** Set the fixed image interpolators. */
  virtual void
  SetFixedImageInterpolator(FixedImageInterpolatorType * _arg, unsigned int pos);

  /** Set the first fixed image interpolator. */
  virtual void
  SetFixedImageInterpolator(FixedImageInterpolatorType * _arg)
  {
    return this->SetFixedImageInterpolator(_arg, 0);
  }


  /** Get the fixed image interpolators. */
  virtual FixedImageInterpolatorType *
  GetFixedImageInterpolator(unsigned int pos) const;

  /** Get the first fixed image interpolator. */
  virtual FixedImageInterpolatorType *
  GetFixedImageInterpolator() const
  {
    return this->GetFixedImageInterpolator(0);
  }


  /** Set the number of fixed image interpolators. */
  itkSetNumberOfMacro(FixedImageInterpolator);

  /** Get the number of fixed image interpolators. */
  itkGetConstMacro(NumberOfFixedImageInterpolators, unsigned int);

  /** ******************** Other public functions ******************** */

  /** Initialisation. */
  void
  Initialize() override;

protected:
  /** Constructor. */
  MultiInputImageToImageMetricBase() = default;

  /** Destructor. */
  ~MultiInputImageToImageMetricBase() override = default;

  /** Typedef's from the Superclass. */
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::MovingImageContinuousIndexType;

  /** Typedef's for the moving image interpolators. */
  using typename Superclass::BSplineInterpolatorType;
  using BSplineInterpolatorPointer = typename BSplineInterpolatorType::Pointer;
  using BSplineInterpolatorVectorType = std::vector<BSplineInterpolatorPointer>;

  /** Initialize variables related to the image sampler; called by Initialize. */
  void
  InitializeImageSampler() override;

  /** Check if all interpolators (for the moving image) are of type
   * BSplineInterpolateImageFunction.
   */
  virtual void
  CheckForBSplineInterpolators();

  /** Check if mappedPoint is inside all moving images.
   * If so, the moving image value and possibly derivative are computed.
   */
  bool
  EvaluateMovingImageValueAndDerivative(const MovingImagePointType & mappedPoint,
                                        RealType &                   movingImageValue,
                                        MovingImageDerivativeType *  gradient) const override;

  /** IsInsideMovingMask: Returns the AND of all moving image masks. */
  bool
  IsInsideMovingMask(const MovingImagePointType & mappedPoint) const override;

  /** Protected member variables. */
  FixedImageVectorType             m_FixedImageVector;
  FixedImageMaskVectorType         m_FixedImageMaskVector;
  FixedImageRegionVectorType       m_FixedImageRegionVector;
  MovingImageVectorType            m_MovingImageVector;
  MovingImageMaskVectorType        m_MovingImageMaskVector;
  InterpolatorVectorType           m_InterpolatorVector;
  FixedImageInterpolatorVectorType m_FixedImageInterpolatorVector;

  bool                          m_InterpolatorsAreBSpline{ false };
  BSplineInterpolatorVectorType m_BSplineInterpolatorVector;

private:
  /// Avoids accidentally calling `this->FastEvaluateMovingImageValueAndDerivative(mappedPoint, ..., threadId)`, when
  /// `*this` is derived from `MultiInputImageToImageMetricBase`. (The non-virtual member function
  /// `AdvancedImageToImageMetric::FastEvaluateMovingImageValueAndDerivative` does not entirely replace the
  /// `MultiInputImageToImageMetricBase::EvaluateMovingImageValueAndDerivative` override.)
  void
  FastEvaluateMovingImageValueAndDerivative(...) const = delete;

  /** Private member variables. */
  FixedImageRegionType m_DummyFixedImageRegion;

  unsigned int m_NumberOfFixedImages{ 0 };
  unsigned int m_NumberOfFixedImageMasks{ 0 };
  unsigned int m_NumberOfFixedImageRegions{ 0 };
  unsigned int m_NumberOfMovingImages{ 0 };
  unsigned int m_NumberOfMovingImageMasks{ 0 };
  unsigned int m_NumberOfInterpolators{ 0 };
  unsigned int m_NumberOfFixedImageInterpolators{ 0 };
};

} // end namespace itk

#undef itkSetNumberOfMacro

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiInputImageToImageMetricBase.hxx"
#endif

#endif // end #ifndef itkMultiInputImageToImageMetricBase_h

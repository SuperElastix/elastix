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
  Module:    $RCSfile: itkGradientDifferenceImageToImageMetric2.h,v $
  Language:  C++
  Date:      $Date: 2011-29-04 14:33 $
  Version:   $Revision: 2.0 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkGradientDifferenceImageToImageMetric2_h
#define itkGradientDifferenceImageToImageMetric2_h

#include "itkAdvancedImageToImageMetric.h"

#include "itkSobelOperator.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkPoint.h"
#include "itkCastImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkOptimizer.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedRayCastInterpolateImageFunction.h"

namespace itk
{
/** \class GradientDifferenceImageToImageMetric
 * \brief Computes similarity between two objects to be registered
 *
 * This Class is templated over the type of the Images to be compared and
 * over the type of transformation and Iterpolator to be used.
 *
 * This metric computes the sum of squared differences between pixels in
 * the derivatives of the moving and fixed images after passing the squared
 * difference through a function of type \f$ \frac{1}{1+x} \f$.
 *
 *
 * Spatial correspondance between both images is established through a
 * Transform. Pixel values are taken from the Moving image. Their positions
 * are mapped to the Fixed image and result in general in non-grid position
 * on it. Values at these non-grid position of the Fixed image are
 * interpolated using a user-selected Interpolator.
 *
 * Implementation of this class is based on:
 * Hipwell, J. H., et. al. (2003), "Intensity-Based 2-D-3D Registration of
 * Cerebral Angiograms,", IEEE Transactions on Medical Imaging,
 * 22(11):1417-1426.
 *
 * \ingroup RegistrationMetrics
 */
template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT GradientDifferenceImageToImageMetric
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef GradientDifferenceImageToImageMetric                  Self;
  typedef AdvancedImageToImageMetric<TFixedImage, TMovingImage> Superclass;

  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GradientDifferenceImageToImageMetric, ImageToImageMetric);

/** Types transferred from the base class */
/** Work around a Visual Studio .NET bug */
#if defined(_MSC_VER) && (_MSC_VER == 1300)
  typedef double RealType;
#else
  typedef typename Superclass::RealType RealType;
#endif

  typedef typename Superclass::TransformType           TransformType;
  typedef typename TransformType::ScalarType           ScalarType;
  typedef typename Superclass::TransformPointer        TransformPointer;
  typedef typename Superclass::TransformParametersType TransformParametersType;
  typedef typename Superclass::TransformJacobianType   TransformJacobianType;
  typedef typename Superclass::InterpolatorType        InterpolatorType;
  typedef typename InterpolatorType::Pointer           InterpolatorPointer;
  typedef typename Superclass::MeasureType             MeasureType;
  typedef typename Superclass::DerivativeType          DerivativeType;
  typedef typename Superclass::FixedImageType          FixedImageType;
  typedef typename Superclass::MovingImageType         MovingImageType;
  typedef typename Superclass::FixedImageConstPointer  FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer MovingImageConstPointer;
  typedef typename TFixedImage::PixelType              FixedImagePixelType;
  typedef typename TMovingImage::PixelType             MovedImagePixelType;
  typedef typename MovingImageType::RegionType         MovingImageRegionType;
  typedef typename itk::Optimizer                      OptimizerType;
  typedef typename OptimizerType::ScalesType           ScalesType;

  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovedImageDimension, unsigned int, MovingImageType::ImageDimension);

  typedef typename itk::AdvancedCombinationTransform<ScalarType, FixedImageDimension>  CombinationTransformType;
  typedef typename CombinationTransformType::Pointer                                   CombinationTransformPointer;
  typedef itk::Image<FixedImagePixelType, itkGetStaticConstMacro(FixedImageDimension)> TransformedMovingImageType;
  typedef itk::ResampleImageFilter<MovingImageType, TransformedMovingImageType>        TransformMovingImageFilterType;
  typedef typename itk::AdvancedRayCastInterpolateImageFunction<MovingImageType, ScalarType> RayCastInterpolatorType;
  typedef typename RayCastInterpolatorType::Pointer                                          RayCastInterpolatorPointer;
  typedef itk::Image<RealType, itkGetStaticConstMacro(FixedImageDimension)>                  FixedGradientImageType;
  typedef itk::CastImageFilter<FixedImageType, FixedGradientImageType>                       CastFixedImageFilterType;
  typedef typename CastFixedImageFilterType::Pointer                               CastFixedImageFilterPointer;
  typedef typename FixedGradientImageType::PixelType                               FixedGradientPixelType;
  typedef itk::Image<RealType, itkGetStaticConstMacro(MovedImageDimension)>        MovedGradientImageType;
  typedef itk::CastImageFilter<TransformedMovingImageType, MovedGradientImageType> CastMovedImageFilterType;
  typedef typename CastMovedImageFilterType::Pointer                               CastMovedImageFilterPointer;
  typedef typename MovedGradientImageType::PixelType                               MovedGradientPixelType;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & derivative) const override;

  /**  Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /**  Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                derivative) const override;

  void
  Initialize(void) override;

  /** Write gradient images to a files for debugging purposes. */
  void
  WriteGradientImagesToFiles(void) const;

  /** Set/Get Scales  */
  itkSetMacro(Scales, ScalesType);
  itkGetConstReferenceMacro(Scales, ScalesType);

  /** Set/Get the value of Delta used for computing derivatives by finite
   * differences in the GetDerivative() method */
  itkSetMacro(DerivativeDelta, double);
  itkGetConstReferenceMacro(DerivativeDelta, double);

protected:
  GradientDifferenceImageToImageMetric();
  ~GradientDifferenceImageToImageMetric() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Compute the range of the moved image gradients. */
  void
  ComputeMovedGradientRange(void) const;

  /** Compute the variance and range of the moving image gradients. */
  void
  ComputeVariance(void) const;

  /** Compute the similarity measure using a specified subtraction factor. */
  MeasureType
  ComputeMeasure(const TransformParametersType & parameters, const double * subtractionFactor) const;

  typedef NeighborhoodOperatorImageFilter<FixedGradientImageType, FixedGradientImageType> FixedSobelFilter;

  typedef NeighborhoodOperatorImageFilter<MovedGradientImageType, MovedGradientImageType> MovedSobelFilter;

private:
  GradientDifferenceImageToImageMetric(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** The variance of the moving image gradients. */
  mutable MovedGradientPixelType m_Variance[FixedImageDimension];

  /** The range of the moving image gradients. */
  mutable MovedGradientPixelType m_MinMovedGradient[MovedImageDimension];
  mutable MovedGradientPixelType m_MaxMovedGradient[MovedImageDimension];

  /** The range of the fixed image gradients. */
  mutable FixedGradientPixelType m_MinFixedGradient[FixedImageDimension];
  mutable FixedGradientPixelType m_MaxFixedGradient[FixedImageDimension];

  /** The filter for transforming the moving image. */
  typename TransformMovingImageFilterType::Pointer m_TransformMovingImageFilter;

  /** The Sobel gradients of the fixed image */
  CastFixedImageFilterPointer m_CastFixedImageFilter;

  SobelOperator<FixedGradientPixelType, itkGetStaticConstMacro(FixedImageDimension)>
    m_FixedSobelOperators[FixedImageDimension];

  typename FixedSobelFilter::Pointer m_FixedSobelFilters[itkGetStaticConstMacro(FixedImageDimension)];

  ZeroFluxNeumannBoundaryCondition<MovedGradientImageType> m_MovedBoundCond;
  ZeroFluxNeumannBoundaryCondition<FixedGradientImageType> m_FixedBoundCond;

  /** The Sobel gradients of the moving image */
  CastMovedImageFilterPointer m_CastMovedImageFilter;

  SobelOperator<MovedGradientPixelType, itkGetStaticConstMacro(MovedImageDimension)>
    m_MovedSobelOperators[MovedImageDimension];

  typename MovedSobelFilter::Pointer m_MovedSobelFilters[itkGetStaticConstMacro(MovedImageDimension)];

  ScalesType                  m_Scales;
  double                      m_DerivativeDelta;
  double                      m_Rescalingfactor;
  CombinationTransformPointer m_CombinationTransform;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGradientDifferenceImageToImageMetric2.hxx"
#endif

#endif

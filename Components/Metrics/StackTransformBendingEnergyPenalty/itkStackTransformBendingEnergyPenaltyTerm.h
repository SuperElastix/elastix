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
#ifndef __itkStackTransformBendingEnergyPenaltyTerm_h
#define __itkStackTransformBendingEnergyPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"
#include "itkImageGridSampler.h"

#include "itkStackTransform.h"

namespace itk
{

/**
 * \class StackTransformBendingEnergyPenalty
 * \brief A penalty term based on the bending energy of a thin metal sheet.
 *
 *
 * [1]: D. Rueckert, L. I. Sonoda, C. Hayes, D. L. G. Hill,
 *      M. O. Leach, and D. J. Hawkes, "Nonrigid registration
 *      using free-form deformations: Application to breast MR
 *      images", IEEE Trans. Med. Imaging 18, 712-721, 1999.\n
 * [2]: M. Staring and S. Klein,
 *      "Itk::Transforms supporting spatial derivatives"",
 *      Insight Journal, http://hdl.handle.net/10380/3215.
 * [3]: M. Polfliet, et al. "Intrasubject multimodal groupwise
 *      registration with the conditional template entropy."
 *      Medical image analysis 46 (2018): 15-25.\n
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "StackTransformBendingEnergyPenalty")</tt>
 *
 * \ingroup Metrics
 *
 */

template< class TFixedImage, class TScalarType >
class StackTransformBendingEnergyPenaltyTerm :
  public TransformPenaltyTerm< TFixedImage, TScalarType >
{
public:

  /** Standard ITK stuff. */
  typedef StackTransformBendingEnergyPenaltyTerm Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TScalarType >                  Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StackTransformBendingEnergyPenaltyTerm, TransformPenaltyTerm );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer           MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename Superclass::FixedImagePointer            FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;
  typedef typename FixedImageType::SizeType                 FixedImageSizeType;
  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass::NumberOfParametersType       NumberOfParametersType;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef typename Superclass::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass::RealType                     RealType;
  typedef typename Superclass::GradientPixelType            GradientPixelType;
  typedef typename Superclass::GradientImageType            GradientImageType;
  typedef typename Superclass::GradientImagePointer         GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType      GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer   GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType           FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer        FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType          MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer       MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename Superclass::DerivativeValueType          DerivativeValueType;
  typedef typename Superclass::ParametersType               ParametersType;
  typedef typename Superclass::FixedImagePixelType          FixedImagePixelType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::ScalarType                   ScalarType;
  typedef typename Superclass::ThreaderType                 ThreaderType;
  typedef typename Superclass::ThreadInfoType               ThreadInfoType;

  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );

  itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

  itkStaticConstMacro( ReducedFixedImageDimension, unsigned int, FixedImageType::ImageDimension - 1 );

  itkStaticConstMacro( ReducedMovingImageDimension, unsigned int, MovingImageType::ImageDimension - 1 );

  typedef itk::StackTransform< ScalarType, FixedImageDimension, MovingImageDimension > StackTransformType;
  typedef typename Superclass::BSplineTransformType                                    BSplineTransformType;
  typedef typename Superclass::CombinationTransformType                                CombinationTransformType;

  typedef typename Superclass::SpatialJacobianType           SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType            SpatialHessianType;
  typedef typename Superclass::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType            InternalMatrixType;
  typedef typename Superclass::HessianValueType              HessianValueType;
  typedef typename Superclass::HessianType                   HessianType;

  virtual MeasureType GetValue(
    const ParametersType & parameters ) const;

  virtual void GetDerivative(
    const ParametersType & parameters,
    DerivativeType & derivative ) const;

  virtual void GetValueAndDerivativeSingleThreaded(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  inline void ThreadedGetValueAndDerivative(
    ThreadIdType threadID );

  inline void AfterThreadedGetValueAndDerivative(
    MeasureType & value,
    DerivativeType & derivative ) const;

  itkSetMacro( SubtractMean, bool );
  itkSetMacro( TransformIsStackTransform, bool );
  itkSetMacro( TransformIsBSpline, bool );
  itkSetMacro( SubTransformIsBSpline, bool );
  itkSetMacro( GridSize, FixedImageSizeType );

protected:

  typedef typename Superclass::FixedImageIndexType                                           FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType                                      FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType                                          MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType                                           FixedImagePointType;
  typedef typename Superclass::MovingImagePointType                                          MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType                                MovingImageContinuousIndexType;
  typedef typename Superclass::NonZeroJacobianIndicesType                                    NonZeroJacobianIndicesType;
  typedef typename itk::ContinuousIndex< CoordinateRepresentationType, FixedImageDimension > FixedImageContinuousIndexType;

  void SampleRandom( const int n, const int m, std::vector< int > & numbers ) const;

  StackTransformBendingEnergyPenaltyTerm();

  virtual ~StackTransformBendingEnergyPenaltyTerm() {}

private:

  StackTransformBendingEnergyPenaltyTerm( const Self & ); // purposely not implemented
  void operator=( const Self & );                         // purposely not implemented

  bool                       m_TransformIsStackTransform;
  bool                       m_SubTransformIsBSpline;
  bool                       m_TransformIsBSpline;
  bool                       m_SubtractMean;

  FixedImageSizeType m_GridSize;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStackTransformBendingEnergyPenaltyTerm.hxx"
#endif

#endif // #ifndef __itkStackTransformBendingEnergyPenaltyTerm_h

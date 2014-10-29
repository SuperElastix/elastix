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
#ifndef __elxViolaWellsMutualInformationMetric_H__
#define __elxViolaWellsMutualInformationMetric_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMutualInformationImageToImageMetric.h"


namespace elastix
{

/**
 * \class ViolaWellsMutualInformationMetric
 * \brief A metric based on the itk::MutualInformationImageToImageMetric.
 *
 * \warning: this metric is not very well tested in elastix.
 * \warning: this metric is not based on the AdvancedImageToImageMetric so
 * does not support the ImageSampler framework and might be very slow in
 * combination with B-spline transform.
 * \warning: this metric uses stochastic sampling of the images. Do not use
 * a quasi-Newton optimizer or a conjugate gradient. The StandardGradientDescent
 * is a better choice.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "ViolaWellsMutualInformation")</tt>
 * \parameter NumberOfSpatialSamples: for each resolution the number of samples
 *    used to calculate this metrics value and its derivative. \n
 *    example: <tt>(NumberOfSpatialSamples 5000 5000 10000)</tt> \n
 *    The default is 10000 for each resolution.
 * \parameter FixedImageStandardDeviation: for each resolution the standard
 *    deviation of the fixed image. \n
 *    example: <tt>(FixedImageStandardDeviation 1.3 1.9 1.0)</tt> \n
 *    The default is 0.4 for each resolution.
 * \parameter MovingImageStandardDeviation: for each resolution the standard
 *    deviation of the moving image. \n
 *    example: <tt>(MovingImageStandardDeviation 1.3 1.9 1.0)</tt> \n
 *    The default is 0.4 for each resolution.
 *
 * \sa MutualInformationImageToImageMetric
 * \ingroup Metrics
 */

template< class TElastix >
class ViolaWellsMutualInformationMetric :
  public
  itk::MutualInformationImageToImageMetric<
  typename MetricBase< TElastix >::FixedImageType,
  typename MetricBase< TElastix >::MovingImageType >,
  public MetricBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef ViolaWellsMutualInformationMetric Self;
  typedef itk::MutualInformationImageToImageMetric<
    typename MetricBase< TElastix >::FixedImageType,
    typename MetricBase< TElastix >::MovingImageType >    Superclass1;
  typedef MetricBase< TElastix >          Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ViolaWellsMutualInformationMetric,
    itk::MutualInformationImageToImageMetric );

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "ViolaWellsMutualInformation")</tt>\n
   */
  elxClassNameMacro( "ViolaWellsMutualInformation" );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::TransformType            TransformType;
  typedef typename Superclass1::TransformPointer         TransformPointer;
  typedef typename Superclass1::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass1::InterpolatorType         InterpolatorType;
  typedef typename Superclass1::MeasureType              MeasureType;
  typedef typename Superclass1::DerivativeType           DerivativeType;
  typedef typename Superclass1::ParametersType           ParametersType;
  typedef typename Superclass1::FixedImageType           FixedImageType;
  typedef typename Superclass1::MovingImageType          MovingImageType;
  typedef typename Superclass1::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass1::MovingImageConstPointer  MovingImageCosntPointer;
  typedef typename Superclass1::FixedImageIndexType      FixedImageIndexType;
  typedef typename Superclass1::FixedImageIndexValueType FixedImageIndexValueType;
  typedef typename Superclass1::MovingImageIndexType     MovingImageIndexType;
  typedef typename Superclass1::FixedImagePointType      FixedImagePointType;
  typedef typename Superclass1::MovingImagePointType     MovingImagePointType;

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Typedef's inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Execute stuff before each new pyramid resolution:
   * \li Set the number of spatial samples.
   * \li Set the standard deviation of the fixed image.
   * \li Set the standard deviation of the moving image.
   */
  virtual void BeforeEachResolution( void );

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  virtual void Initialize( void ) throw ( itk::ExceptionObject );

protected:

  /** The constructor. */
  ViolaWellsMutualInformationMetric();
  /** The destructor. */
  virtual ~ViolaWellsMutualInformationMetric() {}

private:

  /** The private constructor. */
  ViolaWellsMutualInformationMetric( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );                     // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxViolaWellsMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __elxViolaWellsMutualInformationMetric_H__

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
#ifndef __elxActiveRegistrationModelIntensityMetric_h__
#define __elxActiveRegistrationModelIntensityMetric_h__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkActiveRegistrationModelIntensityMetric.h"

#include "itkDirectory.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkStatisticalModel.h"
#include "itkStandardImageRepresenter.h"
#include "itkPCAModelBuilder.h"
#include "itkReducedVarianceModelBuilder.h"
#include "itkStatismoIO.h"

namespace elastix
{

/**
 * \class AdvancedMeanSquaresMetric
 * \brief An metric based on the itk::AdvancedMeanSquaresImageToImageMetric.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "AdvancedMeanSquares")</tt>
 * \parameter UseNormalization: Bool to use normalization or not.\n
 *    If true, the MeanSquares is divided by a factor (range/10)^2,
 *    where range represents the maximum gray value range of the images.\n
 *    <tt>(UseNormalization "true")</tt>\n
 *    The default value is false.
 *
 * \ingroup Metrics
 *
 */

template< class TElastix >
class ActiveRegistrationModelIntensityMetric :
  public
  itk::ActiveRegistrationModelIntensityMetric<
  typename MetricBase< TElastix >::FixedImageType,
  typename MetricBase< TElastix >::MovingImageType >,
  public MetricBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef ActiveRegistrationModelIntensityMetric Self;
  typedef itk::ActiveRegistrationModelIntensityMetric<
    typename MetricBase< TElastix >::FixedImageType,
    typename MetricBase< TElastix >::MovingImageType >    Superclass1;
  typedef MetricBase< TElastix >          Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(ActiveRegistrationModelIntensityMetric, itk::ImageIntensityMetric );

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "AdvancedMeanSquares")</tt>\n
   */
  elxClassNameMacro( "ActiveRegistrationModelIntensityMetric" );

  /** Typedefs from the superclass. */
  typedef typename
    Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::MovingImageType            MovingImageType;
  typedef typename Superclass1::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass1::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass1::FixedImageType             FixedImageType;
  typedef typename Superclass1::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass1::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass1::TransformType              TransformType;
  typedef typename Superclass1::TransformPointer           TransformPointer;
  typedef typename Superclass1::InputPointType             InputPointType;
  typedef typename Superclass1::OutputPointType            OutputPointType;
  typedef typename Superclass1::TransformParametersType    TransformParametersType;
  typedef typename Superclass1::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass1::InterpolatorType           InterpolatorType;
  typedef typename Superclass1::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass1::RealType                   RealType;
  typedef typename Superclass1::GradientPixelType          GradientPixelType;
  typedef typename Superclass1::GradientImageType          GradientImageType;
  typedef typename Superclass1::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass1::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass1::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass1::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType                MeasureType;
  typedef typename Superclass1::DerivativeType             DerivativeType;
  typedef typename Superclass1::ParametersType             ParametersType;
  typedef typename Superclass1::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass1::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass1::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass1::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass1::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass1::ImageSampleContainerPointer ImageSampleContainerPointer;
  typedef typename Superclass1::FixedImageLimiterType  FixedImageLimiterType;
  typedef typename Superclass1::MovingImageLimiterType MovingImageLimiterType;
  typedef typename
    Superclass1::FixedImageLimiterOutputType FixedImageLimiterOutputType;
  typedef typename
    Superclass1::MovingImageLimiterOutputType MovingImageLimiterOutputType;
  typedef typename
    Superclass1::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

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

  typedef typename Superclass1::FixedImagePointType                           FixedImagePointType;
  typedef typename Superclass1::MovingImagePointType                          MovingImagePointType;

  typedef typename Superclass1::StatisticalModelImageType                     StatisticalModelImageType;
  typedef typename StatisticalModelImageType::Pointer                         StatisticalModelImagePointer;

  typedef typename itk::ImageFileReader< StatisticalModelImageType >          ImageReaderType;
  typedef typename ImageReaderType::Pointer                                   ImageReaderPointer;

  typedef typename Superclass1::StatisticalModelIdType                        StatisticalModelIdType;
  typedef typename Superclass1::StatisticalModelPointer                       StatisticalModelPointer;

  typedef typename Superclass1::StatisticalModelVectorType                    StatisticalModelVectorType;
  typedef typename Superclass1::StatisticalModelMatrixType                    StatisticalModelMatrixType;
  typedef vector< std::string >                                               StatisticalModelPathVectorType;

  typedef typename Superclass1::MovingImagePointer                            MovingImagePointer;

  typedef typename Superclass1::StatisticalModelRepresenterType               StatisticalModelRepresenterType;
  typedef typename Superclass1::StatisticalModelRepresenterPointer            StatisticalModelRepresenterPointer;

  typedef typename Superclass1::StatisticalModelModelBuilderType              StatisticalModelBuilderType;
  typedef typename Superclass1::StatisticalModelBuilderPointer                StatisticalModelBuilderPointer;

  typedef typename Superclass1::ReducedVarianceModelBuilderType               ReducedVarianceModelBuilderType;
  typedef typename Superclass1::ReducedVarianceModelBuilderPointer            ReducedVarianceModelBuilderPointer;

  typedef typename Superclass1::StatisticalModelMatrixContainerType           StatisticalModelMatrixContainerType;
  typedef typename Superclass1::StatisticalModelMatrixContainerPointer        StatisticalModelMatrixContainerPointer;

  typedef typename Superclass1::StatisticalModelVectorContainerType           StatisticalModelVectorContainerType;
  typedef typename Superclass1::StatisticalModelVectorContainerPointer        StatisticalModelVectorContainerPointer;

  typedef typename Superclass1::StatisticalModelScalarContainerType           StatisticalModelScalarContainerType;
  typedef typename Superclass1::StatisticalModelScalarContainerPointer        StatisticalModelScalarContainerPointer;

  typedef typename Superclass1::StatisticalModelRepresenterContainerType     StatisticalModelRepresenterContainerType;
  typedef typename Superclass1::StatisticalModelRepresenterContainerPointer  StatisticalModelRepresenterContainerPointer;

  typedef typename Superclass1::StatisticalModelDataManagerType               StatisticalModelDataManagerType;
  typedef typename Superclass1::StatisticalModelDataManagerPointer            StatisticalModelDataManagerPointer;

  typedef itk::ImageFileWriter< FixedImageType >                              FixedImageFileWriterType;
  typedef typename FixedImageFileWriterType::Pointer                          FixedImageFileWriterPointer;
  typedef itk::ImageFileWriter< MovingImageType >                             MovingImageFileWriterType;
  typedef typename MovingImageFileWriterType::Pointer                         MovingImageFileWriterPointer;

  itkSetMacro( MetricNumber, std::string );
  itkGetMacro( MetricNumber, std::string );

  itkSetMacro( NoiseVariance, StatisticalModelVectorType );
  itkGetMacro( NoiseVariance, StatisticalModelVectorType );

  StatisticalModelDataManagerPointer ReadImagesFromDirectory( std::string imageDataDirectory,
                                                              std::string referenceFilename );

  bool ReadImage( const std::string & imageFilename, StatisticalModelImagePointer & image );

  StatisticalModelPathVectorType ReadPath( std::string parameter );

  StatisticalModelVectorType ReadNoiseVariance();

  StatisticalModelVectorType ReadTotalVariance();

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  virtual void Initialize( void );

  // Load data
  virtual int BeforeAllBase( void );

  // Build models
  virtual void BeforeRegistration( void );

  virtual void AfterEachResolution( void );

  virtual void AfterRegistration( void );

  // Write model reconstruction to disk
  virtual void AfterEachIteration( void );

protected:

  /** The constructor. */
  ActiveRegistrationModelIntensityMetric(){}
  /** The destructor. */
  virtual ~ActiveRegistrationModelIntensityMetric() {}

private:

  /** The private constructor. */
  ActiveRegistrationModelIntensityMetric(const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );               // purposely not implemented

  std::string m_MetricNumber;
  StatisticalModelPathVectorType m_LoadIntensityModelFileNames;
  StatisticalModelPathVectorType m_SaveIntensityModelFileNames;
  StatisticalModelPathVectorType m_ImageDirectories;
  StatisticalModelPathVectorType m_ReferenceFilenames;
  StatisticalModelVectorType m_NoiseVariance;

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxActiveRegistrationModelIntensityMetric.hxx"
#endif

#endif // end #ifndef __elxActiveRegistrationModelIntensityMetric_h__

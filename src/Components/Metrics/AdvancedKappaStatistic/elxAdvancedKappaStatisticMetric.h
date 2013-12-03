/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxAdvancedKappaStatisticMetric_H__
#define __elxAdvancedKappaStatisticMetric_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedKappaStatisticImageToImageMetric.h"
#include "elxTimer.h"

namespace elastix
{

  /**
   * \class AdvancedKappaStatisticMetric
   * \brief An metric based on the itk::AdvancedKappaStatisticImageToImageMetric.
   *
   * The parameters used in this class are:
   * \parameter Metric: Select this metric as follows:\n
   *    <tt>(Metric "AdvancedKappaStatistic")</tt>
   * \parameter UseComplement: Bool to use the complement of the metric or not.\n
   *    If true, the 1 - KappaStatistic is returned, which is useful since most 
   *    optimizers search by default for a minimum.\n
   *    <tt>(UseComplement "true")</tt>\n
   *    The default value is true.
   * \parameter ForeGroundvalue: the overlap of structures with this value is
   *    calculated.\n
   *    <tt>(ForeGroundvalue 3.5)</tt>\n
   *    The default value is 1.0.
   *
   * \ingroup Metrics
   *
   */

  template <class TElastix >
    class AdvancedKappaStatisticMetric:
    public
      itk::AdvancedKappaStatisticImageToImageMetric<
        typename MetricBase<TElastix>::FixedImageType,
        typename MetricBase<TElastix>::MovingImageType >,
    public MetricBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef AdvancedKappaStatisticMetric                  Self;
    typedef itk::AdvancedKappaStatisticImageToImageMetric<
      typename MetricBase<TElastix>::FixedImageType,
      typename MetricBase<TElastix>::MovingImageType >    Superclass1;
    typedef MetricBase<TElastix>                          Superclass2;
    typedef itk::SmartPointer<Self>                       Pointer;
    typedef itk::SmartPointer<const Self>                 ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( AdvancedKappaStatisticMetric, itk::AdvancedKappaStatisticImageToImageMetric );

    /** Name of this class.
     * Use this name in the parameter file to select this specific metric. \n
     * example: <tt>(Metric "AdvancedKappaStatistic")</tt>\n
     */
    elxClassNameMacro( "AdvancedKappaStatistic" );

    /** Typedefs from the superclass. */
    typedef typename
      Superclass1::CoordinateRepresentationType              CoordinateRepresentationType;
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
      Superclass1::ImageSampleContainerPointer               ImageSampleContainerPointer;
    typedef typename Superclass1::FixedImageLimiterType      FixedImageLimiterType;
    typedef typename Superclass1::MovingImageLimiterType     MovingImageLimiterType;
    typedef typename
      Superclass1::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
    typedef typename
      Superclass1::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
    typedef typename
      Superclass1::MovingImageDerivativeScalesType           MovingImageDerivativeScalesType;

    /** The fixed image dimension. */
    itkStaticConstMacro( FixedImageDimension, unsigned int,
      FixedImageType::ImageDimension );

    /** The moving image dimension. */
    itkStaticConstMacro( MovingImageDimension, unsigned int,
      MovingImageType::ImageDimension );

    /** Typedef's inherited from Elastix. */
    typedef typename Superclass2::ElastixType               ElastixType;
    typedef typename Superclass2::ElastixPointer            ElastixPointer;
    typedef typename Superclass2::ConfigurationType         ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
    typedef typename Superclass2::RegistrationType          RegistrationType;
    typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
    typedef typename Superclass2::ITKBaseType               ITKBaseType;

    /** Typedef for timer. */
    typedef tmr::Timer          TimerType;
    /** Typedef for timer. */
    typedef TimerType::Pointer  TimerPointer;

    /** Sets up a timer to measure the initialisation time and
     * calls the Superclass' implementation.
     */
    virtual void Initialize(void) throw (itk::ExceptionObject);

    /**
     * Do some things before registration:
     * \li Set the UseComplement setting
     * \li Set the ForeGroundvalue setting
     */
    virtual void BeforeRegistration( void );

    /**
     * Do some things before each resolution:
     * \li Set the CheckNumberOfSamples setting
     */
    virtual void BeforeEachResolution(void);

  protected:

    /** The constructor. */
    AdvancedKappaStatisticMetric(){};
    /** The destructor. */
    virtual ~AdvancedKappaStatisticMetric() {}

  private:

    /** The private constructor. */
    AdvancedKappaStatisticMetric( const Self& );// purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );              // purposely not implemented

  };


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAdvancedKappaStatisticMetric.hxx"
#endif

#endif // end #ifndef __elxAdvancedKappaStatisticMetric_H__


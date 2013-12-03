/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxMutualInformationHistogramMetric_H__
#define __elxMutualInformationHistogramMetric_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMutualInformationHistogramImageToImageMetric.h"

#include "elxTimer.h"

namespace elastix
{

  /**
   * \class MutualInformationHistogramMetric
   * \brief A metric based on the itk::MutualInformationHistogramImageToImageMetric.
   *
   * This metric is not yet fully supported. But with a little effort it is!
   *
   * \warning: this metric is not very well tested in elastix.
   * \warning: this metric is not based on the AdvancedImageToImageMetric so
   * does not support the ImageSampler framework and might be very slow in
   * combination with B-spline transform.
   *
   * The parameters used in this class are:
   * \parameter Metric: Select this metric as follows:\n
   *    <tt>(Metric "MutualInformationHistogram")</tt>
   *
   * \ingroup Metrics
   */

  template <class TElastix >
    class MutualInformationHistogramMetric :
    public
      itk::MutualInformationHistogramImageToImageMetric<
        typename MetricBase<TElastix>::FixedImageType,
        typename MetricBase<TElastix>::MovingImageType >,
    public MetricBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef MutualInformationHistogramMetric              Self;
    typedef itk::MutualInformationHistogramImageToImageMetric<
      typename MetricBase<TElastix>::FixedImageType,
      typename MetricBase<TElastix>::MovingImageType >    Superclass1;
    typedef MetricBase<TElastix>                          Superclass2;
    typedef itk::SmartPointer<Self>                       Pointer;
    typedef itk::SmartPointer<const Self>                 ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( MutualInformationHistogramMetric,
      itk::MutualInformationHistogramImageToImageMetric );

    /** Name of this class.
     * Use this name in the parameter file to select this specific metric. \n
     * example: <tt>(Metric "MutualInformationHistogram")</tt>\n
     */
    elxClassNameMacro( "MutualInformationHistogram" );

    /** Typedefs inherited from the superclass. */
    typedef typename Superclass1::TransformType             TransformType;
    typedef typename Superclass1::TransformPointer          TransformPointer;
    typedef typename Superclass1::TransformJacobianType     TransformJacobianType;
    typedef typename Superclass1::InterpolatorType          InterpolatorType;
    typedef typename Superclass1::MeasureType               MeasureType;
    typedef typename Superclass1::DerivativeType            DerivativeType;
    typedef typename Superclass1::ParametersType            ParametersType;
    typedef typename Superclass1::FixedImageType            FixedImageType;
    typedef typename Superclass1::MovingImageType           MovingImageType;
    typedef typename Superclass1::FixedImageConstPointer    FixedImageConstPointer;
    typedef typename Superclass1::MovingImageConstPointer   MovingImageCosntPointer;
    typedef typename Superclass1::ScalesType                ScalesType;

    /** The moving image dimension. */
    itkStaticConstMacro( MovingImageDimension, unsigned int,
      MovingImageType::ImageDimension );

    /** Typedef's inherited from Elastix. */
    typedef typename Superclass2::ElastixType           ElastixType;
    typedef typename Superclass2::ElastixPointer        ElastixPointer;
    typedef typename Superclass2::ConfigurationType     ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer  ConfigurationPointer;
    typedef typename Superclass2::RegistrationType      RegistrationType;
    typedef typename Superclass2::RegistrationPointer   RegistrationPointer;
    typedef typename Superclass2::ITKBaseType           ITKBaseType;

    /** Typedef's for timer. */
    typedef tmr::Timer          TimerType;
    typedef TimerType::Pointer  TimerPointer;

    /** Execute stuff before the actual registration:
     * \li Nothing yet: still to be implemented.
     */
    virtual void BeforeRegistration(void);

    /** Execute stuff before each new pyramid resolution:
     * \li Nothing yet: still to be implemented.
     */
    virtual void BeforeEachResolution(void);

    /** Sets up a timer to measure the intialisation time and
     * calls the Superclass' implementation.
     */
    virtual void Initialize(void) throw (itk::ExceptionObject);

  protected:

    /** The constructor. */
    MutualInformationHistogramMetric();
    /** The destructor. */
    virtual ~MutualInformationHistogramMetric() {}

  private:

    /** The private constructor. */
    MutualInformationHistogramMetric( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );                    // purposely not implemented

  };


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMutualInformationHistogramMetric.hxx"
#endif

#endif // end #ifndef __elxMutualInformationHistogramMetric_H__

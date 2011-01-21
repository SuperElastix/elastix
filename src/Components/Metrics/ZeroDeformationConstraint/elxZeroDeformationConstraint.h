/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxZeroDeformationConstraint_H__
#define __elxZeroDeformationConstraint_H__

#include "elxIncludes.h"
#include "itkZeroDeformationConstraint.h"

#include "../../Registrations/MultiMetricMultiResolutionRegistration/elxMultiMetricMultiResolutionRegistration.h"
#include "elxTimer.h"

namespace elastix
{
using namespace itk;

  /** \class ZeroDeformationConstraint
   * \brief Penalizes deformation using an augmented Lagrangian penalty term.
   *
   * This Class is templated over the type of the fixed and moving
   * images to be compared.
   *
   * \ingroup RegistrationMetrics
   * \ingroup Metrics
   */

  template <class TElastix >
    class ZeroDeformationConstraint:
    public
      ZeroDeformationConstraintMetric<
        ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
        double >,
    public MetricBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef ZeroDeformationConstraint                        Self;
    typedef ZeroDeformationConstraintMetric<
      typename MetricBase<TElastix>::FixedImageType,
      double >                                            Superclass1;
    typedef MetricBase<TElastix>                          Superclass2;
    typedef SmartPointer<Self>                            Pointer;
    typedef SmartPointer<const Self>                      ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ZeroDeformationConstraint, ZeroDeformationConstraintMetric );

    /** Set functions. */
    itkSetMacro( InitialPenaltyTermMultiplier, double );
    itkSetMacro( PenaltyTermMultiplierFactor, double );
    itkSetMacro( NumSubIterations, unsigned int );

    /** Get functions. */
    itkGetConstMacro( InitialPenaltyTermMultiplier, double );
    itkGetConstMacro( PenaltyTermMultiplierFactor, double );
    itkGetConstMacro( NumSubIterations, unsigned int );

    /** Name of this class.
     * Use this name in the parameter file to select this specific metric. \n
     * example: <tt>(Metric "ZeroDeformationConstraint")</tt>\n
     */
    elxClassNameMacro( "ZeroDeformationConstraint" );

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
      Superclass1::MovingImageDerivativeScalesType          MovingImageDerivativeScalesType;

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

    /** Multi metric typedefs. */
    typedef elx::MultiMetricMultiResolutionRegistration<ElastixType> MultiMetricRegistrationType;

    /** Typedef for timer. */
    typedef tmr::Timer          TimerType;
    /** Typedef for timer. */
    typedef TimerType::Pointer  TimerPointer;

    /** Sets up a timer to measure the initialisation time and
     * calls the Superclass' implementation.
     */
    virtual void Initialize(void) throw (ExceptionObject);

    /**
    * Do some things before registration
    */
    virtual void BeforeRegistration(void);

    /**
     * Do some things before each resolution
     */
    virtual void BeforeEachResolution(void);

    /**
     * Do some things after each iteration:
     * \li Track number of iterations
     * \li Update penaltyTermMultiplier
     * \li Update lagrange multipliers every n iterations
     */
    virtual void AfterEachIteration(void);

  protected:

    /** The constructor. */
    ZeroDeformationConstraint(){};
    /** The destructor. */
    virtual ~ZeroDeformationConstraint() {}

  private:

    /** The private constructor. */
    ZeroDeformationConstraint( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );              // purposely not implemented

    /** Start lagrangian multiplier and penalty multiplier alpha. */
    double m_InitialPenaltyTermMultiplier;
    double m_PenaltyTermMultiplierFactor;

    /** Number of sub-iterations, after which the new lagrangian multipliers are determined. */
    unsigned int m_NumSubIterations;
    unsigned int m_CurrentIteration;

    /** Bool for setting to update param_a or not. */
    bool m_UpdateParam_a;

    /** Previous maximum magnitude value and required decrease factor. */
    double m_PreviousMaximumAbsoluteDisplacement;
    double m_RequiredConstraintDecreaseFactor;
    unsigned int m_NumPenaltyTermUpdates;

    /** Average lagrange multiplier variable for output. */
    double m_AverageLagrangeMultiplier;

    inline void DetermineNewLagrangeMultipliers( );
    inline double DetermineNewPenaltyTermMultiplier( const int iterationNumber ) const;

  }; // end class ZeroDeformationConstraint


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxZeroDeformationConstraint.hxx"
#endif

#endif // end #ifndef __elxZeroDeformationConstraint_H__


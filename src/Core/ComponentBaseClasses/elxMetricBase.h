/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxMetricBase_h
#define __elxMetricBase_h

/** Needed for the macros. */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkAdvancedImageToImageMetric.h"
#include "itkImageFullSampler.h"

#include "elxTimer.h"


namespace elastix
{
using namespace itk;

  /**
   * \class MetricBase
   * \brief This class is the elastix base class for all  Metrics.
   *
   * This class contains the common functionality for all Metrics.
   *
   * The parameters used in this class are:
   * \parameter ShowExactMetricValue: Flag that can set to "true" or "false". If "true" the 
   *		metric computes the exact metric value (computed on all voxels rather than on the set of
   *		spatial samples) and shows it each iteration. Must be given for each resolution. \n
   *		example: <tt>(ShowExactMetricValue "true" "true" "false")</tt> \n
   *		Default is "false" for all resolutions.
   * \parameter CheckNumberOfSamples: Whether the metric checks if at least 1/4 of the 
   *    samples map inside the moving image. Must be given for each resolution or for all
   *    resolutions at once. \n
   *    example: <tt>(CheckNumberOfSamples "false" "true" "false")</tt> \n
   *    The default is true. In general it is wise to set this to true, since it detects
   *    if the registration is going really bad.
   *
   * \ingroup Metrics
   * \ingroup ComponentBaseClasses
   */

  template <class TElastix>
    class MetricBase : public BaseComponentSE<TElastix>
  {
  public:

    /** Standard ITK stuff. */
    typedef MetricBase									Self;
    typedef BaseComponentSE<TElastix>		Superclass;

    /** Run-time type information (and related methods). */
    itkTypeMacro( MetricBase, BaseComponentSE );

    /** Typedef's inherited from Elastix. */
    typedef typename Superclass::ElastixType						ElastixType;
    typedef typename Superclass::ElastixPointer					ElastixPointer;
    typedef typename Superclass::ConfigurationType			ConfigurationType;
    typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
    typedef typename Superclass::RegistrationType				RegistrationType;
    typedef typename Superclass::RegistrationPointer		RegistrationPointer;

    /** Other typedef's. */
    typedef typename ElastixType::FixedImageType		FixedImageType;
    typedef typename ElastixType::MovingImageType		MovingImageType;
    
    /** ITKBaseType. */
    typedef ImageToImageMetric<
      FixedImageType, MovingImageType >				ITKBaseType;
    typedef AdvancedImageToImageMetric<
      FixedImageType, MovingImageType >				AdvancedMetricType;

    /** Typedefs for sampler support. */
    typedef typename AdvancedMetricType::ImageSamplerType   ImageSamplerBaseType;

    /** Cast to ITKBaseType. */
    virtual ITKBaseType * GetAsITKBaseType(void)
    {
      return dynamic_cast<ITKBaseType *>(this);
    }

    /** Cast to ITKBaseType, to use in const functions. */
    virtual const ITKBaseType * GetAsITKBaseType(void) const
    {
      return dynamic_cast<const ITKBaseType *>(this);
    }
    
    /** Get	the dimension of the fixed image. */
    itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
    /** Get	the dimension of the moving image. */
    itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

    /** Execute stuff before each resolution:
     * \li Check if the exact metric value should be computed
     * (to monitor the progress of the registration) */
    virtual void BeforeEachResolutionBase(void);

    /** Execute stuff after each iteration:
     * \li Optionally compute the exact metric value and plot it to screen */
    virtual void AfterEachIterationBase(void);
    
    /*** Force the metric to base its computation on a new subset of image samples.
     * Not every metric may have implemented this. */
    virtual void SelectNewSamples(void);

    /** Returns whether the metric uses a sampler. When the metric is not of
     * AdvancedMetricType, the function returns false immediately. */
    virtual bool GetAdvancedMetricUseImageSampler( void ) const;

    /** Method to set the image sampler. The image sampler is only used when
     * the metric is of type AdvancedMetricType, and has UseImageSampler set
     * to true. In other cases, the function does nothing. */
    virtual void SetAdvancedMetricImageSampler( ImageSamplerBaseType * sampler );

    /** Methods to get the image sampler. The image sampler is only used when
     * the metric is of type AdvancedMetricType, and has UseImageSampler set
     * to true. In other cases, the function returns 0. */
    virtual ImageSamplerBaseType * GetAdvancedMetricImageSampler( void ) const;

  protected:

    /** The type returned by the GetValue methods. Used by the GetExactValue method. */
    typedef typename ITKBaseType::MeasureType       MeasureType;
    typedef typename ITKBaseType::ParametersType    ParametersType;

    /** The full sampler used by the GetExactValue method */    
    typedef itk::ImageFullSampler<FixedImageType>   ImageFullSamplerType;
    
    /** The constructor. */
    MetricBase();
    /** The destructor. */
    virtual ~MetricBase() {}

    /**  Get the exact value. Mutual information computed over all points.
     * It is meant in situations when you optimise using just a subset of pixels, 
     * but are interested in the exact value of the metric. 
     *
     * This method only works when the itkYourMetric inherits from 
     * the AdvancedMetricType.
     * In other cases it returns 0. You may reimplement this method in 
     * the elxYourMetric, if you like. 
     */
    virtual MeasureType GetExactValue( const ParametersType& parameters );
    /** \todo the method GetExactDerivative could as well be added here. */
  
    bool m_ShowExactMetricValue;
    typename ImageFullSamplerType::Pointer m_ExactMetricSampler;


  private:

    /** The private constructor. */
    MetricBase( const Self& );			// purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );	// purposely not implemented


  }; // end class MetricBase


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMetricBase.hxx"
#endif

#endif // end #ifndef __elxMetricBase_h


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
	 * \brief This class is the base for all Metrics.
	 *
	 * This class contains the common functionality for all Metrics.
	 *
	 * The parameters used in this class are:
   * \parameter ShowExactMetricValue: Flag that can set to "true" or "false". If "true" the 
	 *		metric computes the exact metric value (computed on all voxels rather than on the set of
	 *		spatial samples) and shows it each iteration. Must be given for each resolution. \n
	 *		example: <tt>(ShowExactMetricValue "true" "true" "false")</tt> \n
	 *		Default is "false" for all resolutions.\n
   * \parameter ImageSampler: The name of the image sampler that is used to select voxels in the fixed image. \n
   *    Choose one of: Full, Random, RandomSparseMask, RandomCoordinate, Grid.\n
   *    Can be specified for each resolution, or for all resolutions at once.\n
   *    example: <tt>(ImageSampler "Random" "Full")</tt> \n
   *    Default is "Random".\n
   *    See also the NewSamplesEveryIteration parameter (defined in the elx::OptimizerBase)\n
   *    \li Full: All pixels of the fixed image;
   *    \li Random: Random pixels are selected; the parameter NumberOfSpatialSamples says how many;
   *    \li RandomSparseMask: Same as random, but more efficient when the FixedMask is sparse (many 0s);
   *    \li RandomCoordinate: Random coordinates are selected (so not only at voxel locations);
   *    \li Grid: Voxels are selected on a uniform regular grid; this ImageSampler is NOT RECOMMENDED; The grid size can be specified by the parameter SampleGridSpacing;
   * \parameter NumberOfSpatialSamples: The number of image voxels used for computing the \n
	 *		metric value and its derivative in each iteration. Must be given for each resolution.\n
   *    this parameter makes sense with a Random, RandomSparseMask, and RandomCoordinate ImageSampler.\n
	 *		example: <tt>(NumberOfSpatialSamples 2048 2048 4000)</tt> \n
	 *		The default is 10000.\n
   * \parameter SampleGridSpacing: Defines the sampling grid in case of a Grid ImageSampler.\n
   *    An integer downsampling factor must be specified for each dimension, for each resolution.\n
   *    example: <tt>(SampleGridSpacing 4 4 2 2)</tt>\n
   *    Default is 2 for each dimension for each resolution. \n
   *    NB: a Grid ImageSampler is NOT RECOMMENDED!\n
   * \parameter UseRandomSampleRegion: Defines whether to randomly select a subregion of the image
   *    in each iteration. This option is only recognised by the RandomCoordinate sampler and
   *    the MultiInputRandomCoordinate. When set to "true", also specify the SampleRegionSize.
   *    By setting this option to "true", in combination with the NewSamplesEveryIteration parameter,
   *    a "localised" similarity measure is obtained.\n
   *    example: <tt>(UseRandomSampleRegion "true")</tt>\n
   *    Default: false.
   * \parameter SampleRegionSize: the size of the subregions that are selected when using 
   *    the UseRandomSampleRegion option. The size should be specified in mm, for each dimension.
   *    As a rule of thumb, you may try a value ~1/3 of the image size.\n
   *    example: <tt>(SampleRegionSize 50.0 50.0 50.0)</tt>\n    
   *    You can also specify one number, which will be used for all dimensions. Also, you
   *    can specify different values for each resolution:\n
   *    example: <tt>(SampleRegionSize 50.0 50.0 50.0 30.0 30.0 30.0)</tt>\n    
   *    In this example, in the first resolution 50mm is used for each of the 3 dimensions,
   *    and in the second resolution 30mm.
   * \parameter FixedImageBSplineInterpolationOrder: When using a RandomCoordinate or 
   *    MultiInputRandomCoordinate sampler, the fixed image needs to be interpolated. This
   *    is done using B-spline interpolator. With this option you can specify the order of interpolation.\n
   *    example: <tt>(FixedImageBSplineInterpolationOrder 0 0 1)</tt>\n
   *    Default value: 1. The parameter can be specified for each resolution.
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
	   * \li Configure the image sampler
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

    /** ConfigureImageSampler. */
    void ConfigureImageSampler( void );

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


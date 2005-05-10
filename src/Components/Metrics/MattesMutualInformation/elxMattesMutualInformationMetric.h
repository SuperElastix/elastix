#ifndef __elxMattesMutualInformationMetric_H__
#define __elxMattesMutualInformationMetric_H__

#include "elxIncludes.h"
#include "itkMattesMutualInformationImageToImageMetricWithMask.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class MattesMutualInformationMetric
	 * \brief A metric based on mutual information...
	 *
	 * This metric is based on an adapted version of the
	 * itk::MattesMutualInformationImageToImageMetric. 
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 * <tt> (Metric MattesMutualInformation) </tt>
	 * \parameter NumberOfHistogramBins: The size of the histogram. Must be given for each 
	 * resolution. \n
	 *   example: <tt> (NumberOfHistogramBins 32 32 64)</tt>
	 * \parameter NumberOfSpatialSamples: The number of image voxels used for computing the
	 * metric value and its derivative in each iteration. Must be given for each resolution.\n
	 *  example: <tt> (NumberOfSpatialSamples 2048 2048 4000) </tt>
	 * \parameter NumberOfResolutions: The number of resolutions.\n
	 *   example: <tt> (NumberOfResolutions 3) </tt>
	 * \parameter	UseAllPixels: Flag to force the metric to use ALL voxels for 
	 * computing the metric value and its derivative in each iteration. Must be given for each
	 * resolution. Can have values "true" or "false".\n
	 *   example: <tt> (UseAllPixels "true" "false" "true") </tt>
	 * \parameter ShowExactMetricValue: Flag that can set to "true" or "false". If "true" the 
	 * metric computes the exact metric value (computed on all voxels rather than on the set of
	 * spatial samples) and shows it each iteration. Must be given for each resolution.\n
	 * NB: If the UseallPixels flag is set to "true", this option is ignored.\n
	 *   example: <tt> (ShowExactMetricValue "true" "true" "false") </tt>
	 *
   * \sa MattesMutualInformationImageToImageMetricWithMask
	 * \ingroup Metrics
	 */
	
	template <class TElastix >	
		class MattesMutualInformationMetric :
		public
			MattesMutualInformationImageToImageMetricWithMask<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef MattesMutualInformationMetric									Self;
		typedef MattesMutualInformationImageToImageMetricWithMask<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationMetric,
			MattesMutualInformationImageToImageMetricWithMask );
		
		/** Name of this class.*/
		elxClassNameMacro( "MattesMutualInformation" );

		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass1::TransformType							TransformType;
		typedef typename Superclass1::TransformPointer 					TransformPointer;
		typedef typename Superclass1::TransformJacobianType			TransformJacobianType;
		typedef typename Superclass1::InterpolatorType 					InterpolatorType;
		typedef typename Superclass1::MeasureType								MeasureType;
		typedef typename Superclass1::DerivativeType 						DerivativeType;
		typedef typename Superclass1::ParametersType 						ParametersType;
		typedef typename Superclass1::FixedImageType 						FixedImageType;
		typedef typename Superclass1::MovingImageType						MovingImageType;
		typedef typename Superclass1::FixedImageConstPointer 		FixedImageConstPointer;
		typedef typename Superclass1::MovingImageConstPointer		MovingImageConstPointer;
		typedef typename Superclass1::FixedImageIndexType				FixedImageIndexType;
		typedef typename Superclass1::FixedImageIndexValueType 	FixedImageIndexValueType;
		typedef typename Superclass1::MovingImageIndexType 			MovingImageIndexType;
		typedef typename Superclass1::FixedImagePointType				FixedImagePointType;
		typedef typename Superclass1::MovingImagePointType 			MovingImagePointType;
		

		/** The fixed image dimension */
		itkStaticConstMacro (FixedImageDimension, unsigned int,
			FixedImageType::ImageDimension);
		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Typedef's inherited from Elastix. */
		typedef typename Superclass2::ElastixType								ElastixType;
		typedef typename Superclass2::ElastixPointer						ElastixPointer;
		typedef typename Superclass2::ConfigurationType					ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer			ConfigurationPointer;
		typedef typename Superclass2::RegistrationType					RegistrationType;
		typedef typename Superclass2::RegistrationPointer				RegistrationPointer;
		typedef typename Superclass2::ITKBaseType								ITKBaseType;
			
		/** Typedef for timer.*/
		typedef tmr::Timer					TimerType;
		/** Typedef for timer.*/
		typedef TimerType::Pointer	TimerPointer;

		/** Typedefs for support of user defined grid spacing for the spatial samples */
		typedef typename FixedImageType::OffsetType							SampleGridSpacingType;
		typedef typename SampleGridSpacingType::OffsetValueType SampleGridSpacingValueType;
		typedef typename FixedImageType::SizeType								SampleGridSizeType;
		typedef FixedImageIndexType															SampleGridIndexType;
		typedef typename FixedImageType::SizeType 							FixedImageSizeType;
		
		/** Method that takes care of setting the parameters and showing information.*/
		virtual int BeforeAll(void);
		/** Method that takes care of setting the parameters and showing information.*/
		virtual void BeforeEachResolution(void);
		/** Method that takes care of setting the parameters and showing information.*/
		virtual void AfterEachIteration(void);

		/** Sets up a timer to measure the intialisation time and calls the Superclass'
		 * implementation */
		virtual void Initialize(void) throw (ExceptionObject);

		/** Select a new sample set on request */
		virtual void SelectNewSamples(void);
		
	protected:

		MattesMutualInformationMetric(); 
		virtual ~MattesMutualInformationMetric() {}

		/** Uniformly select a sample set from the fixed image domain.
		 * This version adds the functionality to select the samples on a 
		 * uniform grid.
		 *
		 * Mainly for testing purposes. Does not take spacings into account. */
		typedef typename Superclass1::FixedImageSpatialSampleContainer
			FixedImageSpatialSampleContainer;
		virtual void SampleFixedImageDomain( 
			FixedImageSpatialSampleContainer& samples );
		
		/** Flag */
		bool m_ShowExactMetricValue;

		/** The grid spacing of the spatial samples. Only used when the user asked
		 * for a regular sampling grid, instead of randomly placed samples. */
		SampleGridSpacingType			m_SampleGridSpacing;

		/** Flag, whether to put the spatial sample son a uniform grid. */
		bool											m_SamplesOnUniformGrid;
				
	private:

		MattesMutualInformationMetric( const Self& );	// purposely not implemented
		void operator=( const Self& );								// purposely not implemented
		
	}; // end class MattesMutualInformationMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMattesMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __elxMattesMutualInformationMetric_H__

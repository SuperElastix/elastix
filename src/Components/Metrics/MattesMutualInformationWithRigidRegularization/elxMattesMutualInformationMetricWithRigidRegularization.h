#ifndef __elxMattesMutualInformationMetricWithRigidRegularization_H__
#define __elxMattesMutualInformationMetricWithRigidRegularization_H__

#include "elxIncludes.h"
#include "itkMattesMutualInformationImageToImageMetricWithRigidRegularization.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class MattesMutualInformationMetricWithRigidRegularization
	 * \brief A metric based on mutual information and a rigid penalty term.
	 *
	 * This metric adds two metrics, namely an adapted version of the
	 * itk::MattesMutualInformationImageToImageMetric with the
	 * itk::RigidRegulizerMetric.
	 *
	 * This metric only works with B-splines as a transformation model.
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 *		<tt>(Metric "MattesMutualInformationWithRigidRegularization")</tt>
	 * \parameter NumberOfHistogramBins: The size of the histogram. Must be given for each 
	 *		resolution. \n
	 *		example: <tt>(NumberOfHistogramBins 32 32 64)</tt> \n
	 *		The default is 32 for each resolution.
	 * \parameter NumberOfSpatialSamples: The number of image voxels used for computing the
	 *		metric value and its derivative in each iteration. Must be given for each resolution.\n
	 *		example: <tt>(NumberOfSpatialSamples 2048 2048 4000)</tt> \n
	 *		The default is 10000.
	 * \parameter	UseAllPixels: Flag to force the metric to use ALL voxels for 
	 *		computing the metric value and its derivative in each iteration. Must be given for each
	 *		resolution. Choose one of {"true", "false"}. \n
	 *		example: <tt>(UseAllPixels "true" "false" "true")</tt> \n
	 *		Default is "false" for all resolutions.
	 * \parameter ShowExactMetricValue: Flag that can set to "true" or "false". If "true" the 
	 *		metric computes the exact metric value (computed on all voxels rather than on the set of
	 *		spatial samples) and shows it each iteration. Must be given for each resolution. \n
	 *		NB: If the UseallPixels flag is set to "true", this option is ignored. \n
	 *		example: <tt>(ShowExactMetricValue "true" "true" "false")</tt> \n
	 *		Default is "false" for all resolutions.
	 * \parameter SamplesOnUniformGrid: Flag to choose the samples on a uniform grid. \n
	 *		example: <tt>(SamplesOnUniformGrid "true")</tt> \n
	 *		Default is "false".
	 * \parameter SampleGridSpacing: if the SamplesOnUniformGrid is set to "true", this parameter
	 *		controls the spacing of the uniform grid in all dimensions. This should be given in
	 *		index coordinates. \n
	 *		example: <tt>(SampleGridSpacing 4 4 4)</tt> \n
	 *		Default is 2 in each dimension.
	 * \parameter RigidPenaltyWeight: A parameter to weigh the rigidity penalty
	 *		term against the mutual information metric. \n
	 *		example: <tt>(RigidPenaltyWeight 0.1)</tt> \n
	 *		example: <tt>(RigidPenaltyWeight 1.0 0.5 0.1)</tt> \n
	 *    If only one argument is given, that value is used for all resolutions.
	 *		If more than one argument is given, then the number of arguments should be
	 *		equal to the number of resolutions: for each resolution its rigid penalty weight.
	 *		If this parameter option is not used, by default the rigid penalty weight is set
	 *		to 1.0 for each resolution.
	 * \parameter SecondOrderWeight: A parameter to weigh the second order terms
	 *		of the rigidity term against its first order terms. \n
	 *		example: <tt>(SecondOrderWeight 2.0)</tt> \n
	 *		Default is 1.0.
	 * \parameter UseImageSpacing: flag to specify the use of the spacing of voxels
	 *		when calculating the rigidity term. \n
	 *		example: <tt>(UseImageSpacing "false")</tt> \n
	 *		Default is "true".
	 * \parameter UseFixedRigidityImage: flag to specify the use of the fixed rigidity
	 *		image when calculating the rigidity coefficient image. \n
	 *		example: <tt>(UseFixedRigidityImage "false")</tt> \n
	 *		Default is "true".
	 * \parameter FixedRigidityImageName: the name of a coefficient image to specify
	 *		the rigidity index of voxels in the fixed image. \n
	 *		example: <tt>(FixedRigidityImageName "fixedRigidityImage.mhd")</tt> \n
	 *		This argument is mandatory.
	 * \parameter UseMovingRigidityImage: flag to specify the use of the moving rigidity
	 *		image when calculating the rigidity coefficient image. \n
	 *		example: <tt>(UseMovingRigidityImage "false")</tt> \n
	 *		Default is "true".
	 * \parameter MovingRigidityImageName: the name of a coefficient image to specify
	 *		the rigidity index of voxels in the moving image. \n
	 *		example: <tt>(MovingRigidityImageName "movingRigidityImage.mhd")</tt> \n
	 *		This argument is mandatory.
	 * \parameter DilateRigidityImages: flag to specify the dilation of the rigidity
	 *		coefficient images. With this the region of rigidity can be extended to
	 *		force rigidity of the inner region. \n
	 *		example: <tt>(DilateRigidityImages )</tt> \n
	 *		Default is "true".
	 * \parameter DilationRadiusMultiplier: the dilation radius is a muliplier times the
	 *		gridspacing of the B-spline transform. \n
	 *		example: <tt>(DilationRadiusMultiplier 2.0)</tt> \n
	 *		Default is 1.0.
	 *
   * \sa MattesMutualInformationImageToImageMetricWithMask
	 * \sa MattesMutualInformationImageToImageMetricWithRigidRegularization
	 * \sa RigidRegulizerMetric
	 * \sa BSplineTransform
	 * \ingroup Metrics
	 */
	
	template <class TElastix >	
		class MattesMutualInformationMetricWithRigidRegularization :
		public
			MattesMutualInformationImageToImageMetricWithRigidRegularization<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef MattesMutualInformationMetricWithRigidRegularization									Self;
		typedef MattesMutualInformationImageToImageMetricWithRigidRegularization<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationMetricWithRigidRegularization,
			MattesMutualInformationImageToImageMetricWithRigidRegularization );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific metric. \n
		 * example: <tt>(Metric "MattesMutualInformationWithRigidRegularization")</tt>\n
		 */
		elxClassNameMacro( "MattesMutualInformationWithRigidRegularization" );

		/** Typedefs inherited from the superclass. */
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
		
		/** The fixed image dimension. */
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
			
		/** Typedef for timer. */
		typedef tmr::Timer					TimerType;
		/** Typedef for timer. */
		typedef TimerType::Pointer	TimerPointer;

		/** Typedefs for support of user defined grid spacing for the spatial samples. */
		typedef typename FixedImageType::OffsetType							SampleGridSpacingType;
		typedef typename SampleGridSpacingType::OffsetValueType SampleGridSpacingValueType;
		typedef typename FixedImageType::SizeType								SampleGridSizeType;
		typedef FixedImageIndexType															SampleGridIndexType;
		typedef typename FixedImageType::SizeType 							FixedImageSizeType;

		/** Typedefs for the rigidity image. The rigidity images are scalar double
		 * images of dimension (Fixed/Moving)ImageDimension.
		 */
		typedef typename Superclass1::RigidityImageType					RigidityImageType;
		typedef ImageFileReader< RigidityImageType >						RigidityImageReaderType;
		typedef typename RigidityImageReaderType::Pointer				RigidityImageReaderPointer;
		
		/** Execute stuff before the actual registration:
		 * \li Set the rigid penalty weight.
		 * \li Set the weight of the second order term of the penalty term.
		 * \li Set the weight of the orthonormality term.
		 * \li Set the weight of the properness term.
		 * \li Set the flag to use the image spacing for calculations.
		 * \li Set the flag to dilate the rigidity images.
		 * \li Set the dilation radius multiplier
		 * \li Set the output directory name.
		 * \li Set the rigidity coefficients of the fixed image.
		 * \li Set the rigidity coefficients of the moving image.
		 * \li Set the flag to use a fixed rigidity image.
		 * \li Set the flag to use a moving rigidity image.
		 * \li Setup the output to the logfile.
		 */
		virtual void BeforeRegistration(void);

		/** Execute stuff before each new pyramid resolution:
		 * \li Set the number of histogram bins.
		 * \li Set the number of spatial samples.
		 * \li Set the flag to use all samples.
		 * \li Set the flag to calculate and show the exact metric value.
		 * \li Set the flag to take samples on a uniform grid.
		 * \li Set the grid spacing of the sampling grid.
		 * \li Set the rigid penalty weight of this level.
		 */
		virtual void BeforeEachResolution(void);

		/** Execute stuff after each iteration:
		 * \li Show the exact metric value if desired.
		 * \li Show the metric values of the MI and the rigidity term.
		 */
		virtual void AfterEachIteration(void);

		/** Set up a timer to measure the intialisation time and call the Superclass'
		 * implementation.
		 */
		virtual void Initialize(void) throw (ExceptionObject);

		/** Select a new sample set on request. */
		virtual void SelectNewSamples(void);
		
	protected:

		/** The constructor. */
		MattesMutualInformationMetricWithRigidRegularization();
		/** The destructor. */
		virtual ~MattesMutualInformationMetricWithRigidRegularization() {}

		/** Uniformly select a sample set from the fixed image domain.
		 * This version adds the functionality to select the samples on a 
		 * uniform grid.
		 *
		 * Mainly for testing purposes. Does not take spacings into account. */
		typedef typename Superclass1::FixedImageSpatialSampleContainer
			FixedImageSpatialSampleContainer;
		virtual void SampleFixedImageDomain( 
			FixedImageSpatialSampleContainer& samples );
		
		/** Flag. */
		bool m_ShowExactMetricValue;

		/** The grid spacing of the spatial samples. Only used when the user asked
		 * for a regular sampling grid, instead of randomly placed samples. */
		SampleGridSpacingType			m_SampleGridSpacing;

		/** Flag, whether to put the spatial sample son a uniform grid. */
		bool											m_SamplesOnUniformGrid;
				
	private:

		/** The private constructor. */
		MattesMutualInformationMetricWithRigidRegularization( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );								// purposely not implemented
		
		/** Member variables. */
		RigidityImageReaderPointer			m_FixedRigidityImageReader;
		RigidityImageReaderPointer			m_MovingRigidityImageReader;

		std::vector< double >						m_RigidPenaltyWeightVector;
		std::vector< bool >							m_DilateRigidityImagesVector;

	}; // end class MattesMutualInformationMetricWithRigidRegularization


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMattesMutualInformationMetricWithRigidRegularization.hxx"
#endif

#endif // end #ifndef __elxMattesMutualInformationMetricWithRigidRegularization_H__

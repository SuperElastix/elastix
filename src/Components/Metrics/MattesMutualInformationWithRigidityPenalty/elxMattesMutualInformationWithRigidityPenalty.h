#ifndef __elxMattesMutualInformationWithRigidityPenalty_H__
#define __elxMattesMutualInformationWithRigidityPenalty_H__

#include "elxIncludes.h"
#include "itkMattesMutualInformationImageToImageMetricWithRigidityPenalty.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class MattesMutualInformationWithRigidityPenalty
	 * \brief A metric based on mutual information and a rigid penalty term.
	 *
	 * This metric adds two metrics, namely an adapted version of the
	 * itk::MattesMutualInformationImageToImageMetric with the
	 * itk::RigidityPenaltyTermMetric.
	 *
	 * This metric only works with B-splines as a transformation model.
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 *		<tt>(Metric "MattesMutualInformationWithRigidityPenalty")</tt>
	 * \parameter NumberOfHistogramBins: The size of the histogram. Must be given for each 
	 *		resolution. \n
	 *		example: <tt>(NumberOfHistogramBins 32 32 64)</tt> \n
	 *		The default is 32 for each resolution.
	 * \parameter RigidityPenaltyWeight: A parameter to weigh the rigidity penalty
	 *		term against the mutual information metric. \n
	 *		example: <tt>(RigidityPenaltyWeight 0.1)</tt> \n
	 *		example: <tt>(RigidityPenaltyWeight 1.0 0.5 0.1)</tt> \n
	 *    If only one argument is given, that value is used for all resolutions.
	 *		If more than one argument is given, then the number of arguments should be
	 *		equal to the number of resolutions: for each resolution its rigid penalty weight.
	 *		If this parameter option is not used, by default the rigid penalty weight is set
	 *		to 1.0 for each resolution.
	 * \parameter LinearityConditionWeight: A parameter to weigh the linearity
   *    condition term of the rigidity term. \n
	 *		example: <tt>(LinearityConditionWeight 2.0)</tt> \n
	 *		Default is 1.0.
 	 * \parameter OrthonormalityConditionWeight: A parameter to weigh the orthonormality
   *    condition term of the rigidity term. \n
	 *		example: <tt>(OrthonormalityConditionWeight 2.0)</tt> \n
	 *		Default is 1.0.
	 * \parameter PropernessConditionWeight: A parameter to weigh the properness
   *    condition term of the rigidity term. \n
	 *		example: <tt>(PropernessConditionWeight 2.0)</tt> \n
	 *		Default is 1.0.
   * \parameter UseLinearityCondition: A flag to specify the usage of the linearity
   *    condition term for optimisation. \n
	 *		example: <tt>(UseLinearityCondition "false")</tt> \n
	 *		Default is "true".
   * \parameter UseOrthonormalityCondition: A flag to specify the usage of the orthonormality
   *    condition term for optimisation. \n
	 *		example: <tt>(UseOrthonormalityCondition "false")</tt> \n
	 *		Default is "true".
   * \parameter UsePropernessCondition: A flag to specify the usage of the properness
   *    condition term for optimisation. \n
	 *		example: <tt>(UsePropernessCondition "false")</tt> \n
	 *		Default is "true".
   * \parameter CalculateLinearityCondition: A flag to specify if the linearity
   *    condition should still be calcualted, even if it is not used for optimisation. \n
	 *		example: <tt>(CalculateLinearityCondition "false")</tt> \n
	 *		Default is "true".
   * \parameter CalculateOrthonormalityCondition: A flag to specify if the orthonormality
   *    condition should still be calcualted, even if it is not used for optimisation. \n
	 *		example: <tt>(CalculateOrthonormalityCondition "false")</tt> \n
	 *		Default is "true".
   * \parameter CalculatePropernessCondition: A flag to specify if the Properness
   *    condition should still be calcualted, even if it is not used for optimisation. \n
	 *		example: <tt>(CalculatePropernessCondition "false")</tt> \n
	 *		Default is "true".
	 * \parameter UseFixedRigidityImage: flag to specify the use of the fixed rigidity
	 *		image when calculating the rigidity coefficient image. \n
	 *		example: <tt>(UseFixedRigidityImage "false")</tt> \n
	 *		Default is "true". If neither UseFixedRigidityImage nor UseMovingRigidityImage
   *    are true, the RigidityPenaltyTerm is evaluated on the whole transform input domain.
	 * \parameter FixedRigidityImageName: the name of a coefficient image to specify
	 *		the rigidity index of voxels in the fixed image. \n
	 *		example: <tt>(FixedRigidityImageName "fixedRigidityImage.mhd")</tt> \n
	 *		This argument is mandatory whenever UseFixedRigidityImage is "true".
	 * \parameter UseMovingRigidityImage: flag to specify the use of the moving rigidity
	 *		image when calculating the rigidity coefficient image. \n
	 *		example: <tt>(UseMovingRigidityImage "false")</tt> \n
	 *		Default is "true". If neither UseFixedRigidityImage nor UseMovingRigidityImage
   *    are true, the RigidityPenaltyTerm is evaluated on the whole transform input domain.
	 * \parameter MovingRigidityImageName: the name of a coefficient image to specify
	 *		the rigidity index of voxels in the moving image. \n
	 *		example: <tt>(MovingRigidityImageName "movingRigidityImage.mhd")</tt> \n
	 *		This argument is mandatory whenever UseMovingRigidityImage is "true".
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
   * \todo: read these parameters using the new ReadParameter command
	 *
   * \sa MattesMutualInformationImageToImageMetricWithMask
	 * \sa MattesMutualInformationImageToImageMetricWithRigidRegularization
	 * \sa RigidRegulizerMetric
	 * \sa BSplineTransform
	 * \ingroup Metrics
	 */
	
	template <class TElastix >	
		class MattesMutualInformationWithRigidityPenalty :
		public
			MattesMutualInformationImageToImageMetricWithRigidityPenalty<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef MattesMutualInformationWithRigidityPenalty		Self;
		typedef MattesMutualInformationImageToImageMetricWithRigidityPenalty<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationWithRigidityPenalty,
			MattesMutualInformationImageToImageMetricWithRigidityPenalty );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific metric. \n
		 * example: <tt>(Metric "MattesMutualInformationWithRigidRegularization")</tt>\n
		 */
		elxClassNameMacro( "MattesMutualInformationWithRigidityPenalty" );

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

		/** Typedefs for the rigidity image. The rigidity images are scalar double
		 * images of dimension (Fixed/Moving)ImageDimension.
		 */
		typedef typename Superclass1::RigidityImageType					RigidityImageType;
		typedef ImageFileReader< RigidityImageType >						RigidityImageReaderType;
		typedef typename RigidityImageReaderType::Pointer				RigidityImageReaderPointer;
		
		/** Execute stuff before the actual registration:
		 * \li Set the rigidity coefficients of the fixed image.
		 * \li Set the rigidity coefficients of the moving image.
		 * \li Set the flag to use a fixed rigidity image.
		 * \li Set the flag to use a moving rigidity image.
		 * \li Setup the output to the logfile.
		 */
		virtual void BeforeRegistration(void);

		/** Execute stuff before each new pyramid resolution:
		 * \li Set the number of histogram bins.
		 * \li Set the flag to dilate the rigidity images.
		 * \li Set the dilation radius multiplier
		 * \li Set the rigidity penalty weight of this level.
     * \li Set the weight, usage and calculation of the linearity condition of this level.
		 * \li Set the weight, usage and calculation of the orthonormality condition of this level.
		 * \li Set the weight, usage and calculation of the properness condition of this level.
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

	protected:

		/** The constructor. */
		MattesMutualInformationWithRigidityPenalty();
		/** The destructor. */
		virtual ~MattesMutualInformationWithRigidityPenalty() {}
		
	private:

		/** The private constructor. */
		MattesMutualInformationWithRigidityPenalty( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );								// purposely not implemented
		
		/** Member variables. */
		RigidityImageReaderPointer			m_FixedRigidityImageReader;
		RigidityImageReaderPointer			m_MovingRigidityImageReader;

	}; // end class MattesMutualInformationWithRigidityPenalty


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMattesMutualInformationWithRigidityPenalty.hxx"
#endif

#endif // end #ifndef __elxMattesMutualInformationWithRigidityPenalty_H__

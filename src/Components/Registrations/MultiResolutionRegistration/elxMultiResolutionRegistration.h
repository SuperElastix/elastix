#ifndef __elxMultiResolutionRegistration_H__
#define __elxMultiResolutionRegistration_H__

#include "itkMultiResolutionImageRegistrationMethod.h"

/** Mask support. */
#include "itkImageMaskSpatialObject2.h"
#include "itkErodeMaskImageFilter.h"

#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class MultiResolutionRegistration
	 * \brief A registration framework based on the itk::MultiResolutionImageRegistrationMethod.
	 *
	 * This MultiResolutionRegistration givees a framework for registration with a
	 * multi-resolution approach.
	 *
	 * The parameters used in this class are:
	 * \parameter Registration: Select this registration framework as follows:\n
	 *		<tt>(Registration "MultiResolutionRegistration")</tt>
	 * \parameter NumberOfResolutions: the number of resolutions used. \n
	 *		example: <tt>(NumberOfResolutions 4)</tt> \n
	 *		The default is 3.
   * \parameter ErodeMask: a flag to determine if the masks should be eroded
	 *		from one resolution level to another. Choose from {"true", "false"} \n
	 *		example: <tt>(ErodeMask "false")</tt> \n
	 *		The default is "true". The parameter may be specified for each
   *    resolution differently, but that's not obliged. \n
   *    Erosion of the mask prevents the border / edge of the mask taken into account.
	 *    This can be useful for example for ultrasound images,
	 *    where you don't want to take into account values outside
	 *    the US-beam, but where you also don't want to match the
	 *    edge / border of this beam.
	 *    For example for MRI's of the head, the borders of the head
	 *    may be wanted to match, and there erosion should be avoided.
	 *
	 * \ingroup Registrations
	 */

	template <class TElastix>
		class MultiResolutionRegistration :
		public	
			RegistrationBase<TElastix>::ITKBaseType,
		public
			RegistrationBase<TElastix>
	{
	public:

		/** Standard ITK. */
		typedef MultiResolutionRegistration									Self;
		typedef	typename RegistrationBase<TElastix>
			::ITKBaseType																			Superclass1;
		typedef RegistrationBase<TElastix>									Superclass2;
		typedef SmartPointer<Self>													Pointer;
		typedef SmartPointer<const Self>										ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MultiResolutionRegistration, MultiResolutionImageRegistrationMethod );

		/** Name of this class.
		 * Use this name in the parameter file to select this specific registration framework. \n
		 * example: <tt>(Registration "MultiResolutionRegistration")</tt>\n
		 */
		elxClassNameMacro( "MultiResolutionRegistration" );
		
		/** Typedef's inherited from Superclass1. */
		
		/**  Type of the Fixed image. */
		typedef typename Superclass1::FixedImageType						FixedImageType;
		typedef typename Superclass1::FixedImageConstPointer		FixedImageConstPointer;
		typedef typename Superclass1::FixedImageRegionType			FixedImageRegionType;
		
		/**  Type of the Moving image. */
		typedef typename Superclass1::MovingImageType						MovingImageType;
		typedef typename Superclass1::MovingImageConstPointer		MovingImageConstPointer;
		
		/**  Type of the metric. */
		typedef typename Superclass1::MetricType								MetricType;
		typedef typename Superclass1::MetricPointer							MetricPointer;
		
		/**  Type of the Transform . */
		typedef typename Superclass1::TransformType							TransformType;
		typedef typename Superclass1::TransformPointer					TransformPointer;
		
		/**  Type of the Interpolator. */
		typedef typename Superclass1::InterpolatorType					InterpolatorType;
		typedef typename Superclass1::InterpolatorPointer				InterpolatorPointer;
		
		/**  Type of the optimizer. */
		typedef typename Superclass1::OptimizerType							OptimizerType;
		
		/** Type of the Fixed image multiresolution pyramid. */
		typedef typename Superclass1::FixedImagePyramidType			FixedImagePyramidType;
		typedef typename Superclass1::FixedImagePyramidPointer	FixedImagePyramidPointer;
		
		/** Type of the moving image multiresolution pyramid. */
		typedef typename Superclass1::MovingImagePyramidType		MovingImagePyramidType ;
		typedef typename Superclass1::MovingImagePyramidPointer	MovingImagePyramidPointer;
		
		/** Type of the Transformation parameters. This is the same type used to
		 *  represent the search space of the optimization algorithm.
		 */
		typedef typename Superclass1::ParametersType						ParametersType;
		
		/** Typedef's from Elastix. */
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;

    /** Get	the dimension of the fixed image. */
		itkStaticConstMacro( FixedImageDimension, unsigned int, Superclass2::FixedImageDimension );
		/** Get	the dimension of the moving image. */
		itkStaticConstMacro( MovingImageDimension, unsigned int, Superclass2::MovingImageDimension );

    /** Typedef's for mask support. */
    typedef typename ElastixType::MaskPixelType           MaskPixelType;
    typedef typename ElastixType::FixedMaskType           FixedMaskImageType;
    typedef typename ElastixType::MovingMaskType          MovingMaskImageType;
    typedef typename FixedMaskImageType::Pointer          FixedMaskImagePointer;
    typedef typename MovingMaskImageType::Pointer         MovingMaskImagePointer;
		typedef ImageMaskSpatialObject2<
			itkGetStaticConstMacro( FixedImageDimension ) >			FixedMaskSpatialObjectType;
		typedef ImageMaskSpatialObject2<
			itkGetStaticConstMacro( MovingImageDimension ) >		MovingMaskSpatialObjectType;
		typedef typename 
      FixedMaskSpatialObjectType::Pointer             		FixedMaskSpatialObjectPointer;
		typedef typename 
      MovingMaskSpatialObjectType::Pointer      	    		MovingMaskSpatialObjectPointer;

		/** Execute stuff before the actual registration:
		 * \li Connect all components to the registration framework.
		 * \li Set the number of resolution levels.
		 * \li Set the fixed image region.
		 */
		virtual void BeforeRegistration(void);

    /** Execute stuff before each resolution:
		 * \li Update masks with an erosion. */
    virtual void BeforeEachResolution(void);
				
	protected:

		/** The constructor. */
    MultiResolutionRegistration(){};
		/** The destructor. */
		virtual ~MultiResolutionRegistration() {};
    
    /** Typedef for timer.*/
		typedef tmr::Timer					TimerType;
		/** Typedef for timer.*/
		typedef TimerType::Pointer	TimerPointer;
    
		/** Some typedef's used for eroding the masks*/
    typedef ErodeMaskImageFilter< FixedMaskImageType >   FixedMaskErodeFilterType;
    typedef typename FixedMaskErodeFilterType::Pointer   FixedMaskErodeFilterPointer;
    typedef ErodeMaskImageFilter< MovingMaskImageType >  MovingMaskErodeFilterType;
    typedef typename MovingMaskErodeFilterType::Pointer  MovingMaskErodeFilterPointer;

    /** Function to update masks. */
		void UpdateMasks( unsigned int level );

		/** Read the components from m_Elastix and set them in the Registration class. */
		virtual void SetComponents(void);		
		
	private:

		/** The private constructor. */
		MultiResolutionRegistration( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );							// purposely not implemented

	}; // end class MultiResolutionRegistration


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMultiResolutionRegistration.hxx"
#endif

#endif // end #ifndef __elxMultiResolutionRegistration_H__

#ifndef __elxElastixTemplate_h
#define __elxElastixTemplate_h

#include "elxElastixBase.h"
#include "itkObject.h"

#include "itkObjectFactory.h"
#include "itkCommand.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageToImageMetric.h"

#include "elxRegistrationBase.h"
#include "elxFixedImagePyramidBase.h"
#include "elxMovingImagePyramidBase.h"
#include "elxInterpolatorBase.h"
#include "elxMetricBase.h"
#include "elxOptimizerBase.h"
#include "elxResamplerBase.h"
#include "elxResampleInterpolatorBase.h"
#include "elxTransformBase.h"

#include "elxTimer.h"

#include <sstream>

/** Like itkSetObjectMacro, but in this macro also the m_elx_... member 
 * is set (to the same value, but casted to a pointer to _elxtype).
 * This macro is #undef'ed at the end of this header file.
 */

#define elxSetObjectMacro(_name,_type,_elxtype) \
	virtual void Set##_name (_type * _arg) \
	{ \
		if (this->m_##_name != _arg) \
			{ \
			this->m_##_name = _arg; \
			this->m_elx_##_name = dynamic_cast<_elxtype *>( m_##_name .GetPointer() ); \
			this->Modified(); \
			} \
	}
// end elxSetObjectMacro


#define elxGetBaseMacro(_name,_elxbasetype) \
	virtual _elxbasetype * GetElx##_name##Base (void) \
	{ \
	  return this->m_elx_##_name ; \
	}
//end elxGetBaseMacro


namespace elastix
{
	using namespace itk;
	
	/**
	 * \class ElastixTemplate
	 * \brief ???
	 *
	 * The ElastixTemplate class ....
	 *
	 * \ingroup Kernel
	 */
	
	template <class TFixedImage, class TMovingImage>
		class ElastixTemplate : public Object, public ElastixBase
	{
	public:
		
		/** Standard itk.*/
		typedef ElastixTemplate						Self;
		typedef Object										Superclass1;
		typedef ElastixBase								Superclass2;
		typedef SmartPointer<Self>				Pointer;
		typedef SmartPointer<const Self>	ConstPointer;
		
		/** Method for creation through the object factory.*/
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods).*/
		itkTypeMacro( ElastixTemplate, Object );
		
		/** Typedefs inherited from Superclass2.*/
		typedef Superclass2::ConfigurationType														ConfigurationType;
		typedef Superclass2::ConfigurationPointer													ConfigurationPointer;
		typedef Superclass2::ObjectType																		ObjectType; //for the components
		typedef Superclass2::DataObjectType																DataObjectType; //for the images
		typedef ObjectType::Pointer																				ObjectPointer;
		typedef DataObjectType::Pointer																		DataObjectPointer;
		
		/** Typedef's for this class.*/
		typedef TFixedImage																								FixedImageType;
		typedef TMovingImage																							MovingImageType;
		typedef typename FixedImageType::Pointer													FixedImagePointer;
		typedef typename MovingImageType::Pointer													MovingImagePointer;
		
		/** PixelType for internal calculations.*/
		typedef float																											FixedInternalPixelType;
		typedef float																											MovingInternalPixelType;
		
		/** For using the Dimensions.*/
		itkStaticConstMacro( Dimension,				unsigned int, FixedImageType::ImageDimension );
		itkStaticConstMacro( FixedDimension,	unsigned int, FixedImageType::ImageDimension );
		itkStaticConstMacro( MovingDimension, unsigned int, MovingImageType::ImageDimension );
		
		/** ImageType for internal calculations.*/
		typedef Image<	FixedInternalPixelType,
			::itk::GetImageDimension<FixedImageType>::ImageDimension >			FixedInternalImageType;
		typedef Image<	MovingInternalPixelType,
			::itk::GetImageDimension<MovingImageType>::ImageDimension >			MovingInternalImageType;
		typedef typename FixedInternalImageType::Pointer									FixedInternalImagePointer;
		typedef typename MovingInternalImageType::Pointer									MovingInternalImagePointer;
		
		/** Type for representation of the transform coordinates.*/
		typedef CostFunction::ParametersValueType		CoordRepType; // double
		
		/** Image File Readers.*/
		typedef ImageFileReader< FixedImageType  >												FixedImageReaderType;
		typedef ImageFileReader< MovingImageType >												MovingImageReaderType;
		typedef typename FixedImageReaderType::Pointer										FixedImageReaderPointer;
		typedef typename MovingImageReaderType::Pointer										MovingImageReaderPointer;
		
		/** Casters, to cast the 'DiskImageTypes' to the InternalImageType.*/
		typedef CastImageFilter< 
			FixedImageType, FixedInternalImageType >												FixedImageCasterType;
		typedef CastImageFilter< 
			MovingImageType, MovingInternalImageType >											MovingImageCasterType;
		typedef typename FixedImageCasterType::Pointer										FixedImageCasterPointer;
		typedef typename MovingImageCasterType::Pointer										MovingImageCasterPointer;
		
		/** A Pointer to a memberfunction of and elx::BaseComponent.*/
		typedef void (BaseComponent::*PtrToMemberFunction)(void);
		typedef int (BaseComponent::*PtrToMemberFunction2)(void);
		
		/** Commands that react on Events and call Self::Function(void).*/
		typedef SimpleMemberCommand<Self>																	BeforeEachResolutionCommandType;
		typedef SimpleMemberCommand<Self>																	AfterEachResolutionCommandType;
		typedef SimpleMemberCommand<Self>																	AfterEachIterationCommandType;
		typedef typename BeforeEachResolutionCommandType::Pointer					BeforeEachResolutionCommandPointer;
		typedef typename AfterEachResolutionCommandType::Pointer					AfterEachResolutionCommandPointer;
		typedef typename AfterEachIterationCommandType::Pointer						AfterEachIterationCommandPointer;
		
		
		/** BaseComponent.*/
		typedef BaseComponent																							BaseComponentType;
		
		/** The elastix basecomponent types.*/
		typedef FixedImagePyramidBase<Self> 															FixedImagePyramidBaseType;
		typedef MovingImagePyramidBase<Self>															MovingImagePyramidBaseType;
		typedef InterpolatorBase<Self>																		InterpolatorBaseType;
		typedef MetricBase<Self>																					MetricBaseType;
		typedef OptimizerBase<Self> 																			OptimizerBaseType;
		typedef RegistrationBase<Self>																		RegistrationBaseType;
		typedef ResamplerBase<Self> 																			ResamplerBaseType;
		typedef ResampleInterpolatorBase<Self>														ResampleInterpolatorBaseType;
		typedef TransformBase<Self> 																			TransformBaseType;

		/** Typedef's for Timer class.*/
		typedef tmr::Timer																								TimerType;
		typedef TimerType::Pointer																				TimerPointer;
		
		/** Typedef's for ApplyTransform.*/
		typedef TMovingImage																							InputImageType;
		typedef TFixedImage																								OutputImageType;
		typedef ImageFileReader< InputImageType >													InputImageReaderType;
		typedef ImageFileWriter< OutputImageType >												OutputImageWriterType;
		

		/** Functions to set/get pointers to the elastix components.
		 *
		 * These macros override the pure virtual get/set functions,
		 * which were declared in the class ElastixBase:
		 * virtual ObjectType * GetRegistration(void) = 0;
		 * etc.
		 */
		elxSetObjectMacro( FixedImagePyramid, ObjectType, FixedImagePyramidBaseType );
		elxSetObjectMacro( MovingImagePyramid, ObjectType, MovingImagePyramidBaseType );
		elxSetObjectMacro( Interpolator, ObjectType, InterpolatorBaseType );
		elxSetObjectMacro( Metric, ObjectType, MetricBaseType );
		elxSetObjectMacro( Optimizer, ObjectType, OptimizerBaseType );
		elxSetObjectMacro( Registration, ObjectType, RegistrationBaseType );
		elxSetObjectMacro( Resampler, ObjectType, ResamplerBaseType );
		elxSetObjectMacro( ResampleInterpolator, ObjectType, ResampleInterpolatorBaseType );
		elxSetObjectMacro( Transform, ObjectType, TransformBaseType );
		
		/** Get the components as pointers to ObjectType */
		itkGetObjectMacro( FixedImagePyramid, ObjectType );
		itkGetObjectMacro( MovingImagePyramid, ObjectType );
		itkGetObjectMacro( Interpolator, ObjectType );
		itkGetObjectMacro( Metric, ObjectType );
		itkGetObjectMacro( Optimizer, ObjectType );
		itkGetObjectMacro( Registration, ObjectType );
		itkGetObjectMacro( Resampler, ObjectType );
		itkGetObjectMacro( ResampleInterpolator, ObjectType );
		itkGetObjectMacro( Transform, ObjectType );

		/** Get the components as pointers to elxBaseType */
		elxGetBaseMacro( FixedImagePyramid, FixedImagePyramidBaseType );
		elxGetBaseMacro( MovingImagePyramid, MovingImagePyramidBaseType );
		elxGetBaseMacro( Interpolator, InterpolatorBaseType );
		elxGetBaseMacro( Metric, MetricBaseType );
		elxGetBaseMacro( Optimizer, OptimizerBaseType );
		elxGetBaseMacro( Registration, RegistrationBaseType );
		elxGetBaseMacro( Resampler, ResamplerBaseType );
		elxGetBaseMacro( ResampleInterpolator, ResampleInterpolatorBaseType );
		elxGetBaseMacro( Transform, TransformBaseType );
		
		virtual void SetFixedImage( DataObjectType * _arg );
		virtual void SetMovingImage( DataObjectType * _arg );
		itkGetObjectMacro( FixedImage, DataObjectType );
		itkGetObjectMacro( MovingImage, DataObjectType );
		
		virtual void SetFixedInternalImage( DataObjectType * _arg );
		virtual void SetMovingInternalImage( DataObjectType * _arg );
		itkGetObjectMacro( FixedInternalImage, DataObjectType );
		itkGetObjectMacro( MovingInternalImage, DataObjectType );

		/** Set/Get the initial transform
		 *
		 * The type is ObjectType, but the pointer should actually point 
		 * to an itk::Transform type (or inherited from that one).
		 */
		itkSetObjectMacro( InitialTransform, ObjectType );
		itkGetObjectMacro( InitialTransform, ObjectType );
		
		/** Main functions:
		 * Run() for registration, and ApplyTransform() for just
		 * applying a transform to an image.
		 */
		virtual int Run(void);
		virtual int ApplyTransform(void);
		
		/** The Callback functions. */
		virtual int BeforeAll(void);
		virtual int BeforeAllTransformix(void);
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachResolution(void);
		virtual void AfterEachIteration(void);
		virtual void AfterRegistration(void);

		itkGetConstMacro(IterationCounter, unsigned int);
		
	protected:

		ElastixTemplate(); 
		virtual ~ElastixTemplate(); 
		
		FixedImagePointer	 m_FixedImage;
		MovingImagePointer m_MovingImage;
		FixedInternalImagePointer	 m_FixedInternalImage;
		MovingInternalImagePointer m_MovingInternalImage;
		
		/** The components as smartpointers to itkObjects.*/
		ObjectPointer m_FixedImagePyramid;
		ObjectPointer m_MovingImagePyramid;
		ObjectPointer m_Interpolator;
		ObjectPointer m_Metric;
		ObjectPointer m_Optimizer;
		ObjectPointer m_Registration;
		ObjectPointer m_Resampler;
		ObjectPointer m_ResampleInterpolator;
		ObjectPointer m_Transform;
		
		/** The components as pointers to elx...Base objects.*/
		FixedImagePyramidBaseType *			m_elx_FixedImagePyramid;
		MovingImagePyramidBaseType *		m_elx_MovingImagePyramid;
		InterpolatorBaseType *					m_elx_Interpolator;
		MetricBaseType *								m_elx_Metric;
		OptimizerBaseType *							m_elx_Optimizer;
		RegistrationBaseType *					m_elx_Registration;
		ResamplerBaseType *							m_elx_Resampler;
		ResampleInterpolatorBaseType *	m_elx_ResampleInterpolator;
		TransformBaseType *							m_elx_Transform;
			
		/** Readers and casters.*/
		FixedImageReaderPointer		m_FixedImageReader;
		MovingImageReaderPointer	m_MovingImageReader;
		FixedImageCasterPointer		m_FixedImageCaster;
		MovingImageCasterPointer	m_MovingImageCaster;
		
		/** The initial transform.*/
		ObjectPointer m_InitialTransform;
		
		/** CallBack commands.*/
		BeforeEachResolutionCommandPointer	m_BeforeEachResolutionCommand;
		AfterEachIterationCommandPointer		m_AfterEachIterationCommand;
		AfterEachResolutionCommandPointer		m_AfterEachResolutionCommand;

		/** Timers */
		TimerPointer m_Timer0;
		TimerPointer m_IterationTimer;
		TimerPointer m_ResolutionTimer;

		
		/** Count the number of iterations. */
		unsigned int m_IterationCounter;
		
		/** CreateTransformParameterFile.*/
		virtual void CreateTransformParameterFile( std::string FileName, bool ToLog );

		/** Used by the callback functions, beforeeachresolution() etc.).*/
		void CallInEachComponent( PtrToMemberFunction func );
		int CallInEachComponentInt( PtrToMemberFunction2 func );
		
	private:

		ElastixTemplate( const Self& );	// purposely not implemented
		void operator=( const Self& );	// purposely not implemented
		
	}; // end class ElastixTemplate


} // end namespace elastix

#undef elxSetObjectMacro
#undef elxGetBaseMacro

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxElastixTemplate.hxx"
#endif

#endif // end #ifndef __elxElastixTemplate_h


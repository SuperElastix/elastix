#ifndef __elxMattesMutualInformationMetric_H__
#define __elxMattesMutualInformationMetric_H__

/** For easy changing the pixel type of the mask images: */
#define __MaskFilePixelType char

#include "elxIncludes.h"
#include "itkMattesMutualInformationImageToImageMetricWithMask.h"

#include "elxTimer.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "math.h"
#include <string>

namespace elastix
{
using namespace itk;

	/**
	 * \class MattesMutualInformationMetric
	 * \brief An metric based on mutual information...
	 *
	 * This metric ...
	 *
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
		typedef typename Superclass1::MovingImageConstPointer		MovingImageCosntPointer;
		typedef typename Superclass1::FixedImageIndexType				FixedImageIndexType;
		typedef typename Superclass1::FixedImageIndexValueType 	FixedImageIndexValueType;
		typedef typename Superclass1::MovingImageIndexType 			MovingImageIndexType;
		typedef typename Superclass1::FixedImagePointType				FixedImagePointType;
		typedef typename Superclass1::MovingImagePointType 			MovingImagePointType;
		
		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Other typedef's.*/
		typedef typename Superclass1::MaskPixelType							MaskPixelType;
		typedef typename Superclass1::FixedCoordRepType					FixedCoordRepType;
		typedef typename Superclass1::MovingCoordRepType 				MovingCoordRepType;
		typedef typename Superclass1::FixedMaskImageType 				FixedMaskImageType;
		typedef typename Superclass1::MovingMaskImageType				MovingMaskImageType;
		typedef typename Superclass1::FixedMaskImagePointer			FixedMaskImagePointer;
		typedef typename Superclass1::MovingMaskImagePointer		MovingMaskImagePointer;
		
		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
			
		/** Typedef's for the suppport of masks.*/
		typedef __MaskFilePixelType	MaskFilePixelType; //defined at the top of this file
		typedef FixedCoordRepType		MaskCoordinateType;
		
		typedef MaskImage<
			MaskFilePixelType,
			itkGetStaticConstMacro(MovingImageDimension),
			FixedCoordRepType >				FixedMaskFileImageType;
		typedef MaskImage<
			MaskFilePixelType,
			itkGetStaticConstMacro(MovingImageDimension),
			MovingCoordRepType >			MovingMaskFileImageType;
		
		typedef ImageFileReader<
			FixedMaskFileImageType > 	FixedMaskImageReaderType;
		typedef ImageFileReader<
			MovingMaskFileImageType >	MovingMaskImageReaderType;
		
		typedef typename FixedMaskImageReaderType::Pointer		FixedMaskImageReaderPointer;
		typedef typename MovingMaskImageReaderType::Pointer		MovingMaskImageReaderPointer;

		/** Typedef's for timer.*/
		typedef tmr::Timer					TimerType;
		typedef TimerType::Pointer	TimerPointer;
		
		/** Methods that have to be present everywhere.*/
		virtual int BeforeAll(void);
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachIteration(void);

		virtual void Initialize(void) throw (ExceptionObject);
		
	protected:

		MattesMutualInformationMetric(); 
		virtual ~MattesMutualInformationMetric() {}

		/** Declaration of member variables.*/
		FixedMaskImageReaderPointer		m_FixedMaskImageReader;
		MovingMaskImageReaderPointer	m_MovingMaskImageReader;

		bool m_NewSamplesEveryIteration;
		bool m_ShowExactMetricValue;
				
	private:

		MattesMutualInformationMetric( const Self& );	// purposely not implemented
		void operator=( const Self& );								// purposely not implemented
		
	}; // end class MattesMutualInformationMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMattesMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __elxMattesMutualInformationMetric_H__

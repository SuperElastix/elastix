#ifndef __elxNormalizedCorrelationMetric_H__
#define __elxNormalizedCorrelationMetric_H__

/** For easy changing the pixel type of the mask images: */
#define __MaskFilePixelType char

#include "elxIncludes.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"

#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "math.h"
#include <string>

namespace elastix
{
using namespace itk;

	/**
	 * \class NormalizedCorrelationMetric
	 * \brief An metric based on normalized correlation...
	 *
	 * This metric ...
	 *
	 * \ingroup Metrics
	 */

// TO DO: Add masks.

	template <class TElastix >
		class NormalizedCorrelationMetric:
		public
			NormalizedCorrelationImageToImageMetric<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef NormalizedCorrelationMetric										Self;
		typedef NormalizedCorrelationImageToImageMetric<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( NormalizedCorrelationMetric, NormalizedCorrelationImageToImageMetric );
		
		/** Name of this class.*/
		elxClassNameMacro( "NormalizedCorrelation" );

		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass1::TransformType							TransformType;
		typedef typename Superclass1::TransformPointer 					TransformPointer;
		typedef typename Superclass1::TransformParametersType		TransformParametersType;
		typedef typename Superclass1::TransformJacobianType			TransformJacobianType;
		typedef typename Superclass1::GradientPixelType					GradientPixelType;
		typedef typename Superclass1::MeasureType								MeasureType;
		typedef typename Superclass1::DerivativeType 						DerivativeType;
		typedef typename Superclass1::FixedImageType 						FixedImageType;
		typedef typename Superclass1::MovingImageType						MovingImageType;
		typedef typename Superclass1::FixedImageConstPointer 		FixedImageConstPointer;
		typedef typename Superclass1::MovingImageConstPointer		MovingImageConstPointer;
		
/*
		typedef typename Superclass1::InterpolatorType 					InterpolatorType;
		typedef typename Superclass1::ParametersType 						ParametersType;
		typedef typename Superclass1::FixedImageIndexType				FixedImageIndexType;
		typedef typename Superclass1::FixedImageIndexValueType 	FixedImageIndexValueType;
		typedef typename Superclass1::MovingImageIndexType 			MovingImageIndexType;
		typedef typename Superclass1::FixedImagePointType				FixedImagePointType;
		typedef typename Superclass1::MovingImagePointType 			MovingImagePointType;*/
		
		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Other typedef's.*/
/*		typedef typename Superclass1::MaskPixelType							MaskPixelType;
		typedef typename Superclass1::FixedCoordRepType					FixedCoordRepType;
		typedef typename Superclass1::MovingCoordRepType 				MovingCoordRepType;
		typedef typename Superclass1::FixedMaskImageType 				FixedMaskImageType;
		typedef typename Superclass1::MovingMaskImageType				MovingMaskImageType;
		typedef typename Superclass1::FixedMaskImagePointer			FixedMaskImagePointer;
		typedef typename Superclass1::MovingMaskImagePointer		MovingMaskImagePointer;*/
		
		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
			
		/** Typedef's for the suppport of masks.*/
/*		typedef __MaskFilePixelType	MaskFilePixelType; //defined at the top of this file
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
		typedef CastImageFilter< 
			FixedMaskFileImageType,
			FixedMaskImageType >			FixedMaskCastFilterType;
		typedef CastImageFilter< 
			MovingMaskFileImageType,
			MovingMaskImageType >			MovingMaskCastFilterType;
		
		typedef typename FixedMaskImageReaderType::Pointer		FixedMaskImageReaderPointer;
		typedef typename MovingMaskImageReaderType::Pointer		MovingMaskImageReaderPointer;
		typedef typename FixedMaskCastFilterType::Pointer			FixedMaskCastFilterPointer;
		typedef typename MovingMaskCastFilterType::Pointer		MovingMaskCastFilterPointer;
*/		

		/** Methods that have to be present everywhere.*/
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);

	protected:

		NormalizedCorrelationMetric(); 
		virtual ~NormalizedCorrelationMetric() {}

		/** Declaration of member variables.*/
/*		FixedMaskImageReaderPointer		m_FixedMaskImageReader;
		MovingMaskImageReaderPointer	m_MovingMaskImageReader;
		FixedMaskCastFilterPointer		m_FixedMaskCaster;
		MovingMaskCastFilterPointer		m_MovingMaskCaster;*/
				
	private:

		NormalizedCorrelationMetric( const Self& );	// purposely not implemented
		void operator=( const Self& );							// purposely not implemented
		
	}; // end class NormalizedCorrelationMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxNormalizedCorrelationMetric.hxx"
#endif

#endif // end #ifndef __elxNormalizedCorrelationMetric_H__

#ifndef __elxResamplerBase_h
#define __elxResamplerBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkResampleImageFilter.h"

#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"

namespace elastix
{
	using namespace itk;
	

	/**
	 * ********************** ResamplerBase *************************
	 *
	 * This class 
	 */

	template <class TElastix>
		class ResamplerBase : public BaseComponentSE<TElastix>
	{
	public:
		
		/** Standard stuff.*/
		typedef ResamplerBase								Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedef's from superclass.*/
		typedef typename Superclass::ElastixType					ElastixType;
		typedef typename Superclass::ElastixPointer				ElastixPointer;
		typedef typename Superclass::ConfigurationType		ConfigurationType;
		typedef typename Superclass::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass::RegistrationType			RegistrationType;
		typedef typename Superclass::RegistrationPointer	RegistrationPointer;
		
		/** Typedef's from elastix.*/
		typedef typename ElastixType::MovingImageType			InputImageType;
		typedef typename ElastixType::FixedImageType			OutputImageType;
		typedef typename ElastixType::CoordRepType				CoordRepType;
		
		/** Other typedef's.*/
		typedef ResampleImageFilter<
			InputImageType, OutputImageType, CoordRepType>	ITKBaseType;

		/** Typedef's from ResampleImageFiler.*/
		typedef typename ITKBaseType::TransformType							TransformType;
		typedef typename ITKBaseType::InterpolatorType					InterpolatorType;
		typedef typename ITKBaseType::SizeType									SizeType;
		typedef typename ITKBaseType::IndexType									IndexType;
		typedef typename ITKBaseType::SpacingType								SpacingType;
		typedef typename ITKBaseType::OriginPointType						OriginPointType;
		
		/** Typedefs for Saving.
		 *
		 * SavePixelType is chosen the same as the OutputImage::PixelType,
		 * so in this case the caster is useless.
		 */
		typedef typename ITKBaseType::PixelType				PixelType;
		//typedef typename ITKBaseType::OutputImageType				OutputImageType;

		/** Get the ImageDimension.*/
		itkStaticConstMacro( ImageDimension, unsigned int,
			OutputImageType::ImageDimension );

		typedef PixelType															SavePixelType;
		typedef Image< SavePixelType,
			itkGetStaticConstMacro( ImageDimension ) >	SaveImageType;
		typedef ImageFileWriter< SaveImageType >			WriterType;
		typedef CastImageFilter< 
			OutputImageType, SaveImageType >						CasterType;
		typedef typename WriterType::Pointer					WriterPointer;
		typedef typename CasterType::Pointer					CasterPointer;

		/** ...*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

		/** Methods that have to be present everywhere.*/
		virtual int BeforeAllTransformix(void){ return 0;};
		virtual void BeforeRegistrationBase(void);
		virtual void AfterRegistrationBase(void);

		/** Read/Write Resampler specific things from/to file.*/
		virtual void ReadFromFile(void);
		virtual void WriteToFile(void);

	protected:

		ResamplerBase();
		virtual ~ResamplerBase() {}
		
		/** For saving.*/
		CasterPointer					m_Caster;
		WriterPointer					m_Writer;

		/** Method that sets the transform, the interpolator and the inputImage.*/
		virtual void SetComponents(void);

	private:

		ResamplerBase(const Self&);		// purposely not implemented
		void operator=(const Self&);	// purposely not implemented
		
	}; // end class ResamplerBase
	
	
} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxResamplerBase.hxx"
#endif

#endif // end #ifndef __elxResamplerBase_h

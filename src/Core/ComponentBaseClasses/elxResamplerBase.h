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
	 * \class ResampleBase
	 * \brief This class is the base for all Resamplers.
	 *
	 * This class contains all the common functionality for Resamplers.
	 *
	 * \ingroup Resamplers
	 * \ingroup ComponentBaseClasses
	 */

	template <class TElastix>
		class ResamplerBase : public BaseComponentSE<TElastix>
	{
	public:
		
		/** Standard ITK stuff. */
		typedef ResamplerBase								Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Run-time type information (and related methods). */
		itkTypeMacro( ResamplerBase, BaseComponentSE );

		/** Typedef's from superclass. */
		typedef typename Superclass::ElastixType					ElastixType;
		typedef typename Superclass::ElastixPointer				ElastixPointer;
		typedef typename Superclass::ConfigurationType		ConfigurationType;
		typedef typename Superclass::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass::RegistrationType			RegistrationType;
		typedef typename Superclass::RegistrationPointer	RegistrationPointer;
		
		/** Typedef's from elastix. */
		typedef typename ElastixType::MovingImageType			InputImageType;
		typedef typename ElastixType::FixedImageType			OutputImageType;
		typedef typename ElastixType::CoordRepType				CoordRepType;
		
		/** Other typedef's. */
		typedef ResampleImageFilter<
			InputImageType, OutputImageType, CoordRepType>	ITKBaseType;

		/** Typedef's from ResampleImageFiler. */
		typedef typename ITKBaseType::TransformType							TransformType;
		typedef typename ITKBaseType::InterpolatorType					InterpolatorType;
		typedef typename ITKBaseType::SizeType									SizeType;
		typedef typename ITKBaseType::IndexType									IndexType;
		typedef typename ITKBaseType::SpacingType								SpacingType;
		typedef typename ITKBaseType::OriginPointType						OriginPointType;
		typedef typename ITKBaseType::PixelType									OutputPixelType;

		/** Get the ImageDimension. */
		itkStaticConstMacro( ImageDimension, unsigned int,
			OutputImageType::ImageDimension );

		/** Typedef's for wrinting the output image. */
		typedef ImageFileWriter< OutputImageType >		WriterType;
		typedef typename WriterType::Pointer					WriterPointer;

		/** Cast to ITKBaseType. */
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

		/** Execute stuff before the actual transformation:
		 * \li nothing here
		 */
		virtual int BeforeAllTransformix(void){ return 0;};

		/** Execute stuff before the actual registration:
		 * \li Set all components into the resampler, such as the transform
		 *		interpolator, input.
		 * \li Set output image information, such as size, spacing, etc.
		 * \li Set the default pixel value.
		 */
		virtual void BeforeRegistrationBase(void);

		/** Execute stuff after the registration:
		 * \li Write the resulting output image.
		 */
		virtual void AfterEachResolutionBase(void);

		/** Execute stuff after the registration:
		 * \li Write the resulting output image.
		 */
		virtual void AfterRegistrationBase(void);

		/** Function to read transform-parameters from a file. */
		virtual void ReadFromFile(void);
		/** Function to write transform-parameters to a file. */
		virtual void WriteToFile(void);

		/** Function to write the result output image to a file. */
		virtual void WriteResultImage( const char * filename );

	protected:

		/** The constructor. */
		ResamplerBase();
		/** The destructor. */
		virtual ~ResamplerBase() {}

		/** Method that sets the transform, the interpolator and the inputImage. */
		virtual void SetComponents(void);

	private:

		/** The private constructor. */
		ResamplerBase(const Self&);		// purposely not implemented
		/** The private copy constructor. */
		void operator=(const Self&);	// purposely not implemented
		
	}; // end class ResamplerBase
	
	
} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxResamplerBase.hxx"
#endif

#endif // end #ifndef __elxResamplerBase_h

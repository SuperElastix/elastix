#ifndef __elxMyStandardResampler_h
#define __elxMyStandardResampler_h

#include "itkResampleImageFilter.h"
#include "elxIncludes.h"

namespace elastix
{
	using namespace itk;
	
	/**
	 * \class MyStandardResampler
	 * \brief A MyStandardResampler...
	 *
	 * This MyStandardResampler ...
	 *
	 * \ingroup Resamplers
	 */

	template < class TElastix >	
		class MyStandardResampler :
			public ResamplerBase<TElastix>::ITKBaseType,
			public ResamplerBase<TElastix>
	{
	public:
		
		/** Standard ITK-stuff.*/
		typedef MyStandardResampler															Self;
		typedef typename ResamplerBase<TElastix>::ITKBaseType		Superclass1;
		typedef ResamplerBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>															Pointer;
		typedef SmartPointer<const Self>												ConstPointer;
		
		/** Method for creation through the object factory.*/
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods).*/
		itkTypeMacro( MyStandardResampler, ResampleImageFilter );

		/** Name of this class.*/
		elxClassNameMacro( "DefaultResampler" );
		
		/** Typedef's inherited from the superclass.*/
		typedef typename Superclass1::InputImageType						InputImageType;
		typedef typename Superclass1::OutputImageType						OutputImageType;
		typedef typename Superclass1::InputImagePointer					InputImagePointer;
		typedef typename Superclass1::OutputImagePointer				OutputImagePointer;
		typedef typename Superclass1::InputImageRegionType			InputImageRegionType;
		typedef typename Superclass1::TransformType							TransformType;
		typedef typename Superclass1::TransformPointerType			TransformPointerType;
		typedef typename Superclass1::InterpolatorType					InterpolatorType;
		typedef typename Superclass1::InterpolatorPointerType		InterpolatePointerType;
		typedef typename Superclass1::SizeType									SizeType;
		typedef typename Superclass1::IndexType									IndexType;
		typedef typename Superclass1::PointType									PointType;
		typedef typename Superclass1::PixelType									PixelType;
		typedef typename Superclass1::OutputImageRegionType			OutputImageRegionType;
		typedef typename Superclass1::SpacingType								SpacingType;
		typedef typename Superclass1::OriginPointType						OriginPointType;
		
		/** Typedef's from the ResamplerBase.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
		

		
		/* Nothing to add. In the baseclass already everything is done what should be done */
	
	

	protected:

		MyStandardResampler() {};
		virtual ~MyStandardResampler() {};
		
	private:

		MyStandardResampler( const Self& );	// purposely not implemented
		void operator=( const Self& );			// purposely not implemented
		
	}; // end class MyStandardResampler
	
	
} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMyStandardResampler.hxx"
#endif

#endif // end #ifndef __elxMyStandardResampler_h

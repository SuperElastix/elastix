#ifndef __elxFixedRecursivePyramid_h
#define __elxFixedRecursivePyramid_h

#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class FixedRecursivePyramid
	 * \brief A FixedRecursivePyramid...
	 *
	 * This FixedRecursivePyramid ...
	 *
	 * \ingroup FixedImagePyramids
	 */

	template <class TElastix>	
		class FixedRecursivePyramid :
		public
			RecursiveMultiResolutionPyramidImageFilter<
				ITK_TYPENAME FixedImagePyramidBase<TElastix>::InputImageType,
				ITK_TYPENAME FixedImagePyramidBase<TElastix>::OutputImageType >,
		public
			FixedImagePyramidBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef FixedRecursivePyramid																		Self;
		typedef RecursiveMultiResolutionPyramidImageFilter<
				typename FixedImagePyramidBase<TElastix>::InputImageType,
				typename FixedImagePyramidBase<TElastix>::OutputImageType >	Superclass1;		
		typedef FixedImagePyramidBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>																			Pointer;
		typedef SmartPointer<const Self>																ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( FixedRecursivePyramid, RecursiveMultiResolutionPyramidImageFilter );
		
		/** Name of this class.*/
		elxClassNameMacro( "FixedRecursiveImagePyramid" );

		/** Get the ImageDimension.*/
		itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );
		
		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass1::InputImageType						InputImageType; 
		typedef typename Superclass1::OutputImageType						OutputImageType;
		typedef typename Superclass1::InputImagePointer					InputImagePointer;
		typedef typename Superclass1::OutputImagePointer				OutputImagePointer;
		typedef typename Superclass1::InputImageConstPointer		InputImageConstPointer;

		/** Typedefs inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
		
	protected:

		  FixedRecursivePyramid() {}
			virtual ~FixedRecursivePyramid() {}
			
	private:

		  FixedRecursivePyramid( const Self& );	// purposely not implemented
			void operator=( const Self& );				// purposely not implemented
			
	}; // end class FixedRecursivePyramid
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxFixedRecursivePyramid.hxx"
#endif

#endif // end #ifndef __elxFixedRecursivePyramid_h


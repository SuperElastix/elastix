#ifndef __elxMovingRecursivePyramid_h
#define __elxMovingRecursivePyramid_h

#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;


	/**
	 * ******************** MovingRecursivePyramid ******************
	 *
	 * The MovingRecursivePyramid class ....
	 */

	template <class TElastix>	
		class MovingRecursivePyramid :
		public
			RecursiveMultiResolutionPyramidImageFilter<
				ITK_TYPENAME MovingImagePyramidBase<TElastix>::InputImageType,
				ITK_TYPENAME MovingImagePyramidBase<TElastix>::OutputImageType >,
		public
			MovingImagePyramidBase<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef MovingRecursivePyramid																		Self;
		typedef RecursiveMultiResolutionPyramidImageFilter<
				typename MovingImagePyramidBase<TElastix>::InputImageType,
				typename MovingImagePyramidBase<TElastix>::OutputImageType >	Superclass1;		
		typedef MovingImagePyramidBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>																				Pointer;
		typedef SmartPointer<const Self>																	ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MovingRecursivePyramid, RecursiveMultiResolutionPyramidImageFilter );
		
		/** Name of this class.*/
		elxClassNameMacro( "MovingRecursiveImagePyramid" );

		/** Get the ImageDimension.*/
		itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );
		
		/** Typedefs inherited from Superclass1.*/
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

		MovingRecursivePyramid() {}
		virtual ~MovingRecursivePyramid() {}
		
	private:

		MovingRecursivePyramid( const Self& );	// purposely not implemented
		void operator=( const Self& );					// purposely not implemented
			
	}; // end class MovingRecursivePyramid
		

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMovingRecursivePyramid.hxx"
#endif

#endif // end #ifndef __elxMovingRecursivePyramid_h

#ifndef __elxLinearInterpolator_h
#define __elxLinearInterpolator_h

#include "itkLinearInterpolateImageFunction.h"
#include "elxIncludes.h"

namespace elastix
{

using namespace itk;

	/**
	 * \class LinearInterpolator
	 * \brief An interpolator based on the itkLinearInterpolateImageFunction.
	 *
	 * This interpolator interpolates images using linear interpolation.
	 *
	 * The parameters used in this class are:
	 * \parameter Interpolator: Select this interpolator as follows:\n
	 *		<tt>(Interpolator "LinearInterpolator")</tt>
	 *
	 * \ingroup Interpolators
	 */

	template < class TElastix >
		class LinearInterpolator :
		public
			LinearInterpolateImageFunction<
				ITK_TYPENAME InterpolatorBase<TElastix>::InputImageType,
				ITK_TYPENAME InterpolatorBase<TElastix>::CoordRepType >, 
		public
			InterpolatorBase<TElastix>
	{	
	public:
	
		/** Standard ITK-stuff. */
		typedef LinearInterpolator									Self;
		typedef	LinearInterpolateImageFunction<
			typename InterpolatorBase<TElastix>::InputImageType,
			typename InterpolatorBase<TElastix>::CoordRepType >	Superclass1;		
		typedef InterpolatorBase<TElastix>					Superclass2;		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( LinearInterpolator, LinearInterpolateImageFunction );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific interpolator. \n
		 * example: <tt>(Interpolator "LinearInterpolator")</tt>\n
		 */
		elxClassNameMacro( "LinearInterpolator" );

		/** Get the ImageDimension. */
		itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );
		
		/** Typedefs inherited from the superclass. */
		typedef typename Superclass1::OutputType								OutputType;
		typedef typename Superclass1::InputImageType						InputImageType;
		typedef typename Superclass1::IndexType									IndexType;
		typedef typename Superclass1::ContinuousIndexType				ContinuousIndexType;
		typedef typename Superclass1::PointType									PointType;		
		
		/** Typedefs inherited from Elastix. */
		typedef typename Superclass2::ElastixType								ElastixType;
		typedef typename Superclass2::ElastixPointer						ElastixPointer;
		typedef typename Superclass2::ConfigurationType					ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer			ConfigurationPointer;
		typedef typename Superclass2::RegistrationType					RegistrationType;
		typedef typename Superclass2::RegistrationPointer				RegistrationPointer;
		typedef typename Superclass2::ITKBaseType								ITKBaseType;

	protected:

		/** The constructor. */
		LinearInterpolator() {}
		/** The destructor. */
		virtual ~LinearInterpolator() {}
		
	private:

		/** The private constructor. */
		LinearInterpolator( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );			// purposely not implemented
			
	}; // end class LinearInterpolator


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxLinearInterpolator.hxx"
#endif

#endif // end #ifndef __elxLinearInterpolator_h


#ifndef __elxBSplineInterpolator_h
#define __elxBSplineInterpolator_h

#include "itkBSplineInterpolateImageFunction.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class BSplineInterpolator
	 * \brief An interpolator based on the itkBSplineInterpolateImageFunction.
	 *
	 * This interpolator interpolates images with an underlying B-spline
	 * polynomial.
	 *
	 * The parameters used in this class are:
	 * \parameter Interpolator: Select this interpolator as follows:\n
	 *		<tt>(Interpolator "BSplineInterpolator")</tt>
	 * \parameter BSplineInterpolationOrder: the order of the B-spline polynomial. \n
	 *		example: <tt>(BSplineInterpolationOrder 3)</tt> \n
	 *		The default order is 1.
	 *
	 * \ingroup Interpolators
	 */

	template < class TElastix >
		class BSplineInterpolator :
		public
			BSplineInterpolateImageFunction<
				ITK_TYPENAME InterpolatorBase<TElastix>::InputImageType,
				ITK_TYPENAME InterpolatorBase<TElastix>::CoordRepType,
				double > , //CoefficientType 
		public
			InterpolatorBase<TElastix>
	{	
	public:
	
		/** Standard ITK-stuff.*/
		typedef BSplineInterpolator									Self;
		typedef	BSplineInterpolateImageFunction<
			typename InterpolatorBase<TElastix>::InputImageType,
			typename InterpolatorBase<TElastix>::CoordRepType,
			double >																	Superclass1;		
		typedef InterpolatorBase<TElastix>					Superclass2;		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( BSplineInterpolator, BSplineInterpolateImageFunction );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific interpolator. \n
		 * example: <tt>(Interpolator "BSplineInterpolator")</tt>\n
		 */
		elxClassNameMacro( "BSplineInterpolator" );

		/** Get the ImageDimension.*/
		itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );
		
		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass1::OutputType								OutputType;
		typedef typename Superclass1::InputImageType						InputImageType;
		typedef typename Superclass1::IndexType									IndexType;
		typedef typename Superclass1::ContinuousIndexType				ContinuousIndexType;
		typedef typename Superclass1::PointType									PointType;
		
		typedef typename Superclass1::Iterator									Iterator;
		typedef typename Superclass1::CoefficientDataType				CoefficientDataType;
		typedef typename Superclass1::CoefficientImageType			CoefficientImageType;
		typedef typename Superclass1::CoefficientFilter					CoefficientFilter;
		typedef typename Superclass1::CoefficientFilterPointer	CoefficientFilterPointer;
		typedef typename Superclass1::CovariantVectorType				CovariantVectorType;
		
		/** Typedefs inherited from Elastix.*/
		typedef typename Superclass2::ElastixType								ElastixType;
		typedef typename Superclass2::ElastixPointer						ElastixPointer;
		typedef typename Superclass2::ConfigurationType					ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer			ConfigurationPointer;
		typedef typename Superclass2::RegistrationType					RegistrationType;
		typedef typename Superclass2::RegistrationPointer				RegistrationPointer;
		typedef typename Superclass2::ITKBaseType								ITKBaseType;

		/** Overriding some inherited functions */
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);

	protected:

		BSplineInterpolator() {}
		virtual ~BSplineInterpolator() {}
		
	private:

		BSplineInterpolator( const Self& );	// purposely not implemented
		void operator=( const Self& );			// purposely not implemented
			
	}; // end class BSplineInterpolator


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineInterpolator.hxx"
#endif

#endif // end #ifndef __elxBSplineInterpolator_h


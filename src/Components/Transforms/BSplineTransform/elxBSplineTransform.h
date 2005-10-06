#ifndef __elxBSplineTransform_h
#define __elxBSplineTransform_h

/* For easy changing the BSplineOrder: */
#define __VSplineOrder 3
/* For easy changing the PixelType of the saved deformation fields: */
#define __CoefficientOutputType float

#include "itkBSplineDeformableTransform.h"
#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"

#include "elxIncludes.h"

#include "itkBSplineTransformGrouper.h"

#include <sstream>

namespace elastix
{
using namespace itk;

	/**
	 * \class BSplineTransform
	 * \brief A transform based on the itkBSplineTransform.
	 *
	 * This transform is a B-spline transformation, commonly used for nonrigid registration.
	 *
	 * The parameters used in this class are:
	 * \parameter Transform: Select this transform as follows:\n
	 *		<tt>(Transform "BSplineTransform")</tt>
	 * \parameter FinalGridSpacing: the grid spacing of the B-spline transform for each dimension. \n
	 *		example: <tt>(FinalGridSpacing 8.0 8.0 8.0)</tt> \n
	 *    If only one argument is given, that factor is used for each dimension. The spacing
	 *    is not in millimeters, but in "voxel size units".
	 *		The default is 8.0 in every dimension.
	 * \parameter UpsampleGridOption: whether or not the B-spline grid should
	 *		be upsampled from one resolution level to another. Choose from {true, false}. \n
	 *		example: <tt>(UpsampleGridOption "true")</tt>
	 *		The default is "true".
	 * 
	 * The transform parameters necessary for transformix, additionally defined by this class, are:
	 * \transformparameter GridSize: stores the size of the B-spline grid. \n
	 *		example: <tt>(GridSize 16 16 16)</tt>
	 * \transformparameter GridIndex: stores the index of the B-spline grid. \n
	 *		example: <tt>(GridIndex 0 0 0)</tt>
	 * \transformparameter GridSpacing: stores the spacing of the B-spline grid. \n
	 *		example: <tt>(GridSpacing 16.0 16.0 16.0)</tt>
	 * \transformparameter GridOrigin: stores the origin of the B-spline grid. \n
	 *		example: <tt>(GridOrigin 0.0 0.0 0.0)</tt>
	 *
	 * \ingroup Transforms
	 */

	template < class TElastix >
		class BSplineTransform:
	public
		BSplineTransformGrouper< 
			BSplineDeformableTransform<
			  ITK_TYPENAME elx::TransformBase<TElastix>::CoordRepType,
				elx::TransformBase<TElastix>::FixedImageDimension,
				__VSplineOrder >		>,
	public
		TransformBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef BSplineTransform 										Self;
		typedef BSplineDeformableTransform<
			typename elx::TransformBase<TElastix>::CoordRepType,
			elx::TransformBase<TElastix>::FixedImageDimension,
			__VSplineOrder >													Superclass1;
		typedef elx::TransformBase<TElastix>				Superclass2;		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( BSplineTransform, BSplineDeformableTransform );

		/** Name of this class.
		 * Use this name in the parameter file to select this specific transform. \n
		 * example: <tt>(Transform "BSplineTransform")</tt>\n
		 */
		elxClassNameMacro( "BSplineTransform" );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
		
		/** The BSpline order. */
		itkStaticConstMacro( SplineOrder, unsigned int, __VSplineOrder );
		
		/** Typedefs inherited from the superclass. */
		typedef typename Superclass1::ScalarType 								ScalarType;
		typedef typename Superclass1::ParametersType 						ParametersType;
		typedef typename Superclass1::JacobianType 							JacobianType;
		typedef typename Superclass1::InputVectorType						InputVectorType;
		typedef typename Superclass1::OutputVectorType 					OutputVectorType;
		typedef typename Superclass1::InputCovariantVectorType 	InputCovariantVectorType;
		typedef typename Superclass1::OutputCovariantVectorType	OutputCovariantVectorType;
		typedef typename Superclass1::InputVnlVectorType 				InputVnlVectorType;
		typedef typename Superclass1::OutputVnlVectorType				OutputVnlVectorType;
		typedef typename Superclass1::InputPointType 						InputPointType;
		typedef typename Superclass1::OutputPointType						OutputPointType;
		
		/** Typedef's specific for the BSplineTransform. */
		typedef typename Superclass1::PixelType									PixelType;
		typedef typename Superclass1::ImageType									ImageType;
		typedef typename Superclass1::ImagePointer							ImagePointer;
		typedef typename Superclass1::RegionType								RegionType;
		typedef typename Superclass1::IndexType									IndexType;
		typedef typename Superclass1::SizeType									SizeType;
		typedef typename Superclass1::SpacingType								SpacingType;
		typedef typename Superclass1::OriginType								OriginType;
		typedef typename Superclass1::BulkTransformType					BulkTransformType;
		typedef typename Superclass1::BulkTransformPointer			BulkTransformPointer;
		typedef typename Superclass1::WeightsFunctionType				WeightsFunctionType;
		typedef typename Superclass1::WeightsType								WeightsType;
		typedef typename Superclass1::ContinuousIndexType				ContinuousIndexType;
		typedef typename Superclass1::ParameterIndexArrayType		ParameterIndexArrayType;

		/** Typedef's from TransformBase. */
		typedef typename Superclass2::ElastixType								ElastixType;
		typedef typename Superclass2::ElastixPointer						ElastixPointer;
		typedef typename Superclass2::ConfigurationType					ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer			ConfigurationPointer;
		typedef typename Superclass2::RegistrationType					RegistrationType;
		typedef typename Superclass2::RegistrationPointer				RegistrationPointer;
		typedef typename Superclass2::CoordRepType							CoordRepType;
		typedef typename Superclass2::FixedImageType						FixedImageType;
		typedef typename Superclass2::MovingImageType						MovingImageType;
		typedef typename Superclass2::ITKBaseType								ITKBaseType;

				
		/** Typedefs for saving the deformation fields. */
		typedef __CoefficientOutputType								CoefficientOutputType;
		typedef Image< CoefficientOutputType,
			itkGetStaticConstMacro(SpaceDimension) >		CoefficientOutputImageType;
		typedef CastImageFilter< 
			ImageType, CoefficientOutputImageType >			TransformCastFilterType;
		typedef ImageFileWriter<
			CoefficientOutputImageType >								TransformWriterType;

		/** Methods in which parameters can be changed; automatically called by elastix. */
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		
		/** These two methods are called by BeforeEachResolution(). */
		virtual void SetInitialGrid( bool upsampleGridOption );
		virtual void IncreaseScale(void);
		
		/** Function to read/write transform-parameters from/to a file. */
		virtual void ReadFromFile(void);
		virtual void WriteToFile( const ParametersType & param );
		
	protected:

		BSplineTransform();
		virtual ~BSplineTransform() {}
		
		/** Member variables. */
		typename TransformCastFilterType::Pointer		m_Caster;
		typename TransformWriterType::Pointer				m_Writer;
		SpacingType																	m_GridSpacingFactor;

		ParametersType * m_Parameterspointer;
	  ParametersType * m_Parameterspointer_out;
	
		ImagePointer m_Coeffs1;
	  ImagePointer m_Coeffs2;

	private:
		BSplineTransform( const Self& );	// purposely not implemented
		void operator=( const Self& );		// purposely not implemented
		
	}; // end class BSplineTransform
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineTransform.hxx"
#endif

#endif // end #ifndef __elxBSplineTransform_h


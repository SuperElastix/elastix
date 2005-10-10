#ifndef __elxDeformationFieldTransform_H__
#define __elxDeformationFieldTransform_H__

#include "itkDeformationVectorFieldTransform.h"

#include "elxIncludes.h"
#include "itkBSplineTransformGrouper.h"


namespace elastix
{
using namespace itk;

	/**
	 * \class DeformationFieldTransform
	 * \brief A transform based on a DeformationVectorField.
	 *
	 * This transform models the transformation by a deformation vector field.
	 *
	 * The parameters used in this class are:
	 * \parameter Transform: Select this transform as follows:\n
	 *		<tt>(Transform "DeformationFieldTransform")</tt>
	 *
	 * The transform parameters necessary for transformix, additionally defined by this class, are:
	 * \transformparameter DeformationFieldFileName: stores the name of the deformation field. \n
	 *		example: <tt>(DeformationFieldFileName "defField.mhd")</tt>
	 *
	 * \sa DeformationVectorFieldTransform
	 *
	 * \ingroup Transforms
	 */

	template < class TElastix >
		class DeformationFieldTransform:
	public
		BSplineTransformGrouper< 
			DeformationVectorFieldTransform<
			  ITK_TYPENAME elx::TransformBase<TElastix>::CoordRepType,			
				elx::TransformBase<TElastix>::FixedImageDimension > >,
	public
		TransformBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef DeformationFieldTransform											Self;
		typedef DeformationVectorFieldTransform<
			typename elx::TransformBase< TElastix >::CoordRepType,
			elx::TransformBase< TElastix >::FixedImageDimension >		Superclass1;
		typedef elx::TransformBase< TElastix >								Superclass2;		
		typedef SmartPointer< Self >													Pointer;
		typedef SmartPointer< const Self >										ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( DeformationFieldTransform, DeformationVectorFieldTransform );

		/** Name of this class.
		 * Use this name in the parameter file to select this specific transform. \n
		 * example: <tt>(Transform "DeformationFieldTransform")</tt>\n
		 */
		elxClassNameMacro( "DeformationFieldTransform" );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
		
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
		
		/** Typedef's specific for the DeformationVectorFieldTransform. */
		typedef typename Superclass1::PixelType									PixelType;
		typedef typename Superclass1::ImageType									ImageType;
		typedef typename Superclass1::ImagePointer							ImagePointer;
		typedef typename Superclass1::VectorPixelType						VectorPixelType;
		typedef typename Superclass1::VectorImageType						VectorImageType;
		typedef typename Superclass1::VectorImagePointer				VectorImagePointer;

		/** Typedef's for BulkTransform. */
		typedef typename Superclass1::BulkTransformType					BulkTransformType;
		typedef typename Superclass1::BulkTransformPointer			BulkTransformPointer;

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

		/** Execute stuff before the actual registration:
		 * \li nothing here
		 */
		//virtual void BeforeRegistration(void);
		//virtual void BeforeEachResolution(void);
	
		/** Function to read transform-parameters from a file. */
		virtual void ReadFromFile(void);
		/** Function to write transform-parameters to a file. */
		virtual void WriteToFile( const ParametersType & param );

	protected:

		/** The constructor. */
		DeformationFieldTransform();
		/** The destructor. */
		virtual ~DeformationFieldTransform() {};
		
	private:

		/** The private constructor. */
		DeformationFieldTransform( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );						// purposely not implemented
		
	}; // end class DeformationFieldTransform
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxDeformationFieldTransform.hxx"
#endif

#endif // end #ifndef __elxDeformationFieldTransform_H__


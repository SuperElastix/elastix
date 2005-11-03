#ifndef __elxAffineTransform_H_
#define __elxAffineTransform_H_

#include "itkAffineTransform.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class AffineTransformElastix
	 * \brief A transform based on the itk::AffineTransform.
	 *
	 * This transform is an affine transformation.
	 *
	 * The first couple of parameters (4 in 2D and 9 in 3D) define the affine
	 * matrix, the last couple (2 in 2D and 3 in 3D) define the translation.
	 *
	 * The parameters used in this class are:
	 * \parameter Transform: Select this transform as follows:\n
	 *		<tt>(Transform "AffineTransform")</tt>
	 * \parameter Scales: the scale factor between the rotations and translations,
	 *		used in the optimizer. \n
	 *		example: <tt>(Scaler 200000.0)</tt> \n
	 *		example: <tt>(Scaler 100000.0 60000.0 ... 80000.0)</tt> \n
	 *    If only one argument is given, that factor is used for the rotations.
	 *		If more than one argument is given, then the number of arguments should be
	 *		equal to the number of parameters: for each parameter its scale factor.
	 *		If this parameter option is not used, by default the rotations are scaled
	 *		by a factor of 100000.0.
	 * \parameter CenterOfRotation: an index around which the image is rotated. \n
	 *		example: <tt>(CenterOfRotation 128 128 90)</tt> \n
	 *		By default the CenterOfRotation is set to the center of the image.
	 *
	 * The transform parameters necessary for transformix, additionally defined by this class, are:
	 * \transformparameter CenterOfRotation: stores the center of rotation. \n
	 *		example: <tt>(CenterOfRotation 128 128 90)</tt>
	 *
	 * \ingroup Transforms
	 */
	
	template < class TElastix >
		class AffineTransformElastix :
			public TransformGrouper<
				AffineTransform<
					ITK_TYPENAME elx::TransformBase<TElastix>::CoordRepType,
					elx::TransformBase<TElastix>::FixedImageDimension >	>,
			public elx::TransformBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef AffineTransformElastix														Self;
		typedef AffineTransform<
			typename elx::TransformBase<TElastix>::CoordRepType,
			elx::TransformBase<TElastix>::FixedImageDimension >			Superclass1;
		typedef elx::TransformBase<TElastix>											Superclass2;
		typedef SmartPointer<Self>																Pointer;
		typedef SmartPointer<const Self>													ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( AffineTransformElastix, AffineTransform );

		/** Name of this class.
		 * Use this name in the parameter file to select this specific transform. \n
		 * example: <tt>(Transform "AffineTransform")</tt>\n
		 */
		elxClassNameMacro( "AffineTransform" );
		
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
		
		/** Typedef's from the TransformBase class. */
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
		
		/** Other typedef's. */
		typedef typename FixedImageType::IndexType							IndexType;
		typedef typename IndexType::IndexValueType							IndexValueType;
		typedef typename FixedImageType::SizeType								SizeType;
		typedef typename FixedImageType::PointType							PointType;
		typedef typename FixedImageType::SpacingType						SpacingType;
		typedef typename FixedImageType::RegionType							RegionType;
		typedef typename RegistrationType::ITKBaseType					ITKRegistrationType;
		typedef typename ITKRegistrationType::OptimizerType			OptimizerType;
		typedef typename OptimizerType::ScalesType							ScalesType;
		typedef typename Superclass2::DummyImageType						DummyImageType;
		
		/** Execute stuff before the actual registration:
		 * \li Create initial registration parameters.
		 * \li Set the center of rotation.
		 * \li Set the scale of the parameters
		 */
		virtual void BeforeRegistration(void);
		
		/** Function to read transform-parameters from a file. */
		virtual void ReadFromFile(void);
		/** Function to write transform-parameters to a file. */
		virtual void WriteToFile( const ParametersType & param );

		/** Function to calculate the center of rotation. */
		void CalculateRotationPoint( InputPointType & rotationPoint );
		
	protected:

		/** The constructor. */
		AffineTransformElastix();
		/** The destructor. */
		virtual ~AffineTransformElastix() {};
		
	private:

		/** The private constructor. */
		AffineTransformElastix( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );					// purposely not implemented
		
	}; // end class AffineTransformElastix
	
	
} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAffineTransform.hxx"
#endif

#endif // end #ifndef __elxAffineTransform_H_

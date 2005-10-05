#ifndef __elxEulerTransform_H__
#define __elxEulerTransform_H__

#include "itkEulerTransform.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class EulerTransformElastix
	 * \brief A transform based on the itk EulerTransforms.
	 *
	 * This transform is a rigid body transformation.
	 *
	 * The parameters used in this class are:
	 * \parameter Transform: Select this transform as follows:\n
	 *		<tt>(Transform "EulerTransform")</tt>
	 * \parameter Scaler: the scale factor between the rotations and translations,
	 *		used in the optimizer. \n
	 *		example: <tt>(Scaler 100000.0)</tt> \n
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
		class EulerTransformElastix:
	    public 	TransformGrouper<
		    EulerTransform<
		    ITK_TYPENAME elx::TransformBase< TElastix >::CoordRepType,
			  elx::TransformBase< TElastix >::FixedImageDimension >	>,
			public elx::TransformBase< TElastix >
	{
	public:
		
		/** Standard ITK-stuff.*/
		typedef EulerTransformElastix																Self;
		typedef EulerTransform<
			typename elx::TransformBase< TElastix >::CoordRepType,
			elx::TransformBase< TElastix >::FixedImageDimension >			Superclass1;
		typedef elx::TransformBase< TElastix >											Superclass2;
		typedef SmartPointer<Self>																	Pointer;
		typedef SmartPointer<const Self>														ConstPointer;
		
		/** Method for creation through the object factory.*/
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods).*/
		itkTypeMacro( EulerTransformElastix, EulerTransform );

		/** Name of this class.
		 * Use this name in the parameter file to select this specific transform. \n
		 * example: <tt>(Transform "EulerTransform")</tt>\n
		 */
		elxClassNameMacro( "EulerTransform" );
		
		/** Dimension of the domain space.*/
		itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
		
		/** Typedefs inherited from the superclass.*/

		/** These are both in Euler2D and Euler3D.*/
		typedef typename Superclass1::ScalarType									ScalarType;
		typedef typename Superclass1::ParametersType							ParametersType;
		typedef typename Superclass1::JacobianType								JacobianType;
		typedef typename Superclass1::OffsetType									OffsetType;
		typedef typename Superclass1::InputPointType							InputPointType;
		typedef typename Superclass1::OutputPointType							OutputPointType;
		typedef typename Superclass1::InputVectorType							InputVectorType;
		typedef typename Superclass1::OutputVectorType						OutputVectorType;
		typedef typename Superclass1::InputCovariantVectorType		InputCovariantVectorType;
		typedef typename Superclass1::OutputCovariantVectorType		OutputCovariantVectorType;
		typedef typename Superclass1::InputVnlVectorType					InputVnlVectorType;
		typedef typename Superclass1::OutputVnlVectorType					OutputVnlVectorType;
		
		/** NOTE: use this one only in 3D (otherwise it's just an int).*/
		typedef typename Superclass1::AngleType										AngleType;
		
		/** Typedef's inherited from TransformBase.*/
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
		
		/** Other typedef's.*/
		typedef typename RegistrationType::ITKBaseType					ITKRegistrationType;
		typedef typename ITKRegistrationType::OptimizerType			OptimizerType;
		typedef typename OptimizerType::ScalesType							ScalesType;

		typedef typename FixedImageType::IndexType							IndexType;
		typedef typename IndexType::IndexValueType							IndexValueType;
		typedef typename FixedImageType::SizeType								SizeType;
		typedef typename FixedImageType::PointType							PointType;
		typedef typename FixedImageType::SpacingType						SpacingType;
		typedef typename FixedImageType::RegionType							RegionType;
		
		/** Methods that have to be present in each version of MyTransform.*/
		virtual void BeforeRegistration(void);

		/** Calculate the center of rotation or use a user specified one. */
		void CalculateRotationPoint( InputPointType & rotationPoint );

		/** Functions to read/write transform-parameters from/to a file. */
		virtual void ReadFromFile(void);
		virtual void WriteToFile( const ParametersType & param );

	protected:

		EulerTransformElastix();
		virtual ~EulerTransformElastix() {};
		
		/** Variables that will store the program arguments.*/
		
	private:

		EulerTransformElastix( const Self& );	// purposely not implemented
		void operator=( const Self& );				// purposely not implemented
		
	}; // end class EulerTransformElastix
	
	
} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxEulerTransform.hxx"
#endif

#endif // end #ifndef __elxEulerTransform_H__


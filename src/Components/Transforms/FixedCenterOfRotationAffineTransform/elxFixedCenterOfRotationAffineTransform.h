#ifndef __elxFixedCenterOfRotationAffineTransform_H_
#define __elxFixedCenterOfRotationAffineTransform_H_

#include "itkFixedCenterOfRotationAffineTransform.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class FixedCenterOfRotationAffineTransformElastix
	 * \brief A transform based on the itk FixedCenterOfRotationAffineTransform
	 *
	 * This transform is an affine transformation...
	 *
	 * \ingroup Transforms
	 */
	
	template < class TElastix >
		class FixedCenterOfRotationAffineTransformElastix :
			public TransformGrouper<
				FixedCenterOfRotationAffineTransform<
					ITK_TYPENAME elx::TransformBase<TElastix>::CoordRepType,
					elx::TransformBase<TElastix>::FixedImageDimension >	>,
			public elx::TransformBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef FixedCenterOfRotationAffineTransformElastix			Self;
		typedef FixedCenterOfRotationAffineTransform<
			typename elx::TransformBase<TElastix>::CoordRepType,
			elx::TransformBase<TElastix>::FixedImageDimension >				Superclass1;
		typedef elx::TransformBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>															Pointer;
		typedef SmartPointer<const Self>												ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( FixedCenterOfRotationAffineTransformElastix, TranslationTransform );

		/** Name of this class.*/
		elxClassNameMacro( "FixedCenterOfRotationAffineTransform" );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);
		
		/** Typedefs inherited from the superclass.*/
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
		
		/** Typedef's from the TransformBase class.*/
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
		
		/** Methods that have to be present in each version of MyTransform.*/
		virtual void BeforeRegistration(void);
		
		/** Function to read/write transform-parameters from/to a file.*/
		virtual void ReadFromFile(void);
		virtual void WriteToFile( const ParametersType & param );

		/** To Set the center of rotation.*/
		void CalculateRotationPoint( InputPointType & rotationPoint );
		
	protected:

		FixedCenterOfRotationAffineTransformElastix();
		virtual ~FixedCenterOfRotationAffineTransformElastix() {};
		
	private:

		FixedCenterOfRotationAffineTransformElastix( const Self& );	// purposely not implemented
		void operator=( const Self& );															// purposely not implemented
		
	}; // end class FixedCenterOfRotationAffineTransformElastix
	
	
} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxFixedCenterOfRotationAffineTransform.hxx"
#endif

#endif // end #ifndef __elxFixedCenterOfRotationAffineTransform_H_

#ifndef __elxAffineTransform_H_
#define __elxAffineTransform_H_

#include "itkAffineTransform.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class AffineTransformElastix
	 * \brief A transform based on the itkAffineTransform
	 *
	 * This transform is an affine transformation...
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

		/** Standard ITK-stuff.*/
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

		/** Name of this class.*/
		elxClassNameMacro( "AffineTransform" );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
		
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

		AffineTransformElastix();
		virtual ~AffineTransformElastix() {};
		
	private:

		AffineTransformElastix( const Self& );	// purposely not implemented
		void operator=( const Self& );					// purposely not implemented
		
	}; // end class AffineTransformElastix
	
	
} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAffineTransform.hxx"
#endif

#endif // end #ifndef __elxAffineTransform_H_

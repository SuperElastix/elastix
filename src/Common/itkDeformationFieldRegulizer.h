#ifndef __itkDeformationFieldRegulizer_H__
#define __itkDeformationFieldRegulizer_H__

#include "itkDeformationVectorFieldTransform.h"
#include "itkImageRegionIterator.h"


namespace itk
{
	
	/**
	 * \class DeformationFieldRegulizer
	 * \brief This class combines any itk transform with the
	 * DeformationFieldTransform.
	 *
	 * This class is a base class for Transforms that also use
	 * a diffusion / regularization of the deformation field.
	 *
	 * \ingroup Transforms
	 * \ingroup Common
	 */
	
	template <class TAnyITKTransform>
	class DeformationFieldRegulizer :
		public TAnyITKTransform
	{
	public:
		
		/** Standard itk. */
		typedef DeformationFieldRegulizer		Self;
		typedef TAnyITKTransform						Superclass;
		typedef SmartPointer< Self >				Pointer;
		typedef SmartPointer< const Self >	ConstPointer;
		
		/** New method for creating an object using a factory. */
		itkNewMacro( Self );
		
		/** Itk Type info. */
		itkTypeMacro( DeformationFieldRegulizer, TAnyITKTransform );
		
		/** Input and Output space dimension. */
		itkStaticConstMacro( InputSpaceDimension, unsigned int, Superclass::InputSpaceDimension );
		itkStaticConstMacro( OutputSpaceDimension, unsigned int, Superclass::OutputSpaceDimension );
		
		/** Typedef's inherited from Superclass. */
		typedef typename Superclass::ScalarType 								ScalarType;
		typedef typename Superclass::ParametersType 						ParametersType;
		typedef typename Superclass::JacobianType 							JacobianType;
		typedef typename Superclass::InputVectorType						InputVectorType;
		typedef typename Superclass::OutputVectorType 					OutputVectorType;
		typedef typename Superclass::InputCovariantVectorType 	InputCovariantVectorType;
		typedef typename Superclass::OutputCovariantVectorType	OutputCovariantVectorType;
		typedef typename Superclass::InputVnlVectorType 				InputVnlVectorType;
		typedef typename Superclass::OutputVnlVectorType				OutputVnlVectorType;
		typedef typename Superclass::InputPointType 						InputPointType;
		typedef typename Superclass::OutputPointType						OutputPointType;

		/** Typedef's needed in this class. */
		typedef DeformationVectorFieldTransform<
			ScalarType, InputSpaceDimension >											IntermediaryDFTransformType;
		typedef typename IntermediaryDFTransformType::VectorImageType				VectorImageType;
		typedef typename IntermediaryDFTransformType::VectorImagePointer		VectorImagePointer;
		typedef typename VectorImageType::PixelType							VectorPixelType;
		typedef ImageRegionIterator< VectorImageType >					IteratorType;

		/** Typedef's for the vectorImage. */
		typedef typename VectorImageType::RegionType						RegionType;
		typedef typename VectorImageType::SpacingType						SpacingType;
		typedef typename VectorImageType::PointType							OriginType;

		/** Function to update the intermediary deformation field by adding
		 * a diffused deformation field to it.
		 */
		virtual void UpdateIntermediaryDeformationFieldTransform( VectorImagePointer vecImage );

		/** itk Set macro for the region of the deformation field. */
		itkSetMacro( DeformationFieldRegion, RegionType );

		/** itk Set macro for the spacing of the deformation field. */
		itkSetMacro( DeformationFieldSpacing, SpacingType );

		/** itk Set macro for the origin of the deformation field. */
		itkSetMacro( DeformationFieldOrigin, OriginType );

		/** Function to create and initialze the IntermediaryDF. */
		void InitializeIntermediaryDeformationField( void );

		/** Method to transform a point. */
		virtual OutputPointType TransformPoint( const InputPointType & point ) const;
		
	protected:
		
		/** The constructor. */
		DeformationFieldRegulizer();
		/** The destructor. */
		virtual ~DeformationFieldRegulizer();
		
	private:

		/** The private constructor. */
		DeformationFieldRegulizer( const Self& );	// purposely not implemented
		void operator=( const Self& );						// purposely not implemented
		
		/** Declaration of members. */
		typename IntermediaryDFTransformType::Pointer		m_IntermediaryDeformationFieldTransform;
		VectorImagePointer										m_IntermediaryDeformationField;
		bool		m_Initialized;

		/** Declarations of region things. */
		RegionType															m_DeformationFieldRegion;
		OriginType															m_DeformationFieldOrigin;
		SpacingType															m_DeformationFieldSpacing;

	}; // end class DeformationFieldRegulizer
		
		
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDeformationFieldRegulizer.hxx"
#endif

#endif // end #ifndef __itkDeformationFieldRegulizer_H__


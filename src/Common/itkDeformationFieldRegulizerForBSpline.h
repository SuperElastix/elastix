#ifndef __itkDeformationFieldRegulizerForBSpline_H__
#define __itkDeformationFieldRegulizerForBSpline_H__

#include "itkDeformationFieldRegulizer.h"


namespace itk
{
		
	/**
	 * \class DeformationFieldRegulizerForBSpline
	 * \brief This class combines a B-spline transform with the
	 * DeformationFieldTransform.
	 *
	 * This class inherits from the DeformationFieldRegulizer and only
	 * overwrites the TransformPoint() function. This is necessary for
	 * the Mattes MI metric. This class is templated over TAnyITKTransform,
	 * but it is in fact only the BSplineDeformableTransform.
	 *
	 * \ingroup Transforms
	 * \ingroup Common
	 */
	
	template < class TBSplineTransform >
	class DeformationFieldRegulizerForBSpline :
		public DeformationFieldRegulizer<
		TBSplineTransform >
	{
	public:
		
		/** Standard itk.*/
		typedef DeformationFieldRegulizerForBSpline			Self;
		typedef DeformationFieldRegulizer<
			TBSplineTransform >														Superclass;
		typedef SmartPointer< Self >										Pointer;
		typedef SmartPointer< const Self >							ConstPointer;
		
		/** New method for creating an object using a factory.*/
		itkNewMacro( Self );
		
		/** Itk Type info */
		itkTypeMacro( DeformationFieldRegulizerForBSpline, DeformationFieldRegulizer );
		
		/** Get	the dimension of the input space. */
		itkStaticConstMacro( InputSpaceDimension, unsigned int, Superclass::InputSpaceDimension );
		/** Get	the dimension of the output space. */
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

		/** Typedef's inherited from the BSplineTransform. */
		typedef TBSplineTransform			BSplineTransformType;
		typedef typename BSplineTransformType::PixelType									PixelType;
		typedef typename BSplineTransformType::ImageType									ImageType;
		typedef typename BSplineTransformType::ImagePointer								ImagePointer;
		typedef typename BSplineTransformType::RegionType									RegionType;
		typedef typename BSplineTransformType::IndexType									IndexType;
		typedef typename BSplineTransformType::SizeType										SizeType;
		typedef typename BSplineTransformType::SpacingType								SpacingType;
		typedef typename BSplineTransformType::OriginType									OriginType;
		typedef typename BSplineTransformType::BulkTransformType					BulkTransformType;
		typedef typename BSplineTransformType::BulkTransformPointer				BulkTransformPointer;
		typedef typename BSplineTransformType::WeightsFunctionType				WeightsFunctionType;
		typedef typename BSplineTransformType::WeightsType								WeightsType;
		typedef typename BSplineTransformType::ContinuousIndexType				ContinuousIndexType;
		typedef typename BSplineTransformType::ParameterIndexArrayType		ParameterIndexArrayType;

		/** Method to transform a point, 1 argument. */
		virtual OutputPointType TransformPoint( const InputPointType & point ) const;

		/**  Method to transform a point, 5 arguments. */
		virtual void TransformPoint(
			const InputPointType &inputPoint,
			OutputPointType &outputPoint,
			WeightsType &weights,
			ParameterIndexArrayType &indices,
			bool &inside ) const;
		
	protected:
		
		/** The constructor. */
		DeformationFieldRegulizerForBSpline();
		/** The destructor. */
		virtual ~DeformationFieldRegulizerForBSpline();
		
	private:
		
		/** The private constructor. */
		DeformationFieldRegulizerForBSpline( const Self& );	// purposely not implemented
		void operator=( const Self& );											// purposely not implemented
		
	}; // end class DeformationFieldRegulizerForBSpline
		
		
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDeformationFieldRegulizerForBSpline.hxx"
#endif

#endif // end #ifndef __itkDeformationFieldRegulizerForBSpline_H__


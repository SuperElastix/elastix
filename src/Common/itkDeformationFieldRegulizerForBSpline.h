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
		typedef typename TBSplineTransform::PixelType									PixelType;
		typedef typename TBSplineTransform::ImageType									ImageType;
		typedef typename TBSplineTransform::ImagePointer							ImagePointer;
		typedef typename TBSplineTransform::RegionType								RegionType;
		typedef typename TBSplineTransform::IndexType									IndexType;
		typedef typename TBSplineTransform::SizeType									SizeType;
		typedef typename TBSplineTransform::SpacingType								SpacingType;
		typedef typename TBSplineTransform::OriginType								OriginType;
		typedef typename TBSplineTransform::BulkTransformType					BulkTransformType;
		typedef typename TBSplineTransform::BulkTransformPointer			BulkTransformPointer;
		typedef typename TBSplineTransform::WeightsFunctionType				WeightsFunctionType;
		typedef typename TBSplineTransform::WeightsType								WeightsType;
		typedef typename TBSplineTransform::ContinuousIndexType				ContinuousIndexType;
		typedef typename TBSplineTransform::ParameterIndexArrayType		ParameterIndexArrayType;

		/**  Method to transform a point. */
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


#ifndef __itkBSplineTransformGrouper_h
#define __itkBSplineTransformGrouper_h

#include "itkTransformGrouper.h"


namespace itk
{
	

	/**
	 * ******************* BSplineTransformGrouper ******************
	 *
	 * This class
	 */

	template <class TBSplineTransform>
		class BSplineTransformGrouper :
			public TransformGrouper<TBSplineTransform>
		{
		public:
			
			/** Standard itk.*/
			typedef BSplineTransformGrouper								Self;
			typedef TransformGrouper<TBSplineTransform>		Superclass;
			typedef SmartPointer< Self >									Pointer;
			typedef SmartPointer< const Self >						ConstPointer;
			
			/** New method for creating an object using a factory. */
			itkNewMacro( Self );
			
			/** Itk Type info */
			itkTypeMacro( BSplineTransformGrouper, TransformGrouper );
			
			/** Input and Output space dimension */
			itkStaticConstMacro( InputSpaceDimension, unsigned int, Superclass::InputSpaceDimension );
			itkStaticConstMacro( OutputSpaceDimension, unsigned int, Superclass::OutputSpaceDimension );
			
			/** typedefs inherited from Superclass */			
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

			typedef typename Superclass::ObjectType									ObjectType;
			typedef typename Superclass::GrouperDescriptionType			GrouperDescriptionType;
			typedef typename Superclass::PtrToGrouper								PtrToGrouper;
			typedef typename Superclass::GrouperMapType							GrouperMapType; 
			typedef typename Superclass::MapEntryType								MapEntryType;
			typedef typename Superclass::InitialTransformType				InitialTransformType;
			typedef typename Superclass::InitialTransformPointer		InitialTransformPointer;
			typedef typename Superclass::InitialInputPointType			InitialInputPointType;
			typedef typename Superclass::InitialOutputPointType			InitialOutputPointType;
			
			/** Other typedef's.*/
			typedef typename TBSplineTransform::BulkTransformType			BulkTransformType;
			typedef typename TBSplineTransform::BulkTransformPointer	BulkTransformPointer;
			
		protected:
			
			BSplineTransformGrouper();
			virtual ~BSplineTransformGrouper() {};
			
			/** Overrided by this class: */
			virtual void SetCurrentGrouper( const GrouperDescriptionType & name );
		
		private:
			
			BSplineTransformGrouper( const Self& );	// purposely not implemented
			void operator=( const Self& );					// purposely not implemented

		}; // end class BSplineTransformGrouper
		
		
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineTransformGrouper.hxx"
#endif


#endif // end #ifndef __itkBSplineTransformGrouper_h


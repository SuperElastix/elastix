#ifndef __itkTransformGrouper_h
#define __itkTransformGrouper_h

#include "itkTransformGrouperInterface.h"
#include "itkTransform.h"

#include <map>


namespace itk
{
	
	
	/**
	 * ********************* TransformGrouper ***********************
	 *
	 * This class
	 */
	
	template <class TAnyITKTransform>
	class TransformGrouper :
		public TAnyITKTransform,
		public TransformGrouperInterface
	{
	public:
		
		/** Standard itk.*/
		typedef TransformGrouper						Self;
		typedef TAnyITKTransform						Superclass1;
		typedef TransformGrouperInterface				Superclass2;
		typedef SmartPointer< Self >				Pointer;
		typedef SmartPointer< const Self >	ConstPointer;
		
		/** New method for creating an object using a factory.*/
		itkNewMacro( Self );
		
		/** Itk Type info */
		itkTypeMacro( TransformGrouper, TransformGrouperInterface );
		
		/** Input and Output space dimension.*/
		itkStaticConstMacro( InputSpaceDimension, unsigned int, Superclass1::InputSpaceDimension );
		itkStaticConstMacro( OutputSpaceDimension, unsigned int, Superclass1::OutputSpaceDimension );
		
		/** typedefs inherited from Superclass1.*/			
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
		
		/** typedefs inherited from Superclass2.*/
		typedef Superclass2::ObjectType								ObjectType;
		typedef Superclass2::GrouperDescriptionType		GrouperDescriptionType;
		
		/** A pointer to a function that 'eats' a const InputPointType  & 
		 * and spits out an OutputPointType.
		 */
		typedef OutputPointType (Self::*PtrToGrouper)(const InputPointType  & ) const;
		
		/** A map of pointers to groupers and their description.*/
		typedef std::map< GrouperDescriptionType, PtrToGrouper>		GrouperMapType; 
		typedef typename GrouperMapType::value_type								MapEntryType;			
		typedef itk::Transform<
			ScalarType,
			itkGetStaticConstMacro( InputSpaceDimension ),
			itkGetStaticConstMacro( OutputSpaceDimension ) >				InitialTransformType;
		typedef typename InitialTransformType::Pointer						InitialTransformPointer;
		typedef typename InitialTransformType::InputPointType			InitialInputPointType;
		typedef typename InitialTransformType::OutputPointType		InitialOutputPointType;
		
		/**  Method to transform a point. Calls the appropriate Grouper */
		virtual OutputPointType TransformPoint( const InputPointType  & point ) const;
		
		/** Get a pointer to the InitialTransform */
		itkGetObjectMacro( InitialTransform, ObjectType );
		
		/** Set the InitialTransform */
		virtual void SetInitialTransform( ObjectType * _arg );
		
		/** Set the desired grouper. Add, Concatenate and
		 * NoInitialTransform are supported by default.
		 */
		virtual int SetGrouper( const GrouperDescriptionType & name );
		
		/** Get the name of the desired grouper  */
		virtual const GrouperDescriptionType & GetNameOfDesiredGrouper(void) const;
		
		/** Get the name of the actual (currently used) grouper */
		virtual const GrouperDescriptionType & GetNameOfCurrentGrouper(void) const;
		
		/** Adds a Grouper that could be used; returns 0 if successful */
		virtual int AddGrouperToMap( const GrouperDescriptionType & name, PtrToGrouper funcptr );
		
	protected:
		
		TransformGrouper();
		virtual ~TransformGrouper();
		
		/** .*/
		virtual void SetCurrentGrouper(const GrouperDescriptionType & name);
		
		/** .*/
		inline OutputPointType TransformPoint0(const InputPointType  & point ) const;
		
		/** Declaration of members.*/
		InitialTransformPointer m_InitialTransform;
		
		/** the map of grouper functions */
		GrouperMapType m_GrouperMap;
		
		/** The name of the grouper desired by the user. */
		GrouperDescriptionType m_NameOfDesiredGrouper;
		GrouperDescriptionType m_NameOfCurrentGrouper;
		
	private:
		
		TransformGrouper( const Self& );	// purposely not implemented
		void operator=( const Self& );		// purposely not implemented
		
		/** Methods to combine the TransformPoint functions of the 
		 * initial and the current transform.
		 *
		 * WARNING: these only work when the inputpointtype and the 
		 * outputpointtype of both transforms are the same!
		 */
		
		/** ADD: u(x) = u0(x) + u1(x) */
		inline OutputPointType Add( const InputPointType  & point ) const;
		
		/** CONCATENATE: u(x) = u1( u0(x) ) */
		inline OutputPointType Concatenate( const InputPointType  & point ) const;
		
		/** CURRENT ONLY: u(x) = u1(x) */
		inline OutputPointType NoInitialTransform( const InputPointType & point ) const;
		
		/**  A pointer to one of the functions Add, Concatenate and NoInitialTransform.*/
		PtrToGrouper m_Grouper;
		
	}; // end class TransformGrouper
		
		
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTransformGrouper.hxx"
#endif


#endif // end #ifndef __itkTransformGrouper_h


#ifndef __itkCombinationTransform_h
#define __itkCombinationTransform_h

#include "itkTransform.h"
#include "itkExceptionObject.h"


namespace itk
{
	
	/**
	 * \class CombinationTransform
	 *
	 * \brief This class combines two transforms: an 'initial transform'
	 * with a 'current transform'.
	 *
	 * The CombinationTransform class combines an initial transform \f$T_0\f$ with a
	 * current transform \f$T_1\f$.
	 *
	 * Two methods of combining the transforms are supported:
	 * \li Addition: \f$T(x) = T_0(x) + T_1(x)\f$
	 * \li Composition: \f$T(x) = T_1( T_0(x) )\f$
	 *
	 * The TransformPoint(), the GetJacobian() and the GetInverse() methods
	 * depend on this setting.
	 *
	 * If the transform is used in a registration framework,
	 * the initial transform is assumed constant, and the current
	 * transform is assumed to be the transform that is optimised.
	 * So, the transform parameters of the CombinationTransform are the
	 * parameters of the CurrentTransform \f$T_1\f$.
	 *
	 * Note: It is mandatory to set a current transform. An initial transform
	 * is not mandatory. 
	 * 
	 * \ingroup Transforms
	 */
	
	template < typename TScalarType, unsigned int NDimensions = 3 >
	class CombinationTransform :
		public Transform<TScalarType, NDimensions, NDimensions>
	{
	public:
		
		/** Standard itk.*/
		typedef CombinationTransform				Self;
		typedef Transform< TScalarType,
			NDimensions, NDimensions > 				Superclass;
		typedef SmartPointer< Self >				Pointer;
		typedef SmartPointer< const Self >	ConstPointer;
		
		/** New method for creating an object using a factory.*/
		itkNewMacro( Self );
		
		/** Itk Type info */
		itkTypeMacro( CombinationTransform, Transform );
		
		/** Input and Output space dimension.*/
		itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );
		
		/** Typedefs inherited from Superclass.*/			
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
				
		/** A pointer to a function that 'eats' a const InputPointType  & 
		 * and spits out an OutputPointType. */
		typedef OutputPointType (Self::*TransformPointFunctionPointer)( const InputPointType & ) const;

		/** A pointer to a function that 'eats' a const InputPointType &
		 * and spits out a const JacobianType & */
		typedef const JacobianType & (Self::*GetJacobianFunctionPointer)( const InputPointType & ) const;
		
		/** Typedefs for the InitialTransform */
		typedef Superclass																				InitialTransformType;
		typedef typename InitialTransformType::Pointer						InitialTransformPointer;
		typedef typename InitialTransformType::ConstPointer				InitialTransformConstPointer;
		
		/** Typedefs for the CurrentTransform */
		typedef Superclass																				CurrentTransformType;
		typedef typename CurrentTransformType::Pointer						CurrentTransformPointer;
				
		/** Set/Get a pointer to the InitialTransform */
		itkGetConstObjectMacro( InitialTransform, InitialTransformType );
		virtual void SetInitialTransform( const InitialTransformType * _arg );
		
		/** Set/Get a pointer to the CurrentTransform. Make sure to set
		 * the CurrentTransform before functions like TransformPoint(),
		 * GetJacobian(), SetParameters() etc. are called. */
		itkGetObjectMacro( CurrentTransform, CurrentTransformType );
		virtual void SetCurrentTransform( CurrentTransformType * _arg );

		/** Control the way transforms are combined. */
		virtual void SetUseAddition(bool _arg);
		itkGetMacro(UseAddition, bool);

		/** Control the way transforms are combined. */
		virtual void SetUseComposition(bool _arg);
		itkGetMacro(UseComposition, bool);

		/**  Method to transform a point. */
		virtual OutputPointType TransformPoint( const InputPointType  & point ) const;
		
		/** Compute the Jacobian of the transformation */
		virtual const JacobianType & GetJacobian(const InputPointType & point ) const;

		/** Return the number of parameters that completely define the CurrentTransform  */
		virtual unsigned int GetNumberOfParameters(void) const;
		
		/** Get the Transformation Parameters from the CurrentTransform. */
		virtual const ParametersType& GetParameters(void) const;

		/** Set the transformation parameters in the CurrentTransform	*/
		virtual void SetParameters( const ParametersType & param);

		/** Set the transformation parameters in the CurrentTransform.  
		* This method forces the transform to copy the parameters. */
		virtual void SetParametersByValue ( const ParametersType & param );

		/** Return the inverse \f$T^{-1}\f$ of the transform.
		*  This is only possible when:
		* - both the inverses of the initial and the current transform
		*   are defined, and Composition is used:
		*   \f$T^{-1}(y) = T_0^{-1} ( T_1^{-1}(y) )\f$
		* - No initial transform is used and the current transform is defined.
		* In all other cases this function returns false and does not provide
		* an inverse transform. An exception is thrown when no CurrentTransform
		* is set.
		*/
		virtual bool GetInverse(Self* inverse) const;
					
	protected:
		
		CombinationTransform();
		virtual ~CombinationTransform(){};
				
		/** Declaration of members.*/
		InitialTransformConstPointer m_InitialTransform;
		CurrentTransformPointer m_CurrentTransform;

		/** Set the SelectedTransformPointFunction and the 
		 * SelectedGetJacobianFunction */
		virtual void UpdateCombinationMethod(void);

		/** Throw an exception. */
		virtual void NoCurrentTransformSet(void) const throw (ExceptionObject);

	  /**  A pointer to one of the following functions:
		 * - TransformPointUseAddition,
		 * - TransformPointUseComposition,
		 * - TransformPointNoCurrentTransform
		 * - TransformPointNoInitialTransform. 
		 */
		TransformPointFunctionPointer m_SelectedTransformPointFunction;

		/**  A pointer to one of the following functions:
		 * - GetJacobianUseAddition,
		 * - GetJacobianUseComposition,
		 * - GetJacobianNoCurrentTransform
		 * - GetJacobianNoInitialTransform. 
		 */
		GetJacobianFunctionPointer m_SelectedGetJacobianFunction;

		/** Methods to combine the TransformPoint functions of the 
		 * initial and the current transform.	 */
		
		/** ADDITION: \f$T(x) = T_0(x) + T_1(x) - x\f$ */
		inline OutputPointType TransformPointUseAddition( const InputPointType  & point ) const;
		
		/** COMPOSITION: \f$T(x) = T_1( T_0(x) )\f$ 
		 * \warning: assumes that input and output point type are the same */
		inline OutputPointType TransformPointUseComposition( const InputPointType  & point ) const;
		
		/** CURRENT ONLY: \f$T(x) = T_1(x)\f$ */
		inline OutputPointType TransformPointNoInitialTransform( const InputPointType & point ) const;
		
		/** NO CURRENT TRANSFORM SET: throw an exception. */
		inline OutputPointType TransformPointNoCurrentTransform( const InputPointType & point ) const;
	
		/** Methods to compute the Jacobian */
		
		/** ADDITION: \f$J(x) = J_1(x)\f$ */
		inline const JacobianType & GetJacobianUseAddition( const InputPointType  & point ) const;
		
		/** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$ 
		 * \warning: assumes that input and output point type are the same */
		inline const JacobianType & GetJacobianUseComposition( const InputPointType  & point ) const;
		
		/** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
		inline const JacobianType & GetJacobianNoInitialTransform( const InputPointType & point ) const;
		
		/** NO CURRENT TRANSFORM SET: throw an exception */
		inline const JacobianType & GetJacobianNoCurrentTransform( const InputPointType & point ) const;
	
		/** How to combine the transforms.  */
		bool m_UseAddition;
		bool m_UseComposition;
				
	private:
		
		CombinationTransform( const Self& );	// purposely not implemented
		void operator=( const Self& );		// purposely not implemented
				
	}; // end class CombinationTransform
		
		
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCombinationTransform.hxx"
#endif


#endif // end #ifndef __itkCombinationTransform_h


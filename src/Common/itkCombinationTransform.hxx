/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkCombinationTransform_hxx
#define __itkCombinationTransform_hxx

#include "itkCombinationTransform.h"


namespace itk
{
	
	/**
	 * ************************ Constructor	*************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		CombinationTransform<TScalarType, NDimensions>
		::CombinationTransform() : Superclass(NDimensions,1)
	{
		/** Initialize.*/
		this->m_InitialTransform = 0;
		this->m_CurrentTransform = 0;
				
		this->m_UseAddition = true;
		this->m_UseComposition = false;
		this->m_SelectedTransformPointFunction = 
			&Self::TransformPointNoCurrentTransform;
		this->m_SelectedGetJacobianFunction =
			&Self::GetJacobianNoCurrentTransform;

	} // end Constructor
	

	/**
   * ***************** GetNumberOfParameters **************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		unsigned int CombinationTransform<TScalarType, NDimensions>::	
		GetNumberOfParameters(void) const
	{ 
		/** Return the number of parameters that completely define the
		 * m_CurrentTransform  */

		if ( this->m_CurrentTransform.IsNotNull() )
		{
			return this->m_CurrentTransform->GetNumberOfParameters();
		}
		else
		{
			/** Throw an exception */
			this->NoCurrentTransformSet();
			/** dummy return */
			return this->m_Parameters.GetSize();
		}

	} //end GetNumberOfParameters

  /**
   * ***************** IsLinear **************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		bool CombinationTransform<TScalarType, NDimensions>::	
		IsLinear(void) const
	{ 
		bool currentLinear = true;
		if ( this->m_CurrentTransform.IsNotNull() )
		{
	 	  currentLinear = this->m_CurrentTransform->IsLinear();
		}
		
    bool initialLinear = true;
    if ( this->m_InitialTransform.IsNotNull() )
    {
      initialLinear = this->m_InitialTransform->IsLinear();
    }

    return ( currentLinear && initialLinear ); 
  } //end IsLinear


	/**
   * ***************** GetParameters **************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		const typename CombinationTransform<TScalarType, NDimensions>::ParametersType &
		CombinationTransform<TScalarType, NDimensions>::	
		GetParameters(void) const
	{ 
		/** Return the parameters that completely define the m_CurrentTransform  */

		if ( this->m_CurrentTransform.IsNotNull() )
		{
			return this->m_CurrentTransform->GetParameters();
		}
		else
		{
			/** Throw an exception */
			this->NoCurrentTransformSet();
			/** dummy return */
			return this->m_Parameters;
		}

	} //end GetParameters

	
	/**
   * ***************** SetParameters **************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>::	
		SetParameters(const ParametersType & param)
	{ 
		/** Set the parameters in the m_CurrentTransfom  */

		if ( this->m_CurrentTransform.IsNotNull() )
		{
			this->Modified();
			this->m_CurrentTransform->SetParameters(param);
		}
		else
		{
			/** Throw an exception */
			this->NoCurrentTransformSet();
		}

	} //end SetParameters


	/**
   * ***************** SetParametersByValue **************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>::	
		SetParametersByValue(const ParametersType & param)
	{ 
		/** Set the parameters in the m_CurrentTransfom. */

		if ( this->m_CurrentTransform.IsNotNull() )
		{
			this->Modified();
			this->m_CurrentTransform->SetParametersByValue(param);
		}
		else
		{
			/** Throw an exception */
			this->NoCurrentTransformSet();
		}

	} //end SetParametersByValue


	/**
   * ***************** GetInverse **************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		bool CombinationTransform<TScalarType, NDimensions>::	
		GetInverse(Self * inverse) const
	{ 
		if(!inverse)
    {
			/** Inverse transformation cannot be returned into nothingness */
      return false;
		}
		else if ( this->m_CurrentTransform.IsNull() )
		{ 
			/** No current transform has been set. 
			 * Throw an exception */
			this->NoCurrentTransformSet();
			return false;
		}
		else if ( this->m_InitialTransform.IsNull() )
		{
			/** No Initial transform, so call the CurrentTransform's
			 * implementation */
			return this->m_CurrentTransform->GetInverse(inverse);
    }
		else if ( this->m_UseAddition )
		{
			/** No generic expression exists for the inverse of (T0+T1)(x) */
			return false;
		}
		else //UseComposition
		{
      /** The initial transform and the current transform have been set
			 * and UseComposition is set to true. 
			 * The inverse transform IT is defined by:
			 *	IT ( T1(T0(x) ) = x
			 * So:
			 *	IT(y) = T0^{-1} ( T1^{-1} (y) )
			 * which is of course only defined when the inverses of both
			 * the initial and the current transforms are defined.
			 */

			/** Try create the inverse of the initial transform */
			InitialTransformPointer inverseT0 = InitialTransformType::New();
			bool T0invertable = this->m_InitialTransform->GetInverse(inverseT0);

			if (T0invertable)
			{				
				/** Try to create the inverse of the current transform */
				CurrentTransformPointer inverseT1 = CurrentTransformType::New();
				bool T1invertable = this->m_CurrentTransform->GetInverse(inverseT1);

				if (T1invertable)
				{
					/** The transform can be inverted! */
					inverse->SetUseComposition(true);
					inverse->SetInitialTransform(inverseT1);
					inverse->SetCurrentTransform(inverseT0);
					return true;
				}
				else
				{
					/** The initial transform is invertible, but the current one not. */
					return false;
				}
			}
			else
			{
				/** The initial transform is not invertible */
				return false;
			}

		} //end else: UseComposition.

	} //end GetInverse


	/**
   * ****************** TransformPoint ****************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		typename CombinationTransform<TScalarType, NDimensions>::OutputPointType
		CombinationTransform<TScalarType, NDimensions>::	
		TransformPoint(const InputPointType  & point ) const
	{ 
		/** Call the selected TransformPointFunction */
		return ((*this).*m_SelectedTransformPointFunction)(point);

	} // end TransformPoint


	/**
   * ****************** GetJacobian ****************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		const typename CombinationTransform<TScalarType, NDimensions>::JacobianType &
		CombinationTransform<TScalarType, NDimensions>::	
		GetJacobian(const InputPointType  & point) const
	{ 
		/** Call the selected Grouper */
		return ((*this).*m_SelectedGetJacobianFunction)(point);

	} // end TransformPoint


	/**
	 * ******************* SetInitialTransform **********************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>
		::SetInitialTransform( const InitialTransformType * _arg )
	{
		/** Set the the initial transform and call the 
		 * the UpdateCombinationMethod */
		if ( this->m_InitialTransform != _arg )
		{
			this->m_InitialTransform = _arg;
			this->Modified();
			this->UpdateCombinationMethod();
		}

	} // end SetInitialTransform


	/**
	 * ******************* SetCurrentTransform **********************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>
		::SetCurrentTransform( CurrentTransformType * _arg )
	{
		/** Set the the current transform and call the 
		 * the UpdateCombinationMethod */
		if ( this->m_CurrentTransform != _arg )
		{
			this->m_CurrentTransform = _arg;
			this->Modified();
			this->UpdateCombinationMethod();
		}

	} // end SetCurrentTransform


	/**
	 * ********************** SetUseAddition **********************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>
		::SetUseAddition( bool _arg )
	{
		/** Set the UseAddition and UseComposition bools and call the 
		 * the UpdateCombinationMethod */
		if ( this->m_UseAddition != _arg )
		{
			this->m_UseAddition = _arg;
			this->m_UseComposition = !_arg;
			this->Modified();
			this->UpdateCombinationMethod();
		}

	} // end SetUseAddition


	/**
	 * ********************** SetUseComposition *******************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>
		::SetUseComposition( bool _arg )
	{
		/** Set the UseAddition and UseComposition bools and call the 
		 * the UpdateCombinationMethod */
		if ( this->m_UseComposition != _arg )
		{
			this->m_UseComposition = _arg;
			this->m_UseAddition = !_arg;
			this->Modified();
			this->UpdateCombinationMethod();
		}

	} // end SetUseComposition


	/**
	 * ****************** UpdateCombinationMethod ********************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>
		::UpdateCombinationMethod( void )
	{
    /** Update the m_SelectedTransformPointFunction and the
		 * the m_SelectedGetJacobianFunction
		 */
		if ( this->m_CurrentTransform.IsNull() )
		{
			this->m_SelectedTransformPointFunction = 
				&Self::TransformPointNoCurrentTransform;
			this->m_SelectedGetJacobianFunction = 
				&Self::GetJacobianNoCurrentTransform;
		}
		else if ( this->m_InitialTransform.IsNull() )
		{
			this->m_SelectedTransformPointFunction = 
				&Self::TransformPointNoInitialTransform;
			this->m_SelectedGetJacobianFunction = 
				&Self::GetJacobianNoInitialTransform;
		}
		else if ( this->m_UseAddition )
		{
			this->m_SelectedTransformPointFunction = 
				&Self::TransformPointUseAddition;
			this->m_SelectedGetJacobianFunction = 
				&Self::GetJacobianUseAddition;
		}
		else
		{
			this->m_SelectedTransformPointFunction = 
				&Self::TransformPointUseComposition;
			this->m_SelectedGetJacobianFunction = 
				&Self::GetJacobianUseComposition;
		}

	} // end UpdateCombinationMethod


	/**
	 * ************* NoCurrentTransformSet **********************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		void CombinationTransform<TScalarType, NDimensions>::
	  NoCurrentTransformSet( void ) const throw (ExceptionObject)
	{				
		itkExceptionMacro(
			<< "No current transform set in the CombinationTransform" );
	} //end NoCurrentTransformSet

	
	/**
	 * ************* ADDITION: T(x) = T0(x) + T1(x) - x **********************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		typename CombinationTransform<TScalarType, NDimensions>::OutputPointType
		CombinationTransform<TScalarType, NDimensions>::
	  TransformPointUseAddition( const InputPointType  & point ) const
	{				
		/** The Initial transform */		 
		OutputPointType out0 = 
			this->m_InitialTransform->TransformPoint( point );
		
		/** The Current transform */
		OutputPointType out =
			this->m_CurrentTransform->TransformPoint( point );
		
		/** Both added together */
		for ( unsigned int i=0; i < SpaceDimension; i++ )
		{
			out[ i ] += ( out0[ i ] - point[ i ] );
		}
		
		return out;
	} // end TransformPointUseAddition


	/**
	 * **************** COMPOSITION: T(x) = T1( T0(x) ) *************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		typename CombinationTransform<TScalarType, NDimensions>::OutputPointType
		CombinationTransform<TScalarType, NDimensions>::
		TransformPointUseComposition( const InputPointType  & point ) const
		{
			return this->m_CurrentTransform->TransformPoint( 
				this->m_InitialTransform->TransformPoint( point ) );
		} // end TransformPointUseComposition
			
			
	/**
	 * **************** CURRENT ONLY: T(x) = T1(x) ******************
	 */

	template <typename TScalarType, unsigned int NDimensions>
	typename CombinationTransform<TScalarType, NDimensions>::OutputPointType
		CombinationTransform<TScalarType, NDimensions>::	
		TransformPointNoInitialTransform( const InputPointType & point ) const
	{
		return this->m_CurrentTransform->TransformPoint( point );
	} // end TransformPointNoInitialTransform


	/**
	 * ******** NO CURRENT TRANSFORM SET: throw an exception ******************
	 */

	template <typename TScalarType, unsigned int NDimensions>
	typename CombinationTransform<TScalarType, NDimensions>::OutputPointType
		CombinationTransform<TScalarType, NDimensions>::	
		TransformPointNoCurrentTransform( const InputPointType & point ) const
	{
		/** throw an exception */
		this->NoCurrentTransformSet(); 
		/** dummy return */
		return point;
	} // end TransformPointNoCurrentTransform


	/**
	 * ************* ADDITION: J(x) = J1(x) ***************************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		const typename CombinationTransform<TScalarType, NDimensions>::JacobianType &
		CombinationTransform<TScalarType, NDimensions>::
	  GetJacobianUseAddition( const InputPointType  & point ) const
	{				
		return this->m_CurrentTransform->GetJacobian( point );
	} // end GetJacobianUseAddition


	/**
	 * **************** COMPOSITION: J(x) = J1( T0(x) ) *************
	 */

	template <typename TScalarType, unsigned int NDimensions>
		const typename CombinationTransform<TScalarType, NDimensions>::JacobianType &
		CombinationTransform<TScalarType, NDimensions>::
		GetJacobianUseComposition( const InputPointType  & point ) const
		{
			return this->m_CurrentTransform->GetJacobian( 
				this->m_InitialTransform->TransformPoint( point ) );
		} // end GetJacobianUseComposition
			
			
	/**
	 * **************** CURRENT ONLY: J(x) = J1(x) ******************
	 */

	template <typename TScalarType, unsigned int NDimensions>
	const typename CombinationTransform<TScalarType, NDimensions>::JacobianType &
		CombinationTransform<TScalarType, NDimensions>::	
		GetJacobianNoInitialTransform( const InputPointType & point ) const
	{
		return this->m_CurrentTransform->GetJacobian( point );
	} // end GetJacobianNoInitialTransform


	/**
	 * ******** NO CURRENT TRANSFORM SET: throw an exception ******************
	 */

	template <typename TScalarType, unsigned int NDimensions>
	const typename CombinationTransform<TScalarType, NDimensions>::JacobianType &
		CombinationTransform<TScalarType, NDimensions>::	
		GetJacobianNoCurrentTransform( const InputPointType & point ) const
	{
		/** throw an exception */
		this->NoCurrentTransformSet(); 
		/** dummy return */
		return this->m_Jacobian;
	} // end GetJacobianNoCurrentTransform



} // end namespace itk


#endif // end #ifndef __itkCombinationTransform_hxx


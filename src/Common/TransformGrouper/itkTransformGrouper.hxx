#ifndef __itkTransformGrouper_hxx
#define __itkTransformGrouper_hxx

#include "itkTransformGrouper.h"


namespace itk
{
	
	/**
	 * ************************ Constructor	*************************
	 */

	template <class TAnyITKTransform>
		TransformGrouper<TAnyITKTransform>::TransformGrouper()
	{
		/** Initialize.*/
		m_InitialTransform = 0;
				
		/** Add the default groupers to the map.*/
		this->AddGrouperToMap( "NoInitialTransform", &Self::NoInitialTransform );
		this->AddGrouperToMap( "Add", &Self::Add );
		this->AddGrouperToMap( "Concatenate", &Self::Concatenate );
				
		/** Set the default grouper.*/
		this->SetGrouper( "Add" ); 

	} // end Constructor
	
	/**
	 * *********************** Destructor ***************************
	 */

	template <class TAnyITKTransform>
		TransformGrouper<TAnyITKTransform>::~TransformGrouper()
	{
		//nothing
	} // end Destructor


	/**
	 * *********************** SetCurrentGrouper ********************
	 */

	template <class TAnyITKTransform>
		void TransformGrouper<TAnyITKTransform>::
		SetCurrentGrouper( const GrouperDescriptionType & name )
	{
		/** Set the name and put it in GrouperMap.*/
		m_NameOfCurrentGrouper = name;
		m_Grouper = m_GrouperMap[ name ];

	} // end SetCurrentGrouper


	/**
	 * *********************** TransformPoint0 **********************
	 */

	template <class TAnyITKTransform>
		typename TransformGrouper<TAnyITKTransform>::OutputPointType
		TransformGrouper<TAnyITKTransform>::
		TransformPoint0( const InputPointType  & point ) const
	{
		return m_InitialTransform->TransformPoint( point );

	} // end TransformPoint0


	/**
	 * ************* ADD: u(x) = u0(x) + u1(x) **********************
	 */

	template <class TAnyITKTransform>
		typename TransformGrouper<TAnyITKTransform>::OutputPointType
		TransformGrouper<TAnyITKTransform>::
	  Add( const InputPointType  & point ) const
	{				
		/** The initial transform
		 *
		 * It is assumed that the InitialTransform has been set.*/
		InitialOutputPointType out0 = this->TransformPoint0( point );
		
		/** The Current transform */
		OutputPointType out = this->Superclass1::TransformPoint( point );
		
		/** Both added together */
		for ( unsigned int i=0; i < InputSpaceDimension; i++ )
		{
			out[ i ] += ( out0[ i ] - point[ i ] );
		}
		
		return out;
				
	} // end Add


	/**
	 * **************** CONCATENATE: u(x) = u1( u0(x) ) *************
	 */

	template <class TAnyITKTransform>
		typename TransformGrouper<TAnyITKTransform>::OutputPointType
		TransformGrouper<TAnyITKTransform>::
		Concatenate( const InputPointType  & point ) const
		{
			InitialOutputPointType out0 = this->TransformPoint0( point );
			return this->Superclass1::TransformPoint( out0 );

		} // end Concatenate
			
			
	/**
	 * **************** CURRENT ONLY: u(x) = u1(x) ******************
	 */

	template <class TAnyITKTransform>
	typename TransformGrouper<TAnyITKTransform>::OutputPointType
		TransformGrouper<TAnyITKTransform>::	
		NoInitialTransform( const InputPointType & point ) const
	{
		return this->Superclass1::TransformPoint( point );

	} // end NoInitialTransform


	/**
   * ****************** TransformPoint ****************************
	 *
	 * Method to transform a point. Calls the appropriate Grouper 
	 */

	template <class TAnyITKTransform>
		typename TransformGrouper<TAnyITKTransform>::OutputPointType
		TransformGrouper<TAnyITKTransform>::	
		TransformPoint(const InputPointType  & point ) const
	{ 
		/** Call the selected Grouper */
		return ((*this).*m_Grouper)(point);

	} // end TransformPoint


	/**
	 * ******************* SetInitialTransform **********************
	 */

	template <class TAnyITKTransform>
		void TransformGrouper<TAnyITKTransform>
		::SetInitialTransform( ObjectType * _arg )
	{
		/** .*/
		if ( m_InitialTransform != _arg )
		{
			m_InitialTransform = dynamic_cast<InitialTransformType *>( _arg );
			this->Modified();
			if ( _arg )
			{
				/** if not zero, try to set the DesiredGrouper  */
				this->SetGrouper( m_NameOfDesiredGrouper );
			}
			else
			{
				/** if set to zero, set the Grouper temporarily back to "NoInitialTransform" */
				this->SetCurrentGrouper( "NoInitialTransform" );
				/** but don't set the name of the desired grouper!
				* because it is not desired by the user! */
			}
		}

	} // end SetInitialTransform


	/**
	 * *********************** SetGrouper ***************************
	 */

	template <class TAnyITKTransform>
		int TransformGrouper<TAnyITKTransform>::	
		SetGrouper( const GrouperDescriptionType & name )
	{
		/** .*/
		if ( m_GrouperMap.count( name ) == 0 ) 
		{
			std::cerr << "Error: " << std::endl;
			std::cerr << name << " - This grouper is not installed!" << std::endl;
			return 1;
		}
		else
		{
			m_NameOfDesiredGrouper = name;

			/** Set the Grouper to the desired grouper, but only if the Transform
			* is non-zero, or if the desired grouper is "NoInitialTransform" */
			if ( m_InitialTransform ) 
			{
				this->SetCurrentGrouper( name );
			}
			else 
			{
				this->SetCurrentGrouper( "NoInitialTransform" );
			}
	
			return 0;
		}
													
	} // end SetGrouper


	/**
	 * *************** GetNameOfDesiredGrouper **********************
	 */

	template <class TAnyITKTransform>
		const typename TransformGrouper<TAnyITKTransform>::GrouperDescriptionType &
		TransformGrouper<TAnyITKTransform>::	
		GetNameOfDesiredGrouper(void) const
	{
		return m_NameOfDesiredGrouper;

	} // end GetNameOfDesiredGrouper


	/**
	 * ****************** GetNameOfCurrentGrouper *******************
	 */

	template <class TAnyITKTransform>
		const typename TransformGrouper<TAnyITKTransform>::GrouperDescriptionType &
		TransformGrouper<TAnyITKTransform>::	
		GetNameOfCurrentGrouper(void) const
	{
		return m_NameOfCurrentGrouper;

	} // end GetNameOfCurrentGrouper


	/**
	 * ********************* AddGrouperToMap ************************
	 *
	 * Adds a Grouper that could be used; returns 0 if successful.
	 */

	template <class TAnyITKTransform>
		int TransformGrouper<TAnyITKTransform>::	
		AddGrouperToMap( const GrouperDescriptionType & name, PtrToGrouper funcptr )
	{
				
		if ( m_GrouperMap.count( name ) ) //==1
		{
			std::cerr << "Error: " << std::endl;
			std::cerr << name << " - This grouper has already been installed!" << std::endl;
			return 1;
		}
		else
		{
			m_GrouperMap.insert(	MapEntryType( name,	funcptr	)	);
			return 0;
		}
		
	} // end AddGrouperToMap


} // end namespace itk


#endif // end #ifndef __itkTransformGrouper_hxx


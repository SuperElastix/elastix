#ifndef __itkBSplineTransformGrouper_hxx
#define __itkBSplineTransformGrouper_hxx

#include "itkBSplineTransformGrouper.h"


namespace itk
{
	

	/**
	 * ************************ Constructor	*************************
	 */

	template <class TBSplineTransform>
		BSplineTransformGrouper<TBSplineTransform>::BSplineTransformGrouper()
	{
		/** Add a key to the GrouperMap, pointing to the same grouper
		 * as "NoInitialTransform".
		 */

		this->AddGrouperToMap(
			"InternalBSplineTransformAdder",
			this->m_GrouperMap[ "NoInitialTransform" ] );
					
	} // end Constructor
	

	/**
	 * *********************** SetCurrentGrouper ********************
	 */

	template <class TBSplineTransform>
		void BSplineTransformGrouper<TBSplineTransform>::
		SetCurrentGrouper( const GrouperDescriptionType & name )
	{
		if ( name == "Add" )
		{
			this->Superclass::SetCurrentGrouper( "InternalBSplineTransformAdder" );
			this->SetBulkTransform( dynamic_cast<BulkTransformType *>(
				this->GetInitialTransform() )    );
		}
		else
		{
			this->Superclass::SetCurrentGrouper( name );
			this->SetBulkTransform( 0 );
		}

	} // end SetCurrentGrouper


} // end namespace itk


#endif // end #ifndef __itkBSplineTransformGrouper_hxx


#ifndef __elxMultiResolutionRegistration_HXX__
#define __elxMultiResolutionRegistration_HXX__

#include "elxMultiResolutionRegistration.h"

namespace elastix
{
using namespace itk;
	

	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MultiResolutionRegistration<TElastix>
		::MultiResolutionRegistration()
	{
		//nothing	
	} // end Constructor
	

	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void MultiResolutionRegistration<TElastix>
		::BeforeRegistration(void)
	{	
		/** Get the components from m_Elastix and set them.*/
		this->SetComponents();

		/** Set the number of resolutions.*/		
		unsigned int numberOfResolutions = 3;
		m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
		this->SetNumberOfLevels( numberOfResolutions );
				
		/** Set the FixedImageRegion.*/
		
		/** Make the fixedImagePointer non-const, to allow for calling ->Update() */
		FixedImageType * nonconstFixedImage = const_cast<FixedImageType *>( this->GetFixedImage() );
		
		/** Update and Set the image region. */
		try
		{
			nonconstFixedImage->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			/** Add information to the exception. */
			excp.SetLocation( "MultiResolutionRegistration - BeforeRegistration()" );
			std::string err_str = excp.GetDescription();
			err_str += "\nError occured while updating region info of the fixed image.\n";
			excp.SetDescription( err_str );
			/** Pass the exception to an higher level. */
			throw excp;
		}

		/** Set the fixedImageRegion. */
		this->SetFixedImageRegion( nonconstFixedImage->GetBufferedRegion() );
		
	} // end BeforeRegistration
	
	
	/**
	 * *********************** SetComponents ************************
	 */

	template <class TElastix>
		void MultiResolutionRegistration<TElastix>
		::SetComponents(void)
	{	
		/** Get the component from m_Elastix (as Object::Pointer),
		 * cast it to the appropriate type and set it in 'this'.
		 */
		this->SetFixedImagePyramid(
			dynamic_cast<FixedImagePyramidType *>(
				m_Elastix->GetFixedImagePyramid() 	)
		);

		this->SetInterpolator(
			dynamic_cast<InterpolatorType *>(
				m_Elastix->GetInterpolator()	)
		);

		this->SetMetric(
			dynamic_cast<MetricType *>(
				m_Elastix->GetMetric()	)
		);

		this->SetMovingImagePyramid(
			dynamic_cast<MovingImagePyramidType *>(
				m_Elastix->GetMovingImagePyramid()		)
		);

		this->SetOptimizer(
			dynamic_cast<OptimizerType *>(
				m_Elastix->GetOptimizer() 	)
		);


		this->SetTransform(
			dynamic_cast<TransformType *>(
				m_Elastix->GetTransform() 	)
		);


	} // end SetComponents


} // end namespace elastix

#endif // end #ifndef __elxMultiResolutionRegistration_HXX__


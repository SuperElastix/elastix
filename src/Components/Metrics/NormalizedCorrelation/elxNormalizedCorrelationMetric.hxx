#ifndef __elxNormalizedCorrelationMetric_HXX__
#define __elxNormalizedCorrelationMetric_HXX__

#include "elxNormalizedCorrelationMetric.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		NormalizedCorrelationMetric<TElastix>::NormalizedCorrelationMetric()
	{
		/** Initialize.*/
/*		m_FixedMaskImageReader = 0;
		m_MovingMaskImageReader = 0;
		m_FixedMaskCaster = 0;
		m_MovingMaskCaster = 0;*/

	} // end Constructor

	
	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void NormalizedCorrelationMetric<TElastix>::
		BeforeRegistration(void)
	{		
		/** Read masks if necessary.*/
		
		/** Read fixed mask.*/
/*		std::string fixedMaskFileName = m_Configuration->
			GetCommandLineArgument( "-fMask" );
		if ( !( fixedMaskFileName.empty() ) )
		{
			m_FixedMaskImageReader	= FixedMaskImageReaderType::New();
			m_FixedMaskCaster				= FixedMaskCastFilterType::New();
			
			m_FixedMaskImageReader->SetFileName( fixedMaskFileName.c_str() );
			m_FixedMaskCaster->SetInput( m_FixedMaskImageReader->GetOutput() );
			/** Do the casting.*
			try
			{
				m_FixedMaskCaster->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

		}  // end if ( fixed mask present )*/
		
		/** Read moving mask.*/
/*		std::string movingMaskFileName = m_Configuration->
			GetCommandLineArgument( "-mMask" );
		if ( !( movingMaskFileName.empty() ) )
		{
			m_MovingMaskImageReader	= MovingMaskImageReaderType::New();
			m_MovingMaskCaster			= MovingMaskCastFilterType::New();
			
			m_MovingMaskImageReader->SetFileName( movingMaskFileName.c_str() );
			m_MovingMaskCaster->SetInput( m_MovingMaskImageReader->GetOutput() );
			/** Do the casting.*
			try
			{
				m_MovingMaskCaster->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

		}  // end if ( moving mask present )*/
			
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void NormalizedCorrelationMetric<TElastix>::
		BeforeEachResolution(void)
	{
		/** \todo adapt SecondOrderRegularisationMetric.
		 * Set alpha, which balances the similarity and deformation energy
		 * E_total = (1-alpha)*E_sim + alpha*E_def.
		 * metric->SetAlpha( config.GetAlpha(level) );
		 */

	/** Erode and Set masks if necessary.*/

		/** Read the number of resolutions from the ParameterFile.*
		unsigned int numberOfResolutions = 3;
		m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
		
		/** Erode and Set the fixed Mask if necessary *
		if ( this->GetFixedMask() )
		{
			/**
			*  If more resolution levels are used, the image is subsampled. Before
			*  subsampling the image is smoothed with a Gaussian filter, with variance
			*  (schedule/2)^2. The 'schedule' depends on the resolution level.
			*  The lowest resolution level has a schedule of 2^(nr_of_levels-1).
			*  The 'radius' of the convolution filter is roughly twice the standard deviation.
			*	 Thus, the parts in the edge with size 'radius' are influenced by the background.
			*
		
			this->SetFixedMask
			(
				( m_FixedMaskCaster->GetOutput() )->Erode
				( 
					static_cast<unsigned long>
					(
						ceil( pow( 2.0,	numberOfResolutions - level - 1 ) ) + 1
					),
					1
				)
			);
		} // end if fixedmask present

		/** Erode and Set the moving Mask if necessary *
		if ( this->GetMovingMask() )
		{
			/**
			*	Same story as before. Now the size the of the eroding element is doubled.
			* This is because the gradient of the moving image is used for calculating
			* the derivative of the metric. 
			*
			
			this->SetMovingMask
			(
				( m_MovingMaskCaster->GetOutput() )->Erode
				( 
					static_cast<unsigned long>
					(
						ceil( pow ( 2.0, numberOfResolutions - level ) ) + 1
					),
					1
				)
			);
		} // end if movingmask present
		*/
		
	} // end BeforeEachResolution
	
	
} // end namespace elastix


#endif // end #ifndef __elxNormalizedCorrelationMetric_HXX__


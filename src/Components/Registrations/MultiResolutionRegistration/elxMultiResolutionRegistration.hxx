#ifndef __elxMultiResolutionRegistration_HXX__
#define __elxMultiResolutionRegistration_HXX__

#include "elxMultiResolutionRegistration.h"


namespace elastix
{
using namespace itk;
	
	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void MultiResolutionRegistration<TElastix>
		::BeforeRegistration(void)
	{	
		/** Get the components from this->m_Elastix and set them.*/
		this->SetComponents();

		/** Set the number of resolutions.*/		
		unsigned int numberOfResolutions = 3;
		this->m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
		this->SetNumberOfLevels( numberOfResolutions );
				
		/** Set the FixedImageRegion.*/
		
		/** Make sure the fixed image is up to date. */
		try
		{
			this->GetElastix()->GetFixedImage()->Update();
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
		this->SetFixedImageRegion( this->GetElastix()->GetFixedImage()->GetBufferedRegion() );
		
	} // end BeforeRegistration


  /**
	 * ******************* BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MultiResolutionRegistration<TElastix>
		::BeforeEachResolution(void)
	{	
    /** Get the current resolution level. */
		unsigned int level = this->GetCurrentLevel();

    /** Do erosion, or just reset the original masks in the metric, or
     * do nothing when no masks are used */
    this->UpdateMasks(level);		

  } // end BeforeEachResolution
	
	
	/**
	 * *********************** SetComponents ************************
	 */

	template <class TElastix>
		void MultiResolutionRegistration<TElastix>
		::SetComponents(void)
	{	
		/** Get the component from this-GetElastix() (as elx::...BaseType *),
		 * cast it to the appropriate type and set it in 'this'. */

    this->SetFixedImage( this->GetElastix()->GetFixedImage() );
    this->SetMovingImage( this->GetElastix()->GetMovingImage() );

    this->SetFixedImagePyramid( this->GetElastix()->
      GetElxFixedImagePyramidBase()->GetAsITKBaseType() );

    this->SetMovingImagePyramid( this->GetElastix()->
      GetElxMovingImagePyramidBase()->GetAsITKBaseType() );

    this->SetInterpolator( this->GetElastix()->
      GetElxInterpolatorBase()->GetAsITKBaseType() );

    this->SetMetric( this->GetElastix()->
      GetElxMetricBase()->GetAsITKBaseType() );

    this->SetOptimizer(  dynamic_cast<OptimizerType*>(
      this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType() )   );

    this->SetTransform( this->GetElastix()->
      GetElxTransformBase()->GetAsITKBaseType() );

    if ( this->GetElastix()->GetFixedMask() )
    {
      FixedMaskSpatialObjectPointer fixedMask = FixedMaskSpatialObjectType::New();
      fixedMask->SetImage( this->GetElastix()->GetFixedMask() );
      this->GetMetric()->SetFixedImageMask( fixedMask );
    }

    if ( this->GetElastix()->GetMovingMask() )
    {
      MovingMaskSpatialObjectPointer movingMask = MovingMaskSpatialObjectType::New();
      movingMask->SetImage( this->GetElastix()->GetMovingMask() );
      this->GetMetric()->SetMovingImageMask( movingMask );
    }

	} // end SetComponents


 	/**
	 * ************************* UpdateMasks ************************
   **/

  template <class TElastix>
		void MultiResolutionRegistration<TElastix>
    ::UpdateMasks( unsigned int level )
	{
    if ( ( ! this->GetElastix()->GetFixedMask()  ) && 
         ( ! this->GetElastix()->GetMovingMask() )    )
    {
      return;
    }

    /** Read whether mask erosion is wanted */
    std::string erosionOrNot = "true";
		this->m_Configuration->ReadParameter( erosionOrNot, "ErodeMask", 0, true );
    this->m_Configuration->ReadParameter( erosionOrNot, "ErodeMask", level );

    if ( erosionOrNot != "true" )
    {
      /** Set the original masks again, in case they were replaced by
       * eroded versions in the previous resolution */
      if ( this->GetElastix()->GetFixedMask() )
      {
        FixedMaskSpatialObjectPointer fixedMask = FixedMaskSpatialObjectType::New();
        fixedMask->SetImage( this->GetElastix()->GetFixedMask() );
        this->GetMetric()->SetFixedImageMask( fixedMask );
      }

      if ( this->GetElastix()->GetMovingMask() )
      {
        MovingMaskSpatialObjectPointer movingMask = MovingMaskSpatialObjectType::New();
        movingMask->SetImage( this->GetElastix()->GetMovingMask() );
        this->GetMetric()->SetMovingImageMask( movingMask );
      }

      return ;
    }
    
    /** Create and start timer, to time the erosion. */
    TimerPointer timer = TimerType::New();
    timer->StartTimer();

		/** Erode and set the fixed mask if necessary.  */
		if ( this->GetElastix()->GetFixedMask() )
		{	
      FixedMaskErodeFilterPointer erosion = FixedMaskErodeFilterType::New();
      erosion->SetInput( this->GetElastix()->GetFixedMask() );
      erosion->SetSchedule( this->GetFixedImagePyramid()->GetSchedule() );
      erosion->SetIsMovingMask( false );
      erosion->SetResolutionLevel( level );

			/** Set output of the erosion to fixedImageMaskAsImage. */
      FixedMaskImagePointer fixedMaskAsImage = erosion->GetOutput();
						
			/** Do the erosion. */
			try
			{
				fixedMaskAsImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MultiResolutionRegistration - UpdateMasks()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError while eroding the fixed mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

      /** Release some memory */
      fixedMaskAsImage->DisconnectPipeline();
     
      /** Convert image to spatial object and put it in the metric. */
      FixedMaskSpatialObjectPointer fixedMask = FixedMaskSpatialObjectType::New();
      fixedMask->SetImage( fixedMaskAsImage );
      this->GetMetric()->SetFixedImageMask( fixedMask );
	
		} // end if fixed mask present

    /** Erode and set the moving mask if necessary.  */
		if ( this->GetElastix()->GetMovingMask() )
		{	
      MovingMaskErodeFilterPointer erosion = MovingMaskErodeFilterType::New();
      erosion->SetInput( this->GetElastix()->GetMovingMask() );
      erosion->SetSchedule( this->GetMovingImagePyramid()->GetSchedule() );
      erosion->SetIsMovingMask( true );
      erosion->SetResolutionLevel( level );

			/** Set output of the erosion to movingImageMaskAsImage. */
      MovingMaskImagePointer movingMaskAsImage = erosion->GetOutput();
						
			/** Do the erosion. */
			try
			{
				movingMaskAsImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MultiResolutionRegistration - UpdateMasks()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError while eroding the moving mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

      /** Release some memory */
      movingMaskAsImage->DisconnectPipeline();
     
      /** Convert image to spatial object and put it in the metric. */
      MovingMaskSpatialObjectPointer movingMask = MovingMaskSpatialObjectType::New();
      movingMask->SetImage( movingMaskAsImage );
      this->GetMetric()->SetMovingImageMask( movingMask );
	
		} // end if moving mask present

    /** Stop timer and print the elapsed time. */
		timer->StopTimer();
		elxout << "Eroding the masks took: "
		  << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;
		
	} // end UpdateMasks


} // end namespace elastix

#endif // end #ifndef __elxMultiResolutionRegistration_HXX__


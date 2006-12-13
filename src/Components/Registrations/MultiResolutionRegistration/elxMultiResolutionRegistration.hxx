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
    	/** Create timer. */
		TimerPointer timer = TimerType::New();

		/** Get the current resolution level. */
		unsigned int level = this->GetCurrentLevel();

		std::string erosionOrNot = "true";
		this->m_Configuration->ReadParameter( erosionOrNot, "ErodeMask", 0 );
    this->m_Configuration->ReadParameter( erosionOrNot, "ErodeMask", level );

		/** If there are any masks and if wanted, update them by erosion. */
		if ( this->GetElastix()->GetFixedMask() || this->GetElastix()->GetMovingMask() )
		{
			if ( erosionOrNot == "true" )
			{
				/** Start timer. */
				timer->StartTimer();

				/** Erode masks. */
				this->UpdateMasks( level );

				/** Stop timer and print the elapsed time. */
				timer->StopTimer();
				elxout << "Eroding the masks took: "
					<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;
			}
      else
      {
        /** Set them again, in case they were replaced by eroded versions in the
         * previous resolution */
        FixedMaskSpatialObjectPointer fixedMask = FixedMaskSpatialObjectType::New();
        fixedMask->SetImage( this->GetElastix()->GetFixedMask() );
        this->GetMetric()->SetFixedImageMask( fixedMask );

        MovingMaskSpatialObjectPointer movingMask = MovingMaskSpatialObjectType::New();
        movingMask->SetImage( this->GetElastix()->GetMovingMask() );
        this->GetMetric()->SetMovingImageMask( movingMask );
      }
		}
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

    FixedMaskSpatialObjectPointer fixedMask = FixedMaskSpatialObjectType::New();
    fixedMask->SetImage( this->GetElastix()->GetFixedMask() );
    this->GetMetric()->SetFixedImageMask( fixedMask );

    MovingMaskSpatialObjectPointer movingMask = MovingMaskSpatialObjectType::New();
    movingMask->SetImage( this->GetElastix()->GetMovingMask() );
    this->GetMetric()->SetMovingImageMask( movingMask );

	} // end SetComponents


 	/**
	 * ************************* UpdateMasks ************************
   * \todo this function is not really nicely programmed. Some
   * functionality could be defined in a separate function.
	 */

  template <class TElastix>
		void MultiResolutionRegistration<TElastix>
    ::UpdateMasks( unsigned int level )
	{
		/** Erode and set the fixed mask if necessary.  */
		if ( this->GetElastix()->GetFixedMask() )
		{
			/**
			 *  If more resolution levels are used, the image is subsampled. Before
			 *  subsampling the image is smoothed with a Gaussian filter, with variance
			 *  (schedule/2)^2. The 'schedule' depends on the resolution level.
			 *  The 'radius' of the convolution filter is roughly twice the standard deviation.
			 *	Thus, the parts in the edge with size 'radius' are influenced by the background.
			 */

			/** Create erosion-filters. */
			typename ErodeFilterTypeF::Pointer erosionF[ FixedImageDimension ];
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				erosionF[ i ] = ErodeFilterTypeF::New();
			}
      
			/** Declare radius-array and structuring element. */
			RadiusTypeF								radiusarrayF;
			StructuringElementTypeF		S_ballF;
			
			/** Setup the erosion pipeline. */
			erosionF[ 0 ]->SetInput( this->GetElastix()->GetFixedMask() );
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				/** Create the radius array. */
				radiusarrayF.Fill( 0 );
				unsigned int schedule = this->GetFixedImagePyramid()->GetSchedule()[ level ][ i ];
				unsigned long radius = static_cast<unsigned long>( schedule + 1 );
				radiusarrayF.SetElement( i, radius );

				/** Create the structuring element and set it into the erosion filter. */
				S_ballF.SetRadius( radiusarrayF );
				S_ballF.CreateStructuringElement();
				erosionF[ i ]->SetKernel( S_ballF );
        erosionF[ i ]->SetForegroundValue( itk::NumericTraits<MaskPixelType>::One );
        erosionF[ i ]->SetBackgroundValue( itk::NumericTraits<MaskPixelType>::Zero );
								
				/** Connect the pipeline. */
				if ( i > 0 ) erosionF[ i ]->SetInput( erosionF[ i - 1 ]->GetOutput() );			
			}

			/** Set output of the erosion to fixedImageMaskAsImage. */
      typename FixedMaskImageType::Pointer fixedMaskAsImage = 
        erosionF[ FixedImageDimension - 1 ]->GetOutput();
						
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
     
      /** Convert image to spatial object and put it in the metric. */
      FixedMaskSpatialObjectPointer fixedMask = FixedMaskSpatialObjectType::New();
      fixedMask->SetImage( fixedMaskAsImage );
      this->GetMetric()->SetFixedImageMask( fixedMask );
	
		} // end if fixed mask present

		/** Erode and set the moving mask if necessary.  */
		if ( this->GetElastix()->GetMovingMask() )
		{
			/**
			 *	Same story as before. Now the size the of the eroding element is doubled.
			 * This is because the gradient of the moving image is used for calculating
			 * the derivative of the metric. 
			 */

			/** Create erosion-filters. */
			typename ErodeFilterTypeM::Pointer erosionM[ MovingImageDimension ];
			for ( unsigned int i = 0; i < MovingImageDimension; i++ )
			{
				erosionM[ i ] = ErodeFilterTypeM::New();
			}
      
			/** Declare radius-array and structuring element. */
			RadiusTypeM								radiusarrayM;
			StructuringElementTypeM		S_ballM;
			
			/** Setup the erosion pipeline. */
			erosionM[ 0 ]->SetInput( this->GetElastix()->GetMovingMask() );
			for ( unsigned int i = 0; i < MovingImageDimension; i++ )
			{
				/** Create the radius array. */
				radiusarrayM.Fill( 0 );
				unsigned int schedule = this->GetMovingImagePyramid()->GetSchedule()[ level ][ i ];
				unsigned long radius = static_cast<unsigned long>( 2 * schedule + 1 );
				radiusarrayM.SetElement( i, radius );

				/** Create the structuring element and set it into the erosion filter. */
				S_ballM.SetRadius( radiusarrayM );
				S_ballM.CreateStructuringElement();
				erosionM[ i ]->SetKernel( S_ballM );
        erosionM[ i ]->SetForegroundValue( itk::NumericTraits<MaskPixelType>::One );
        erosionM[ i ]->SetBackgroundValue( itk::NumericTraits<MaskPixelType>::Zero );
								
				/** Connect the pipeline. */
				if ( i > 0 ) erosionM[ i ]->SetInput( erosionM[ i - 1 ]->GetOutput() );			
			}

			/** Set output of the erosion to movingImageMaskAsImage. */
      typename MovingMaskImageType::Pointer movingMaskAsImage = 
        erosionM[ MovingImageDimension - 1 ]->GetOutput();
						
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

			/** Convert image to spatial object and put it in the metric. */
      MovingMaskSpatialObjectPointer movingMask = MovingMaskSpatialObjectType::New();
      movingMask->SetImage( movingMaskAsImage );
      this->GetMetric()->SetMovingImageMask( movingMask );
	
		} // end if moving mask present

	} // end UpdateMasks


} // end namespace elastix

#endif // end #ifndef __elxMultiResolutionRegistration_HXX__

